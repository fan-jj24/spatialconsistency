#!/usr/bin/env python3
"""Transformers backend for the R4 summary reward model."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import sys
import threading


LOGGER = logging.getLogger(__name__)


class TransformersRewardModel:
    """Transformers adapter matching ``reward_model.RewardModel``'s batch API."""

    def __init__(self, reward_model_module, model_path, device="auto"):
        self._config = reward_model_module
        self._model_path = model_path
        self._requested_device = device
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._choice_token_ids = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @staticmethod
    def _normalize_summary(summary):
        return " ".join(summary.split()).casefold()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import (
                    AutoConfig,
                    AutoModelForCausalLM,
                    AutoTokenizer,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Transformers 回退后端需要安装 torch 和 transformers"
                ) from exc

            if self._requested_device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self._requested_device
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "指定了 Transformers CUDA 后端，但 torch 未检测到 CUDA"
                )

            LOGGER.info(
                "Loading R4 reward model with Transformers on %s: %s",
                device,
                self._model_path,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise RuntimeError(
                        "奖励模型 tokenizer 没有 pad_token 或 eos_token"
                    )
                tokenizer.pad_token = tokenizer.eos_token

            model_config = AutoConfig.from_pretrained(
                self._model_path,
                trust_remote_code=True,
            )
            if getattr(model_config, "vision_config", None) is not None:
                try:
                    from transformers import AutoModelForMultimodalLM
                except ImportError as exc:
                    raise RuntimeError(
                        "当前模型是完整多模态 Qwen3.5 检查点；请升级 "
                        "transformers 以获得 AutoModelForMultimodalLM"
                    ) from exc
                model_loader = AutoModelForMultimodalLM
            else:
                model_loader = AutoModelForCausalLM
            model = model_loader.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            model.to(device)
            model.eval()

            choice_token_ids = []
            all_letters = (*self._config.CHOICE_LETTERS, "U")
            for letter in all_letters:
                token_ids = tokenizer.encode(letter, add_special_tokens=False)
                if len(token_ids) != 1:
                    raise RuntimeError(
                        f"选项 {letter!r} 不是单个 tokenizer token: {token_ids!r}"
                    )
                token_id = token_ids[0]
                decoded = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if decoded != letter:
                    raise RuntimeError(
                        f"选项 token {token_id} 解码为 {decoded!r}，"
                        f"而不是 {letter!r}"
                    )
                choice_token_ids.append(token_id)
            if len(set(choice_token_ids)) != len(choice_token_ids):
                raise RuntimeError(f"A/B/C/D/U token 不唯一: {choice_token_ids!r}")

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._choice_token_ids = choice_token_ids

    def load(self):
        """Load the model before the HTTP service reports ready."""
        self._ensure_loaded()

    def close(self):
        """Release model references and the Transformers CUDA cache."""
        with self._load_lock:
            torch_module = self._torch
            device = self._device
            self._model = None
            self._tokenizer = None
            self._choice_token_ids = None
            self._torch = None
            self._device = None
        if (
            torch_module is not None
            and isinstance(device, str)
            and device.startswith("cuda")
        ):
            torch_module.cuda.empty_cache()

    def _truncate_summary(self, summary):
        token_ids = self._tokenizer.encode(
            summary,
            add_special_tokens=False,
            truncation=True,
            max_length=self._config.MAX_SUMMARY_TOKENS,
        )
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def _truncate_conclusion(self, sentence):
        token_ids = self._tokenizer.encode(
            sentence,
            add_special_tokens=False,
            truncation=True,
            max_length=self._config.REASONING_GATE_MAX_SENTENCE_TOKENS,
        )
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def _build_prompt(self, pred_summary, gt_summary):
        messages = [
            {"role": "system", "content": self._config.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._config.INPUT_TEMPLATE.format(
                    gt=gt_summary,
                    pred=pred_summary,
                ),
            },
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _build_reasoning_gate_prompt(self, sentence, option_a, option_b):
        messages = [
            {
                "role": "system",
                "content": self._config.REASONING_GATE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self._config.REASONING_GATE_INPUT_TEMPLATE.format(
                    sentence=sentence,
                    option_a=option_a,
                    option_b=option_b,
                ),
            },
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _infer_probabilities(self, prompts, letters, task_name):
        encoded = self._tokenizer(
            prompts,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for index, length in enumerate(lengths):
            if length > self._config.MAX_INPUT_TOKENS:
                raise RuntimeError(
                    f"{task_name} prompt {index} 截断后仍有 {length} tokens，超过 "
                    f"R4_MAX_INPUT_TOKENS={self._config.MAX_INPUT_TOKENS}；"
                    "请减小 R4_MAX_SUMMARY_TOKENS"
                )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._infer_lock, self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        if not generated.logits or len(generated.logits) != 1:
            raise RuntimeError(
                f"Transformers {task_name} 未返回一个 token 的原始 logits"
            )
        next_token_logits = generated.logits[0].float()
        all_letters = (*self._config.CHOICE_LETTERS, "U")
        token_id_by_letter = dict(zip(all_letters, self._choice_token_ids))
        selected_token_ids = [token_id_by_letter[letter] for letter in letters]
        choice_logits = next_token_logits[:, selected_token_ids]
        choice_probabilities = self._torch.softmax(choice_logits, dim=-1)

        log_normalizer = self._torch.logsumexp(next_token_logits, dim=-1)
        choice_log_mass = (
            self._torch.logsumexp(choice_logits, dim=-1) - log_normalizer
        )
        choice_masses = choice_log_mass.exp().detach().cpu().tolist()
        for index, mass in enumerate(choice_masses):
            if (
                not math.isfinite(mass)
                or mass < self._config.MIN_CHOICE_MASS
            ):
                raise RuntimeError(
                    f"{task_name} prompt {index} 的标签总概率 {mass:.6g} 低于 "
                    f"R4_MIN_CHOICE_MASS={self._config.MIN_CHOICE_MASS:.6g}"
                )

        return (
            choice_probabilities.detach().cpu().tolist(),
            choice_masses,
        )

    def _infer_scores(self, prompts):
        probabilities, _ = self._infer_probabilities(
            prompts, self._config.CHOICE_LETTERS, "R4"
        )
        return [
            sum(
                probability * weight
                for probability, weight in zip(
                    row, self._config.CHOICE_WEIGHTS
                )
            )
            for row in probabilities
        ]

    def score_summary(self, pred_summary, gt_summary):
        return self.score_summaries([(pred_summary, gt_summary)])[0]

    def score_summaries(self, pairs):
        if not pairs:
            return []
        results = [0.0] * len(pairs)
        pending = []
        for index, pair in enumerate(pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise TypeError(
                    f"pairs[{index}] 必须是 (pred_summary, gt_summary)"
                )
            pred_summary, gt_summary = pair
            if not isinstance(pred_summary, str) or not isinstance(gt_summary, str):
                raise TypeError(f"pairs[{index}] 的 summary 必须都是字符串")
            normalized_pred = self._normalize_summary(pred_summary)
            normalized_gt = self._normalize_summary(gt_summary)
            if not normalized_pred or not normalized_gt:
                continue
            if normalized_pred == normalized_gt:
                results[index] = 1.0
                continue
            pending.append((index, pred_summary, gt_summary))

        if not pending:
            return results
        self._ensure_loaded()
        prompts = [
            self._build_prompt(
                self._truncate_summary(pred_summary),
                self._truncate_summary(gt_summary),
            )
            for _, pred_summary, gt_summary in pending
        ]
        scores = self._infer_scores(prompts)
        if len(scores) != len(pending):
            raise RuntimeError("Transformers R4 返回数量错误")
        for (index, _, _), score in zip(pending, scores):
            results[index] = float(score)
        return results

    def classify_option_support(self, sentence, option_a, option_b):
        return self.classify_option_support_batch(
            [(sentence, option_a, option_b)]
        )[0]

    def classify_option_support_batch(self, items):
        if not items:
            return []
        normalized_items = []
        for index, item in enumerate(items):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise TypeError(
                    f"items[{index}] 必须是 (sentence, option_a, option_b)"
                )
            sentence, option_a, option_b = item
            if not all(isinstance(value, str) for value in item):
                raise TypeError(f"items[{index}] 的 reasoning gate 输入必须都是字符串")
            if not sentence.strip():
                raise ValueError(f"items[{index}] 的最后一句为空")
            if not option_a.strip() or not option_b.strip():
                raise ValueError(f"items[{index}] 的 A/B 选项为空")
            normalized_items.append((sentence, option_a, option_b))

        self._ensure_loaded()
        prompts = [
            self._build_reasoning_gate_prompt(
                self._truncate_conclusion(sentence), option_a, option_b
            )
            for sentence, option_a, option_b in normalized_items
        ]
        probabilities, choice_masses = self._infer_probabilities(
            prompts,
            self._config.REASONING_GATE_CHOICE_LETTERS,
            "reasoning gate",
        )
        results = []
        for row, choice_mass in zip(probabilities, choice_masses):
            class_probabilities = tuple(float(x) for x in row)
            supported_index = max(
                range(len(class_probabilities)),
                key=class_probabilities.__getitem__,
            )
            results.append(
                self._config.OptionSupportScore(
                    supported_option=self._config.REASONING_GATE_CHOICE_LETTERS[
                        supported_index
                    ],
                    unclear_probability=class_probabilities[2],
                    probabilities=class_probabilities,
                    choice_mass=float(choice_mass),
                )
            )
        return results


def create_reward_model(
    reward_model_module,
    model_path=None,
    backend="auto",
    device="auto",
):
    """Select vLLM when usable, otherwise return the Transformers adapter."""
    if backend not in {"auto", "vllm", "transformers"}:
        raise ValueError(f"未知 R4 后端: {backend!r}")

    selected = backend
    fallback_reason = None
    if selected == "auto":
        if sys.platform == "win32":
            fallback_reason = "检测到 Windows"
            selected = "transformers"
        else:
            try:
                import vllm  # noqa: F401 - verify importability
            except (ImportError, OSError, RuntimeError) as exc:
                fallback_reason = f"vLLM 不可用（{exc}）"
                selected = "transformers"
            else:
                selected = "vllm"

    if selected == "vllm":
        if model_path is not None:
            # RewardModel currently reads its path from the module-level
            # setting, so keep the explicit CLI path effective for vLLM too.
            reward_model_module.MODEL_LOCAL_PATH = model_path
        return reward_model_module.RewardModel(), selected

    configured_local_path = reward_model_module.MODEL_LOCAL_PATH
    if (
        model_path is None
        and configured_local_path
        and "R4_MODEL_LOCAL_PATH" not in os.environ
        and not Path(configured_local_path).exists()
    ):
        LOGGER.warning(
            "默认 R4 本地模型目录不存在（%s），改用模型名 %s",
            configured_local_path,
            reward_model_module.MODEL_NAME,
        )
        configured_local_path = None
    resolved_model_path = (
        model_path or configured_local_path or reward_model_module.MODEL_NAME
    )
    if fallback_reason is not None:
        LOGGER.warning("%s，自动回退到 Transformers", fallback_reason)
    return (
        TransformersRewardModel(
            reward_model_module,
            resolved_model_path,
            device=device,
        ),
        selected,
    )
