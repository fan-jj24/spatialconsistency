#!/usr/bin/env python3
"""Recompute validation rewards locally and render a self-contained HTML report.

The script reads ``{step}.jsonl`` files dumped by VERL, matches each prompt back
to the source JSONL/Parquet data, and uses the repository's *current* reward
logic.  R4 is evaluated directly with a local reward model through vLLM when
available, with a Transformers fallback for Windows and environments without
vLLM; no training-time HTTP reward server is required.

Example (one RTX 6000 Ada)::

    python3 plot_val_trends.py \
      --val_dir /data/run/val_generations \
      --data_root /data/RL1 \
      --model_path /models/Qwen3.5-9B \
      --cuda_visible_devices 0 \
      --out /data/run/val_trends.html
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import glob
import html as html_mod
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REWARD_MODULE = SCRIPT_DIR / "json_answer_reward.py"
DEFAULT_REWARD_MODEL_MODULE = SCRIPT_DIR / "reward_model.py"

METRICS = ["reward", "recorded_reward", "C", "R2", "R3", "R4"]
METRIC_DESC = {
    "reward": "按当前 json_answer_reward.py 公式重算的总奖励",
    "recorded_reward": "val JSONL 中原始记录的 score（用于对照）",
    "C": "answer 门控正确率（重算，0/1 均值）",
    "R2": "框区域奖励（bbox 并集 IoU；humanref 为匈牙利匹配 IoU）",
    "R3": "label 关键词召回 × IoU × 方向系数（-1=格式门控）",
    "R4": "本地奖励模型计算的 summary 语义一致性",
    "n": "匹配到的验证样本数",
}

# 12 色分类调色板（中等饱和度，明暗主题下均可读）
PALETTE = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#9333ea", "#b45309",
    "#475569", "#0d9488",
]

# ---------------------------------------------------------------------------
# Loading and matching source data
# ---------------------------------------------------------------------------


def load_module(path, name):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Python 模块不存在: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Python 模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def discover_steps(val_dir):
    """扫描 val_dir 下所有 {step}.jsonl，返回按 step 升序的文件列表。"""
    files = []
    for f in glob.glob(os.path.join(val_dir, "*.jsonl")):
        name = os.path.splitext(os.path.basename(f))[0]
        if re.fullmatch(r"\d+", name):
            files.append((int(name), f))
    files.sort()
    return files


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} JSON 无效: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no} 必须是 JSON object")
                records.append(value)
    return records


def _plain(value):
    """Convert Arrow/numpy containers to ordinary Python objects."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    return value


def _text_parts(value):
    value = _plain(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return [value["text"]]
        parts = []
        for key in ("content", "prompt", "messages"):
            if key in value:
                parts.extend(_text_parts(value[key]))
        return parts
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            parts.extend(_text_parts(item))
        return parts
    return []


_TEMPLATE_TOKEN_RE = re.compile(
    r"<\|(?:im|vision)_(?:start|end)\|>|<\|image_pad\|>"
    r"|<\|(?:eot_id|endoftext|end)\|>|<\|(?:assistant|user|system)\|>"
    r"|<image>|\[image\]|\s+",
    re.IGNORECASE,
)


def prompt_key(value):
    """Canonical text key shared by source prompts and decoded VERL inputs."""
    text = "\n".join(_text_parts(value)).strip().lower()
    # skip_special_tokens=True removes chat boundary tokens but commonly leaves
    # the role names as standalone lines in the decoded validation input.
    text = re.sub(r"(?m)^\s*(?:system|user|assistant)\s*$", "", text)
    text = re.sub(
        r"<\|im_start\|>\s*(?:system|user|assistant)?", "", text,
        flags=re.IGNORECASE,
    )
    return _TEMPLATE_TOKEN_RE.sub("", text)


def _ground_truth(row):
    for key in ("gts", "ground_truth", "gt"):
        if row.get(key) is not None:
            return _plain(row[key])
    reward_model = _plain(row.get("reward_model"))
    if isinstance(reward_model, dict):
        return _plain(reward_model.get("ground_truth"))
    return None


def _as_reward_string(value):
    value = _plain(value)
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _dataset_source(root, path):
    """Return IVG's source label: the dataset directory below data_root."""
    if root.is_file():
        return root.stem
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else path.stem


def _iter_source_rows(data_root):
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"数据地址不存在: {root}")
    files = ([root] if root.is_file() else sorted(root.rglob("*.jsonl"))
             + sorted(root.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"{root} 下未找到 .jsonl 或 .parquet")
    for path in files:
        if path.suffix.lower() == ".jsonl":
            rows = load_jsonl(path)
        elif path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("读取 Parquet 需要安装 pyarrow") from exc
            rows = pq.read_table(path).to_pylist()
        else:
            continue
        # Follow inspect_val_generations.py's grouping convention: ``source``
        # is the dataset directory directly below data_root, not data_source
        # and not an optional field embedded in a row.  A file directly under
        # data_root has no dataset directory, so use its stem as a fallback.
        dataset_source = _dataset_source(root, path)
        for row in rows:
            if isinstance(row, dict):
                yield row, dataset_source, path


@dataclass(frozen=True)
class SourceEntry:
    key: str
    data_source: str
    source: str
    ground_truth: str
    path: str


class SourceIndex:
    def __init__(self, entries):
        self.by_key = defaultdict(list)
        for entry in entries:
            self.by_key[entry.key].append(entry)
        self.keys_by_length = sorted(self.by_key, key=len, reverse=True)
        self.cache = {}

    def match(self, record):
        # A val_generations JSONL usually contains data_source, but that only
        # selects the reward route.  It cannot identify the original dataset
        # directory, so always trace the prompt back through the source index.
        explicit = record.get("data_source")
        explicit = explicit if isinstance(explicit, str) and explicit else ""
        key = prompt_key(record.get("input", record.get("prompt", "")))
        gt = _as_reward_string(_ground_truth(record))
        cache_key = (key, gt, explicit)
        if cache_key in self.cache:
            return self.cache[cache_key]

        candidates = list(self.by_key.get(key, ()))
        if not candidates and key:
            # A decoded chat-template input normally contains the original source
            # prompt plus role markers.  Longest match is the least ambiguous.
            for source_key in self.keys_by_length:
                if len(source_key) >= 24 and source_key in key:
                    candidates.extend(self.by_key[source_key])
                    break
        if gt and len(candidates) > 1:
            same_gt = [item for item in candidates if item.ground_truth == gt]
            if same_gt:
                candidates = same_gt
        # Match IVG's original behavior: after prompt and GT filtering, take
        # the first source hit when duplicate rows remain.
        result = None
        if candidates:
            item = candidates[0]
            data_source = explicit or item.data_source
            if data_source:
                result = SourceEntry(
                    item.key, data_source, item.source,
                    item.ground_truth, item.path,
                )
        self.cache[cache_key] = result
        return result


def build_index(data_root):
    entries = []
    skipped = 0
    for row, dataset_source, path in _iter_source_rows(data_root):
        key = prompt_key(row.get("prompt", row.get("input", row.get("messages", ""))))
        data_source = row.get("data_source")
        if not key:
            skipped += 1
            continue
        if not isinstance(data_source, str):
            data_source = ""
        entries.append(SourceEntry(
            key, data_source, dataset_source,
            _as_reward_string(_ground_truth(row)), str(path)
        ))
    if not entries:
        raise RuntimeError("源数据中没有可索引的 prompt 记录")
    print(f"  索引条目: {len(entries)}，跳过: {skipped}")
    return SourceIndex(entries)


# ---------------------------------------------------------------------------
# Current reward decomposition and batched local R4 inference
# ---------------------------------------------------------------------------


@dataclass
class Case:
    step: int
    source: str
    data_source: str
    output: str
    ground_truth: str
    recorded_reward: float | None
    metrics: dict = field(default_factory=dict)
    r4_pair: tuple | None = None


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def decompose_case(case, reward):
    ds = case.data_source
    output = case.output
    gt = case.ground_truth
    pred_obj = reward._parse_json_obj(output)
    gt_obj = reward._parse_json_obj(gt)

    if ds in reward.JSON_ANSWER_SOURCES:
        c = reward.score_json_answer(output, gt)
        case.metrics.update(C=c, reward=c)
        return
    if ds in reward.HUNGARIAN_IOU_SOURCES:
        r2 = reward.score_hungarian_iou(output, gt)
        case.metrics.update(R2=r2, reward=r2)
        return

    gt_entries = reward._parse_bbox_entries(gt)
    pred_entries = reward._parse_bbox_entries(output)
    matches = reward._hungarian_match(gt_entries, pred_entries)

    if ds in reward.SPATIAL_CONSISTENCY_BBOX_SOURCES:
        gt_answer = reward._normalize_answer(
            gt_obj.get("answer") if isinstance(gt_obj, dict) else None)
        pred_answer = reward._normalize_answer(
            pred_obj.get("answer") if isinstance(pred_obj, dict) else None)
        c = float(gt_answer is not None and pred_answer is not None
                  and gt_answer == pred_answer)
        case.metrics["C"] = c
        if not gt_entries:  # Positive branch never calls R4 in training.
            case.metrics["reward"] = c * (1.0 if not pred_entries else 0.2)
            return
        # For negative bbox cases, compute diagnostic R2/R3/R4 even when C=0.
        # The final reward remains zero through the C multiplier below.
        case.metrics["R2"] = reward._score_r2(gt_entries, pred_entries)
        case.metrics["R3"] = reward._score_r3(gt_entries, pred_entries, matches)
    elif ds in reward.SPATIAL_DETECTION_SOURCES:
        case.metrics["R2"] = reward._score_r2(gt_entries, pred_entries)
        case.metrics["R3"] = reward._score_r3(gt_entries, pred_entries, matches)
    else:
        raise ValueError(f"未知 data_source: {ds!r}")

    if not isinstance(pred_obj, dict) or not isinstance(gt_obj, dict):
        case.metrics["R4"] = 0.0
        return
    pred_summary = pred_obj.get("summary")
    gt_summary = gt_obj.get("summary")
    if not isinstance(pred_summary, str) or not isinstance(gt_summary, str):
        case.metrics["R4"] = 0.0
    elif not pred_summary.strip() or not gt_summary.strip():
        case.metrics["R4"] = 0.0
    else:
        case.r4_pair = (pred_summary, gt_summary)


class TransformersRewardModel:
    """Transformers adapter matching reward_model.RewardModel's batch API.

    The training reward server keeps its vLLM-only tensor-parallel behavior;
    this offline script can additionally run natively on Windows. Prompt and
    scoring constants come from reward_model.py to keep both paths aligned.
    """

    def __init__(self, reward_model_module, model_path, device="auto"):
        self._config = reward_model_module
        self._model_path = model_path
        self._requested_device = device
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._choice_token_ids = None

    @staticmethod
    def _normalize_summary(summary):
        return " ".join(summary.split()).casefold()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers 回退后端需要安装 torch 和 transformers"
            ) from exc

        if self._requested_device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self._requested_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("指定了 Transformers CUDA 后端，但 torch 未检测到 CUDA")

        print(f"加载 Transformers R4 模型到 {device}: {self._model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("奖励模型 tokenizer 没有 pad_token 或 eos_token")
            tokenizer.pad_token = tokenizer.eos_token

        model_config = AutoConfig.from_pretrained(
            self._model_path, trust_remote_code=True,
        )
        if getattr(model_config, "vision_config", None) is not None:
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError as exc:
                raise RuntimeError(
                    "当前模型是完整多模态 Qwen3.5 检查点；请升级 transformers "
                    "以获得 AutoModelForMultimodalLM"
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
        for letter in self._config.CHOICE_LETTERS:
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
                    f"选项 token {token_id} 解码为 {decoded!r}，而不是 {letter!r}"
                )
            choice_token_ids.append(token_id)
        if len(set(choice_token_ids)) != len(choice_token_ids):
            raise RuntimeError(f"A/B/C/D token 不唯一: {choice_token_ids!r}")

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._choice_token_ids = choice_token_ids

    def _truncate_summary(self, summary):
        token_ids = self._tokenizer.encode(
            summary,
            add_special_tokens=False,
            truncation=True,
            max_length=self._config.MAX_SUMMARY_TOKENS,
        )
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def _build_prompt(self, pred_summary, gt_summary):
        messages = [
            {"role": "system", "content": self._config.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._config.INPUT_TEMPLATE.format(
                    gt=gt_summary, pred=pred_summary,
                ),
            },
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _infer_scores(self, prompts):
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
                    f"R4 prompt {index} 截断后仍有 {length} tokens，超过 "
                    f"R4_MAX_INPUT_TOKENS={self._config.MAX_INPUT_TOKENS}；"
                    "请减小 R4_MAX_SUMMARY_TOKENS"
                )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
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
            raise RuntimeError("Transformers R4 未返回一个 token 的原始 logits")
        next_token_logits = generated.logits[0].float()
        choice_logits = next_token_logits[:, self._choice_token_ids]
        choice_probabilities = self._torch.softmax(choice_logits, dim=-1)

        # Same fail-closed check as the vLLM implementation: the four choices
        # must carry enough probability mass in the complete vocabulary.
        log_normalizer = self._torch.logsumexp(next_token_logits, dim=-1)
        choice_log_mass = self._torch.logsumexp(choice_logits, dim=-1) - log_normalizer
        choice_masses = choice_log_mass.exp().detach().cpu().tolist()
        for index, mass in enumerate(choice_masses):
            if not math.isfinite(mass) or mass < self._config.MIN_CHOICE_MASS:
                raise RuntimeError(
                    f"R4 prompt {index} 的 A/B/C/D 总概率 {mass:.6g} 低于 "
                    f"R4_MIN_CHOICE_MASS={self._config.MIN_CHOICE_MASS:.6g}"
                )

        weights = self._torch.tensor(
            self._config.CHOICE_WEIGHTS,
            dtype=choice_probabilities.dtype,
            device=choice_probabilities.device,
        )
        return (choice_probabilities * weights).sum(dim=-1).detach().cpu().tolist()

    def score_summaries(self, pairs):
        if not pairs:
            return []
        results = [0.0] * len(pairs)
        pending = []
        for index, pair in enumerate(pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise TypeError(f"pairs[{index}] 必须是 (pred_summary, gt_summary)")
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


def create_local_reward_model(reward_model_module, model_path, backend, device):
    """Select vLLM when usable, otherwise return the Transformers adapter."""
    selected = backend
    vllm_error = None
    if selected == "auto":
        if sys.platform == "win32":
            selected = "transformers"
        else:
            try:
                import vllm  # noqa: F401 - verify importability, not presence
            except (ImportError, OSError, RuntimeError) as exc:
                vllm_error = exc
                selected = "transformers"
            else:
                selected = "vllm"
    if selected == "vllm":
        return reward_model_module.RewardModel(), selected
    if vllm_error is not None:
        print(f"vLLM 不可用（{vllm_error}），自动回退到 Transformers")
    elif backend == "auto" and sys.platform == "win32":
        print("检测到 Windows，自动使用 Transformers R4 后端")
    return TransformersRewardModel(
        reward_model_module, model_path, device=device,
    ), selected


def score_r4_locally(cases, reward_model, reward, batch_size):
    pair_to_score = {}
    unique_pairs = []
    for case in cases:
        if case.r4_pair is not None and case.r4_pair not in pair_to_score:
            pair_to_score[case.r4_pair] = None
            unique_pairs.append(case.r4_pair)
    if unique_pairs:
        print(f"本地 R4 推理: {len(unique_pairs)} 个唯一 summary 对")
    for start in range(0, len(unique_pairs), batch_size):
        batch = unique_pairs[start:start + batch_size]
        scores = reward_model.score_summaries(batch)
        if len(scores) != len(batch):
            raise RuntimeError("reward_model.score_summaries 返回数量错误")
        for pair, score in zip(batch, scores):
            score = float(score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"R4 返回无效分数: {score!r}")
            pair_to_score[pair] = score
        print(f"  R4 {min(start + len(batch), len(unique_pairs))}/{len(unique_pairs)}")

    for case in cases:
        if "R4" not in case.metrics and case.r4_pair is not None:
            case.metrics["R4"] = pair_to_score[case.r4_pair]
        if "R4" not in case.metrics:
            continue
        if case.data_source in reward.SPATIAL_CONSISTENCY_BBOX_SOURCES:
            case.metrics["reward"] = case.metrics["C"] * (
                0.1 + 0.25 * case.metrics["R2"]
                + 0.25 * case.metrics["R3"] + 0.4 * case.metrics["R4"])
        elif case.data_source in reward.SPATIAL_DETECTION_SOURCES:
            case.metrics["reward"] = (
                0.25 * case.metrics["R2"] + 0.25 * case.metrics["R3"]
                + 0.5 * case.metrics["R4"])


def collect(val_dir, index, reward, local_model, batch_size, verbose=True):
    step_files = discover_steps(val_dir)
    if not step_files:
        raise SystemExit(f"ERROR: {val_dir} 下没有找到 {{step}}.jsonl 文件")

    cases = []
    unmatched = {}
    total_records = 0
    for step, path in step_files:
        records = load_jsonl(path)
        total_records += len(records)
        step_unmatched = 0
        for record in records:
            source_entry = index.match(record)
            if source_entry is None:
                step_unmatched += 1
                continue
            gt = _as_reward_string(_ground_truth(record)) or source_entry.ground_truth
            output = record.get("output", record.get("response", ""))
            case = Case(
                step=step,
                source=source_entry.source,
                data_source=source_entry.data_source,
                output=str(output or ""),
                ground_truth=gt,
                recorded_reward=_finite_float(record.get("score", record.get("reward"))),
            )
            decompose_case(case, reward)
            cases.append(case)
        unmatched[step] = step_unmatched
        if verbose:
            print(f"  step {step}: {len(records)} 条, "
                  f"匹配 {len(records) - step_unmatched}, 未匹配 {step_unmatched}")

    if not cases:
        raise RuntimeError(
            "没有匹配到任何验证样本；请检查 --data_root 是否对应"
            "生成这些 val JSONL 的源数据"
        )

    score_r4_locally(cases, local_model, reward, batch_size)

    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    ns = defaultdict(lambda: defaultdict(int))
    categories = {}
    for case in cases:
        categories.setdefault(case.source, case.data_source)
        ns[case.source][case.step] += 1
        values = dict(case.metrics)
        if case.recorded_reward is not None:
            values["recorded_reward"] = case.recorded_reward
        for metric, value in values.items():
            sums[case.source][metric][case.step] += value
            counts[case.source][metric][case.step] += 1

    series = {}
    for source, metrics in sums.items():
        series[source] = {}
        for metric, by_step in metrics.items():
            series[source][metric] = {
                step: total / counts[source][metric][step]
                for step, total in by_step.items()
            }
        series[source]["n"] = dict(ns[source])

    meta = {
        "total_records": total_records,
        "unmatched": unmatched,
        "cat_of_source": categories,
    }
    return [step for step, _ in step_files], series, meta

# ============================================================
# SVG 折线图（自包含，无 JS/CDN 依赖）
# ============================================================

def nice_ticks(vmin, vmax, target=5):
    """生成整齐的刻度列表。"""
    if vmax - vmin < 1e-12:
        return [vmin]
    raw = (vmax - vmin) / max(target, 1)
    mag = 10 ** int(math.floor(math.log10(raw)))
    for step in (1, 2, 2.5, 5, 10):
        tick = step * mag
        if raw <= tick:
            break
    start = math.ceil(vmin / tick) * tick
    ticks = []
    v = start
    while v <= vmax + tick * 1e-6:
        ticks.append(round(v, 10))
        v += tick
    return ticks


def build_svg_chart(steps, series, sources, colors, metric,
                    width=1100, height=320):
    """画一张折线图: x=step, 每个 source 一条线。

    返回 SVG 字符串。缺失点直接跳过（折线在缺失处断开）。
    """
    ml, mr, mt, mb = 52, 14, 12, 30

    # 数据范围
    all_vals = []
    for src in sources:
        all_vals.extend(series.get(src, {}).get(metric, {}).values())
    if not all_vals:
        return (f"<div class='empty'>该指标无数据</div>")
    vmin, vmax = min(all_vals), max(all_vals)
    if vmax - vmin < 1e-9:
        pad = max(abs(vmax) * 0.1, 0.05)
        vmin, vmax = vmin - pad, vmax + pad
    else:
        pad = (vmax - vmin) * 0.08
        vmin, vmax = vmin - pad, vmax + pad

    xs = sorted(steps)
    if len(xs) == 1:
        x_of = {xs[0]: ml + (width - ml - mr) / 2}
    else:
        span = width - ml - mr
        x_of = {s: ml + span * i / (len(xs) - 1) for i, s in enumerate(xs)}

    def y_of(v):
        return mt + (height - mt - mb) * (1 - (v - vmin) / (vmax - vmin))

    p = []
    p.append(f"<svg viewBox='0 0 {width} {height}' class='chart' "
             f"role='img' aria-label='{metric} trend'>")

    # 网格 + y 刻度
    for t in nice_ticks(vmin, vmax):
        y = y_of(t)
        p.append(f"<line x1='{ml}' y1='{y:.1f}' x2='{width - mr}' y2='{y:.1f}' "
                 f"class='grid'/>")
        label = f"{t:.2f}".rstrip("0").rstrip(".") if t != 0 else "0"
        p.append(f"<text x='{ml - 8}' y='{y + 4:.1f}' text-anchor='end' "
                 f"class='axis-text'>{label}</text>")

    # x 刻度（step 太多时抽稀）
    stride = max(1, math.ceil(len(xs) / 12))
    for i, s in enumerate(xs):
        if i % stride != 0 and i != len(xs) - 1:
            continue
        p.append(f"<text x='{x_of[s]:.1f}' y='{height - 8}' text-anchor='middle' "
                 f"class='axis-text'>{s}</text>")

    # 折线 + 数据点
    for src in sources:
        pts = series.get(src, {}).get(metric, {})
        if not pts:
            continue
        color = colors[src]
        seq = [(s, pts[s]) for s in xs if s in pts]
        if len(seq) >= 2:
            d = " ".join(f"{x_of[s]:.1f},{y_of(v):.1f}" for s, v in seq)
            p.append(f"<polyline points='{d}' fill='none' stroke='{color}' "
                     f"stroke-width='2' stroke-linejoin='round'/>")
        for s, v in seq:
            p.append(f"<circle cx='{x_of[s]:.1f}' cy='{y_of(v):.1f}' r='3.2' "
                     f"fill='{color}'><title>{html_mod.escape(src)} "
                     f"step={s} {metric}={v:.4f}</title></circle>")

    p.append("</svg>")
    return "".join(p)

# ============================================================
# HTML 页面生成
# ============================================================

CSS = """
:root {
  --bg: #f6f7f9; --card: #ffffff; --ink: #1a2233; --muted: #5b6472;
  --line: #e3e7ee; --accent: #2563eb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12151c; --card: #1a1f2b; --ink: #e8ebf2; --muted: #9aa3b2;
    --line: #2a3140; --accent: #60a5fa;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: var(--bg); color: var(--ink);
  font: 14px/1.5 -apple-system, "Segoe UI", "PingFang SC",
        "Microsoft YaHei", sans-serif;
}
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 8px; }
.sub { color: var(--muted); font-size: 12px; margin-bottom: 20px; }
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 6px; padding: 16px 20px; margin-bottom: 20px;
}
.meta-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px 24px; font-size: 13px;
}
.meta-grid b { font-weight: 600; }
.meta-grid span.k { color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: 12px; }
td.mono, th.mono { font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 10px 0 4px; }
.legend .item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
.legend .swatch { width: 14px; height: 3px; border-radius: 2px; }
.chart { width: 100%; height: auto; display: block; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .axis-text { fill: var(--muted); font-size: 11px;
  font-variant-numeric: tabular-nums; }
.empty { color: var(--muted); padding: 40px; text-align: center; }
.desc { color: var(--muted); font-size: 12px; margin: 2px 0 8px; }
"""


def _fmt(v, digits=4):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def build_summary_table(steps, series, sources, colors, cat_of_source):
    """最新 step 的各 source 指标汇总表。"""
    last = steps[-1]
    rows = []
    for src in sources:
        s = series.get(src, {})
        cells = "".join(
            f"<td class='mono'>{_fmt(s.get(m, {}).get(last))}</td>"
            for m in METRICS)
        n = s.get("n", {}).get(last)
        cat = cat_of_source.get(src, "?")
        color = colors[src]
        rows.append(
            f"<tr><td><span class='legend-swatch' style='display:inline-block;"
            f"width:10px;height:10px;border-radius:2px;background:{color};"
            f"margin-right:6px;vertical-align:-1px'></span>"
            f"{html_mod.escape(src)} "
            f"<span style='color:var(--muted);font-size:11px'>[{cat}]</span>"
            f"</td>{cells}<td class='mono'>{n if n is not None else '-'}</td></tr>")
    header = "".join(f"<th class='mono'>{m}</th>" for m in METRICS)
    return (f"<table><thead><tr><th>source (step={last})</th>{header}"
            f"<th class='mono'>n</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def build_html(steps, series, meta, val_dir, data_root, reward_module,
               model_path, r4_backend, out_path):
    cat_of_source = meta["cat_of_source"]
    # source 按类别 + 名字排序，颜色稳定分配
    sources = sorted(series.keys(),
                     key=lambda s: (cat_of_source.get(s, ""), s))
    colors = {src: PALETTE[i % len(PALETTE)] for i, src in enumerate(sources)}

    legend = "".join(
        f"<span class='item'><span class='swatch' style='background:{colors[s]}'></span>"
        f"{html_mod.escape(s)}</span>" for s in sources)

    unmatched_total = sum(meta["unmatched"].values())
    meta_html = f"""
    <div class='meta-grid'>
      <div><span class='k'>val 目录</span><br><b>{html_mod.escape(val_dir)}</b></div>
      <div><span class='k'>源数据</span><br><b>{html_mod.escape(data_root)}</b></div>
      <div><span class='k'>reward 模块</span><br><b>{html_mod.escape(reward_module)}</b></div>
      <div><span class='k'>R4 本地模型</span><br><b>{html_mod.escape(model_path)}</b></div>
      <div><span class='k'>R4 推理后端</span><br><b>{html_mod.escape(r4_backend)}</b></div>
      <div><span class='k'>step 数</span><br><b>{len(steps)}</b>
        （{steps[0]} → {steps[-1]}）</div>
      <div><span class='k'>source 数</span><br><b>{len(sources)}</b></div>
      <div><span class='k'>总记录数</span><br><b>{meta['total_records']}</b></div>
      <div><span class='k'>未匹配记录</span><br><b>{unmatched_total}</b></div>
    </div>"""

    charts = []
    for m in METRICS + ["n"]:
        svg = build_svg_chart(steps, series, sources, colors, m)
        charts.append(
            f"<div class='card'><h2>{m}</h2>"
            f"<div class='desc'>{METRIC_DESC[m]}</div>{svg}</div>")

    html = f"""<!DOCTYPE html>
<html lang='zh'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>val 指标走势</title>
<style>{CSS}</style>
</head>
<body>
<h1>val_generations 指标走势</h1>
<div class='sub'>各指标按 source 的逐步走势；reward 按当前公式重算，recorded_reward 为 JSONL 原值</div>

<div class='card'>
  <h2>概览</h2>
  {meta_html}
</div>

<div class='card'>
  <h2>最新 step 汇总</h2>
  {build_summary_table(steps, series, sources, colors, cat_of_source)}
</div>

<div class='card'>
  <h2>图例</h2>
  <div class='legend'>{legend}</div>
</div>

{''.join(charts)}

</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    return out_path

# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="用本地 R4 模型重算 val_generations 奖励并绘图")
    ap.add_argument("--val_dir", "--val-dir", required=True,
                    help="val_generations 目录（含 {step}.jsonl）")
    ap.add_argument("--data_root", "--data-root", required=True,
                    help="源数据目录/文件（递归读取 JSONL 和 Parquet）")
    ap.add_argument("--model_path", "--model-path", required=True,
                    help="本地 R4 奖励模型目录")
    ap.add_argument("--reward_module", "--reward-module",
                    default=str(DEFAULT_REWARD_MODULE),
                    help="训练用 json_answer_reward.py 路径（保证子项口径一致）")
    ap.add_argument("--reward_model_module", "--reward-model-module",
                    default=str(DEFAULT_REWARD_MODEL_MODULE),
                    help="本地 reward_model.py 路径")
    ap.add_argument("--cuda_visible_devices", "--cuda-visible-devices",
                    default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                    help="用于 R4 的 GPU，RTX 6000 Ada 单卡通常填 0")
    ap.add_argument("--batch_size", "--batch-size", type=int, default=32,
                    help="R4 推理批大小（默认 32）")
    ap.add_argument("--r4_backend", "--r4-backend",
                    choices=("auto", "vllm", "transformers"), default="auto",
                    help="R4 后端；auto 优先 vLLM，不可用时回退 Transformers")
    ap.add_argument("--transformers_device", "--transformers-device",
                    default="auto",
                    help="Transformers 设备，如 auto、cuda、cuda:0 或 cpu")
    ap.add_argument("--gpu_memory_utilization", "--gpu-memory-utilization",
                    type=float, default=0.8,
                    help="vLLM 单卡显存使用比例（默认 0.8）")
    ap.add_argument("--kv_cache_memory_bytes", "--kv-cache-memory-bytes",
                    type=int, default=0,
                    help="显式 KV cache 字节数；0 表示由 vLLM 自动分配")
    ap.add_argument("--out", default=None,
                    help="输出 HTML 路径（默认: val_dir/trends.html）")
    args = ap.parse_args()

    if args.batch_size <= 0:
        ap.error("--batch_size 必须为正整数")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        ap.error("--gpu_memory_utilization 必须在 (0, 1] 内")
    if args.kv_cache_memory_bytes < 0:
        ap.error("--kv_cache_memory_bytes 不能为负数")

    val_dir = str(Path(args.val_dir).expanduser().resolve())
    data_root = str(Path(args.data_root).expanduser().resolve())
    model_path = str(Path(args.model_path).expanduser().resolve())
    if not Path(val_dir).is_dir():
        ap.error(f"val 目录不存在: {val_dir}")
    if not Path(model_path).is_dir():
        ap.error(f"奖励模型目录不存在: {model_path}")
    if args.out is None:
        args.out = os.path.join(val_dir, "trends.html")
    out_path = str(Path(args.out).expanduser().resolve())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # reward_model.py reads these settings at import time.  TP=1 targets the
    # local RTX 6000 Ada; the training server's TP=8 defaults must not leak in.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["R4_MODEL_LOCAL_PATH"] = model_path
    os.environ["R4_VLLM_TENSOR_PARALLEL_SIZE"] = "1"
    os.environ["R4_VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
    os.environ["R4_VLLM_KV_CACHE_BYTES"] = str(args.kv_cache_memory_bytes)
    os.environ["R4_VLLM_MAX_NUM_SEQS"] = str(args.batch_size)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    reward = load_module(args.reward_module, "_plot_json_answer_reward")
    reward_model_module = load_module(
        args.reward_model_module, "_plot_local_reward_model")
    local_model, r4_backend = create_local_reward_model(
        reward_model_module,
        model_path,
        args.r4_backend,
        args.transformers_device,
    )
    print(f"reward 模块: {Path(args.reward_module).resolve()}")
    if r4_backend == "vllm":
        print(f"R4 本地模型: {model_path} (vLLM, GPU "
              f"{args.cuda_visible_devices}, TP=1)")
    else:
        print(f"R4 本地模型: {model_path} (Transformers, device="
              f"{args.transformers_device})")

    print("建立源数据索引（只读 prompt + reward_model 列）...")
    index = build_index(data_root)

    print(f"扫描 {val_dir} ...")
    steps, series, meta = collect(
        val_dir, index, reward, local_model, args.batch_size)
    print(f"共 {len(steps)} 个 step: {steps}")

    out_path = build_html(
        steps, series, meta, val_dir, data_root,
        str(Path(args.reward_module).resolve()), model_path, r4_backend, out_path)
    print(f"\n完成。报告: {out_path}")


if __name__ == "__main__":
    main()
