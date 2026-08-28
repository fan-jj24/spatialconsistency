#!/usr/bin/env python3
r"""Run Qwen/Gemini/optional InternVL rollouts, score them, and build HTML.

This script is intended for a Windows workstation with one CUDA GPU.  It reads
one VERL ``train.parquet`` through the same Hugging Face ``datasets`` loader
used by VERL, selects 100 rows by default, generates full responses with Qwen
and optional InternVL plus an answer/summary response with Gemini, then
calculates reward details and reuses the interactive human-accuracy page from
``annotate_val_cases.py``.

Example (PowerShell, one line)::

    python rollout_parquet_to_html.py --model_path D:\models\Qwen_RL \
      --internvl_model_path D:\models\InternVL3_5-14B-HF \
      --data_path D:\RL1\train.parquet --num_samples 100 \
      --out_dir D:\eval\train_100

Dependencies::

    pip install torch torchvision "transformers>=4.52.1" accelerate datasets pyarrow pillow requests oss2

Qwen, Gemini, and InternVL results are checkpointed separately.  By default, an
existing local checkpoint is replaced, while an existing complete Gemini
checkpoint is validated and reused without making any Gemini/OSS requests.  If
a local run is interrupted, repeat the command with ``--resume``.

Passing ``--internvl_model_path`` enables a third rollout.  In that mode,
existing Qwen/Gemini checkpoints are reused, and Qwen then InternVL run serially
on the local GPU while Gemini occupies its own parallel branch.

Generating a new Gemini checkpoint reads ``ak``, ``sk``, ``ep``, ``bn``, and
``api_key`` from a non-versioned ``key.py`` beside this script; reusing one does
not need that file.  After every enabled rollout finishes, this script starts
``reward_model_server.py``, waits for its health check, calculates all rewards,
and then stops the reward service.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import annotate_val_cases as annotation
import json_answer_reward as reward


GEMINI_MODEL = "gemini-3.5-flash"
# 本脚本只用于 inconsistent CoT 数据。两路使用独立的固定标识：
# 本地路由计算 C/R2/R3/R4/reward，Gemini 路由只计算 C/R4。
LOCAL_DATA_SOURCE = reward.INCONSISTENT_COT_LOCAL_EVAL_SOURCE
GEMINI_DATA_SOURCE = reward.INCONSISTENT_COT_GEMINI_EVAL_SOURCE
KEY_PATH = Path(__file__).with_name("key.py")


def _gemini_module():
    """Load optional Gemini/OSS dependencies only when Gemini is enabled."""
    try:
        import run_gemini
    except ImportError as exc:
        raise RuntimeError(
            "Gemini 调用需要 requests 和 oss2：pip install requests oss2"
        ) from exc
    return run_gemini


@dataclass
class EvalRow:
    order: int
    source_row: int
    row: dict[str, Any]
    ground_truth: str
    local_data_source: str
    gemini_data_source: str
    prediction: str = ""
    generation_error: str = ""
    gemini_prediction: str = ""
    gemini_error: str = ""
    internvl_prediction: str = ""
    internvl_error: str = ""
    local_scores: dict[str, float] | None = None
    gemini_scores: dict[str, float] | None = None
    internvl_scores: dict[str, float] | None = None
    evaluation_error: str = ""


@dataclass(frozen=True)
class GeminiCredentials:
    ak: str
    sk: str
    ep: str
    bn: str
    api_key: str


def _load_gemini_credentials(path: Path | None = None) -> GeminiCredentials:
    """Load Gemini and OSS credentials only from key.py beside this script."""
    key_path = path or KEY_PATH
    if not key_path.is_file():
        raise FileNotFoundError(
            f"找不到 {key_path}；请在其中定义 ak、sk、ep、bn、api_key"
        )
    spec = importlib.util.spec_from_file_location("_rollout_local_key", key_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {key_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ("ak", "sk", "ep", "bn", "api_key")
    values = {name: str(getattr(module, name, "")).strip() for name in names}
    missing = [name for name in names if not values[name]]
    if missing:
        raise ValueError(f"{key_path} 缺少非空配置: {', '.join(missing)}")
    return GeminiCredentials(**values)


def _copy_reused_gemini_checkpoint(reuse_dir: Path, out_dir: Path) -> Path:
    source = reuse_dir / "gemini_results.jsonl"
    destination = out_dir / "gemini_results.jsonl"
    if not reuse_dir.is_dir():
        raise FileNotFoundError(f"Gemini 复用目录不存在: {reuse_dir}")
    if not source.is_file():
        raise FileNotFoundError(f"Gemini 复用目录中缺少: {source}")
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
        print(f"复制 Gemini checkpoint: {source} -> {destination}")
    return destination


def _require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "按 VERL 方式读取 Parquet 需要 datasets：pip install datasets pyarrow"
        ) from exc
    return datasets


def select_indices(
    total: int,
    count: int,
    selection: str,
    seed: int,
    start_row: int,
) -> list[int]:
    if total <= 0:
        raise ValueError("train.parquet 为空")
    if start_row < 0 or start_row >= total:
        raise ValueError(f"--start_row 必须在 0 到 {total - 1} 之间")
    available = total - start_row
    if count > available:
        raise ValueError(
            f"从 row {start_row} 开始只有 {available} 条，不能抽取 {count} 条"
        )
    population = range(start_row, total)
    if selection == "first":
        return list(range(start_row, start_row + count))
    # Sort sampled indices so the report follows the original Parquet order.
    return sorted(random.Random(seed).sample(population, count))


def load_eval_rows(
    data_path: Path,
    count: int,
    selection: str,
    seed: int,
    start_row: int,
) -> tuple[list[EvalRow], int]:
    datasets = _require_datasets()
    # Keep this call identical to VERL's RLHFDataset._read_files_and_tokenize.
    dataframe = datasets.load_dataset("parquet", data_files=str(data_path))["train"]
    total = len(dataframe)
    names = set(dataframe.column_names)
    if "prompt" not in names:
        raise ValueError(f"{data_path} 缺少 VERL prompt 字段")
    if not ({"reward_model", "gts", "ground_truth", "gt"} & names):
        raise ValueError(f"{data_path} 中找不到 GT 字段")
    indices = select_indices(total, count, selection, seed, start_row)
    selected = dataframe.select(indices)
    rows: list[EvalRow] = []
    for order, (source_row, row) in enumerate(zip(indices, selected, strict=True)):
        if not isinstance(row, dict):
            raise TypeError(f"Parquet row {source_row} 不是 object")
        ground_truth = annotation._json_text(annotation._ground_truth(row))
        if not ground_truth:
            raise ValueError(f"Parquet row {source_row} 的 GT 为空")
        # 输入只包含 inconsistent CoT 数据，因此不读取或信任 Parquet 自带的
        # data_source；本地模型和 Gemini 始终使用各自的显式评测路由。
        rows.append(
            EvalRow(
                order,
                source_row,
                row,
                ground_truth,
                LOCAL_DATA_SOURCE,
                GEMINI_DATA_SOURCE,
            )
        )
    return rows, total


def _content_image(image, original: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(original or {})
    payload.pop("path", None)
    payload.pop("bytes", None)
    payload["type"] = "image"
    payload["image"] = image
    return payload


def build_messages_and_images(
    row: dict[str, Any],
    data_path: Path,
    reveal_gt_answer: bool = False,
) -> tuple[list[dict[str, Any]], list[Any]]:
    prompt = annotation._plain(row.get("prompt"))
    if not isinstance(prompt, list) or not prompt:
        raise TypeError("prompt 必须是非空 messages list")
    messages = copy.deepcopy(prompt)
    image_values = annotation._image_values(row)
    images = [annotation._decode_image(value, data_path) for value in image_values]
    image_offset = 0

    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("prompt 中每条 message 必须是 object")
        content = message.get("content", "")
        if isinstance(content, str):
            if not images:
                continue
            parts = [part for part in re.split(r"(<image>)", content) if part]
            converted = []
            for part in parts:
                if part == "<image>":
                    if image_offset >= len(images):
                        raise ValueError("prompt 的 <image> 数量多于 images 字段")
                    converted.append(_content_image(images[image_offset]))
                    image_offset += 1
                else:
                    converted.append({"type": "text", "text": part})
            message["content"] = converted
        elif isinstance(content, list):
            converted = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    if image_offset >= len(images):
                        raise ValueError("prompt 的 image item 数量多于 images 字段")
                    converted.append(_content_image(images[image_offset], item))
                    image_offset += 1
                else:
                    converted.append(item)
            message["content"] = converted
        else:
            raise TypeError("message.content 必须是字符串或 content list")

    if image_offset != len(images):
        raise ValueError(
            f"prompt 使用了 {image_offset} 幅图，但 images 字段有 {len(images)} 幅"
        )
    if reveal_gt_answer:
        ground_truth = annotation._plain(annotation._ground_truth(row))
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError as exc:
                raise ValueError("GT 不是包含 answer 的 JSON object") from exc
        if not isinstance(ground_truth, dict) or "answer" not in ground_truth:
            raise ValueError("GT 中找不到 answer，无法使用 --reveal_gt_answer")
        answer = annotation._plain(ground_truth["answer"])
        if answer is None:
            raise ValueError("GT 的 answer 为空，无法使用 --reveal_gt_answer")
        answer_text = (
            answer
            if isinstance(answer, str)
            else json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
        )
        reveal_text = (
            f"The correct answer is {answer_text}. "
            "Please complete the requested output based on this answer."
        )
        # Reveal only the answer value; never include other ground-truth fields.
        messages.append({"role": "user", "content": reveal_text})
    return messages, images


GEMINI_SYSTEM_PROMPT = """You are evaluating the same visual-spatial task as a local model.
The role-tagged task prompt and images in the user message are the complete shared input.
Follow that task prompt, but return only the final judgment and summary: output exactly one
JSON object with keys \"answer\" and \"summary\". Use the answer value/format requested by
the shared prompt. Do not output reasoning, chain-of-thought, bounding boxes, objects,
markdown fences, or any other keys or text."""


def build_gemini_user_content(
    messages: list[dict[str, Any]],
    image_urls: list[str],
) -> list[dict[str, Any]]:
    """Convert the exact local-model messages to idealab multimodal content."""
    content: list[dict[str, Any]] = []
    image_offset = 0
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content.append({"type": "text", "text": f"\n[{role}]\n"})
        value = message.get("content", "")
        items = value if isinstance(value, list) else [{"type": "text", "text": str(value)}]
        for item in items:
            if isinstance(item, str):
                content.append({"type": "text", "text": item})
            elif isinstance(item, dict) and item.get("type") == "image":
                if image_offset >= len(image_urls):
                    raise ValueError("Gemini prompt 的图片数量多于已上传 URL")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_urls[image_offset]},
                })
                image_offset += 1
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                content.append({"type": "text", "text": item["text"]})
            else:
                content.append({
                    "type": "text",
                    "text": json.dumps(annotation._plain(item), ensure_ascii=False),
                })
    if image_offset != len(image_urls):
        raise ValueError(
            f"Gemini prompt 使用了 {image_offset} 幅图，但上传了 {len(image_urls)} 幅"
        )
    content.append({
        "type": "text",
        "text": "\nReturn only {\"answer\": ..., \"summary\": \"...\"}.",
    })
    return content


def _gemini_dataset_key(data_path: Path) -> str:
    stat = data_path.stat()
    fingerprint = f"{data_path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def _prepare_gemini_request(
    eval_row: EvalRow,
    data_path: Path,
    out_dir: Path,
    oss_handler,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    messages, images = build_messages_and_images(
        eval_row.row, data_path, args.reveal_gt_answer
    )
    input_dir = out_dir / "gemini_inputs" / f"row_{eval_row.source_row:08d}"
    input_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = _gemini_dataset_key(data_path)
    prefix = args.gemini_oss_prefix.strip("/")
    image_urls = []
    for index, image in enumerate(images, 1):
        local_path = input_dir / f"image_{index}.jpg"
        resized = annotation._resize_image(image, args.gemini_max_image_edge)
        annotation._save_jpeg(resized, local_path, args.jpeg_quality)
        oss_key = (
            f"{prefix}/{dataset_key}/row_{eval_row.source_row:08d}/image_{index}.jpg"
        )
        if not oss_handler.is_file_exist(oss_key):
            oss_handler.upload_file(oss_key, str(local_path))
        image_urls.append(oss_handler.get_oss_url(oss_key))
    return build_gemini_user_content(messages, image_urls)


class TransformersRollout:
    def __init__(
        self,
        args: argparse.Namespace,
        model_path: str | None = None,
        model_label: str = "本地模型",
    ):
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as ModelLoader
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelLoader
        except ImportError as exc:
            raise RuntimeError(
                "rollout 需要 torch、transformers、accelerate；请按脚本开头的命令安装"
            ) from exc

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("--device 指定 CUDA，但 PyTorch 未检测到可用 NVIDIA GPU")
        dtype_by_name = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        selected_model_path = model_path or args.model_path
        print(f"加载 {model_label} processor: {selected_model_path}")
        processor = AutoProcessor.from_pretrained(
            selected_model_path,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"加载 {model_label} 到 {args.device}（dtype={args.dtype}）...")
        try:
            model = ModelLoader.from_pretrained(
                selected_model_path,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=dtype_by_name[args.dtype],
                device_map={"": args.device},
                low_cpu_mem_usage=True,
            )
        except ImportError as exc:
            raise RuntimeError(
                "device_map 加载需要 accelerate：pip install accelerate"
            ) from exc
        model.eval()
        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        self.args = args
        self.torch = torch
        self.processor = processor
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, rows: list[EvalRow], data_path: Path) -> list[str]:
        texts: list[str] = []
        all_images: list[Any] = []
        for eval_row in rows:
            messages, images = build_messages_and_images(
                eval_row.row, data_path, self.args.reveal_gt_answer
            )
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
            all_images.extend(images)

        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if all_images:
            processor_kwargs["images"] = all_images
        inputs = self.processor(**processor_kwargs)
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        overlong = [int(length) for length in prompt_lengths if length > self.args.max_prompt_length]
        if overlong:
            raise ValueError(
                f"多模态 prompt 长度 {max(overlong)} 超过 --max_prompt_length="
                f"{self.args.max_prompt_length}；与 VERL 一样不做会破坏图文对齐的截断，"
                "请提高该参数或减小图片分辨率"
            )

        inputs = {
            name: value.to(self.model.device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        input_width = inputs["input_ids"].shape[1]
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.temperature > 0,
            "repetition_penalty": self.args.repetition_penalty,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.args.temperature > 0:
            generation_kwargs["temperature"] = self.args.temperature
            generation_kwargs["top_p"] = self.args.top_p
            if self.args.top_k > 0:
                generation_kwargs["top_k"] = self.args.top_k

        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        response_ids = generated[:, input_width:]
        return self.processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def close(self) -> None:
        """Release the rollout model before the R4 process claims the GPU."""
        model = self.model
        self.model = None
        self.processor = None
        self.tokenizer = None
        del model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            try:
                self.torch.cuda.ipc_collect()
            except (AttributeError, RuntimeError):
                pass


def _read_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                completed[int(value["source_row"])] = value
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no} checkpoint 无效: {exc}") from exc
    return completed


def _append_checkpoint(
    path: Path,
    eval_row: EvalRow,
    args: argparse.Namespace,
    *,
    prediction_attr: str = "prediction",
    error_attr: str = "generation_error",
) -> None:
    payload = {
        "source_row": eval_row.source_row,
        "sample_order": eval_row.order,
        "data_source": eval_row.local_data_source,
        "ground_truth": eval_row.ground_truth,
        prediction_attr: getattr(eval_row, prediction_attr),
        error_attr: getattr(eval_row, error_attr),
        "model_path": str(Path(args.model_path)),
        "data_path": str(Path(args.data_path)),
        "seed": args.seed,
        "generation_seed": args.generation_seed,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "max_prompt_length": args.max_prompt_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "reveal_gt_answer": args.reveal_gt_answer,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _validate_checkpoint(
    completed: dict[int, dict[str, Any]],
    rows: list[EvalRow],
    args: argparse.Namespace,
) -> None:
    expected = {
        "model_path": str(Path(args.model_path)),
        "data_path": str(Path(args.data_path)),
        "seed": args.seed,
        "generation_seed": args.generation_seed,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "max_prompt_length": args.max_prompt_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }
    selected = {row.source_row: row for row in rows}
    for source_row, saved in completed.items():
        if source_row not in selected:
            continue
        mismatched = [key for key, value in expected.items() if saved.get(key) != value]
        if bool(saved.get("reveal_gt_answer", False)) != args.reveal_gt_answer:
            mismatched.append("reveal_gt_answer")
        if saved.get("data_source") != selected[source_row].local_data_source:
            mismatched.append("data_source")
        if mismatched:
            raise ValueError(
                f"checkpoint 中 Parquet row {source_row} 的运行参数与本次不同: "
                f"{', '.join(mismatched)}；请恢复原参数或使用 --overwrite 重跑"
            )


def _run_transformers_rollout(
    rows: list[EvalRow],
    data_path: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    *,
    prediction_attr: str,
    error_attr: str,
    model_label: str,
) -> None:
    completed = _read_checkpoint(checkpoint_path) if args.resume else {}
    _validate_checkpoint(completed, rows, args)
    pending: list[EvalRow] = []
    for row in rows:
        saved = completed.get(row.source_row)
        if saved is not None and not (args.retry_errors and saved.get(error_attr)):
            setattr(row, prediction_attr, str(saved.get(prediction_attr, "")))
            setattr(row, error_attr, str(saved.get(error_attr, "")))
        else:
            pending.append(row)
    print(
        f"{model_label}：已完成 {len(rows) - len(pending)} 条；"
        f"本次需生成 {len(pending)} 条"
    )
    if not pending:
        return

    engine = TransformersRollout(args, model_label=model_label)
    try:
        engine.torch.manual_seed(args.generation_seed)
        if engine.torch.cuda.is_available():
            engine.torch.cuda.manual_seed_all(args.generation_seed)

        completed_count = len(rows) - len(pending)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            try:
                predictions = engine.generate(batch, data_path)
                for row, prediction in zip(batch, predictions, strict=True):
                    setattr(row, prediction_attr, prediction)
            except Exception as batch_exc:
                if len(batch) == 1:
                    setattr(
                        batch[0], error_attr,
                        f"{type(batch_exc).__name__}: {batch_exc}",
                    )
                    print(
                        f"  [WARN] {model_label} Parquet row "
                        f"{batch[0].source_row} 生成失败: {batch_exc}"
                    )
                else:
                    print(
                        f"  [WARN] {model_label} batch 生成失败，逐条重试: "
                        f"{batch_exc}"
                    )
                    if engine.torch.cuda.is_available():
                        engine.torch.cuda.empty_cache()
                    for row in batch:
                        try:
                            setattr(
                                row,
                                prediction_attr,
                                engine.generate([row], data_path)[0],
                            )
                        except Exception as row_exc:
                            setattr(
                                row,
                                error_attr,
                                f"{type(row_exc).__name__}: {row_exc}",
                            )
                            print(
                                f"  [WARN] {model_label} Parquet row "
                                f"{row.source_row} 生成失败: {row_exc}"
                            )
            for row in batch:
                _append_checkpoint(
                    checkpoint_path,
                    row,
                    args,
                    prediction_attr=prediction_attr,
                    error_attr=error_attr,
                )
                completed_count += 1
            print(f"  {model_label} rollout {completed_count}/{len(rows)}")
    finally:
        print(f"释放 {model_label} rollout 模型和 CUDA 缓存...")
        engine.close()


def run_rollout(
    rows: list[EvalRow],
    data_path: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    _run_transformers_rollout(
        rows,
        data_path,
        checkpoint_path,
        args,
        prediction_attr="prediction",
        error_attr="generation_error",
        model_label="Qwen",
    )


def run_internvl_rollout(
    rows: list[EvalRow],
    data_path: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    if checkpoint_path.is_file() and getattr(args, "overwrite", False):
        checkpoint_path.unlink()
        print(f"覆盖 InternVL checkpoint: {checkpoint_path}")
    internvl_args = copy.copy(args)
    internvl_args.model_path = args.internvl_model_path
    internvl_args.batch_size = args.internvl_batch_size
    # InternVL has its own checkpoint and always resumes it when present.
    internvl_args.resume = checkpoint_path.is_file()
    _run_transformers_rollout(
        rows,
        data_path,
        checkpoint_path,
        internvl_args,
        prediction_attr="internvl_prediction",
        error_attr="internvl_error",
        model_label="InternVL",
    )


def _append_gemini_checkpoint(
    path: Path,
    eval_row: EvalRow,
    args: argparse.Namespace,
) -> None:
    payload = {
        "source_row": eval_row.source_row,
        "sample_order": eval_row.order,
        "data_source": eval_row.gemini_data_source,
        "ground_truth": eval_row.ground_truth,
        "gemini_prediction": eval_row.gemini_prediction,
        "gemini_error": eval_row.gemini_error,
        "data_path": str(Path(args.data_path)),
        "gemini_model": GEMINI_MODEL,
        "gemini_thinking_level": args.gemini_thinking_level,
        "reveal_gt_answer": args.reveal_gt_answer,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _validate_gemini_checkpoint(
    completed: dict[int, dict[str, Any]],
    rows: list[EvalRow],
    args: argparse.Namespace,
) -> None:
    expected = {
        "data_path": str(Path(args.data_path)),
        "gemini_model": GEMINI_MODEL,
        "gemini_thinking_level": args.gemini_thinking_level,
    }
    selected = {row.source_row: row for row in rows}
    for source_row, saved in completed.items():
        if source_row not in selected:
            continue
        mismatched = [key for key, value in expected.items() if saved.get(key) != value]
        if bool(saved.get("reveal_gt_answer", False)) != args.reveal_gt_answer:
            mismatched.append("reveal_gt_answer")
        if saved.get("data_source") != selected[source_row].gemini_data_source:
            mismatched.append("data_source")
        if saved.get("ground_truth") != selected[source_row].ground_truth:
            mismatched.append("ground_truth")
        if mismatched:
            raise ValueError(
                f"Gemini checkpoint 中 Parquet row {source_row} 参数不同: "
                f"{', '.join(mismatched)}；请恢复原参数或移走该文件后重跑"
            )


def run_gemini_rollout(
    rows: list[EvalRow],
    data_path: Path,
    out_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    credentials: GeminiCredentials | None = None,
    oss_handler=None,
    trusted_checkpoint: bool = False,
) -> None:
    checkpoint_exists = checkpoint_path.is_file()
    completed = _read_checkpoint(checkpoint_path) if checkpoint_exists else {}
    if not trusted_checkpoint:
        _validate_gemini_checkpoint(completed, rows, args)
    if checkpoint_exists:
        missing = [row.source_row for row in rows if row.source_row not in completed]
        if missing:
            preview = ", ".join(map(str, missing[:10]))
            if len(missing) > 10:
                preview += ", ..."
            raise ValueError(
                f"Gemini checkpoint 缺少本次抽样的 {len(missing)} 条 row: {preview}；"
                "为避免混用 case，不会调用 Gemini 补齐。请使用匹配的抽样参数，"
                "或移走 gemini_results.jsonl 后重跑"
            )
        for row in rows:
            saved = completed[row.source_row]
            row.gemini_prediction = str(saved.get("gemini_prediction", ""))
            row.gemini_error = str(saved.get("gemini_error", ""))
        print(f"Gemini checkpoint 已存在，直接复用 {len(rows)} 条，不调用 Gemini")
        return

    pending = list(rows)
    print(f"Gemini 已完成 {len(rows) - len(pending)} 条；本次需生成 {len(pending)} 条")
    if not pending:
        return

    gemini_module = _gemini_module()
    credentials = credentials or _load_gemini_credentials()
    oss_handler = oss_handler or gemini_module.OuterOSSHandle(
        credentials.ak, credentials.sk, credentials.ep, credentials.bn
    )

    def generate_one(row: EvalRow) -> tuple[str, str]:
        try:
            user_content = _prepare_gemini_request(
                row, data_path, out_dir, oss_handler, args
            )
            result = gemini_module.call_idealab(
                credentials.api_key,
                GEMINI_SYSTEM_PROMPT,
                user_content,
                max_retries=args.gemini_max_retries,
                thinking_level=args.gemini_thinking_level,
            )
            if not result.get("ok"):
                return "", str(result.get("error", "unknown Gemini error"))
            prediction = str(result.get("text", "")).strip()
            obj = reward._parse_json_obj(prediction)
            if not isinstance(obj, dict):
                return prediction, "Gemini 输出中找不到有效 JSON object"
            if "answer" not in obj or not isinstance(obj.get("summary"), str):
                return prediction, "Gemini JSON 必须包含 answer 和字符串 summary"
            return prediction, ""
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}"

    completed_count = len(rows) - len(pending)
    with ThreadPoolExecutor(max_workers=args.gemini_workers) as executor:
        future_to_row = {executor.submit(generate_one, row): row for row in pending}
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            row.gemini_prediction, row.gemini_error = future.result()
            _append_gemini_checkpoint(checkpoint_path, row, args)
            completed_count += 1
            if row.gemini_error:
                print(f"  [WARN] Gemini row {row.source_row}: {row.gemini_error}")
            print(f"  Gemini {completed_count}/{len(rows)}")


def _stop_reward_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32" and hasattr(signal, "CTRL_BREAK_EVENT"):
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_for_reward_server(
    process: subprocess.Popen,
    health_url: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status = "等待 reward model 加载"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"reward_model_server.py 在就绪前退出（exit={return_code}）"
            )
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
                last_status = f"HTTP {response.status}"
        except HTTPError as exc:
            if exc.code == 503:
                last_status = "reward model 仍在加载"
            else:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"reward service 健康检查失败：HTTP {exc.code}: {body}"
                ) from exc
        except (URLError, TimeoutError) as exc:
            last_status = str(getattr(exc, "reason", exc))
        time.sleep(0.5)
    raise TimeoutError(
        f"reward model 在 {timeout_seconds:g} 秒内未就绪：{last_status}"
    )


@contextmanager
def reward_server_for_scoring(args: argparse.Namespace):
    """Start R4 only after rollout, then always stop the child process."""
    if not args.auto_reward_server:
        print("使用外部 reward service（本脚本不启动或停止它）...")
        yield
        return

    server_script = Path(__file__).resolve().with_name("reward_model_server.py")
    if not server_script.is_file():
        raise FileNotFoundError(f"reward server 脚本不存在: {server_script}")

    host = args.reward_host
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    base_url = f"http://{client_host}:{args.reward_port}"
    command = [
        sys.executable,
        str(server_script),
        "--host",
        host,
        "--port",
        str(args.reward_port),
        "--backend",
        getattr(args, "reward_backend", "auto"),
        "--transformers-device",
        getattr(args, "reward_transformers_device", "auto"),
        "--max-batch-size",
        str(args.reward_max_batch_size),
        "--max-wait-ms",
        str(args.reward_max_wait_ms),
    ]
    reward_model_path = getattr(args, "reward_model_path", None)
    if reward_model_path:
        command.extend(["--model-path", reward_model_path])
    print(f"rollout 已完成，启动 reward model: {base_url}")
    popen_kwargs = {"cwd": str(server_script.parent)}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(command, **popen_kwargs)
    old_reward_url = os.environ.get("R4_REWARD_URL")
    os.environ["R4_REWARD_URL"] = base_url
    try:
        _wait_for_reward_server(
            process,
            f"{base_url}/health",
            args.reward_start_timeout,
        )
        print("reward model 已就绪，开始计算奖励...")
        yield
    finally:
        print("停止本次自动启动的 reward model...")
        try:
            _stop_reward_server(process)
        finally:
            if old_reward_url is None:
                os.environ.pop("R4_REWARD_URL", None)
            else:
                os.environ["R4_REWARD_URL"] = old_reward_url


def calculate_rewards(
    rows: list[EvalRow],
    args: argparse.Namespace,
) -> None:
    """Generate all reward details after every enabled rollout has finished."""
    jobs = []
    for row in rows:
        if row.prediction and not row.generation_error:
            jobs.append((row, "local"))
        if args.gemini and row.gemini_prediction and not row.gemini_error:
            jobs.append((row, "gemini"))
        if (
            args.internvl_model_path
            and row.internvl_prediction
            and not row.internvl_error
        ):
            jobs.append((row, "internvl"))

    def score_one(job):
        row, kind = job
        if kind in ("local", "internvl"):
            prediction = (
                row.prediction if kind == "local" else row.internvl_prediction
            )
            scores = reward.compute_score_details(
                row.local_data_source, prediction, row.ground_truth
            )
        else:
            scores = reward.score_answer_and_summary(
                row.gemini_data_source, row.gemini_prediction, row.ground_truth
            )
        return row, kind, {key: float(value) for key, value in scores.items()}

    errors: dict[int, list[str]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=args.reward_workers) as executor:
        future_to_job = {executor.submit(score_one, job): job for job in jobs}
        for future in as_completed(future_to_job):
            row, kind = future_to_job[future]
            try:
                _, _, scores = future.result()
                if kind == "local":
                    row.local_scores = scores
                elif kind == "gemini":
                    row.gemini_scores = scores
                else:
                    row.internvl_scores = scores
            except Exception as exc:
                errors.setdefault(row.source_row, []).append(
                    f"{kind}: {type(exc).__name__}: {exc}"
                )
            completed += 1
            if completed % 20 == 0 or completed == len(jobs):
                print(f"  奖励 {completed}/{len(jobs)}")
    for row in rows:
        row.local_scores = row.local_scores or {}
        row.gemini_scores = row.gemini_scores or {}
        row.internvl_scores = row.internvl_scores or {}
        row.evaluation_error = "; ".join(errors.get(row.source_row, []))


def write_evaluation_results(
    rows: list[EvalRow],
    path: Path,
    args: argparse.Namespace,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "source_row": row.source_row,
                "sample_order": row.order,
                # data_source 保留为本地路由，兼容旧的汇总结果读取方式。
                "data_source": row.local_data_source,
                "local_data_source": row.local_data_source,
                "gemini_data_source": row.gemini_data_source,
                "ground_truth": row.ground_truth,
                "prediction": row.prediction,
                "generation_error": row.generation_error,
                "local_scores": row.local_scores or {},
                "gemini_prediction": row.gemini_prediction,
                "gemini_error": row.gemini_error,
                "gemini_scores": row.gemini_scores or {},
                "internvl_prediction": row.internvl_prediction,
                "internvl_error": row.internvl_error,
                "internvl_scores": row.internvl_scores or {},
                "evaluation_error": row.evaluation_error,
                "temperature": args.temperature,
                "reveal_gt_answer": args.reveal_gt_answer,
                "gemini_model": GEMINI_MODEL if args.gemini else None,
                "internvl_model_path": args.internvl_model_path,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def materialize_report_images(
    rows: list[EvalRow],
    cases: list[annotation.Case],
    out_dir: Path,
    max_edge: int,
    quality: int,
) -> None:
    """Export images from already loaded VERL rows without source trace-back."""
    assets_dir = out_dir / "assets"
    fallback_count = 0
    failed_count = 0
    for completed, (eval_row, case) in enumerate(
        zip(rows, cases, strict=True), 1
    ):
        try:
            values = annotation._image_values(eval_row.row)
            if not values:
                raise ValueError("Parquet 行没有 images/image 字段")
            images = [
                annotation._resize_image(
                    annotation._decode_image(value, case.source_path),
                    max_edge,
                )
                for value in values
            ]
            target_index = 1 if len(images) >= 2 else len(images) - 1
            if len(images) < 2:
                fallback_count += 1
                case.image_error = "该行不足两幅图，框已画在唯一图片上"
            images[target_index] = annotation.annotate_second_image(
                images[target_index],
                annotation._extract_boxes(case.ground_truth),
                annotation._extract_boxes(case.prediction),
            )
            for image_index, image in enumerate(images):
                name = f"case_{case.order + 1:06d}_img_{image_index + 1}.jpg"
                annotation._save_jpeg(image, assets_dir / name, quality)
                case.image_paths.append(f"assets/{name}")
        except Exception as exc:
            case.image_error = f"图片处理失败: {exc}"
            failed_count += 1
        if completed % 100 == 0 or completed == len(cases):
            print(f"  图片 {completed}/{len(cases)}")
    if fallback_count:
        print(f"  [WARN] {fallback_count} 条不足两幅图，改画在唯一图片上")
    if failed_count:
        print(f"  [WARN] {failed_count} 条图片处理失败；HTML 中会显示原因")


def build_report(
    rows: list[EvalRow],
    data_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    cases = []
    for row in rows:
        prediction = row.prediction
        if row.generation_error:
            prediction = f"[生成失败]\n{row.generation_error}\n\n{prediction}".rstrip()
        gemini_prediction = row.gemini_prediction
        if row.gemini_error and gemini_prediction:
            gemini_prediction = (
                f"[输出不符合要求]\n{row.gemini_error}\n\n{gemini_prediction}"
            ).rstrip()
        internvl_prediction = row.internvl_prediction
        if row.internvl_error:
            internvl_prediction = (
                f"[生成失败]\n{row.internvl_error}\n\n{internvl_prediction}"
            ).rstrip()
        cases.append(annotation.Case(
            order=row.order,
            jsonl_line=row.source_row,
            source=row.local_data_source,
            source_path=data_path,
            source_row=row.source_row,
            ground_truth=row.ground_truth,
            prediction=prediction,
            image_paths=[],
            data_source=row.local_data_source,
            gemini_prediction=gemini_prediction,
            internvl_prediction=internvl_prediction,
            local_scores=row.local_scores or {},
            gemini_scores=row.gemini_scores or {},
            internvl_scores=row.internvl_scores or {},
            evaluation_error=row.evaluation_error,
            gemini_error=row.gemini_error if not gemini_prediction else "",
            internvl_error=row.internvl_error if not internvl_prediction else "",
        ))

    print(
        "导出原图，并把 GT/Qwen 预测框独立画到第二幅图"
        "（Gemini/InternVL 不绘框）..."
    )
    materialize_report_images(
        rows,
        cases,
        out_dir,
        args.max_image_edge,
        args.jpeg_quality,
    )
    out_path = out_dir / "index.html"
    model_name = Path(args.model_path.rstrip("/\\")).name or args.model_path
    internvl_name = (
        Path(args.internvl_model_path.rstrip("/\\")).name
        if args.internvl_model_path
        else "关闭"
    )
    subtitle = (
        f"模型 {model_name} · 数据 {data_path.name} · {len(rows)} 条 · "
        f"temperature={args.temperature:g} · Gemini="
        f"{GEMINI_MODEL if args.gemini else '关闭'} · "
        f"InternVL={internvl_name} · "
        f"透露GT answer={'是' if args.reveal_gt_answer else '否'}"
    )
    annotation.build_html(
        cases,
        data_path,
        "rollout",
        out_path,
        title=f"{data_path.name} Qwen / Gemini / InternVL 对比评测",
        sources=list(dict.fromkeys(row.local_data_source for row in rows)),
        subtitle=subtitle,
        row_label="Parquet row",
        export_filename="rollout_annotations.jsonl",
        show_internvl_column=True,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Windows 单卡运行 Qwen/可选 InternVL，并行调用 Gemini，"
            "统一计算奖励并生成 HTML"
        )
    )
    parser.add_argument("--model_path", "--model-path", required=True,
                        help="Qwen 本地 Hugging Face 模型目录")
    parser.add_argument(
        "--internvl_model_path", "--internvl-model-path",
        help=(
            "InternVL3.5-14B-HF 本地模型目录；指定后启用 InternVL，"
            "并自动复用输出目录中已有的 Qwen/Gemini checkpoint"
        ),
    )
    parser.add_argument("--data_path", "--data-path", required=True,
                        help="输入 VERL train.parquet 地址")
    parser.add_argument("--out_dir", "--out-dir", required=True,
                        help="输出目录（包含 checkpoint、assets 和 index.html）")
    parser.add_argument(
        "--reveal_gt_answer", "--reveal-gt-answer",
        action="store_true",
        help="在原 prompt 后用自然语言透露 GT answer，并要求据此完成输出",
    )
    parser.add_argument("--num_samples", "--num-samples", type=int, default=100,
                        help="评测条数，默认 100")
    parser.add_argument("--selection", choices=("random", "first"), default="random",
                        help="random=固定种子随机抽取；first=从 start_row 连续取，默认 random")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子，默认 42")
    parser.add_argument("--start_row", "--start-row", type=int, default=0,
                        help="抽样起始 row（0-based），默认 0")
    parser.add_argument("--generation_seed", "--generation-seed", type=int, default=42,
                        help="模型采样随机种子，默认 42")
    parser.add_argument("--device", default="cuda:0", help="模型设备，默认 cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"),
                        default="bfloat16", help="模型 dtype，默认 bfloat16")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=10,
                        help="Qwen 生成 batch size，默认 10")
    parser.add_argument(
        "--internvl_batch_size", "--internvl-batch-size",
        type=int,
        default=1,
        help="InternVL 生成 batch size，默认 1",
    )
    parser.add_argument("--max_prompt_length", "--max-prompt-length", type=int, default=2048,
                        help="与当前 VERL rollout.prompt_length 一致，默认 2048")
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=4096,
                        help="每条最大生成 token，默认 4096")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="本地模型生成温度，默认 0.01")
    parser.add_argument("--top_p", "--top-p", type=float, default=1.0,
                        help="默认 1.0，与当前 VERL rollout 一致")
    parser.add_argument("--top_k", "--top-k", type=int, default=-1,
                        help="默认 -1（关闭），与当前 VERL rollout 一致")
    parser.add_argument("--repetition_penalty", "--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true",
                        help="从现有本地 rollout_results.jsonl 继续")
    parser.add_argument("--retry_errors", "--retry-errors", action="store_true",
                        help="与 --resume 一起使用时重试本地模型之前失败的条目")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖本地 checkpoint（默认行为；保留此参数以兼容旧命令）")
    parser.add_argument("--trust_remote_code", "--trust-remote-code", action=argparse.BooleanOptionalAction,
                        default=True, help="是否允许模型仓库代码，默认开启")
    parser.add_argument("--gemini", action=argparse.BooleanOptionalAction, default=True,
                        help="是否并行调用 Gemini，默认开启；离线调试可用 --no-gemini")
    parser.add_argument(
        "--reuse_gemini_dir", "--reuse-gemini-dir",
        help=(
            "旧输出目录；把其中 gemini_results.jsonl 复制到本次输出并跳过 Gemini API"
        ),
    )
    parser.add_argument("--gemini_workers", "--gemini-workers", type=int, default=10,
                        help="Gemini API 并发数，默认 10")
    parser.add_argument("--gemini_max_retries", "--gemini-max-retries", type=int, default=3,
                        help="Gemini 瞬态失败最大尝试次数，默认 3")
    parser.add_argument("--gemini_thinking_level", "--gemini-thinking-level",
                        choices=("low", "medium", "high"), default="high",
                        help="Gemini thinking 等级，默认 high")
    parser.add_argument("--gemini_oss_prefix", "--gemini-oss-prefix",
                        default="yk/ai-material/neo/fjj/rollout",
                        help="上传 Gemini 输入图片使用的 OSS key 前缀")
    parser.add_argument("--gemini_max_image_edge", "--gemini-max-image-edge", type=int,
                        default=2048, help="Gemini 输入图片上传前的最长边，默认 2048")
    parser.add_argument("--reward_workers", "--reward-workers", type=int, default=100,
                        help="奖励计算并发数，默认 100，便于 R4 服务动态合批")
    parser.add_argument(
        "--auto_reward_server", "--auto-reward-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "rollout 后自动启动/停止 reward_model_server.py，默认开启；"
            "使用已有服务时传 --no-auto-reward-server"
        ),
    )
    parser.add_argument("--reward_host", "--reward-host", default="127.0.0.1",
                        help="自动 reward service 监听地址，默认 127.0.0.1")
    parser.add_argument("--reward_port", "--reward-port", type=int, default=8765,
                        help="自动 reward service 端口，默认 8765")
    parser.add_argument(
        "--reward_backend", "--reward-backend",
        choices=("auto", "vllm", "transformers"),
        default="auto",
        help="R4 后端；auto 优先 vLLM，不可用时回退 Transformers",
    )
    parser.add_argument(
        "--reward_model_path", "--reward-model-path",
        default=None,
        help="R4 奖励模型目录；默认读取 R4_MODEL_LOCAL_PATH",
    )
    parser.add_argument(
        "--reward_transformers_device", "--reward-transformers-device",
        default="auto",
        help="Transformers 回退设备，如 auto、cuda、cuda:0 或 cpu",
    )
    parser.add_argument(
        "--reward_start_timeout", "--reward-start-timeout",
        type=float,
        default=900.0,
        help="等待 reward model 加载的超时秒数，默认 900",
    )
    parser.add_argument(
        "--reward_max_batch_size", "--reward-max-batch-size",
        type=int,
        default=100,
        help="R4 服务动态合批上限，默认 100",
    )
    parser.add_argument(
        "--reward_max_wait_ms", "--reward-max-wait-ms",
        type=float,
        default=20.0,
        help="R4 服务动态合批等待时间，默认 20ms",
    )
    parser.add_argument("--max_image_edge", "--max-image-edge", type=int, default=1200)
    parser.add_argument("--jpeg_quality", "--jpeg-quality", type=int, default=88)
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num_samples 必须大于 0")
    if args.batch_size <= 0 or args.internvl_batch_size <= 0:
        parser.error("Qwen/InternVL batch size 必须大于 0")
    if args.gemini_workers <= 0 or args.reward_workers <= 0:
        parser.error("Gemini/reward workers 必须大于 0")
    if not args.reward_host.strip():
        parser.error("--reward_host 不能为空")
    if not 1 <= args.reward_port <= 65535:
        parser.error("--reward_port 必须在 1 到 65535 之间")
    if not math.isfinite(args.reward_start_timeout) or args.reward_start_timeout <= 0:
        parser.error("--reward_start_timeout 必须大于 0")
    if args.reward_max_batch_size <= 0:
        parser.error("--reward_max_batch_size 必须大于 0")
    if not math.isfinite(args.reward_max_wait_ms) or args.reward_max_wait_ms < 0:
        parser.error("--reward_max_wait_ms 不能小于 0")
    if args.gemini_max_retries <= 0:
        parser.error("--gemini_max_retries 必须大于 0")
    if args.max_prompt_length <= 0 or args.max_new_tokens <= 0:
        parser.error("prompt/response 长度必须大于 0")
    if args.temperature < 0:
        parser.error("--temperature 不能小于 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p 必须在 (0, 1] 内")
    if args.repetition_penalty <= 0:
        parser.error("--repetition_penalty 必须大于 0")
    if args.max_image_edge < 128:
        parser.error("--max_image_edge 不能小于 128")
    if args.gemini_max_image_edge < 128:
        parser.error("--gemini_max_image_edge 不能小于 128")
    if not args.gemini_oss_prefix.strip("/"):
        parser.error("--gemini_oss_prefix 不能为空")
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("--jpeg_quality 必须在 30 到 100 之间")
    if args.resume and args.overwrite:
        parser.error("--resume 和 --overwrite 不能同时使用")
    if args.retry_errors and not args.resume:
        parser.error("--retry_errors 必须与 --resume 一起使用")
    if args.reuse_gemini_dir and not args.gemini:
        parser.error("--reuse_gemini_dir 不能与 --no-gemini 同时使用")
    return args


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    internvl_model_path = (
        Path(args.internvl_model_path).expanduser().resolve()
        if args.internvl_model_path
        else None
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(f"ERROR: Parquet 不存在: {data_path}")
    if data_path.suffix.lower() != ".parquet":
        raise SystemExit(f"ERROR: --data_path 必须是 VERL .parquet 文件: {data_path}")
    if not model_path.is_dir():
        raise SystemExit(f"ERROR: 模型目录不存在: {model_path}")
    if internvl_model_path is not None and not internvl_model_path.is_dir():
        raise SystemExit(f"ERROR: InternVL 模型目录不存在: {internvl_model_path}")
    args.data_path = str(data_path)
    args.model_path = str(model_path)
    args.internvl_model_path = (
        str(internvl_model_path) if internvl_model_path is not None else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoint_path = out_dir / "rollout_results.jsonl"
    gemini_checkpoint_path = out_dir / "gemini_results.jsonl"
    internvl_checkpoint_path = out_dir / "internvl_results.jsonl"
    trusted_gemini_checkpoint = False
    if args.reuse_gemini_dir:
        reuse_dir = Path(args.reuse_gemini_dir).expanduser().resolve()
        try:
            _copy_reused_gemini_checkpoint(reuse_dir, out_dir)
        except FileNotFoundError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
        trusted_gemini_checkpoint = True
    reuse_gemini = args.gemini and gemini_checkpoint_path.is_file()
    credentials = None
    oss_handler = None
    if args.gemini and not reuse_gemini:
        try:
            credentials = _load_gemini_credentials()
            gemini_module = _gemini_module()
            oss_handler = gemini_module.OuterOSSHandle(
                credentials.ak, credentials.sk, credentials.ep, credentials.bn
            )
        except Exception as exc:
            raise SystemExit(f"ERROR: 从 key.py 初始化 Gemini/OSS 失败: {exc}") from exc

    print(f"读取数据: {data_path}")
    rows, total = load_eval_rows(
        data_path,
        args.num_samples,
        args.selection,
        args.seed,
        args.start_row,
    )
    print(
        f"数据共 {total} 条；{args.selection} 抽取 {len(rows)} 条；"
        f"row 范围 {rows[0].source_row}..{rows[-1].source_row}"
    )
    if reuse_gemini:
        run_gemini_rollout(
            rows,
            data_path,
            out_dir,
            gemini_checkpoint_path,
            args,
            trusted_checkpoint=trusted_gemini_checkpoint,
        )

    reuse_qwen = bool(
        args.internvl_model_path
        and local_checkpoint_path.is_file()
        and not args.overwrite
    )
    if local_checkpoint_path.exists() and not args.resume and not reuse_qwen:
        local_checkpoint_path.unlink()
        print(f"覆盖本地 checkpoint: {local_checkpoint_path}")

    qwen_args = copy.copy(args)
    if reuse_qwen:
        qwen_args.resume = True
        print("Qwen：检测到已有 checkpoint，启用 InternVL 时直接复用。")

    def run_serial_local_models() -> None:
        run_rollout(rows, data_path, local_checkpoint_path, qwen_args)
        if args.internvl_model_path:
            run_internvl_rollout(
                rows,
                data_path,
                internvl_checkpoint_path,
                args,
            )

    if args.gemini and not reuse_gemini:
        if args.internvl_model_path:
            print("Gemini 独立并行；本地分支按 Qwen -> InternVL 串行生成...")
        else:
            print("Qwen 与 Gemini 并行生成...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_serial_local_models),
                executor.submit(
                    run_gemini_rollout,
                    rows,
                    data_path,
                    out_dir,
                    gemini_checkpoint_path,
                    args,
                    credentials,
                    oss_handler,
                ),
            ]
            for future in as_completed(futures):
                future.result()
    else:
        if reuse_gemini:
            print("Gemini：使用已有 checkpoint；运行本地模型分支...")
        else:
            print("Gemini：关闭；运行本地模型分支...")
        run_serial_local_models()

    print("所有启用的生成分支均已完成。")
    has_scorable_output = any(
        (row.prediction and not row.generation_error)
        or (args.gemini and row.gemini_prediction and not row.gemini_error)
        or (
            args.internvl_model_path
            and row.internvl_prediction
            and not row.internvl_error
        )
        for row in rows
    )
    if has_scorable_output:
        with reward_server_for_scoring(args):
            calculate_rewards(rows, args)
    else:
        print("没有可评分的生成结果，不启动 reward model。")
        calculate_rewards(rows, args)
    evaluation_path = out_dir / "evaluation_results.jsonl"
    write_evaluation_results(rows, evaluation_path, args)
    out_path = build_report(rows, data_path, out_dir, args)
    errors = sum(bool(row.generation_error) for row in rows)
    gemini_errors = sum(bool(row.gemini_error) for row in rows)
    internvl_errors = sum(bool(row.internvl_error) for row in rows)
    reward_errors = sum(bool(row.evaluation_error) for row in rows)
    print(f"完成: {out_path}")
    print(f"本地 checkpoint: {local_checkpoint_path}")
    if args.gemini:
        print(f"Gemini checkpoint: {gemini_checkpoint_path}")
    if args.internvl_model_path:
        print(f"InternVL checkpoint: {internvl_checkpoint_path}")
    print(f"完整结果: {evaluation_path}")
    if errors:
        print(f"[WARN] {errors} 条本地生成失败；可用 --resume --retry_errors 重试")
    if gemini_errors:
        print(f"[WARN] {gemini_errors} 条 Gemini 失败；可用 --resume --retry_errors 重试")
    if internvl_errors:
        print(
            f"[WARN] {internvl_errors} 条 InternVL 生成失败；"
            "可用 --resume --retry_errors 重试"
        )
    if reward_errors:
        print(f"[WARN] {reward_errors} 条奖励计算有失败，详情已写入 HTML/完整结果")
    print("人工标注保存在浏览器 localStorage；请在页面中导出 JSONL 备份。")


if __name__ == "__main__":
    main()
