#!/usr/bin/env python3
r"""Run a reward-free Transformers rollout on Parquet rows and build an HTML report.

This script is intended for a Windows workstation with one CUDA GPU.  It reads
one VERL Parquet shard directly, selects 200 rows by default, generates one
response per row with the supplied local Hugging Face model, and reuses the
interactive human-accuracy page from ``annotate_val_cases.py``.  It never loads
or calls the reward code/model.

Example (PowerShell, one line)::

    python rollout_parquet_to_html.py --model_path D:\models\Qwen_RL --data_path D:\RL1\inconsistent_cot_verl_2500\train_0000.parquet --out_dir D:\eval\train0000_200

Dependencies::

    pip install torch torchvision transformers accelerate pyarrow pillow

Generation results are appended to ``rollout_results.jsonl`` after every
batch.  If a run is interrupted, repeat the command with ``--resume``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import annotate_val_cases as annotation


@dataclass
class EvalRow:
    order: int
    source_row: int
    row: dict[str, Any]
    ground_truth: str
    prediction: str = ""
    generation_error: str = ""


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("读取 Parquet 需要 pyarrow：pip install pyarrow") from exc
    return pq


def select_indices(
    total: int,
    count: int,
    selection: str,
    seed: int,
    start_row: int,
) -> list[int]:
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


def _read_selected_row_groups(
    parquet_file,
    indices: Iterable[int],
    columns: list[str],
) -> dict[int, dict[str, Any]]:
    wanted = set(indices)
    result: dict[int, dict[str, Any]] = {}
    offset = 0
    for group_index in range(parquet_file.num_row_groups):
        count = parquet_file.metadata.row_group(group_index).num_rows
        in_group = sorted(index for index in wanted if offset <= index < offset + count)
        if in_group:
            rows = parquet_file.read_row_group(group_index, columns=columns).to_pylist()
            for index in in_group:
                value = rows[index - offset]
                if not isinstance(value, dict):
                    raise TypeError(f"Parquet row {index} 不是 object")
                result[index] = value
        offset += count
    missing = sorted(wanted - result.keys())
    if missing:
        raise RuntimeError(f"未能读取 Parquet rows: {missing[:10]}")
    return result


def load_eval_rows(
    data_path: Path,
    count: int,
    selection: str,
    seed: int,
    start_row: int,
) -> tuple[list[EvalRow], int]:
    pq = _require_pyarrow()
    parquet_file = pq.ParquetFile(data_path)
    total = parquet_file.metadata.num_rows
    names = set(parquet_file.schema_arrow.names)
    if "prompt" not in names:
        raise ValueError(f"{data_path} 缺少 VERL prompt 字段")
    if not ({"reward_model", "gts", "ground_truth", "gt"} & names):
        raise ValueError(f"{data_path} 中找不到 GT 字段")

    indices = select_indices(total, count, selection, seed, start_row)
    columns = [
        name
        for name in ("prompt", "images", "image", "reward_model", "gts", "ground_truth", "gt")
        if name in names
    ]
    selected = _read_selected_row_groups(parquet_file, indices, columns)
    rows: list[EvalRow] = []
    for order, source_row in enumerate(indices):
        row = selected[source_row]
        ground_truth = annotation._json_text(annotation._ground_truth(row))
        if not ground_truth:
            raise ValueError(f"Parquet row {source_row} 的 GT 为空")
        rows.append(EvalRow(order, source_row, row, ground_truth))
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
    return messages, images


class TransformersRollout:
    def __init__(self, args: argparse.Namespace):
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
        print(f"加载 processor: {args.model_path}")
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"加载模型到 {args.device}（dtype={args.dtype}）...")
        try:
            model = ModelLoader.from_pretrained(
                args.model_path,
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
            messages, images = build_messages_and_images(eval_row.row, data_path)
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


def _append_checkpoint(path: Path, eval_row: EvalRow, args: argparse.Namespace) -> None:
    payload = {
        "source_row": eval_row.source_row,
        "sample_order": eval_row.order,
        "ground_truth": eval_row.ground_truth,
        "prediction": eval_row.prediction,
        "generation_error": eval_row.generation_error,
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
    selected = {row.source_row for row in rows}
    for source_row, saved in completed.items():
        if source_row not in selected:
            continue
        mismatched = [key for key, value in expected.items() if saved.get(key) != value]
        if mismatched:
            raise ValueError(
                f"checkpoint 中 Parquet row {source_row} 的运行参数与本次不同: "
                f"{', '.join(mismatched)}；请恢复原参数或使用 --overwrite 重跑"
            )


def run_rollout(
    rows: list[EvalRow],
    data_path: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    completed = _read_checkpoint(checkpoint_path) if args.resume else {}
    _validate_checkpoint(completed, rows, args)
    pending: list[EvalRow] = []
    for row in rows:
        saved = completed.get(row.source_row)
        if saved is not None and not (args.retry_errors and saved.get("generation_error")):
            row.prediction = str(saved.get("prediction", ""))
            row.generation_error = str(saved.get("generation_error", ""))
        else:
            pending.append(row)
    print(f"已完成 {len(rows) - len(pending)} 条；本次需生成 {len(pending)} 条")
    if not pending:
        return

    engine = TransformersRollout(args)
    engine.torch.manual_seed(args.generation_seed)
    if engine.torch.cuda.is_available():
        engine.torch.cuda.manual_seed_all(args.generation_seed)

    completed_count = len(rows) - len(pending)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        try:
            predictions = engine.generate(batch, data_path)
            for row, prediction in zip(batch, predictions, strict=True):
                row.prediction = prediction
        except Exception as batch_exc:
            if len(batch) == 1:
                batch[0].generation_error = f"{type(batch_exc).__name__}: {batch_exc}"
                print(f"  [WARN] Parquet row {batch[0].source_row} 生成失败: {batch_exc}")
            else:
                print(f"  [WARN] batch 生成失败，逐条重试: {batch_exc}")
                if engine.torch.cuda.is_available():
                    engine.torch.cuda.empty_cache()
                for row in batch:
                    try:
                        row.prediction = engine.generate([row], data_path)[0]
                    except Exception as row_exc:
                        row.generation_error = f"{type(row_exc).__name__}: {row_exc}"
                        print(f"  [WARN] Parquet row {row.source_row} 生成失败: {row_exc}")
        for row in batch:
            _append_checkpoint(checkpoint_path, row, args)
            completed_count += 1
        print(f"  rollout {completed_count}/{len(rows)}")


def build_report(
    rows: list[EvalRow],
    data_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    source = data_path.parent.name or "parquet"
    cases = []
    for row in rows:
        prediction = row.prediction
        if row.generation_error:
            prediction = f"[生成失败]\n{row.generation_error}\n\n{prediction}".rstrip()
        cases.append(annotation.Case(
            order=row.order,
            jsonl_line=row.source_row,
            source=source,
            source_path=data_path,
            source_row=row.source_row,
            ground_truth=row.ground_truth,
            prediction=prediction,
            image_paths=[],
        ))

    print("导出原图，并把全部 GT/预测框独立画到第二幅图...")
    annotation.materialize_images(
        cases,
        out_dir,
        args.max_image_edge,
        args.jpeg_quality,
    )
    out_path = out_dir / "index.html"
    model_name = Path(args.model_path.rstrip("/\\")).name or args.model_path
    subtitle = (
        f"模型 {model_name} · 数据 {data_path.name} · {len(rows)} 条 · "
        f"temperature={args.temperature:g}"
    )
    annotation.build_html(
        cases,
        data_path,
        "rollout",
        out_path,
        title=f"{data_path.name} Rollout 人工评测",
        sources=[source],
        subtitle=subtitle,
        row_label="Parquet row",
        export_filename="rollout_annotations.jsonl",
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows 单卡运行 VERL Parquet rollout，并生成无奖励的交互式 HTML"
    )
    parser.add_argument("--model_path", "--model-path", required=True,
                        help="本地 Hugging Face 模型目录")
    parser.add_argument("--data_path", "--data-path", required=True,
                        help="输入 train_0000.parquet 地址")
    parser.add_argument("--out_dir", "--out-dir", required=True,
                        help="输出目录（包含 checkpoint、assets 和 index.html）")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=200,
                        help="评测条数，默认 200")
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
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1,
                        help="生成 batch size，默认 1；显存充足可提高")
    parser.add_argument("--max_prompt_length", "--max-prompt-length", type=int, default=2048,
                        help="与当前 VERL rollout.prompt_length 一致，默认 2048")
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=4096,
                        help="每条最大生成 token，默认 4096")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="默认 0（贪心），与当前 VERL 验证 rollout 一致；训练采样可设 1.0")
    parser.add_argument("--top_p", "--top-p", type=float, default=1.0,
                        help="默认 1.0，与当前 VERL rollout 一致")
    parser.add_argument("--top_k", "--top-k", type=int, default=-1,
                        help="默认 -1（关闭），与当前 VERL rollout 一致")
    parser.add_argument("--repetition_penalty", "--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true",
                        help="从现有 rollout_results.jsonl 继续")
    parser.add_argument("--retry_errors", "--retry-errors", action="store_true",
                        help="与 --resume 一起使用时重试之前失败的条目")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已有 checkpoint（不删除目录内其他文件）")
    parser.add_argument("--trust_remote_code", "--trust-remote-code", action=argparse.BooleanOptionalAction,
                        default=True, help="是否允许模型仓库代码，默认开启")
    parser.add_argument("--max_image_edge", "--max-image-edge", type=int, default=1200)
    parser.add_argument("--jpeg_quality", "--jpeg-quality", type=int, default=88)
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num_samples 必须大于 0")
    if args.batch_size <= 0:
        parser.error("--batch_size 必须大于 0")
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
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("--jpeg_quality 必须在 30 到 100 之间")
    if args.resume and args.overwrite:
        parser.error("--resume 和 --overwrite 不能同时使用")
    if args.retry_errors and not args.resume:
        parser.error("--retry_errors 必须与 --resume 一起使用")
    return args


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(f"ERROR: Parquet 不存在: {data_path}")
    if data_path.suffix.lower() != ".parquet":
        raise SystemExit(f"ERROR: --data_path 必须是 .parquet: {data_path}")
    if not model_path.is_dir():
        raise SystemExit(f"ERROR: 模型目录不存在: {model_path}")
    args.data_path = str(data_path)
    args.model_path = str(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "rollout_results.jsonl"
    if checkpoint_path.exists() and not args.resume:
        if args.overwrite:
            checkpoint_path.unlink()
        else:
            raise SystemExit(
                f"ERROR: 已存在 {checkpoint_path}；继续请加 --resume，重跑请加 --overwrite"
            )

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
    print("奖励模块：不加载；奖励计算：关闭")
    run_rollout(rows, data_path, checkpoint_path, args)
    out_path = build_report(rows, data_path, out_dir, args)
    errors = sum(bool(row.generation_error) for row in rows)
    print(f"完成: {out_path}")
    print(f"rollout checkpoint: {checkpoint_path}")
    if errors:
        print(f"[WARN] {errors} 条生成失败；修复环境后可用 --resume --retry_errors 重试")
    print("人工标注保存在浏览器 localStorage；请在页面中导出 JSONL 备份。")


if __name__ == "__main__":
    main()
