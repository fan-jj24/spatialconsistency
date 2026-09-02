#!/usr/bin/env python3
"""Run exactly one rollout and write one route-prefixed checkpoint.

Examples::

    # Local Qwen (default local backend)
    python rollout_checkpoint.py --mode local --name qwen_rl_step_100 \
      --model-path /models/qwen --data-path train.parquet --out-dir checkpoints

    # Local InternVL (explicit opt-in)
    python rollout_checkpoint.py --mode local --name internvl_baseline --internvl \
      --model-path /models/internvl --data-path train.parquet --out-dir checkpoints

    # Remote Gemini
    python rollout_checkpoint.py --mode remote --name gemini_35 \
      --data-path train.parquet --out-dir checkpoints

The output is ``<mode>__<name>.jsonl``.  Existing checkpoints are resumed by
default.  Use ``--retry-errors`` to retry failed rows or ``--overwrite`` to
start over.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from typing import Any

import json_answer_reward as reward
import rollout_parquet_to_html as legacy


SCHEMA_VERSION = 1
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def checkpoint_filename(mode: str, name: str) -> str:
    """Return a safe filename whose prefix is also the reward route."""
    if mode not in {"local", "remote"}:
        raise ValueError("mode 必须是 local 或 remote")
    if not NAME_RE.fullmatch(name):
        raise ValueError(
            "--name 只能包含英文字母、数字、点、下划线、连字符，且必须以字母或数字开头"
        )
    return f"{mode}__{name}.jsonl"


def read_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    """Read a checkpoint; a later retry record replaces the earlier record."""
    records: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                source_row = int(value["source_row"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no} checkpoint 无效: {exc}") from exc
            records[source_row] = value
    return records


def checkpoint_metadata(args: argparse.Namespace) -> dict[str, Any]:
    backend = "gemini" if args.mode == "remote" else (
        "internvl" if args.internvl else "qwen"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "name": args.name,
        "backend": backend,
        "data_path": args.data_path,
        "model_path": args.model_path if args.mode == "local" else None,
        "num_samples": args.num_samples,
        "selection": args.selection,
        "seed": args.seed,
        "start_row": args.start_row,
        "generation_seed": args.generation_seed,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "reveal_gt_answer": args.reveal_gt_answer,
        "remote_model": legacy.GEMINI_MODEL if args.mode == "remote" else None,
        "gemini_thinking_level": (
            args.gemini_thinking_level if args.mode == "remote" else None
        ),
    }


def validate_checkpoint(
    completed: dict[int, dict[str, Any]],
    rows: list[legacy.EvalRow],
    args: argparse.Namespace,
) -> None:
    expected = checkpoint_metadata(args)
    selected = {row.source_row: row for row in rows}
    for source_row, saved in completed.items():
        mismatched = [
            key for key, value in expected.items() if saved.get(key) != value
        ]
        if source_row not in selected:
            mismatched.append("source_row(不在本次抽样中)")
        elif saved.get("ground_truth") != selected[source_row].ground_truth:
            mismatched.append("ground_truth")
        if mismatched:
            raise ValueError(
                f"checkpoint 中 Parquet row {source_row} 与本次参数不同: "
                f"{', '.join(mismatched)}；请恢复原参数、换 name 或使用 --overwrite"
            )


def append_checkpoint(
    path: Path,
    row: legacy.EvalRow,
    prediction: str,
    error: str,
    args: argparse.Namespace,
) -> None:
    payload = {
        **checkpoint_metadata(args),
        "source_row": row.source_row,
        "sample_order": row.order,
        # 该字段便于人工查看；评分脚本以文件名前缀为最终路由依据。
        "data_source": (
            legacy.LOCAL_DATA_SOURCE
            if args.mode == "local"
            else legacy.GEMINI_DATA_SOURCE
        ),
        "ground_truth": row.ground_truth,
        "prediction": prediction,
        "error": error,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_local(
    rows: list[legacy.EvalRow],
    pending: list[legacy.EvalRow],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    label = "InternVL" if args.internvl else "Qwen"
    engine = legacy.TransformersRollout(args, model_label=label)
    try:
        engine.torch.manual_seed(args.generation_seed)
        if engine.torch.cuda.is_available():
            engine.torch.cuda.manual_seed_all(args.generation_seed)
        completed_count = len(rows) - len(pending)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            results: dict[int, tuple[str, str]] = {}
            try:
                predictions = engine.generate(batch, Path(args.data_path))
                results.update(
                    (row.source_row, (prediction, ""))
                    for row, prediction in zip(batch, predictions, strict=True)
                )
            except Exception as batch_exc:
                if len(batch) > 1:
                    print(f"  [WARN] {label} batch 失败，逐条重试: {batch_exc}")
                    if engine.torch.cuda.is_available():
                        engine.torch.cuda.empty_cache()
                for row in batch:
                    try:
                        prediction = engine.generate([row], Path(args.data_path))[0]
                        results[row.source_row] = (prediction, "")
                    except Exception as row_exc:
                        error = f"{type(row_exc).__name__}: {row_exc}"
                        results[row.source_row] = ("", error)
                        print(f"  [WARN] {label} row {row.source_row}: {error}")
            for row in batch:
                prediction, error = results[row.source_row]
                append_checkpoint(checkpoint_path, row, prediction, error, args)
                completed_count += 1
            print(f"  {label} rollout {completed_count}/{len(rows)}")
    finally:
        print(f"释放 {label} rollout 模型和 CUDA 缓存...")
        engine.close()


def run_remote(
    rows: list[legacy.EvalRow],
    pending: list[legacy.EvalRow],
    checkpoint_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    gemini = legacy._gemini_module()
    credentials = legacy._load_gemini_credentials()
    oss_handler = gemini.OuterOSSHandle(
        credentials.ak, credentials.sk, credentials.ep, credentials.bn
    )

    def generate_one(row: legacy.EvalRow) -> tuple[str, str]:
        try:
            content = legacy._prepare_gemini_request(
                row, Path(args.data_path), out_dir, oss_handler, args
            )
            result = gemini.call_idealab(
                credentials.api_key,
                legacy.GEMINI_SYSTEM_PROMPT,
                content,
                max_retries=args.gemini_max_retries,
                thinking_level=args.gemini_thinking_level,
            )
            if not result.get("ok"):
                return "", str(result.get("error", "unknown remote error"))
            prediction = str(result.get("text", "")).strip()
            obj = reward._parse_json_obj(prediction)
            if not isinstance(obj, dict):
                return prediction, "远程输出中找不到有效 JSON object"
            if "answer" not in obj or not isinstance(obj.get("summary"), str):
                return prediction, "远程 JSON 必须包含 answer 和字符串 summary"
            return prediction, ""
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}"

    completed_count = len(rows) - len(pending)
    with ThreadPoolExecutor(max_workers=args.remote_workers) as executor:
        futures = {executor.submit(generate_one, row): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            prediction, error = future.result()
            append_checkpoint(checkpoint_path, row, prediction, error, args)
            completed_count += 1
            if error:
                print(f"  [WARN] Remote row {row.source_row}: {error}")
            print(f"  Remote rollout {completed_count}/{len(rows)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行单个 local/remote rollout，并输出统一格式 checkpoint"
    )
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    parser.add_argument("--name", required=True, help="checkpoint 名称（不含路由前缀和扩展名）")
    parser.add_argument("--data-path", required=True, help="输入 VERL Parquet")
    parser.add_argument("--out-dir", required=True, help="checkpoint 输出目录")
    parser.add_argument("--model-path", help="local 模型目录；remote 模式不需要")
    parser.add_argument(
        "--internvl", action="store_true",
        help="local 模式使用 InternVL；不指定时默认按 Qwen 加载",
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--selection", choices=("random", "first"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--reveal-gt-answer", action="store_true")
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remote-workers", type=int, default=10)
    parser.add_argument("--gemini-max-retries", type=int, default=3)
    parser.add_argument(
        "--gemini-thinking-level", choices=("low", "medium", "high"), default="high"
    )
    parser.add_argument(
        "--gemini-oss-prefix", default="yk/ai-material/neo/fjj/rollout"
    )
    parser.add_argument("--gemini-max-image-edge", type=int, default=2048)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    args = parser.parse_args(argv)

    try:
        checkpoint_filename(args.mode, args.name)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "local" and not args.model_path:
        parser.error("local 模式必须指定 --model-path")
    if args.mode == "remote" and args.model_path:
        parser.error("remote 模式不使用 --model-path")
    if args.mode == "remote" and args.internvl:
        parser.error("--internvl 只能用于 local 模式")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.remote_workers <= 0:
        parser.error("num-samples、batch-size、remote-workers 必须大于 0")
    if args.max_prompt_length <= 0 or args.max_new_tokens <= 0:
        parser.error("prompt/response 长度必须大于 0")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature 必须非负，top-p 必须在 (0, 1] 内")
    if args.repetition_penalty <= 0:
        parser.error("repetition-penalty 必须大于 0")
    if args.gemini_max_retries <= 0 or args.gemini_max_image_edge < 128:
        parser.error("Gemini 重试次数必须大于 0，图片最长边不能小于 128")
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality 必须在 30 到 100 之间")
    return args


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not data_path.is_file() or data_path.suffix.lower() != ".parquet":
        raise SystemExit(f"ERROR: Parquet 不存在或扩展名不正确: {data_path}")
    args.data_path = str(data_path)
    if args.mode == "local":
        model_path = Path(args.model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise SystemExit(f"ERROR: 模型目录不存在: {model_path}")
        args.model_path = str(model_path)
    else:
        args.model_path = None
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / checkpoint_filename(args.mode, args.name)
    if checkpoint_path.exists() and args.overwrite:
        checkpoint_path.unlink()
        print(f"覆盖 checkpoint: {checkpoint_path}")

    print(f"读取数据: {data_path}")
    rows, total = legacy.load_eval_rows(
        data_path, args.num_samples, args.selection, args.seed, args.start_row
    )
    completed = read_checkpoint(checkpoint_path)
    validate_checkpoint(completed, rows, args)
    pending = [
        row for row in rows
        if row.source_row not in completed
        or (args.retry_errors and bool(completed[row.source_row].get("error")))
    ]
    print(
        f"数据共 {total} 条；选中 {len(rows)} 条；checkpoint 已完成 "
        f"{len(rows) - len(pending)} 条，本次生成 {len(pending)} 条"
    )
    if pending:
        if args.mode == "local":
            run_local(rows, pending, checkpoint_path, args)
        else:
            run_remote(rows, pending, checkpoint_path, out_dir, args)
    else:
        print("checkpoint 已完整，无需调用模型。")
    print(f"完成: {checkpoint_path}")


if __name__ == "__main__":
    main()
