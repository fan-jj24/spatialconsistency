#!/usr/bin/env python3
"""给 RL_qwen Parquet 的 ``extra_info`` 补齐 ``option_a/option_b``。

脚本扫描 ``rl.sh`` 使用的五个数据集目录。四个 reasoning gate 数据集的
每一行都会从顶层 ``prompt`` 提取 A/B 选项，并写入：

    extra_info["option_a"]
    extra_info["option_b"]

detection 没有 answer/reasoning gate，因此只统计、不修改。默认是只读检查；
只有显式传入 ``--apply`` 才会写临时文件、完整复查后原子替换原 Parquet。
训练和验证文件都需要处理，因为二者使用同一个 reward function。

用法：

    # 先检查全部数据，不修改文件
    python3 fix_parquet_extra_info_options.py \
        --data-root /home/deepspeed/model_output/RL_qwen

    # 检查通过后原地修复；可选地先把原文件备份到指定目录
    python3 fix_parquet_extra_info_options.py \
        --data-root /home/deepspeed/model_output/RL_qwen \
        --apply \
        --backup-dir /home/deepspeed/model_output/RL_qwen_before_option_fix

依赖：pyarrow（训练环境的 VERL requirements 已包含）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import uuid
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("/home/deepspeed/model_output/RL_qwen")
DATASET_NAMES = (
    "consistent_qwen",
    "inconsistent_qwen",
    "detection_qwen",
    "consistent_all",
    "inconsistent_all",
)

# 与 json_answer_reward.py 中当前训练/验证 reasoning gate 路由一致。
REASONING_GATE_SOURCES = {
    "spatial_consistency_pos",
    "spatial_consistency_pos_train",
    "spatial_consistency_neg",
    "spatial_consistency_neg_train",
    "spatial_consistency_bbox_pos",
    "spatial_consistency_bbox_pos_train",
    "spatial_consistency_bbox_neg",
    "spatial_consistency_bbox_neg_train",
    "consistent_qwen_eval",
    "consistent_qwen",
    "inconsistent_qwen_eval",
    "inconsistent_qwen",
    "inconsistent_cot_local_eval",
}
DETECTION_SOURCES = {
    "spatial_detection",
    "spatial_detection_train",
    "detection_qwen_eval",
    "detection_qwen",
}

# 支持 A. / A) / A: / (A) 以及 OPTION A:，每个选项必须独占一行。
_OPTION_LINE_RE = re.compile(
    r"^[ \t]*(?:OPTION[ \t]+)?(?:\(([AB])\)|([AB])[ \t]*[.．):：])"
    r"[ \t]*(.+?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


class DataValidationError(ValueError):
    """训练池内容不满足 reasoning gate 输入要求。"""


@dataclass
class FileStats:
    rows: int = 0
    gated_rows: int = 0
    detection_rows: int = 0
    added_rows: int = 0
    already_present_rows: int = 0

    def add(self, other: "FileStats") -> None:
        self.rows += other.rows
        self.gated_rows += other.gated_rows
        self.detection_rows += other.detection_rows
        self.added_rows += other.added_rows
        self.already_present_rows += other.already_present_rows


def _plain(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    return value


def _prompt_text(value: Any) -> str:
    """把 VERL prompt/messages/multimodal content 展平为文本。"""
    value = _plain(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [_prompt_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key in ("content", "text", "prompt", "messages"):
            if key in value:
                part = _prompt_text(value[key])
                if part:
                    parts.append(part)
        return "\n".join(parts)
    return ""


def extract_ab_options(prompt: Any) -> tuple[str, str]:
    """从一行顶层 prompt 提取唯一的 A/B 选项。"""
    prompt_text = _prompt_text(prompt)
    if not prompt_text.strip():
        raise DataValidationError("prompt 中没有可用文本")

    matches: dict[str, list[str]] = {"A": [], "B": []}
    for parenthesized, plain, text in _OPTION_LINE_RE.findall(prompt_text):
        letter = (parenthesized or plain).upper()
        option = text.strip()
        if option and option not in matches[letter]:
            matches[letter].append(option)

    missing = [letter for letter in ("A", "B") if not matches[letter]]
    if missing:
        raise DataValidationError(
            "无法从 prompt 提取完整 A/B 选项；缺少 " + ", ".join(missing)
        )
    ambiguous = {
        letter: values for letter, values in matches.items() if len(values) != 1
    }
    if ambiguous:
        details = "; ".join(
            f"{letter}={values!r}" for letter, values in ambiguous.items()
        )
        raise DataValidationError(f"prompt 中存在多组不同选项，无法安全判断：{details}")

    option_a, option_b = matches["A"][0], matches["B"][0]
    if option_a == option_b:
        raise DataValidationError("A/B 选项文本完全相同")
    return option_a, option_b


def enrich_extra_info(
    data_source: Any, prompt: Any, extra_info: Any
) -> tuple[dict[str, Any] | None, str]:
    """返回补齐后的 extra_info 及行状态；纯函数，便于独立测试。"""
    data_source = _plain(data_source)
    if not isinstance(data_source, str) or not data_source.strip():
        raise DataValidationError(f"data_source 非法：{data_source!r}")
    data_source = data_source.strip()

    if data_source in DETECTION_SOURCES:
        return _plain(extra_info), "detection"
    if data_source not in REASONING_GATE_SOURCES:
        raise DataValidationError(
            f"data_source {data_source!r} 既不属于 reasoning gate，也不属于 detection"
        )

    option_a, option_b = extract_ab_options(prompt)
    extra_info = _plain(extra_info)
    if extra_info is None:
        enriched: dict[str, Any] = {}
    elif isinstance(extra_info, dict):
        enriched = dict(extra_info)
    else:
        raise DataValidationError(
            f"extra_info 必须是 struct/dict 或 null，实际为 {type(extra_info).__name__}"
        )

    old_a = enriched.get("option_a")
    old_b = enriched.get("option_b")
    had_both = (
        isinstance(old_a, str)
        and bool(old_a.strip())
        and isinstance(old_b, str)
        and bool(old_b.strip())
    )
    if had_both and (old_a.strip(), old_b.strip()) != (option_a, option_b):
        raise DataValidationError(
            "extra_info 已有 A/B 与 prompt 不一致："
            f"extra_info={(old_a.strip(), old_b.strip())!r}，"
            f"prompt={(option_a, option_b)!r}"
        )

    enriched["option_a"] = option_a
    enriched["option_b"] = option_b
    return enriched, "present" if had_both else "added"


def discover_parquet_files(data_root: Path) -> tuple[list[Path], list[str]]:
    """扫描 rl.sh 声明的五个目录内全部 Parquet（含 train/val/val1）。"""
    files: list[Path] = []
    missing: list[str] = []
    for name in DATASET_NAMES:
        dataset_dir = data_root / name
        if not dataset_dir.is_dir():
            missing.append(name)
            continue
        files.extend(sorted(dataset_dir.glob("*.parquet")))
    return sorted(set(files)), missing


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "ERROR: 缺少 pyarrow；请在 VERL 训练环境中运行此脚本"
        ) from exc
    return pa, pq


def _target_schema(pa: Any, schema: Any) -> Any:
    index = schema.get_field_index("extra_info")
    option_fields = [
        pa.field("option_a", pa.string()),
        pa.field("option_b", pa.string()),
    ]
    if index < 0:
        return schema.append(pa.field("extra_info", pa.struct(option_fields)))

    old_field = schema.field(index)
    if not pa.types.is_struct(old_field.type):
        raise DataValidationError(
            f"extra_info 列必须是 struct，实际 schema 为 {old_field.type}"
        )
    fields = list(old_field.type)
    names = {field.name for field in fields}
    for field in option_fields:
        if field.name not in names:
            fields.append(field)
            continue
        existing = old_field.type.field(field.name)
        if not (
            pa.types.is_string(existing.type)
            or pa.types.is_large_string(existing.type)
        ):
            raise DataValidationError(
                f"extra_info.{field.name} 必须是字符串，实际为 {existing.type}"
            )
    new_field = pa.field(
        old_field.name,
        pa.struct(fields),
        nullable=old_field.nullable,
        metadata=old_field.metadata,
    )
    return schema.set(index, new_field)


def _iter_enriched_row_groups(
    parquet_file: Any, target_schema: Any, pa: Any, path: Path
) -> Iterable[tuple[Any, FileStats]]:
    source_schema = parquet_file.schema_arrow
    source_extra_index = source_schema.get_field_index("extra_info")
    target_extra_index = target_schema.get_field_index("extra_info")
    target_extra_field = target_schema.field(target_extra_index)

    for group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(group_index)
        required = {"data_source", "prompt"}
        missing = required.difference(table.column_names)
        if missing:
            raise DataValidationError(
                f"缺少顶层列：{', '.join(sorted(missing))}"
            )

        sources = table.column("data_source").to_pylist()
        prompts = table.column("prompt").to_pylist()
        if source_extra_index >= 0:
            extras = table.column("extra_info").to_pylist()
        else:
            extras = [None] * table.num_rows

        enriched_extras = []
        stats = FileStats(rows=table.num_rows)
        for row_offset, (source, prompt, extra) in enumerate(
            zip(sources, prompts, extras, strict=True)
        ):
            try:
                enriched, status = enrich_extra_info(source, prompt, extra)
            except DataValidationError as exc:
                # row group 的局部行号用于低内存处理；同时给出文件全局行号。
                before = sum(
                    parquet_file.metadata.row_group(i).num_rows
                    for i in range(group_index)
                )
                raise DataValidationError(
                    f"{path}: row={before + row_offset}, data_source={source!r}: {exc}"
                ) from exc
            enriched_extras.append(enriched)
            if status == "detection":
                stats.detection_rows += 1
            else:
                stats.gated_rows += 1
                if status == "added":
                    stats.added_rows += 1
                else:
                    stats.already_present_rows += 1

        extra_array = pa.array(enriched_extras, type=target_extra_field.type)
        if source_extra_index < 0:
            table = table.append_column(target_extra_field, extra_array)
        else:
            table = table.set_column(
                source_extra_index, target_extra_field, extra_array
            )
        yield table.cast(target_schema), stats


def _compression_codec(parquet_file: Any) -> str | None:
    metadata = parquet_file.metadata
    if metadata.num_row_groups and metadata.num_columns:
        codec = metadata.row_group(0).column(0).compression.lower()
        return None if codec == "uncompressed" else codec
    return "snappy"


def process_file(path: Path, output: Path | None = None) -> FileStats:
    """检查一个文件；output 非空时写出修复结果。"""
    pa, pq = _import_pyarrow()
    parquet_file = pq.ParquetFile(path)
    schema = _target_schema(pa, parquet_file.schema_arrow)
    total = FileStats()
    writer = None
    try:
        if output is not None:
            writer = pq.ParquetWriter(
                output,
                schema,
                compression=_compression_codec(parquet_file),
                use_dictionary=True,
            )
        for table, stats in _iter_enriched_row_groups(
            parquet_file, schema, pa, path
        ):
            total.add(stats)
            if writer is not None:
                writer.write_table(table, row_group_size=max(1, table.num_rows))
    finally:
        if writer is not None:
            writer.close()
    return total


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.option-fix-{uuid.uuid4().hex}.tmp")


def _verify_output(original: Path, fixed: Path, expected: FileStats) -> None:
    actual = process_file(fixed)
    if actual.rows != expected.rows or actual.gated_rows != expected.gated_rows:
        raise DataValidationError(
            f"{original}: 写后校验行数不一致，expected={expected}, actual={actual}"
        )
    if actual.added_rows != 0:
        raise DataValidationError(
            f"{original}: 写后仍有 {actual.added_rows} 行缺少 option_a/option_b"
        )


def _print_stats(path: Path, stats: FileStats, data_root: Path) -> None:
    try:
        label = path.relative_to(data_root)
    except ValueError:
        label = path
    print(
        f"  {label}: rows={stats.rows}, gated={stats.gated_rows}, "
        f"need_add={stats.added_rows}, present={stats.already_present_rows}, "
        f"detection={stats.detection_rows}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="RL_qwen 数据根目录"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="原子替换原 Parquet；不传时只执行只读检查",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="--apply 前将原 Parquet 按相对路径备份到此目录",
    )
    parser.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help="允许 rl.sh 声明的部分数据集目录不存在",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    if args.backup_dir is not None and not args.apply:
        raise SystemExit("ERROR: --backup-dir 只能与 --apply 一起使用")
    if not data_root.is_dir():
        raise SystemExit(f"ERROR: 数据根目录不存在：{data_root}")

    files, missing_datasets = discover_parquet_files(data_root)
    if missing_datasets and not args.allow_missing_datasets:
        raise SystemExit(
            "ERROR: 缺少 rl.sh 声明的数据集目录："
            + ", ".join(missing_datasets)
            + "；如属预期请加 --allow-missing-datasets"
        )
    for name in missing_datasets:
        print(f"WARN: 数据集目录不存在，已跳过：{name}", file=sys.stderr)
    if not files:
        raise SystemExit(f"ERROR: {data_root} 的训练池目录下没有 Parquet 文件")

    mode = "写入修复" if args.apply else "只读检查"
    print(f"[{mode}] data_root={data_root}, files={len(files)}")

    total = FileStats()
    prepared: list[tuple[Path, Path, FileStats]] = []
    try:
        for path in files:
            if args.apply:
                temporary = _temporary_path(path)
                try:
                    stats = process_file(path, temporary)
                    _verify_output(path, temporary, stats)
                    if stats.added_rows:
                        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
                        prepared.append((path, temporary, stats))
                    else:
                        # 已完整修复的文件和纯 detection 文件保持字节不变。
                        temporary.unlink()
                except Exception:
                    if temporary.exists():
                        temporary.unlink()
                    raise
            else:
                stats = process_file(path)
            total.add(stats)
            _print_stats(path, stats, data_root)

        if args.apply and args.backup_dir is not None:
            backup_root = args.backup_dir.resolve()
            if backup_root == data_root or data_root in backup_root.parents:
                raise DataValidationError(
                    "backup-dir 不能位于 data-root 内，避免再次扫描或递归备份"
                )
            destinations = [
                backup_root / path.relative_to(data_root)
                for path, _, _ in prepared
            ]
            existing = [
                destination for destination in destinations if destination.exists()
            ]
            if existing:
                raise DataValidationError(
                    "备份文件已存在，未开始备份：" + ", ".join(map(str, existing))
                )
            for path, _, _ in prepared:
                destination = backup_root / path.relative_to(data_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

        if args.apply:
            for path, temporary, _ in prepared:
                os.replace(temporary, path)
    except Exception:
        for _, temporary, _ in prepared:
            if temporary.exists():
                temporary.unlink()
        raise

    print(
        f"完成：rows={total.rows}, gated={total.gated_rows}, "
        f"need_add={total.added_rows}, present={total.already_present_rows}, "
        f"detection={total.detection_rows}"
    )
    if not args.apply:
        if total.added_rows:
            print("未修改文件；确认统计无误后添加 --apply 执行修复。")
        else:
            print("未修改文件；所有 reasoning gate 行均已包含正确 A/B。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataValidationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
