#!/usr/bin/env python3
"""Generate an interactive, reward-free annotation report for one val step.

Only these source datasets are included:

* ``inconsistent_cot_verl_2500``
* ``inconsistent_detection_verl_cot_2500``

Records keep their original order in the validation JSONL.  The source data is
used only to trace each record back to its dataset and to load its images.  No
reward module, reward model, bbox matching, or reward calculation is involved.

Example::

    python3 annotate_val_cases.py \
      --val_dir /data/run/val_generations \
      --step 100 \
      --data_root /data/RL1 \
      --out_dir /data/run/annotations_step_100

The generated ``index.html`` stores annotations in browser localStorage and can
export/import JSONL, so it can be used as a static local report on Windows too.
Pillow is required; Parquet source data additionally requires pyarrow.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import html as html_mod
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


TARGET_SOURCES = (
    "inconsistent_cot_verl_2500",
    "inconsistent_detection_verl_cot_2500",
)

_TEMPLATE_TOKEN_RE = re.compile(
    r"<\|(?:im|vision)_(?:start|end)\|>|<\|image_pad\|>"
    r"|<\|(?:eot_id|endoftext|end)\|>|<\|(?:assistant|user|system)\|>"
    r"|<image>|\[image\]|\s+",
    re.IGNORECASE,
)
_VECTOR_RE = re.compile(r"\(([^()]*)\)")


def _plain(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    return value


def _text_parts(value: Any) -> list[str]:
    value = _plain(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return [value["text"]]
        result: list[str] = []
        for key in ("content", "prompt", "messages"):
            if key in value:
                result.extend(_text_parts(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_text_parts(item))
        return result
    return []


def prompt_key(value: Any) -> str:
    """Build the same normalized prompt key for source and val records."""
    text = "\n".join(_text_parts(value)).strip().lower()
    text = re.sub(r"(?m)^\s*(?:system|user|assistant)\s*$", "", text)
    text = re.sub(
        r"<\|im_start\|>\s*(?:system|user|assistant)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return _TEMPLATE_TOKEN_RE.sub("", text)


def _ground_truth(row: dict[str, Any]) -> Any:
    for key in ("gts", "ground_truth", "gt"):
        if row.get(key) is not None:
            return _plain(row[key])
    reward_model = _plain(row.get("reward_model"))
    if isinstance(reward_model, dict):
        return _plain(reward_model.get("ground_truth"))
    return None


def _json_text(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical_gt(value: Any) -> str:
    text = _json_text(value).strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return re.sub(r"\s+", " ", text)
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 无效: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} 必须是 JSON object")
            value["__jsonl_line__"] = line_no
            records.append(value)
    return records


def _target_directories(data_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if data_root.name in TARGET_SOURCES and data_root.is_dir():
        found[data_root.name] = data_root
    for name in TARGET_SOURCES:
        direct = data_root / name
        if direct.is_dir():
            found[name] = direct
            continue
        hits = sorted(path for path in data_root.rglob(name) if path.is_dir())
        if hits:
            found[name] = hits[0]
    missing = [name for name in TARGET_SOURCES if name not in found]
    if missing:
        raise FileNotFoundError(
            f"源数据目录中找不到: {', '.join(missing)}；data_root={data_root}"
        )
    return found


@dataclass(frozen=True)
class SourceEntry:
    key: str
    canonical_gt: str
    ground_truth: str
    source: str
    path: Path
    row_index: int


class SourceIndex:
    def __init__(self, entries: Iterable[SourceEntry]):
        self.by_key: dict[str, list[SourceEntry]] = defaultdict(list)
        for entry in entries:
            self.by_key[entry.key].append(entry)
        self.keys_by_length = sorted(self.by_key, key=len, reverse=True)

    def match(self, record: dict[str, Any]) -> SourceEntry | None:
        key = prompt_key(record.get("input", record.get("prompt", "")))
        candidates = list(self.by_key.get(key, ()))
        if not candidates and key:
            # Decoded VERL input can contain extra chat-template text.  This is
            # IVG's prompt trace-back with a longest-substring fallback.
            for source_key in self.keys_by_length:
                if len(source_key) >= 24 and source_key in key:
                    candidates = list(self.by_key[source_key])
                    break
        if not candidates:
            return None
        gt = _canonical_gt(_ground_truth(record))
        if gt and len(candidates) > 1:
            same_gt = [entry for entry in candidates if entry.canonical_gt == gt]
            if same_gt:
                candidates = same_gt
        return candidates[0]


def _iter_jsonl_index_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{row_index + 1} JSON 无效: {exc}") from exc
            if isinstance(row, dict):
                yield row_index, row


def _iter_parquet_index_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("读取 Parquet 源数据需要安装 pyarrow") from exc
    parquet_file = pq.ParquetFile(path)
    names = set(parquet_file.schema_arrow.names)
    wanted = [
        name for name in ("prompt", "input", "messages", "reward_model",
                          "gts", "ground_truth", "gt") if name in names
    ]
    if not wanted:
        return
    row_offset = 0
    for group_index in range(parquet_file.num_row_groups):
        rows = parquet_file.read_row_group(group_index, columns=wanted).to_pylist()
        for local_index, row in enumerate(rows):
            if isinstance(row, dict):
                yield row_offset + local_index, row
        row_offset += len(rows)


def build_source_index(data_root: Path) -> SourceIndex:
    entries: list[SourceEntry] = []
    target_dirs = _target_directories(data_root)
    for source in TARGET_SOURCES:
        source_dir = target_dirs[source]
        files = sorted(source_dir.rglob("*.jsonl")) + sorted(source_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{source_dir} 下没有 JSONL 或 Parquet")
        source_count = 0
        for path in files:
            iterator = (_iter_jsonl_index_rows(path) if path.suffix.lower() == ".jsonl"
                        else _iter_parquet_index_rows(path))
            for row_index, row in iterator:
                key = prompt_key(row.get("prompt", row.get("input", row.get("messages", ""))))
                if not key:
                    continue
                gt = _json_text(_ground_truth(row))
                entries.append(SourceEntry(
                    key=key,
                    canonical_gt=_canonical_gt(gt),
                    ground_truth=gt,
                    source=source,
                    path=path,
                    row_index=row_index,
                ))
                source_count += 1
        print(f"  {source}: {source_count} 条可索引记录")
    if not entries:
        raise RuntimeError("两个目标数据集中没有可索引的 prompt")
    return SourceIndex(entries)


@dataclass
class Case:
    order: int
    jsonl_line: int
    source: str
    source_path: Path
    source_row: int
    ground_truth: str
    prediction: str
    image_paths: list[str]
    image_error: str = ""

    @property
    def case_id(self) -> str:
        return f"{self.source}:{self.jsonl_line}:{self.source_path.name}:{self.source_row}"


def collect_cases(records: list[dict[str, Any]], index: SourceIndex) -> tuple[list[Case], int]:
    cases: list[Case] = []
    unmatched = 0
    for record in records:
        entry = index.match(record)
        if entry is None:
            unmatched += 1
            continue
        gt = _json_text(_ground_truth(record)) or entry.ground_truth
        output = record.get("output", record.get("response", record.get("prediction", "")))
        cases.append(Case(
            order=len(cases),
            jsonl_line=int(record["__jsonl_line__"]),
            source=entry.source,
            source_path=entry.path,
            source_row=entry.row_index,
            ground_truth=gt,
            prediction=_json_text(output),
            image_paths=[],
        ))
    return cases, unmatched


def _extract_boxes(text: str) -> list[tuple[list[float], str]]:
    """Parse every valid bbox entry from the last JSON object containing boxes."""
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r"\{", text or ""):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("boxes"), list):
            objects.append(obj)
    if not objects:
        return []
    result = []
    for item in objects[-1]["boxes"]:
        if isinstance(item, dict):
            bbox, label = item.get("bbox"), item.get("label", "")
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            bbox, label = item, ""
        else:
            continue
        bbox = _plain(bbox)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            coords = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in coords):
            result.append((coords, str(label)))
    return result


def _image_values(row: dict[str, Any]) -> list[Any]:
    value = _plain(row.get("images", row.get("image")))
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _read_jsonl_rows(path: Path, row_indices: set[int]) -> dict[int, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index not in row_indices:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                result[row_index] = row
            if len(result) == len(row_indices):
                break
    return result


def _read_parquet_rows(path: Path, row_indices: set[int]) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("读取 Parquet 图片需要安装 pyarrow") from exc
    parquet_file = pq.ParquetFile(path)
    image_column = next(
        (name for name in ("images", "image") if name in parquet_file.schema_arrow.names),
        None,
    )
    if image_column is None:
        return {row_index: {} for row_index in row_indices}
    result = {}
    offset = 0
    for group_index in range(parquet_file.num_row_groups):
        count = parquet_file.metadata.row_group(group_index).num_rows
        wanted = [index for index in row_indices if offset <= index < offset + count]
        if wanted:
            rows = parquet_file.read_row_group(group_index, columns=[image_column]).to_pylist()
            for row_index in wanted:
                row = rows[row_index - offset]
                result[row_index] = row if isinstance(row, dict) else {}
        offset += count
    return result


def _decode_image(value: Any, source_path: Path):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("生成图片需要安装 Pillow: pip install pillow") from exc

    value = _plain(value)
    if isinstance(value, Image.Image):
        image = value.copy()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(bytes(value)))
    elif isinstance(value, str):
        if value.startswith("data:image/") and "," in value:
            image = Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1])))
        else:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = source_path.parent / path
            image = Image.open(path)
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            raw = value["bytes"]
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            image = Image.open(io.BytesIO(bytes(raw)))
        elif value.get("path"):
            path = Path(str(value["path"])).expanduser()
            if not path.is_absolute():
                path = source_path.parent / path
            image = Image.open(path)
        elif value.get("image") is not None:
            return _decode_image(value["image"], source_path)
        else:
            raise ValueError(f"无法识别图片字典字段: {sorted(value)}")
    else:
        raise TypeError(f"不支持的图片类型: {type(value).__name__}")
    return ImageOps.exif_transpose(image).convert("RGB")


def _resize_image(image, max_edge: int):
    if max(image.size) <= max_edge:
        return image.copy()
    try:
        from PIL.Image import Resampling
        resample = Resampling.LANCZOS
    except ImportError:
        from PIL import Image
        resample = Image.LANCZOS
    ratio = max_edge / max(image.size)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(size, resample)


def _scaled_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    return [
        bbox[0] * width / 1000.0,
        bbox[1] * height / 1000.0,
        bbox[2] * width / 1000.0,
        bbox[3] * height / 1000.0,
    ]


def _dashed_line(draw, start, end, fill, width: int, dash: int = 10) -> None:
    x1, y1 = start
    x2, y2 = end
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance <= 0:
        return
    ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
    position = 0.0
    while position < distance:
        finish = min(position + dash, distance)
        draw.line(
            [(x1 + ux * position, y1 + uy * position),
             (x1 + ux * finish, y1 + uy * finish)],
            fill=fill,
            width=width,
        )
        position += dash * 2


def _dashed_rectangle(draw, bbox, fill, width: int) -> None:
    x1, y1, x2, y2 = bbox
    _dashed_line(draw, (x1, y1), (x2, y1), fill, width)
    _dashed_line(draw, (x2, y1), (x2, y2), fill, width)
    _dashed_line(draw, (x2, y2), (x1, y2), fill, width)
    _dashed_line(draw, (x1, y2), (x1, y1), fill, width)


def _direction(label: str, bbox: list[float]) -> tuple[float, float] | None:
    match = _VECTOR_RE.search(label)
    if not match:
        return None
    try:
        values = [float(part) for part in re.split(r"[,\s]+", match.group(1).strip()) if part]
    except ValueError:
        return None
    if len(values) == 2:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return values[0] - cx, values[1] - cy
    if len(values) == 3:
        return values[0], -values[1]
    return None


def _arrow(draw, center, direction, color, width: int, sx: float, sy: float) -> None:
    dx, dy = direction[0] * sx, direction[1] * sy
    magnitude = math.hypot(dx, dy)
    if magnitude < 1e-6:
        return
    scale = min(1.0, 140.0 / magnitude)
    dx, dy = dx * scale, dy * scale
    end = (center[0] + dx, center[1] + dy)
    draw.line([center, end], fill=color, width=width)
    angle = math.atan2(dy, dx)
    head = max(9, width * 3)
    points = [end]
    for delta in (0.52, -0.52):
        points.append((end[0] - head * math.cos(angle + delta),
                       end[1] - head * math.sin(angle + delta)))
    draw.polygon(points, fill=color)


def annotate_second_image(image, gt_boxes, pred_boxes):
    """Draw every GT and prediction independently; there is no bbox matching."""
    from PIL import ImageDraw

    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    line_width = max(2, round(max(width, height) / 350))
    gt_color = (225, 45, 45)
    pred_color = (35, 100, 225)

    for index, (bbox, label) in enumerate(gt_boxes, 1):
        scaled = _scaled_bbox(bbox, width, height)
        draw.rectangle(scaled, outline=gt_color, width=line_width)
        draw.text((scaled[0] + 3, scaled[1] + 3), f"GT {index}", fill=gt_color,
                  stroke_width=2, stroke_fill=(255, 255, 255))
        direction = _direction(label, bbox)
        if direction:
            center = ((scaled[0] + scaled[2]) / 2, (scaled[1] + scaled[3]) / 2)
            _arrow(draw, center, direction, gt_color, line_width,
                   width / 1000.0, height / 1000.0)

    for index, (bbox, label) in enumerate(pred_boxes, 1):
        scaled = _scaled_bbox(bbox, width, height)
        _dashed_rectangle(draw, scaled, pred_color, line_width)
        draw.text((scaled[0] + 3, scaled[1] + 19), f"P {index}", fill=pred_color,
                  stroke_width=2, stroke_fill=(255, 255, 255))
        direction = _direction(label, bbox)
        if direction:
            center = ((scaled[0] + scaled[2]) / 2, (scaled[1] + scaled[3]) / 2)
            _arrow(draw, center, direction, pred_color, line_width,
                   width / 1000.0, height / 1000.0)
    return result


def _save_jpeg(image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, optimize=True)


def materialize_images(cases: list[Case], out_dir: Path, max_edge: int, quality: int) -> None:
    by_path: dict[Path, list[Case]] = defaultdict(list)
    for case in cases:
        by_path[case.source_path].append(case)
    assets_dir = out_dir / "assets"
    total = len(cases)
    completed = 0
    fallback_count = 0
    failed_count = 0

    for source_path, file_cases in sorted(by_path.items(), key=lambda item: str(item[0])):
        indices = {case.source_row for case in file_cases}
        try:
            rows = (_read_jsonl_rows(source_path, indices)
                    if source_path.suffix.lower() == ".jsonl"
                    else _read_parquet_rows(source_path, indices))
        except Exception as exc:
            for case in file_cases:
                case.image_error = f"读取源文件失败: {exc}"
            failed_count += len(file_cases)
            completed += len(file_cases)
            print(f"  图片 {completed}/{total}")
            continue

        for case in file_cases:
            try:
                row = rows.get(case.source_row, {})
                values = _image_values(row)
                if not values:
                    raise ValueError("源行没有 images/image 字段")
                images = [_resize_image(_decode_image(value, source_path), max_edge)
                          for value in values]
                target_index = 1 if len(images) >= 2 else len(images) - 1
                if len(images) < 2:
                    fallback_count += 1
                    case.image_error = "源行不足两幅图，框已画在唯一图片上"
                gt_boxes = _extract_boxes(case.ground_truth)
                pred_boxes = _extract_boxes(case.prediction)
                images[target_index] = annotate_second_image(
                    images[target_index], gt_boxes, pred_boxes)
                for image_index, image in enumerate(images):
                    name = f"case_{case.order + 1:06d}_img_{image_index + 1}.jpg"
                    _save_jpeg(image, assets_dir / name, quality)
                    case.image_paths.append(f"assets/{name}")
            except Exception as exc:
                case.image_error = f"图片处理失败: {exc}"
                failed_count += 1
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"  图片 {completed}/{total}")
    if fallback_count:
        print(f"  [WARN] {fallback_count} 条不足两幅图，改画在唯一图片上")
    if failed_count:
        print(f"  [WARN] {failed_count} 条图片处理失败；HTML 中会显示原因")


CSS = r"""
:root {
  --bg:#f4f6f8; --surface:#fff; --ink:#17202a; --muted:#667085;
  --line:#dfe3e8; --accent:#2457c5; --good:#18794e; --bad:#c43232;
  --warn:#a86400; --radius:8px;
}
@media (prefers-color-scheme:dark) {
  :root { --bg:#12161d; --surface:#1b212b; --ink:#edf0f5; --muted:#9ba5b4;
    --line:#343c48; --accent:#78a2ff; --good:#55c991; --bad:#ff7b72; --warn:#e5ad54; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
button,select,input,textarea { font:inherit; }
.top { position:sticky; top:0; z-index:10; background:color-mix(in srgb,var(--surface) 96%,transparent);
  backdrop-filter:blur(10px); border-bottom:1px solid var(--line); }
.top-inner,.main { max-width:1440px; margin:auto; padding:12px 18px; }
.title-row,.controls,.nav,.verdicts,.stats { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
h1 { font-size:18px; margin:0; }
.grow { flex:1; }
.muted { color:var(--muted); }
.mono { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-variant-numeric:tabular-nums; }
.stats { margin-top:8px; gap:14px; font-size:12px; }
.stat b { font-size:14px; }
button,select,input[type=number] { border:1px solid var(--line); border-radius:6px;
  background:var(--surface); color:var(--ink); padding:7px 10px; }
button { cursor:pointer; }
button:hover { border-color:var(--accent); }
button.primary { background:var(--accent); color:white; border-color:var(--accent); }
.controls { margin-top:10px; }
.main { padding-top:18px; padding-bottom:80px; }
.case { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:16px; }
.meta { display:flex; flex-wrap:wrap; gap:8px 18px; color:var(--muted); margin-bottom:12px; }
.images { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr)); gap:12px; }
.image-wrap { min-height:160px; border:1px solid var(--line); border-radius:var(--radius);
  background:#0d1117; display:flex; flex-direction:column; overflow:hidden; }
.image-label { padding:5px 8px; color:#d1d7e0; background:#171d26; font-size:12px; }
.image-wrap img { display:block; width:100%; height:auto; max-height:68vh; object-fit:contain; flex:1; }
.legend { margin:10px 0; color:var(--muted); font-size:12px; }
.swatch { display:inline-block; width:18px; height:3px; vertical-align:3px; margin:0 4px 0 10px; }
.swatch:first-child { margin-left:0; }
.text-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
.panel { min-width:0; }
.panel h2 { font-size:13px; margin:0 0 5px; }
pre { margin:0; padding:10px; min-height:110px; max-height:300px; overflow:auto;
  border:1px solid var(--line); border-radius:6px; background:var(--bg); white-space:pre-wrap; word-break:break-word; }
.annotation { margin-top:14px; border-top:1px solid var(--line); padding-top:14px; }
.verdict { min-width:86px; font-weight:650; }
.verdict.good.active { background:var(--good); color:white; border-color:var(--good); }
.verdict.bad.active { background:var(--bad); color:white; border-color:var(--bad); }
.verdict.unsure.active { background:var(--warn); color:white; border-color:var(--warn); }
textarea { width:100%; min-height:64px; margin-top:10px; resize:vertical; border:1px solid var(--line);
  border-radius:6px; background:var(--surface); color:var(--ink); padding:8px; }
.error { margin:10px 0; color:var(--bad); }
.empty { text-align:center; padding:80px 20px; color:var(--muted); }
.kbd { border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 4px; font-size:11px; }
@media (max-width:760px) {
  .top-inner,.main { padding-left:10px; padding-right:10px; }
  .text-grid { grid-template-columns:1fr; }
  .controls > * { flex:1 1 auto; }
  .verdicts button { flex:1; }
  pre { max-height:240px; }
}
"""


JS = r"""
const stateKey = `val-case-annotations:${REPORT.id}`;
let annotations = {};
try { annotations = JSON.parse(localStorage.getItem(stateKey) || '{}') || {}; } catch (_) {}
let visible = [];
let position = 0;

const $ = id => document.getElementById(id);

function annotationFor(c) { return annotations[c.id] || {verdict:'', note:''}; }
function save() {
  try { localStorage.setItem(stateKey, JSON.stringify(annotations)); } catch (_) {}
  updateStats();
}
function selectedCase() { return visible.length ? CASES[visible[position]] : null; }

function computeStats(source='') {
  const rows = source ? CASES.filter(c => c.source === source) : CASES;
  let correct=0, incorrect=0, uncertain=0;
  rows.forEach(c => {
    const v = annotationFor(c).verdict;
    if (v === 'correct') correct++;
    else if (v === 'incorrect') incorrect++;
    else if (v === 'uncertain') uncertain++;
  });
  const decided = correct + incorrect;
  return {total:rows.length, correct, incorrect, uncertain,
    labeled:correct+incorrect+uncertain, accuracy:decided ? correct/decided : null};
}
function fmtStats(label, s) {
  const acc = s.accuracy === null ? 'n/a' : `${(100*s.accuracy).toFixed(1)}%`;
  return `<span class="stat"><b>${label}</b> ${s.labeled}/${s.total} 已标 · `+
    `<span style="color:var(--good)">${s.correct} 对</span> · `+
    `<span style="color:var(--bad)">${s.incorrect} 错</span> · 正确率 <b>${acc}</b></span>`;
}
function updateStats() {
  const parts = [fmtStats('总体', computeStats())];
  REPORT.sources.forEach(source => parts.push(fmtStats(source, computeStats(source))));
  $('stats').innerHTML = parts.join('');
}

function applyFilters(keepCurrent=true) {
  const current = keepCurrent ? selectedCase() : null;
  const source = $('sourceFilter').value;
  const status = $('statusFilter').value;
  visible = [];
  CASES.forEach((c, index) => {
    const verdict = annotationFor(c).verdict || 'unlabeled';
    if ((!source || c.source === source) && (!status || verdict === status)) visible.push(index);
  });
  if (current) {
    const found = visible.findIndex(index => CASES[index].id === current.id);
    position = found >= 0 ? found : Math.min(position, Math.max(0, visible.length-1));
  } else position = 0;
  render();
}

function render() {
  $('count').textContent = visible.length ? `${position+1} / ${visible.length}` : '0 / 0';
  $('prev').disabled = position <= 0;
  $('next').disabled = position >= visible.length-1;
  if (!visible.length) {
    $('case').innerHTML = '<div class="empty">当前筛选条件下没有 case</div>';
    return;
  }
  const c = selectedCase();
  const a = annotationFor(c);
  const imageHtml = c.images.length ? c.images.map((src,i) =>
    `<div class="image-wrap"><div class="image-label">第 ${i+1} 幅图${i===1 ? ' · GT 与预测框叠加' : ''}</div>`+
    `<img src="${src}" alt="case ${c.order} image ${i+1}"></div>`).join('') : '';
  $('case').innerHTML = `
    <div class="meta"><b class="mono">JSONL #${c.jsonl_line}</b><span>${escapeHtml(c.source)}</span>`+
    `<span>${escapeHtml(c.source_file)} row ${c.source_row}</span></div>
    ${c.image_error ? `<div class="error">${escapeHtml(c.image_error)}</div>` : ''}
    <div class="images">${imageHtml || '<div class="empty">无可用图片</div>'}</div>
    <div class="legend"><span class="swatch" style="background:#e12d2d"></span>GT：红色实线 `+
    `<span class="swatch" style="background:#2364e1"></span>预测：蓝色虚线（全部独立绘制，不做匹配）</div>
    <div class="text-grid"><div class="panel"><h2>GT</h2><pre id="gt"></pre></div>`+
    `<div class="panel"><h2>预测</h2><pre id="pred"></pre></div></div>
    <div class="annotation"><div class="verdicts">
      <button class="verdict good ${a.verdict==='correct'?'active':''}" data-v="correct">正确 <span class="kbd">1</span></button>
      <button class="verdict bad ${a.verdict==='incorrect'?'active':''}" data-v="incorrect">错误 <span class="kbd">2</span></button>
      <button class="verdict unsure ${a.verdict==='uncertain'?'active':''}" data-v="uncertain">不确定 <span class="kbd">3</span></button>
      <button class="verdict" data-v="">清除</button>
    </div><textarea id="note" placeholder="备注（自动保存在当前浏览器）"></textarea></div>`;
  $('gt').textContent = c.gt;
  $('pred').textContent = c.pred;
  $('note').value = a.note || '';
  document.querySelectorAll('[data-v]').forEach(button => button.onclick = () => setVerdict(button.dataset.v));
  $('note').oninput = event => {
    const old = annotationFor(c);
    annotations[c.id] = {...old, note:event.target.value, updated_at:new Date().toISOString()};
    if (!annotations[c.id].verdict && !annotations[c.id].note) delete annotations[c.id];
    save();
  };
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function setVerdict(verdict) {
  const c = selectedCase();
  if (!c) return;
  const old = annotationFor(c);
  if (verdict || old.note) annotations[c.id] = {...old, verdict, updated_at:new Date().toISOString()};
  else delete annotations[c.id];
  save();
  if ($('autoNext').checked && verdict && position < visible.length-1) position++;
  applyFilters(true);
}
function move(delta) {
  position = Math.max(0, Math.min(visible.length-1, position+delta));
  render(); window.scrollTo({top:0, behavior:'smooth'});
}
function nextUnlabeled() {
  const start = visible.length ? position+1 : 0;
  for (let i=0; i<visible.length; i++) {
    const p = (start+i) % visible.length;
    if (!annotationFor(CASES[visible[p]]).verdict) { position=p; render(); return; }
  }
  alert('当前筛选范围内没有未标注 case');
}
function exportJsonl() {
  const lines = CASES.map(c => JSON.stringify({
    id:c.id, source:c.source, jsonl_line:c.jsonl_line, source_file:c.source_file,
    source_row:c.source_row, verdict:annotationFor(c).verdict || '',
    note:annotationFor(c).note || '', updated_at:annotationFor(c).updated_at || '',
    ground_truth:c.gt, prediction:c.pred
  }));
  const blob = new Blob([lines.join('\n')+'\n'], {type:'application/x-ndjson;charset=utf-8'});
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href=url; a.download=`annotations_step_${REPORT.step}.jsonl`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function importJsonl(file) {
  if (!file) return;
  const valid = new Set(CASES.map(c => c.id));
  let imported=0;
  for (const line of (await file.text()).split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const row=JSON.parse(line);
      if (!valid.has(row.id)) continue;
      const verdict=['correct','incorrect','uncertain'].includes(row.verdict) ? row.verdict : '';
      const note=typeof row.note==='string' ? row.note : '';
      if (verdict || note) annotations[row.id]={verdict,note,updated_at:row.updated_at||new Date().toISOString()};
      else delete annotations[row.id];
      imported++;
    } catch (_) {}
  }
  save(); applyFilters(true); alert(`已导入 ${imported} 条标注`);
}

$('prev').onclick = () => move(-1);
$('next').onclick = () => move(1);
$('nextUnlabeled').onclick = nextUnlabeled;
$('sourceFilter').onchange = () => applyFilters(false);
$('statusFilter').onchange = () => applyFilters(false);
$('export').onclick = exportJsonl;
$('import').onchange = event => importJsonl(event.target.files[0]);
document.addEventListener('keydown', event => {
  if (event.target.matches('textarea,input,select')) return;
  if (event.key==='ArrowLeft') move(-1);
  else if (event.key==='ArrowRight') move(1);
  else if (event.key==='1') setVerdict('correct');
  else if (event.key==='2') setVerdict('incorrect');
  else if (event.key==='3') setVerdict('uncertain');
});

REPORT.sources.forEach(source => {
  const option=document.createElement('option'); option.value=source; option.textContent=source;
  $('sourceFilter').appendChild(option);
});
applyFilters(false); updateStats();
"""


def _safe_script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_html(cases: list[Case], val_jsonl: Path, step: str, out_path: Path) -> None:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.case_id.encode())
        digest.update(case.ground_truth.encode())
        digest.update(case.prediction.encode())
    report = {
        "id": digest.hexdigest()[:20],
        "step": step,
        "val_jsonl": str(val_jsonl),
        "sources": list(TARGET_SOURCES),
        "count": len(cases),
    }
    payload = [{
        "id": case.case_id,
        "order": case.order + 1,
        "jsonl_line": case.jsonl_line,
        "source": case.source,
        "source_file": case.source_path.name,
        "source_row": case.source_row,
        "gt": case.ground_truth,
        "pred": case.prediction,
        "images": case.image_paths,
        "image_error": case.image_error,
    } for case in cases]
    document = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step {html_mod.escape(step)} 人工正确率标注</title><style>{CSS}</style></head>
<body><header class="top"><div class="top-inner">
  <div class="title-row"><h1>Step {html_mod.escape(step)} 人工正确率标注</h1>
    <span class="grow"></span><span id="count" class="mono"></span></div>
  <div id="stats" class="stats"></div>
  <div class="controls">
    <select id="sourceFilter"><option value="">两个数据集</option></select>
    <select id="statusFilter"><option value="">全部状态</option><option value="unlabeled">未标注</option>
      <option value="correct">正确</option><option value="incorrect">错误</option><option value="uncertain">不确定</option></select>
    <button id="prev">← 上一个</button><button id="next">下一个 →</button>
    <button id="nextUnlabeled">下一个未标注</button>
    <label><input id="autoNext" type="checkbox" checked> 标注后自动前进</label>
    <span class="grow"></span><button id="export" class="primary">导出 JSONL</button>
    <label><input id="import" type="file" accept=".jsonl,application/x-ndjson" hidden>
      <span role="button" style="display:inline-block;border:1px solid var(--line);border-radius:6px;padding:7px 10px;cursor:pointer">导入标注</span></label>
  </div>
</div></header><main class="main"><section id="case" class="case"></section>
<p class="muted">标注保存在当前浏览器 localStorage。正确率分母仅含“正确 + 错误”，“不确定”不计入。</p>
</main><script>const REPORT={_safe_script_json(report)};const CASES={_safe_script_json(payload)};{JS}</script>
</body></html>"""
    out_path.write_text(document, encoding="utf-8")


def resolve_val_jsonl(args, parser: argparse.ArgumentParser) -> tuple[Path, str]:
    if args.val_jsonl:
        path = Path(args.val_jsonl).expanduser().resolve()
        inferred = path.stem if re.fullmatch(r"\d+", path.stem) else "unknown"
        step = str(args.step) if args.step is not None else inferred
    else:
        if args.step is None:
            parser.error("使用 --val_dir 时必须同时提供 --step")
        step = str(args.step)
        path = Path(args.val_dir).expanduser().resolve() / f"{step}.jsonl"
    if not path.is_file():
        parser.error(f"验证 JSONL 不存在: {path}")
    return path, step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为两个 inconsistent 数据集生成指定 step 的交互式人工标注 HTML"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--val_jsonl", "--val-jsonl", help="指定 step 的验证 JSONL")
    source.add_argument("--val_dir", "--val-dir", help="val_generations 目录")
    parser.add_argument("--step", type=int, help="step；与 --val_dir 一起使用")
    parser.add_argument("--data_root", "--data-root", required=True,
                        help="包含两个目标数据集目录的源数据根目录")
    parser.add_argument("--out_dir", "--out-dir", default=None,
                        help="输出目录；默认放在验证 JSONL 同级 annotations_step_{step}")
    parser.add_argument("--max_image_edge", "--max-image-edge", type=int, default=1200,
                        help="导出图片最长边，默认 1200")
    parser.add_argument("--jpeg_quality", "--jpeg-quality", type=int, default=88,
                        help="JPEG 质量，默认 88")
    args = parser.parse_args()

    if args.max_image_edge < 128:
        parser.error("--max_image_edge 不能小于 128")
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("--jpeg_quality 必须在 30 到 100 之间")
    val_jsonl, step = resolve_val_jsonl(args, parser)
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        parser.error(f"源数据地址不存在: {data_root}")
    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else val_jsonl.parent / f"annotations_step_{step}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("建立两个目标数据集的源索引（不加载奖励模块）...")
    index = build_source_index(data_root)
    records = read_jsonl(val_jsonl)
    print(f"验证 JSONL: {len(records)} 条")
    cases, unmatched = collect_cases(records, index)
    counts = defaultdict(int)
    for case in cases:
        counts[case.source] += 1
    print(f"目标 case: {len(cases)} 条；其余/未匹配: {unmatched} 条")
    for source_name in TARGET_SOURCES:
        print(f"  {source_name}: {counts[source_name]} 条")
    if not cases:
        raise SystemExit("ERROR: 没有匹配到两个目标数据集的任何 case")

    # Check Pillow before potentially spending time loading source rows.
    try:
        import PIL  # noqa: F401
    except ImportError as exc:
        raise SystemExit("ERROR: 需要 Pillow，请先运行: pip install pillow") from exc
    print("导出原图，并把全部 GT/预测框独立画到第二幅图...")
    materialize_images(cases, out_dir, args.max_image_edge, args.jpeg_quality)
    out_path = out_dir / "index.html"
    build_html(cases, val_jsonl, step, out_path)
    print(f"完成: {out_path}")
    print("标注会保存在浏览器本地；请用页面右上角“导出 JSONL”备份结果。")


if __name__ == "__main__":
    main()
