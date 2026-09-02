#!/usr/bin/env python3
"""Score every route-prefixed rollout checkpoint and build a dynamic HTML report.

By default this script scans ``--checkpoint-dir`` for ``local__*.jsonl`` and
``remote__*.jsonl``.  The filename prefix is authoritative: local checkpoints
receive C/R2/R3/R4, while remote checkpoints receive C/R4.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import html as html_mod
import json
import math
from pathlib import Path
import re
from typing import Any

import annotate_val_cases as annotation
import json_answer_reward as reward
import rollout_checkpoint as checkpoint_io
import rollout_parquet_to_html as legacy


CHECKPOINT_RE = re.compile(r"^(local|remote)__(.+)\.jsonl$")


@dataclass
class ModelRun:
    path: Path
    mode: str
    name: str
    backend: str
    records: dict[int, dict[str, Any]]

    @property
    def model_id(self) -> str:
        return self.path.stem

    @property
    def label(self) -> str:
        return f"{self.mode}:{self.name}"


def discover_checkpoints(directory: Path) -> list[Path]:
    paths = [
        path for path in directory.glob("*.jsonl")
        if CHECKPOINT_RE.fullmatch(path.name)
    ]
    return sorted(paths, key=lambda path: path.name)


def load_model_run(path: Path) -> ModelRun:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if not match:
        raise ValueError(
            f"checkpoint 文件名必须是 local__NAME.jsonl 或 remote__NAME.jsonl: {path}"
        )
    mode, name = match.groups()
    records = checkpoint_io.read_checkpoint(path)
    if not records:
        raise ValueError(f"checkpoint 为空: {path}")
    first = next(iter(records.values()))
    backend = str(first.get("backend", ""))
    expected_count = int(first.get("num_samples", -1))
    if expected_count <= 0 or len(records) != expected_count:
        raise ValueError(
            f"checkpoint 尚未完成: {path}；期望 {expected_count} 条，实际 {len(records)} 条"
        )
    orders: set[int] = set()
    for source_row, record in records.items():
        mismatched = []
        if record.get("schema_version") != checkpoint_io.SCHEMA_VERSION:
            mismatched.append("schema_version")
        if record.get("mode") != mode:
            mismatched.append("mode")
        if record.get("name") != name:
            mismatched.append("name")
        if record.get("backend") != backend:
            mismatched.append("backend")
        try:
            order = int(record["sample_order"])
        except (KeyError, TypeError, ValueError):
            mismatched.append("sample_order")
            order = -1
        if order in orders:
            mismatched.append("sample_order(重复)")
        orders.add(order)
        if mismatched:
            raise ValueError(
                f"{path} row {source_row} 元数据不一致: {', '.join(mismatched)}"
            )
    if orders != set(range(expected_count)):
        raise ValueError(f"{path} 的 sample_order 不是完整的 0..{expected_count - 1}")
    return ModelRun(path, mode, name, backend, records)


def validate_runs(runs: list[ModelRun], explicit_data_path: Path | None) -> Path:
    reference = runs[0]
    source_rows = set(reference.records)
    order_by_row = {
        source_row: int(record["sample_order"])
        for source_row, record in reference.records.items()
    }
    gt_by_row = {
        source_row: str(record.get("ground_truth", ""))
        for source_row, record in reference.records.items()
    }
    data_paths = {
        str(record.get("data_path", ""))
        for run in runs for record in run.records.values()
    }
    if len(data_paths) != 1:
        raise ValueError("checkpoint 的 data_path 不一致，不能放进同一份对比报告")
    recorded_data_path = Path(data_paths.pop()).expanduser().resolve()
    data_path = explicit_data_path or recorded_data_path
    for run in runs[1:]:
        if set(run.records) != source_rows:
            missing = sorted(source_rows - set(run.records))
            extra = sorted(set(run.records) - source_rows)
            raise ValueError(
                f"{run.path.name} 的抽样集合不同；missing={missing[:5]}, extra={extra[:5]}"
            )
        for source_row, record in run.records.items():
            if int(record["sample_order"]) != order_by_row[source_row]:
                raise ValueError(f"{run.path.name} row {source_row} 的 sample_order 不一致")
            if str(record.get("ground_truth", "")) != gt_by_row[source_row]:
                raise ValueError(f"{run.path.name} row {source_row} 的 ground_truth 不一致")
    if not data_path.is_file() or data_path.suffix.lower() != ".parquet":
        raise FileNotFoundError(f"Parquet 不存在或扩展名不正确: {data_path}")
    return data_path


def score_runs(
    runs: list[ModelRun], args: argparse.Namespace
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, dict[int, str]]]:
    scores: dict[str, dict[int, dict[str, float]]] = {
        run.model_id: {} for run in runs
    }
    errors: dict[str, dict[int, str]] = {run.model_id: {} for run in runs}
    jobs = []
    for run in runs:
        for source_row, record in run.records.items():
            prediction = str(record.get("prediction", ""))
            if prediction and not record.get("error"):
                jobs.append((run, source_row, prediction, str(record["ground_truth"])))

    def score_one(job):
        run, source_row, prediction, ground_truth = job
        if run.mode == "local":
            result = reward.compute_score_details(
                legacy.LOCAL_DATA_SOURCE, prediction, ground_truth
            )
        else:
            result = reward.score_answer_and_summary(
                legacy.GEMINI_DATA_SOURCE, prediction, ground_truth
            )
        return run.model_id, source_row, {
            key: float(value) for key, value in result.items()
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=args.reward_workers) as executor:
        future_to_job = {executor.submit(score_one, job): job for job in jobs}
        for future in as_completed(future_to_job):
            run, source_row, _, _ = future_to_job[future]
            try:
                model_id, row_id, result = future.result()
                scores[model_id][row_id] = result
            except Exception as exc:
                errors[run.model_id][source_row] = f"{type(exc).__name__}: {exc}"
            completed += 1
            if completed % 20 == 0 or completed == len(jobs):
                print(f"  奖励 {completed}/{len(jobs)}")
    return scores, errors


def load_parquet_rows(data_path: Path, source_rows: list[int]) -> dict[int, dict[str, Any]]:
    datasets = legacy._require_datasets()
    dataframe = datasets.load_dataset("parquet", data_files=str(data_path))["train"]
    if source_rows and max(source_rows) >= len(dataframe):
        raise ValueError(f"checkpoint row 超出 Parquet 范围（共 {len(dataframe)} 条）")
    selected = dataframe.select(source_rows)
    return {
        source_row: row
        for source_row, row in zip(source_rows, selected, strict=True)
    }


def materialize_images(
    ordered_rows: list[int],
    parquet_rows: dict[int, dict[str, Any]],
    ground_truths: dict[int, str],
    runs: list[ModelRun],
    data_path: Path,
    out_dir: Path,
    max_edge: int,
    quality: int,
) -> tuple[dict[int, list[str]], dict[int, list[dict[str, str]]], dict[int, str]]:
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    images_by_row: dict[int, list[str]] = {}
    overlays_by_row: dict[int, list[dict[str, str]]] = {}
    errors: dict[int, str] = {}
    local_runs = [run for run in runs if run.mode == "local"]
    for case_index, source_row in enumerate(ordered_rows, 1):
        try:
            values = annotation._image_values(parquet_rows[source_row])
            if not values:
                raise ValueError("Parquet 行没有 images/image 字段")
            images = [
                annotation._resize_image(
                    annotation._decode_image(value, data_path), max_edge
                )
                for value in values
            ]
            image_paths = []
            for image_index, image in enumerate(images, 1):
                filename = f"case_{case_index:06d}_img_{image_index}.jpg"
                annotation._save_jpeg(image, assets_dir / filename, quality)
                image_paths.append(f"assets/{filename}")
            images_by_row[source_row] = image_paths

            target_index = 1 if len(images) >= 2 else 0
            overlays = []
            gt_only = annotation.annotate_second_image(
                images[target_index].copy(),
                annotation._extract_boxes(ground_truths[source_row]),
                [],
            )
            gt_filename = f"case_{case_index:06d}_boxes_gt.jpg"
            annotation._save_jpeg(gt_only, assets_dir / gt_filename, quality)
            overlays.append({"label": "GT", "path": f"assets/{gt_filename}"})
            for model_index, run in enumerate(local_runs, 1):
                prediction = str(run.records[source_row].get("prediction", ""))
                if not prediction:
                    continue
                overlay = annotation.annotate_second_image(
                    images[target_index].copy(),
                    annotation._extract_boxes(ground_truths[source_row]),
                    annotation._extract_boxes(prediction),
                )
                filename = (
                    f"case_{case_index:06d}_boxes_model_{model_index:03d}.jpg"
                )
                annotation._save_jpeg(overlay, assets_dir / filename, quality)
                overlays.append({"label": run.label, "path": f"assets/{filename}"})
            overlays_by_row[source_row] = overlays
        except Exception as exc:
            errors[source_row] = f"图片处理失败: {exc}"
        if case_index % 100 == 0 or case_index == len(ordered_rows):
            print(f"  图片 {case_index}/{len(ordered_rows)}")
    return images_by_row, overlays_by_row, errors


def write_evaluation_results(
    path: Path,
    ordered_rows: list[int],
    runs: list[ModelRun],
    scores: dict[str, dict[int, dict[str, float]]],
    score_errors: dict[str, dict[int, str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for source_row in ordered_rows:
            first = runs[0].records[source_row]
            models = {}
            for run in runs:
                record = run.records[source_row]
                models[run.model_id] = {
                    "name": run.name,
                    "mode": run.mode,
                    "backend": run.backend,
                    "checkpoint": run.path.name,
                    "prediction": str(record.get("prediction", "")),
                    "generation_error": str(record.get("error", "")),
                    "scores": scores[run.model_id].get(source_row, {}),
                    "evaluation_error": score_errors[run.model_id].get(source_row, ""),
                }
            payload = {
                "source_row": source_row,
                "sample_order": int(first["sample_order"]),
                "ground_truth": str(first["ground_truth"]),
                "models": models,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


DYNAMIC_CSS = r"""
:root{--bg:#f4f6f8;--surface:#fff;--ink:#17202a;--muted:#65717e;--line:#d8dee5;--accent:#1769aa;--good:#16834a;--bad:#c33535}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.top{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}.top-inner,.main{max-width:1680px;margin:auto;padding:12px 18px}.title{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}.title h1{font-size:19px;margin:0}.muted{color:var(--muted)}.grow{flex:1}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
button,select,.file-button{border:1px solid var(--line);background:#fff;color:var(--ink);padding:7px 10px;border-radius:7px;cursor:pointer}.primary{background:var(--accent);color:#fff;border-color:var(--accent)}.stats{display:flex;gap:7px;overflow:auto;margin-top:9px}.stat{white-space:nowrap;border:1px solid var(--line);border-radius:999px;padding:4px 9px;background:var(--bg)}
.case{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}.meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);margin-bottom:12px}.images,.overlays{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(330px,100%),1fr));gap:10px;margin-top:10px}.image-wrap{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#111}.image-label{color:#dfe6ee;background:#1b222b;padding:5px 8px;font-size:12px}.image-wrap img{display:block;width:100%;max-height:56vh;object-fit:contain}
.section-title{font-size:14px;margin:18px 0 4px}.model-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(360px,100%),1fr));gap:12px;margin-top:14px}.model{border:1px solid var(--line);border-radius:8px;padding:11px;min-width:0}.model h2{font-size:14px;margin:0 0 7px}.route{font-size:11px;color:#fff;background:var(--accent);padding:2px 6px;border-radius:999px}pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:9px;max-height:320px;overflow:auto;min-height:90px}.scores{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.score{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:2px 7px}.error{color:var(--bad);margin:7px 0}.judgments{display:flex;gap:6px;flex-wrap:wrap}.judgment.active.good{background:var(--good);color:#fff}.judgment.active.bad{background:var(--bad);color:#fff}.best.active{background:var(--accent);color:#fff}.gt{margin-top:12px}.annotation{margin-top:15px}textarea{width:100%;min-height:65px;border:1px solid var(--line);border-radius:7px;padding:8px}.empty{text-align:center;color:var(--muted);padding:50px}.legend{font-size:12px;color:var(--muted)}
@media(max-width:700px){.top-inner,.main{padding:9px}.model-grid{grid-template-columns:1fr}.controls>*{flex:1 1 auto}.top{position:static}}
"""


DYNAMIC_JS = r"""
const stateKey=`dynamic-rollout-annotations:${REPORT.id}`;
let annotations={};try{annotations=JSON.parse(localStorage.getItem(stateKey)||'{}')||{}}catch(_){}
let visible=[],position=0;const $=id=>document.getElementById(id);
const esc=v=>String(v).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
function ann(c){const a=annotations[c.id]||{};return{verdicts:a.verdicts&&typeof a.verdicts==='object'?a.verdicts:{},best:Array.isArray(a.best)?a.best:[],note:typeof a.note==='string'?a.note:'',updated_at:a.updated_at||''}}
function save(c,a){a.updated_at=new Date().toISOString();annotations[c.id]=a;localStorage.setItem(stateKey,JSON.stringify(annotations));stats()}
function complete(c,a){return REPORT.models.every(m=>['correct','incorrect'].includes(a.verdicts[m.id]))}
function stats(){const html=REPORT.models.map(m=>{let good=0,bad=0,best=0;CASES.forEach(c=>{const a=ann(c);if(a.verdicts[m.id]==='correct')good++;if(a.verdicts[m.id]==='incorrect')bad++;if(a.best.includes(m.id))best++});const n=good+bad,acc=n?(100*good/n).toFixed(1)+'%':'n/a';return `<span class="stat"><b>${esc(m.label)}</b> ${acc} (${good}/${n}) · 最佳 ${best}</span>`}).join('');$('stats').innerHTML=html}
function apply(keep=true){const current=keep&&visible.length?CASES[visible[position]]:null,model=$('modelFilter').value,status=$('statusFilter').value;visible=[];CASES.forEach((c,i)=>{const a=ann(c),ids=model?[model]:REPORT.models.map(m=>m.id);let ok=true;if(status==='unlabeled')ok=ids.some(id=>!a.verdicts[id]);else if(status==='completed')ok=ids.every(id=>a.verdicts[id]);else if(status==='correct')ok=ids.some(id=>a.verdicts[id]==='correct');else if(status==='incorrect')ok=ids.some(id=>a.verdicts[id]==='incorrect');else if(status==='error')ok=c.models.some(m=>ids.includes(m.id)&&(m.error||m.score_error));if(ok)visible.push(i)});if(current){const n=visible.findIndex(i=>CASES[i].id===current.id);position=n>=0?n:Math.min(position,Math.max(0,visible.length-1))}else position=0;render()}
function scoreHtml(scores){return Object.entries(scores||{}).map(([k,v])=>`<span class="score"><b>${esc(k==='reward'?'总奖励':k)}</b> ${Number(v).toFixed(4)}</span>`).join('')}
function render(){ $('count').textContent=visible.length?`${position+1} / ${visible.length}`:'0 / 0';$('prev').disabled=position<=0;$('next').disabled=position>=visible.length-1;if(!visible.length){$('case').innerHTML='<div class="empty">当前筛选条件下没有 case</div>';return}const c=CASES[visible[position]],a=ann(c);const imgs=c.images.map((p,i)=>`<div class="image-wrap"><div class="image-label">原图 ${i+1}</div><img src="${p}"></div>`).join('');const overlays=c.overlays.map(o=>`<div class="image-wrap"><div class="image-label">${esc(o.label)} · 红色 GT / 蓝色预测</div><img src="${o.path}"></div>`).join('');const cards=c.models.map((m,i)=>`<article class="model"><h2>${esc(m.label)} <span class="route">${esc(m.backend)}</span></h2>${m.error?`<div class="error">生成失败：${esc(m.error)}</div>`:''}${m.score_error?`<div class="error">评分失败：${esc(m.score_error)}</div>`:''}<pre class="prediction" data-index="${i}"></pre><div class="scores">${scoreHtml(m.scores)}</div><div class="judgments"><button data-model="${esc(m.id)}" data-verdict="correct" class="judgment good ${a.verdicts[m.id]==='correct'?'active':''}">正确</button><button data-model="${esc(m.id)}" data-verdict="incorrect" class="judgment bad ${a.verdicts[m.id]==='incorrect'?'active':''}">错误</button><button data-best="${esc(m.id)}" class="best ${a.best.includes(m.id)?'active':''}">本条最佳</button></div></article>`).join('');$('case').innerHTML=`<div class="meta"><b>Parquet row #${c.source_row}</b><span>sample ${c.order+1}</span></div>${c.image_error?`<div class="error">${esc(c.image_error)}</div>`:''}<div class="images">${imgs||'<div class="empty">无图片</div>'}</div>${overlays?`<h2 class="section-title">框图对比</h2><div class="legend">GT 图仅画红框；每个 local 模型分别生成一张红色 GT + 蓝色预测框图，避免多模型框互相遮挡。</div><div class="overlays">${overlays}</div>`:''}<div class="gt"><h2 class="section-title">Ground truth</h2><pre id="gt"></pre></div><div class="model-grid">${cards}</div><div class="annotation"><textarea id="note" placeholder="本条备注（自动保存在浏览器）"></textarea></div>`;$('gt').textContent=c.gt;document.querySelectorAll('.prediction').forEach(el=>el.textContent=c.models[Number(el.dataset.index)].prediction);document.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>{const x=ann(c),id=b.dataset.model,v=b.dataset.verdict;x.verdicts[id]=x.verdicts[id]===v?'':v;save(c,x);render()});document.querySelectorAll('[data-best]').forEach(b=>b.onclick=()=>{const x=ann(c),id=b.dataset.best;x.best=x.best.includes(id)?x.best.filter(v=>v!==id):[...x.best,id];save(c,x);render()});$('note').value=a.note;$('note').oninput=e=>{const x=ann(c);x.note=e.target.value;save(c,x)}}
function move(d){position=Math.max(0,Math.min(visible.length-1,position+d));render();scrollTo({top:0,behavior:'smooth'})}
function exportJsonl(){const lines=CASES.map(c=>JSON.stringify({id:c.id,source_row:c.source_row,ground_truth:c.gt,models:Object.fromEntries(c.models.map(m=>[m.id,{name:m.name,mode:m.mode,backend:m.backend,prediction:m.prediction,scores:m.scores,generation_error:m.error,evaluation_error:m.score_error}])),verdicts:ann(c).verdicts,best_models:ann(c).best,note:ann(c).note,updated_at:ann(c).updated_at}));const blob=new Blob([lines.join('\n')+'\n'],{type:'application/x-ndjson;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=REPORT.export_filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
async function importJsonl(file){if(!file)return;const valid=new Set(CASES.map(c=>c.id));let n=0;for(const line of(await file.text()).split(/\r?\n/)){if(!line.trim())continue;try{const r=JSON.parse(line);if(!valid.has(r.id))continue;annotations[r.id]={verdicts:r.verdicts||{},best:Array.isArray(r.best_models)?r.best_models:[],note:typeof r.note==='string'?r.note:'',updated_at:r.updated_at||new Date().toISOString()};n++}catch(_){}}localStorage.setItem(stateKey,JSON.stringify(annotations));stats();apply();alert(`已导入 ${n} 条标注`)}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('modelFilter').onchange=()=>apply(false);$('statusFilter').onchange=()=>apply(false);$('export').onclick=exportJsonl;$('import').onchange=e=>importJsonl(e.target.files[0]);document.addEventListener('keydown',e=>{if(e.target.matches('textarea,input,select'))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});REPORT.models.forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=m.label;$('modelFilter').appendChild(o)});stats();apply(false);
"""


def safe_script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def build_html(
    out_path: Path,
    data_path: Path,
    ordered_rows: list[int],
    runs: list[ModelRun],
    scores: dict[str, dict[int, dict[str, float]]],
    score_errors: dict[str, dict[int, str]],
    images: dict[int, list[str]],
    overlays: dict[int, list[dict[str, str]]],
    image_errors: dict[int, str],
) -> None:
    digest = hashlib.sha256()
    for run in runs:
        digest.update(run.model_id.encode())
        for source_row in ordered_rows:
            digest.update(str(run.records[source_row].get("prediction", "")).encode())
    report = {
        "id": digest.hexdigest()[:20],
        "models": [
            {
                "id": run.model_id,
                "name": run.name,
                "mode": run.mode,
                "backend": run.backend,
                "label": run.label,
            }
            for run in runs
        ],
        "export_filename": "rollout_annotations.jsonl",
    }
    payload = []
    for source_row in ordered_rows:
        first = runs[0].records[source_row]
        payload.append({
            "id": f"{data_path.name}:{source_row}",
            "order": int(first["sample_order"]),
            "source_row": source_row,
            "gt": str(first["ground_truth"]),
            "images": images.get(source_row, []),
            "overlays": overlays.get(source_row, []),
            "image_error": image_errors.get(source_row, ""),
            "models": [
                {
                    "id": run.model_id,
                    "name": run.name,
                    "mode": run.mode,
                    "backend": run.backend,
                    "label": run.label,
                    "prediction": str(run.records[source_row].get("prediction", "")),
                    "error": str(run.records[source_row].get("error", "")),
                    "scores": scores[run.model_id].get(source_row, {}),
                    "score_error": score_errors[run.model_id].get(source_row, ""),
                }
                for run in runs
            ],
        })
    labels = " / ".join(run.label for run in runs)
    title = f"{data_path.name} 动态 Rollout 对比评测"
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html_mod.escape(title)}</title>
<style>{DYNAMIC_CSS}</style></head><body><header class="top"><div class="top-inner">
<div class="title"><h1>{html_mod.escape(title)}</h1><span class="muted">{len(ordered_rows)} 条 · {html_mod.escape(labels)}</span><span class="grow"></span><span id="count"></span></div>
<div id="stats" class="stats"></div><div class="controls"><select id="modelFilter"><option value="">全部模型</option></select><select id="statusFilter"><option value="">全部状态</option><option value="unlabeled">未标注</option><option value="completed">已标注</option><option value="correct">正确</option><option value="incorrect">错误</option><option value="error">执行失败</option></select><button id="prev">← 上一个</button><button id="next">下一个 →</button><span class="grow"></span><button id="export" class="primary">导出标注</button><label><input id="import" type="file" accept=".jsonl" hidden><span class="file-button">导入标注</span></label></div></div></header>
<main class="main"><section id="case" class="case"></section><p class="muted">每个模型独立标记正确/错误；“本条最佳”可多选以表达并列。标注保存在当前浏览器 localStorage，请导出 JSONL 备份。</p></main>
<script>const REPORT={safe_script_json(report)};const CASES={safe_script_json(payload)};{DYNAMIC_JS}</script></body></html>"""
    out_path.write_text(document, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描并评分所有 local__/remote__ checkpoint，生成动态模型 HTML"
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--checkpoint", action="append",
        help="只评估指定 checkpoint；可重复，默认扫描目录中的全部 checkpoint",
    )
    parser.add_argument("--data-path", help="可省略；默认读取 checkpoint 中的绝对路径")
    parser.add_argument("--out-dir", help="默认 CHECKPOINT_DIR/evaluation_report")
    parser.add_argument("--reward-workers", type=int, default=100)
    parser.add_argument(
        "--auto-reward-server", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--reward-host", default="127.0.0.1")
    parser.add_argument("--reward-port", type=int, default=8765)
    parser.add_argument(
        "--reward-backend", choices=("auto", "vllm", "transformers"), default="auto"
    )
    parser.add_argument("--reward-model-path")
    parser.add_argument("--reward-transformers-device", default="auto")
    parser.add_argument("--reward-start-timeout", type=float, default=900.0)
    parser.add_argument("--reward-max-batch-size", type=int, default=100)
    parser.add_argument("--reward-max-wait-ms", type=float, default=20.0)
    parser.add_argument("--max-image-edge", type=int, default=1200)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    args = parser.parse_args(argv)
    if args.reward_workers <= 0 or args.reward_max_batch_size <= 0:
        parser.error("reward-workers 和 reward-max-batch-size 必须大于 0")
    if not 1 <= args.reward_port <= 65535:
        parser.error("reward-port 必须在 1 到 65535 之间")
    if not math.isfinite(args.reward_start_timeout) or args.reward_start_timeout <= 0:
        parser.error("reward-start-timeout 必须大于 0")
    if not math.isfinite(args.reward_max_wait_ms) or args.reward_max_wait_ms < 0:
        parser.error("reward-max-wait-ms 不能小于 0")
    if args.max_image_edge < 128 or not 30 <= args.jpeg_quality <= 100:
        parser.error("max-image-edge 不能小于 128，jpeg-quality 必须在 30..100")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"ERROR: checkpoint 目录不存在: {checkpoint_dir}")
    if args.checkpoint:
        paths = [
            (checkpoint_dir / value).resolve()
            if not Path(value).expanduser().is_absolute()
            else Path(value).expanduser().resolve()
            for value in args.checkpoint
        ]
    else:
        paths = discover_checkpoints(checkpoint_dir)
    if not paths:
        raise SystemExit("ERROR: 没有找到 local__*.jsonl 或 remote__*.jsonl")
    try:
        runs = [load_model_run(path) for path in paths]
        explicit_data_path = (
            Path(args.data_path).expanduser().resolve() if args.data_path else None
        )
        data_path = validate_runs(runs, explicit_data_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir else checkpoint_dir / "evaluation_report"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print("载入 checkpoint: " + ", ".join(run.label for run in runs))

    has_scorable = any(
        record.get("prediction") and not record.get("error")
        for run in runs for record in run.records.values()
    )
    if has_scorable:
        with legacy.reward_server_for_scoring(args):
            scores, score_errors = score_runs(runs, args)
    else:
        print("没有可评分输出，不启动 reward model。")
        scores, score_errors = score_runs(runs, args)

    ordered_rows = [
        source_row for source_row, _ in sorted(
            runs[0].records.items(), key=lambda item: int(item[1]["sample_order"])
        )
    ]
    parquet_rows = load_parquet_rows(data_path, ordered_rows)
    ground_truths = {
        source_row: str(runs[0].records[source_row]["ground_truth"])
        for source_row in ordered_rows
    }
    for source_row in ordered_rows:
        parquet_gt = annotation._json_text(
            annotation._ground_truth(parquet_rows[source_row])
        )
        if parquet_gt != ground_truths[source_row]:
            raise SystemExit(
                f"ERROR: Parquet row {source_row} 的 ground truth 与 checkpoint 不一致"
            )
    print("导出原图和逐模型框图...")
    images, overlays, image_errors = materialize_images(
        ordered_rows, parquet_rows, ground_truths, runs, data_path, out_dir,
        args.max_image_edge, args.jpeg_quality,
    )
    results_path = out_dir / "evaluation_results.jsonl"
    write_evaluation_results(results_path, ordered_rows, runs, scores, score_errors)
    html_path = out_dir / "index.html"
    build_html(
        html_path, data_path, ordered_rows, runs, scores, score_errors,
        images, overlays, image_errors,
    )
    generation_errors = sum(
        bool(record.get("error")) for run in runs for record in run.records.values()
    )
    evaluation_errors = sum(len(values) for values in score_errors.values())
    print(f"完成: {html_path}")
    print(f"完整结果: {results_path}")
    if generation_errors:
        print(f"[WARN] {generation_errors} 个生成结果失败")
    if evaluation_errors:
        print(f"[WARN] {evaluation_errors} 个奖励计算失败")


if __name__ == "__main__":
    main()
