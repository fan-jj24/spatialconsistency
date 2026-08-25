#!/usr/bin/env python3
"""
val_generations 指标走势报告（单页 HTML，自包含 SVG 折线图）。

读取 val_generations 目录下的所有 {step}.jsonl，对每个 step:
  1. 反查源数据集（复用 inspect_val_generations 的索引/匹配逻辑）
  2. 用训练 reward 模块重算各子项（C / R2 / R3 / R4，不加系数）
  3. 按 source 聚合均值

产出 trends.html，包含:
  - 概览: meta 信息 + 最新 step 各 source 指标汇总表
  - 每个指标一张折线图（x=step，每个 source 一条线）:
      reward / C / R2 / R3 / R4 / 样本数 n
  - 不渲染具体 case（看 case 用 inspect_val_generations.py）

用法:
    python3 plot_val_trends.py \
        --val_dir /home/deepspeed/model_output/rl1_ckpt/val_generations \
        --data_root /home/deepspeed/model_output/RL1 \
        [--reward_module ../verl/json_answer_reward.py] \
        [--out /path/to/trends.html]
"""

import argparse
import glob
import html as html_mod
import json
import math
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import inspect_val_generations as ivg  # noqa: E402

DEFAULT_REWARD_MODULE = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "verl", "json_answer_reward.py"))
DATA_ROOT_DEFAULT = "/home/deepspeed/model_output/RL1"

METRICS = ["reward", "C", "R2", "R3", "R4"]
METRIC_DESC = {
    "reward": "jsonl 记录的训练 reward（含系数/门控，原样均值）",
    "C": "answer 门控正确率（重算，0/1 均值）",
    "R2": "并集 IoU（重算，不加系数）",
    "R3": "label 关键词子集召回（重算，不加系数）",
    "R4": "方向夹角（重算，不加系数；-1=格式门控触发）",
    "n": "匹配到的验证样本数",
}

# 12 色分类调色板（中等饱和度，明暗主题下均可读）
PALETTE = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#9333ea", "#b45309",
    "#475569", "#0d9488",
]

# ============================================================
# 数据收集: 逐 step 读取 + 匹配 + 聚合
# ============================================================

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
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect(val_dir, index, verbose=True):
    """对每个 step 做匹配+聚合。

    返回 (steps, series, meta):
      steps:  升序 step 列表
      series: {source: {metric: {step: value}}}
              metric ∈ METRICS + ["n"]，value 为均值（n 为计数）
      meta:   {"total_records": N, "unmatched": {step: n},
               "cat_of_source": {source: cat}}
    """
    step_files = discover_steps(val_dir)
    if not step_files:
        raise SystemExit(f"ERROR: {val_dir} 下没有找到 {{step}}.jsonl 文件")

    steps = []
    series = {}
    cat_of_source = {}
    unmatched = {}
    total_records = 0

    for step, f in step_files:
        records = load_jsonl(f)
        total_records += len(records)
        matched, mstats = ivg.match_records(records, index)
        unmatched[step] = mstats["unmatched"]
        groups = ivg.aggregate(matched)  # 同时完成子项分解

        steps.append(step)
        for src, g in groups.items():
            cat_of_source.setdefault(src, g["cat"])
            s = series.setdefault(src, {})
            for m in METRICS:
                acc = g[m]
                if acc.n > 0:
                    s.setdefault(m, {})[step] = acc.mean
            s.setdefault("n", {})[step] = g["n"]

        if verbose:
            print(f"  step {step}: {len(records)} 条, 匹配 {len(matched)}, "
                  f"未匹配 {mstats['unmatched']}, sources={len(groups)}")

    meta = {"total_records": total_records, "unmatched": unmatched,
            "cat_of_source": cat_of_source}
    return steps, series, meta

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


def build_html(steps, series, meta, val_dir, reward_module, out_path):
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
      <div><span class='k'>reward 模块</span><br><b>{html_mod.escape(reward_module)}</b></div>
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
<div class='sub'>各指标按 source 的逐步走势（不加系数的重算均值；reward 列为 jsonl 原值）</div>

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
        description="val_generations 指标走势报告（单页 HTML，SVG 折线图）")
    ap.add_argument("--val_dir", required=True,
                    help="val_generations 目录（含 {step}.jsonl）")
    ap.add_argument("--data_root", default=DATA_ROOT_DEFAULT,
                    help="训练数据根目录（用于反查 source）")
    ap.add_argument("--reward_module", default=DEFAULT_REWARD_MODULE,
                    help="训练用 json_answer_reward.py 路径（保证子项口径一致）")
    ap.add_argument("--out", default=None,
                    help="输出 HTML 路径（默认: val_dir/trends.html）")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(args.val_dir, "trends.html")

    # 1. 加载 reward 模块（inspect_val_generations 的全局 R）
    ivg.R = ivg.load_reward_module(args.reward_module)
    print(f"reward 模块: {args.reward_module}")

    # 2. 建索引（只做一次，所有 step 共用）
    print("建立源数据索引（只读 prompt + reward_model 列）...")
    index = ivg.build_index(args.data_root)
    print(f"  索引条目: {len(index)}")

    # 3. 逐 step 收集
    print(f"扫描 {args.val_dir} ...")
    steps, series, meta = collect(args.val_dir, index)
    print(f"共 {len(steps)} 个 step: {steps}")

    # 4. 生成 HTML
    out_path = build_html(steps, series, meta, args.val_dir,
                          args.reward_module, args.out)
    print(f"\n完成。报告: {out_path}")


if __name__ == "__main__":
    main()
