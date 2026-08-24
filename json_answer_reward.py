#!/usr/bin/env python3
"""
JSON RLVR Reward Function（分类奖励框架）
适用于 verl 框架的 custom_reward_function。

按 data_source 路由到四类奖励:

━━━ 第一类: JSON answer 二值奖励 ━━━
  data_source: spatial_consistency_pos / spatial_consistency_neg /
               vst_caption / spatialscore / spatialcorpus_vi /
               viewspatial / vst_oor
  GT:          {"answer": "B"}
  模型输出:    <think>...</think> + {"answer": "B", "summary": "..."}
  逻辑:        解析模型 JSON 的 answer 键，与 GT 的 answer 做 exact match
  值域:        答对=1.0，答错/JSON缺失/无法解析=0.0

━━━ 第二类: bbox 匈牙利匹配 IoU 奖励 ━━━
  data_source: humanref_cot
  GT:          {"boxes": [[x1,y1,x2,y2], ...]}  (纯坐标数组)
  模型输出:    同上
  逻辑:        1) 每个框 clamp 到 [0,1000]
              2) 构造 n×m IoU 矩阵，匈牙利算法求总 IoU 最大的一对一匹配 M*
              3) reward = Σ IoU(M*) / max(n, m)
  值域:        [0, 1] 连续值
  设计:        分母取 max(n,m): 漏检(n大)和多报(m大)都被惩罚，
              模型无法靠多输出框刷分。无匹配阈值。

━━━ 第三类: spatial_consistency_bbox 组合奖励 ━━━
  data_source: spatial_consistency_bbox_pos / spatial_consistency_bbox_neg
  GT:          {"answer": "B", "summary": "...",
               "boxes": [{"bbox": [x1,y1,x2,y2], "label": "..."}, ...]}
               正例时 boxes 为空 []
  逻辑:
    GT boxes 为空（正例）:
      C=0 → R=0
      C=1 且预测 boxes 也为空 → R=1.0
      C=1 且预测 boxes 非空   → R=0.2
    GT boxes 非空（负例）:
      R = C × (0.2 + 0.7×R2 + 0.1×R3)
      C  = answer 门控（0/1）
      R2 = GT 所有框并集 vs 预测所有框并集 的 IoU（栅格化）
      R3 = 关键词子集召回 × IoU × 方向系数（含格式门控），详见下方
  值域:        [0, 1] 连续值

━━━ 第四类: spatial_detection 组合奖励 ━━━
  data_source: spatial_detection
  GT:          同第三类结构（全负例，无正例空框分支）
  逻辑:        R = 0.85×R2 + 0.15×R3
              无 C 门控，无正例分支
  值域:        [0, 1] 连续值

━━━ R2/R3/R4 子项详解（第三、四类共用）━━━
  R2（并集 IoU）:
    将 GT 所有非空框的并集区域、预测所有非空框的并集区域，
    分别栅格化到 1000×1000 布尔掩码，做像素级 IoU。
    天然处理缺口、分离框、重叠框。空集vs空集=1.0，任一为空=0.0。

  R3（关键词子集召回 × IoU × 方向系数，含格式门控）:
    关键词集合: add / delete / rotate / replace / move / background
    匹配方式: 单词前缀匹配（\bmove 匹配 move/moved/moves/movement，
              但不匹配 remove）。

    格式硬门控（前置，遍历所有 pred entries，无条件）:
      任一 pred label 满足以下任一条件 → R3 = -1:
        a) 含 move 但无箭头向量（括号内 2/3 值数值向量）
        b) 不含 move 但有箭头向量
        c) 含 move 且有 >1 组括号数值向量

    通过门控后，对每个匈牙利匹配对 (gt_i, pred_j, iou_ij):
      pred 多报任一关键词 → label_score = 0
      普通关键词命中记 1；move 命中记方向系数 dir_coef（见下）
      label_score = 各 GT 关键词子项得分之和 / |GT 关键词|
      R3_pair = label_score × iou_ij
    R3 = mean(R3_pairs)。格式门控优先于“无匹配”判定；通过门控后无匹配 → 0.0。

    方向系数 dir_coef（原 R4 融入，仅在 move 同时 ∈ gt_kw 且 ∈ pred_kw 时生效）:
      GT label 有箭头向量:
        dir_coef = cosθ（同向=1, 垂直=0, 反向=-1；零向量=0）
        GT 3 值 + pred 2 值 → dir_coef -= 0.2（丢失深度，无条件扣）
      GT label 有 move 但无箭头:
        dir_coef = 1.0（GT 没要求方向，pred 有 move 且格式合规即可）
      否则: dir_coef = 1.0（该对不涉及 move，不缩放）

    方向向量提取:
      二值 (x,y): 位置 → 方向 = (x-cx, cy-y)
                  cx,cy = 该框各自中心; 第一维正=右, 第二维正=上
      三值 (x,y,z): 方向本身 → 取前两维 (x,y)

━━━ R4 子项详解（第三、四类共用）━━━
  R4（summary 一致性校验，Qwen3.5-0.8B 奖励模型）:
    使用 Qwen/Qwen3.5-0.8B（0.8B VLM）作为判官，运行在 CPU 上，
    校验模型输出的 summary 与 GT summary 是否语义一致。
    模型加载和推理封装在中央 reward_model_server.py 服务中，本文件通过
    同目录 reward_model_client.py 发起单条 HTTP 请求，由服务动态攒批。

    调用: reward_model_client.score_summary(pred_summary, gt_summary)
    返回: 1.0×P(A) + 0.5×P(B) + 0.0×P(C)，范围 [0, 1]
    性能: CPU bfloat16, 单样本 ~3s, batch 更快

    输入: pred summary（从模型输出 JSON 的 summary 键提取）
          gt summary（从 GT JSON 的 summary 键提取）
    异常: 客户端导入、HTTP 请求、模型加载或推理失败时直接抛出，中断训练；
          禁止静默去掉 R4 后继续训练。

    在第三/四类中的权重:
      第三类: R = C × (0.15 + 0.65×R2 + 0.1×R3 + 0.1×R4)
      第四类: R = 0.75×R2 + 0.15×R3 + 0.1×R4
━━━ 设计约束 ━━━
  - 第一类保持严格 0/1 二值: DAPO 的 filter_groups.metric=acc 依赖
    组内二值对错来过滤全对/全错 group。
  - 第二/三/四类为连续值: 组内几乎必然存在方差，filter_groups 的
    过滤基本不再触发——这是预期行为，连续奖励任务不依赖该机制。
  - 长度惩罚不在本函数内: 由 DAPO reward manager 的
    Overlong Buffer 单独控制。
  - 路由无兜底: 未知 data_source 直接 raise ValueError。
    历史上静默兜底到第一类曾导致 bbox 任务被悄悄降级为纯 answer
    匹配（reward 退化为 C）而无人察觉，故改为显式报错。

verl 调用签名:
    def compute_score(data_source, solution_str, ground_truth, extra_info=None):
        ...
        return float  # 标量 reward

在 verl 配置中注册:
    custom_reward_function.path = /path/to/json_answer_reward.py
    custom_reward_function.name = compute_score
"""

import importlib.util
import json
from pathlib import Path
import re
import sys
import threading

import numpy as np
from scipy.optimize import linear_sum_assignment

# R4 HTTP 客户端 — 懒导入，首次需要 R4 时才加载模块。
# 用文件路径导入，避免 verl 从任意工作目录动态加载本文件时找不到同目录的
# reward_model_client.py，也避免误导入环境中另一个同名模块。
_r4_score_summary_fn = None
_r4_import_lock = threading.Lock()
_R4_MODULE_NAME = "_spatialconsistency_r4_reward_model_client"


def _get_r4_score_summary():
    """从同目录 reward_model_client.py 懒加载 score_summary。

    任何导入错误都会原样抛出。这里不返回降级 sentinel，防止训练在 R4
    实际失效后仍继续运行。
    """
    global _r4_score_summary_fn
    if _r4_score_summary_fn is not None:
        return _r4_score_summary_fn

    with _r4_import_lock:
        if _r4_score_summary_fn is not None:
            return _r4_score_summary_fn

        module_path = Path(__file__).resolve().with_name("reward_model_client.py")
        if not module_path.is_file():
            raise FileNotFoundError(f"R4 reward client file not found: {module_path}")

        module = sys.modules.get(_R4_MODULE_NAME)
        if module is None:
            spec = importlib.util.spec_from_file_location(
                _R4_MODULE_NAME, module_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load R4 reward client from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[_R4_MODULE_NAME] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                if sys.modules.get(_R4_MODULE_NAME) is module:
                    del sys.modules[_R4_MODULE_NAME]
                raise

        score_fn = getattr(module, "score_summary", None)
        if not callable(score_fn):
            raise AttributeError(
                f"score_summary is missing or not callable in {module_path}"
            )
        _r4_score_summary_fn = score_fn
        return _r4_score_summary_fn


# ============================================================
# 共享工具函数
# ============================================================

def _clamp_box(box, lo=0, hi=1000):
    """将 bbox 的 4 个值 clamp 到 [lo, hi]。"""
    return [max(lo, min(hi, v)) for v in box]


def _iou(box_a, box_b):
    """计算两个 [x1,y1,x2,y2] 框的 IoU。坐标已 clamp 到 [0,1000]。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _normalize_answer(ans):
    """归一化答案: 去空白/括号，统一小写。兼容 "B"/"b"/"(B)"/" B "。"""
    if ans is None:
        return None
    s = str(ans).strip().lower()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s if s else None


def _scan_balanced_json(s, start):
    """从 s[start]（须为 '{'）起做括号平衡扫描，感知 JSON 字符串与转义。

    返回 (obj, end):
      - 从 start 起的平衡片段能 json.loads 且为 dict → (dict, 配对 '}' 的下标)
      - 否则 → (None, 扫描终止下标)
    供 _extract_last_json_obj 枚举每个 '{' 起点使用。
    """
    depth = 0
    in_str = False
    esc = False
    n = len(s)
    k = start
    while k < n:
        ch = s[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:k + 1])
                    except (json.JSONDecodeError, ValueError):
                        return None, k
                    if isinstance(obj, dict):
                        return obj, k
                    return None, k
        k += 1
    return None, n - 1


def _extract_last_json_obj(s):
    """从字符串中提取最后一个合法的 JSON 对象（dict）。

    模型输出可能带 <think> 块，直接 json.loads 会失败。
    用括号平衡扫描（感知 JSON 字符串内的花括号与转义引号）枚举每个
    `{` 起点，逐个 json.loads，返回【结束位置最靠后】的成功解析为
    dict 的对象。结束最靠后保证: 嵌套对象取外层完整对象、并列多个
    对象取最后输出者。失败返回 None。

    历史 bug: 曾用贪婪正则 \\{.*\\} 匹配，think 中一旦出现花括号，
    正则把首个 `{` 到末尾 `}` 整段当作唯一候选，解析必然失败，
    导致第三/四类奖励被整体误判为 0。
    """
    if not s:
        return None
    best_obj = None
    best_end = -1
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        obj, end = _scan_balanced_json(s, i)
        if obj is not None:
            if end > best_end:
                best_obj, best_end = obj, end
            # 成功候选内部的嵌套起点结束更早，不可能更优，直接跳过
            i = end + 1
        else:
            # 该起点解析失败，其内部可能仍藏有有效 JSON，逐个字符后移再试
            i += 1
    return best_obj


def _parse_json_obj(raw):
    """从字符串或 dict 中解析出 JSON 对象（dict）。失败返回 None。

    GT 通常是纯 JSON，json.loads 直接成功。
    模型输出带 <think> 块时，用 _extract_last_json_obj 扫描。
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return _extract_last_json_obj(s)


# ============================================================
# 第一类: JSON answer 二值奖励
# ============================================================

def _parse_gt_answer(ground_truth):
    """从 ground_truth 中提取 answer。

    ground_truth 期望为 JSON 字符串: {"answer": "B"}
    兜底: 若是 dict 直接取 answer 键; JSON 损坏时正则抓;
    若不是 JSON 则把整个字符串当裸答案。
    """
    if ground_truth is None:
        return None
    if isinstance(ground_truth, dict):
        return _normalize_answer(ground_truth.get("answer"))

    gt_str = str(ground_truth).strip()
    if gt_str.startswith("{"):
        try:
            obj = json.loads(gt_str)
            if isinstance(obj, dict) and "answer" in obj:
                return _normalize_answer(obj["answer"])
        except (json.JSONDecodeError, ValueError):
            pass
        m = re.search(r'"answer"\s*:\s*"([^"]*)"', gt_str)
        if m:
            return _normalize_answer(m.group(1))
        return None
    return _normalize_answer(gt_str)


def _extract_json_answer(solution_str):
    """从模型输出中提取最终 JSON 的 answer 键。

    模型输出形如: <think>...</think>\n{"answer": "B", "summary": "..."}
    策略: 扫描所有 {...} 块逐个 json.loads，取最后一个含 answer 键的;
          兜底正则抓 "answer": "X"。
    """
    if not solution_str:
        return None

    candidates = re.findall(r"\{[^{}]*\}", solution_str, re.DOTALL)
    last_answer = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "answer" in obj:
            last_answer = obj["answer"]

    if last_answer is not None:
        return _normalize_answer(last_answer)

    m = re.findall(r'"answer"\s*:\s*"([^"]*)"', solution_str)
    if m:
        return _normalize_answer(m[-1])

    return None


def score_json_answer(solution_str, ground_truth):
    """第一类奖励: JSON answer exact match，二值 0/1。"""
    gt = _parse_gt_answer(ground_truth)
    if gt is None:
        return 0.0
    pred = _extract_json_answer(solution_str)
    if pred is None:
        return 0.0
    return 1.0 if pred == gt else 0.0


# ============================================================
# 第二类: bbox 匈牙利匹配 IoU 奖励（humanref_cot）
# ============================================================

def _parse_boxes_from_json(raw):
    """从 JSON 字符串或 dict 中提取 boxes 列表。

    期望格式: {"boxes": [[x1,y1,x2,y2], ...]}  (纯坐标数组)
    每个框 clamp 到 [0,1000]。
    返回 list[list[float]]，解析失败返回 []。
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        obj = raw
    else:
        s = str(raw).strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            # 兜底: 正则抓所有 [x,y,x,y] 数组
            boxes = []
            for m in re.finditer(r"\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]", s):
                boxes.append(_clamp_box([int(m.group(i)) for i in (1, 2, 3, 4)]))
            return boxes

    boxes_raw = obj.get("boxes") if isinstance(obj, dict) else None
    if not isinstance(boxes_raw, list):
        return []
    result = []
    for b in boxes_raw:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                result.append(_clamp_box([float(v) for v in b]))
            except (ValueError, TypeError):
                continue
    return result


def score_hungarian_iou(solution_str, ground_truth):
    """第二类奖励: bbox 匈牙利匹配 IoU / max(n, m)，连续值 ∈ [0,1]。

    步骤:
      1) 从 GT 和模型输出中提取 boxes，clamp 到 [0,1000]
      2) 构造 n×m IoU 矩阵
      3) 匈牙利算法求总 IoU 最大的一对一匹配
      4) reward = Σ IoU(M*) / max(n, m)

    无匹配阈值; GT 或预测为空时返回 0.0。
    """
    gt_boxes = _parse_boxes_from_json(ground_truth)
    pred_boxes = _parse_boxes_from_json(solution_str)

    n = len(gt_boxes)
    m = len(pred_boxes)
    if n == 0 or m == 0:
        return 0.0

    iou_matrix = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = _iou(gt_boxes[i], pred_boxes[j])

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_iou_sum = float(iou_matrix[row_ind, col_ind].sum())

    return matched_iou_sum / max(n, m)


# ============================================================
# 第三/四类共享: bbox+label 解析、R2/R3/R4 子项、匈牙利匹配
# ============================================================

CANVAS_SIZE = 1000

# R3 关键词（background 为新增子项）
LABEL_KEYWORDS = ("add", "delete", "rotate", "replace", "move", "background")


def _parse_bbox_entries(raw):
    """解析 spatial_consistency_bbox / spatial_detection 的 boxes 字段。

    每个元素是 {"bbox": [x1,y1,x2,y2], "label": "..."}。
    返回 [(bbox_clamped, label_str), ...]，解析失败返回 []。

    模型输出可能带 <think> 块，用 _extract_last_json_obj 提取。
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        obj = raw
    else:
        s = str(raw).strip()
        if not s:
            return []
        obj = None
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            pass
        if obj is None:
            obj = _extract_last_json_obj(s)
        if obj is None:
            return []

    boxes_raw = obj.get("boxes") if isinstance(obj, dict) else None
    if not isinstance(boxes_raw, list):
        return []
    result = []
    for item in boxes_raw:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        label = item.get("label", "")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                bbox_c = _clamp_box([float(v) for v in bbox])
            except (ValueError, TypeError):
                continue
            result.append((bbox_c, str(label)))
    return result


def _rasterize_union(entries, size=CANVAS_SIZE):
    """把 entries 里所有框的并集栅格化到 size×size 布尔掩码。

    entries: [(bbox, label), ...]
    返回 numpy 布尔数组 (size, size)。
    """
    mask = np.zeros((size, size), dtype=bool)
    for bbox, _ in entries:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, min(x1, size)), max(0, min(y1, size))
        x2, y2 = max(0, min(x2, size)), max(0, min(y2, size))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def _score_r2(gt_entries, pred_entries):
    """R2: GT 框并集 vs 预测框并集 的 IoU（栅格化）。

    空集 vs 空集 → 1.0; 任一为空 → 0.0。
    """
    gt_mask = _rasterize_union(gt_entries)
    pred_mask = _rasterize_union(pred_entries)
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def _extract_keyword_set(label):
    """从 label 文本中提取动作关键词集合（单词前缀匹配）。

    用 \\b<kw> 前缀匹配: "move" 匹配 move/moved/moves/movement，
    但不匹配 remove（move 前无词首边界）。
    """
    s = set()
    for kw in LABEL_KEYWORDS:
        if re.search(r"\b" + kw, label):
            s.add(kw)
    return s


def _has_bracket_vector(label):
    """label 中是否含至少一组括号内 2/3 值数值向量。"""
    return _count_bracket_vectors(label) > 0


def _score_r3(gt_entries, pred_entries, matches):
    """R3: 关键词子集召回 × IoU × 方向系数，含格式门控。

    matches: [(gt_idx, pred_idx, iou), ...]

    格式硬门控（前置，遍历所有 pred entries，无条件）:
      任一 pred label 满足以下任一条件 → R3 = -1:
        a) 含 move 但无箭头向量
        b) 不含 move 但有箭头向量
        c) 含 move 且有 >1 组括号数值向量

    通过门控后，对每个匹配对:
      pred 多报任一关键词 → label_score = 0。
      否则，普通关键词命中记 1，move 命中记方向系数（原 R4 融入），
      再除以 GT 关键词数。因此 move 方向只影响 move 子项。
      R3_pair = label_score × iou_ij。
    R3 = mean(R3_pairs)。格式门控优先于无匹配判定；通过后无匹配 → 0.0。
    """
    # ── 格式硬门控: 遍历所有 pred entries（不只匹配上的） ──
    for _, pred_label in pred_entries:
        pl = pred_label.lower()
        has_move = "move" in _extract_keyword_set(pl)
        vec_count = _count_bracket_vectors(pl)
        if has_move and vec_count == 0:
            return -1.0  # 有 move 但无箭头
        if not has_move and vec_count > 0:
            return -1.0  # 无 move 但有箭头
        if has_move and vec_count > 1:
            return -1.0  # 多组向量

    if not matches:
        return 0.0

    # ── 逐匹配对计算 ──
    scores = []
    for gi, pi, iou in matches:
        gt_label = gt_entries[gi][1].lower()
        pred_label = pred_entries[pi][1].lower()
        gt_set = _extract_keyword_set(gt_label)
        pred_set = _extract_keyword_set(pred_label)

        if not gt_set or not pred_set <= gt_set:
            label_score = 0.0  # GT 无可评关键词，或 pred 多报
        else:
            label_points = float(len(pred_set))

            # move 命中时，用方向系数替换该 move 子项原本的 1 分。
            if "move" in gt_set and "move" in pred_set:
                dir_coef = 1.0
                gt_vec, gt_ndim = _extract_direction_vec(
                    gt_label, gt_entries[gi][0]
                )
                pred_vec, pred_ndim = _extract_direction_vec(
                    pred_label, pred_entries[pi][0]
                )
                if gt_vec is not None:
                    # GT 给了箭头时，零向量没有方向，不能保留默认满分。
                    dir_coef = 0.0
                    if pred_vec is not None:
                        norm = np.linalg.norm(gt_vec) * np.linalg.norm(pred_vec)
                        if np.isfinite(norm) and norm >= 1e-12:
                            cos_theta = np.dot(gt_vec, pred_vec) / norm
                            if np.isfinite(cos_theta):
                                dir_coef = max(
                                    -1.0, min(1.0, float(cos_theta))
                                )
                    # GT 三值、pred 二值时无条件扣 0.2，
                    # 包括其中一个是零向量的情况。
                    if gt_ndim == 3 and pred_ndim == 2:
                        dir_coef -= 0.2
                # GT 有 move 但无箭头 → dir_coef 保持 1.0。
                # pred 有 move 但无合法箭头 → 已被格式门控拦截。
                label_points += dir_coef - 1.0

            label_score = label_points / len(gt_set)

        scores.append(label_score * iou)

    return sum(scores) / len(scores)


def _extract_bracket_vectors(label):
    """提取 label 中格式严格的 2/3 值有限数值向量。

    括号内所有分量都必须是数字；不忽略非数字内容。
    """
    vectors = []
    for match in re.finditer(r"\(([^)]+)\)", label):
        parts = re.split(r"[,\s]+", match.group(1).strip())
        if len(parts) not in (2, 3):
            continue
        try:
            nums = tuple(float(part) for part in parts)
        except ValueError:
            continue
        if all(np.isfinite(value) for value in nums):
            vectors.append(nums)
    return vectors


def _count_bracket_vectors(label):
    """统计 label 中括号内数值向量（2 值或 3 值）的组数。

    用于 R3 格式门控。
    """
    return len(_extract_bracket_vectors(label))


def _extract_direction_vec(label, bbox):
    """从 label 文本中提取括号内的数值向量，转成 2D 方向。

    两种格式:
      二值 (x, y): 位置 → 方向 = (x - cx, cy - y)
                   cx,cy = bbox 中心; 第一维正=右, 第二维正=上
      三值 (x, y, z): 方向本身 → 取前两维 (x, y)
    返回 ((dx, dy), n_dims)，n_dims ∈ {2, 3}；无法提取返回 (None, 0)。
    """
    vectors = _extract_bracket_vectors(label)
    if not vectors:
        return None, 0
    nums = vectors[0]
    if len(nums) == 2:
        x, y = nums
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        dx = x - cx
        dy = cy - y  # 图像 y 向下，取反使正=上
        return (dx, dy), 2
    elif len(nums) == 3:
        return (nums[0], nums[1]), 3
    return None, 0


# ============================================================
# R4: summary 一致性校验 (Qwen3.5-0.8B 奖励模型)
# ============================================================

def _score_r4(pred_obj, gt_obj):
    """R4: 用 Qwen3.5-0.8B 校验 pred summary 与 GT summary 是否语义一致。

    输入为调用方已经解析的 JSON 对象。预测 JSON/summary 无效属于普通坏
    样本，R4 记 0；客户端、网络、模型加载和推理异常则直接向上抛出。

    Returns:
        float ∈ [0, 1]: 1.0×P(A) + 0.5×P(B) + 0.0×P(C)。
    """
    if not isinstance(pred_obj, dict) or not isinstance(gt_obj, dict):
        return 0.0

    gt_summary = gt_obj.get("summary", "")
    pred_summary = pred_obj.get("summary", "")
    if not isinstance(gt_summary, str) or not isinstance(pred_summary, str):
        return 0.0
    if not gt_summary.strip() or not pred_summary.strip():
        return 0.0

    score = float(_get_r4_score_summary()(pred_summary, gt_summary))
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"R4 returned an invalid score: {score!r}")
    return score


def _hungarian_match(gt_entries, pred_entries):
    """对 GT 和 pred 的框做匈牙利匹配（按 IoU 最大化）。

    返回 [(gt_idx, pred_idx, iou), ...]。
    """
    n = len(gt_entries)
    m = len(pred_entries)
    if n == 0 or m == 0:
        return []
    iou_matrix = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = _iou(gt_entries[i][0], pred_entries[j][0])
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    return [(int(r), int(c), float(iou_matrix[r, c]))
            for r, c in zip(row_ind, col_ind)]


# ============================================================
# 第三类: spatial_consistency_bbox 组合奖励
# ============================================================

def score_spatial_consistency_bbox(solution_str, ground_truth):
    """第三类奖励: spatial_consistency_bbox 组合奖励。

    GT boxes 为空（正例）:
      C=0 → R=0;  C=1 且预测空 → 1.0;  C=1 且预测非空 → 0.2
    GT boxes 非空（负例）:
      R = C × (0.15 + 0.65×R2 + 0.1×R3 + 0.1×R4)

    R3 已融入 IoU 缩放和方向系数（原方向 R4）。
    R4 为 summary 一致性校验（Qwen3.5-0.8B 奖励模型）。
    """
    # C: answer 门控
    gt_obj = _parse_json_obj(ground_truth)
    pred_obj = _parse_json_obj(solution_str)
    gt_answer = _normalize_answer(gt_obj.get("answer") if gt_obj else None)
    pred_answer = _normalize_answer(pred_obj.get("answer") if pred_obj else None)
    c = 1.0 if (gt_answer is not None and pred_answer is not None
                and gt_answer == pred_answer) else 0.0
    if c == 0.0:
        return 0.0

    # 解析 boxes entries
    gt_entries = _parse_bbox_entries(ground_truth)
    pred_entries = _parse_bbox_entries(solution_str)

    # GT boxes 为空（正例）
    if len(gt_entries) == 0:
        return 1.0 if len(pred_entries) == 0 else 0.2

    # GT boxes 非空（负例）: 走公式
    r2 = _score_r2(gt_entries, pred_entries)
    matches = _hungarian_match(gt_entries, pred_entries)
    r3 = _score_r3(gt_entries, pred_entries, matches)
    r4 = _score_r4(pred_obj, gt_obj)
    return c * (0.15 + 0.65 * r2 + 0.1 * r3 + 0.1 * r4)


# ============================================================
# 第四类: spatial_detection 组合奖励
# ============================================================

def score_spatial_detection(solution_str, ground_truth):
    """第四类奖励: spatial_detection 组合奖励（全负例，无门控）。

    R = 0.75×R2 + 0.15×R3 + 0.1×R4
    GT 结构同第三类，但无 C 门控、无正例空框分支。

    R3 已融入 IoU 缩放和方向系数（原方向 R4）。
    R4 为 summary 一致性校验（Qwen3.5-0.8B 奖励模型）。
    """
    gt_obj = _parse_json_obj(ground_truth)
    pred_obj = _parse_json_obj(solution_str)
    gt_entries = _parse_bbox_entries(ground_truth)
    pred_entries = _parse_bbox_entries(solution_str)
    r2 = _score_r2(gt_entries, pred_entries)
    matches = _hungarian_match(gt_entries, pred_entries)
    r3 = _score_r3(gt_entries, pred_entries, matches)
    r4 = _score_r4(pred_obj, gt_obj)
    return 0.75 * r2 + 0.15 * r3 + 0.1 * r4


# ============================================================
# 分类路由（按 data_source 分发）
# ============================================================

JSON_ANSWER_SOURCES = {
    "spatial_consistency_pos",
    "spatial_consistency_neg",
    # 其余第一类数据源（run_rl1.sh 中使用）
    "vst_caption",
    "spatialscore",
    "spatialcorpus_vi",
    "viewspatial",
    "vst_oor",
}

HUNGARIAN_IOU_SOURCES = {
    "humanref_cot",
}

SPATIAL_CONSISTENCY_BBOX_SOURCES = {
    "spatial_consistency_bbox_pos",
    "spatial_consistency_bbox_neg",
}

SPATIAL_DETECTION_SOURCES = {
    "spatial_detection",
}

KNOWN_SOURCES = (JSON_ANSWER_SOURCES | HUNGARIAN_IOU_SOURCES
                 | SPATIAL_CONSISTENCY_BBOX_SOURCES | SPATIAL_DETECTION_SOURCES)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl 标准 reward function 入口。按 data_source 路由到对应奖励类别。

    Args:
        data_source: 数据源标识
        solution_str: 模型生成的完整回复
        ground_truth: 标准答案（JSON 字符串）
        extra_info: 额外元数据 dict

    Returns:
        float: 标量 reward

    Raises:
        ValueError: data_source 不在任何已知集合中。
            不做静默兜底——历史上兜底到第一类曾导致 bbox 任务被
            悄悄降级为纯 answer 匹配而无人察觉。宁可当场报错。
    """
    if data_source in SPATIAL_DETECTION_SOURCES:
        return score_spatial_detection(solution_str, ground_truth)

    if data_source in SPATIAL_CONSISTENCY_BBOX_SOURCES:
        return score_spatial_consistency_bbox(solution_str, ground_truth)

    if data_source in HUNGARIAN_IOU_SOURCES:
        return score_hungarian_iou(solution_str, ground_truth)

    if data_source in JSON_ANSWER_SOURCES:
        return score_json_answer(solution_str, ground_truth)

    # 未匹配的 data_source: 直接报错，禁止静默兜底
    raise ValueError(
        f"未知的 data_source: {data_source!r}，不在任何奖励路由集合中。"
        f"已知值: {sorted(KNOWN_SOURCES)}。"
        f"请检查 parquet 的 data_source 列与 reward 路由键名是否一致。"
    )


# ============================================================
# 本地测试
# ============================================================
if __name__ == "__main__":
    think = "<think>\n分析图片中的人物位置关系...\n</think>\n"
    good_json = '{"answer": "B", "summary": "inconsistent orientation."}'

    # --- 第一类测试 ---
    print("=== 第一类: JSON answer 二值奖励 ===\n")
    cases_1 = [
        (think + good_json, '{"answer": "B"}', 1.0, "标准格式答对"),
        (think + good_json, '{"answer": "A"}', 0.0, "标准格式答错"),
        ('{"answer": "A"}', '{"answer": "A"}', 1.0, "无 think 直接 JSON"),
        ('{"answer": "a"}', '{"answer": "A"}', 1.0, "小写答案"),
        ('{"answer": "(B)"}', '{"answer": "B"}', 1.0, "答案带括号"),
        ('{"answer":"C"}', '{"answer": "C"}', 1.0, "紧凑 JSON"),
        ('{"answer": "A"} 然后 {"answer": "C"}',
         '{"answer": "C"}', 1.0, "多个 JSON 取最后一个"),
        ('{"answer": "yes"}', '{"answer": "yes"}', 1.0, "yes 判断题"),
        ('{"answer": "no"}', '{"answer": "yes"}', 0.0, "yes/no 答错"),
        ('{"answer": "B", "summary": "截断',
         '{"answer": "B"}', 1.0, "JSON 损坏但 answer 可抓"),
        ("我觉得是第二个", '{"answer": "B"}', 0.0, "无 JSON 输出"),
        ("", '{"answer": "B"}', 0.0, "空输出"),
        ('{"summary": "无 answer"}', '{"answer": "B"}', 0.0, "缺 answer 键"),
        ('{"answer": "B"}', "B", 1.0, "GT 是裸答案"),
    ]
    passed = 0
    for i, (sol, gt, exp, desc) in enumerate(cases_1):
        score = compute_score("spatialscore", sol, gt)
        status = "PASS" if abs(score - exp) < 1e-6 else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] {i+1:2d}. {desc}: reward={score:.1f} (exp {exp:.1f})")

    # --- 第二类测试 ---
    print("\n=== 第二类: bbox 匈牙利匹配 IoU 奖励 ===\n")
    cases_2 = [
        ('{"boxes": [[0,0,100,100]]}',
         '{"boxes": [[0,0,100,100]]}', 1.0, "单框完全匹配"),
        ('{"boxes": [[0,0,100,100]]}',
         '{"boxes": [[50,50,150,150]]}', None, "单框部分重叠"),
        ('{"boxes": [[0,0,100,100]]}',
         '{"boxes": [[200,200,300,300]]}', 0.0, "单框不重叠"),
        ('{"boxes": [[0,0,100,100],[200,200,300,300]]}',
         '{"boxes": [[0,0,100,100],[200,200,300,300]]}', 1.0, "双框完全匹配"),
        ('{"boxes": [[0,0,100,100]]}',
         '{"boxes": [[0,0,100,100],[200,200,300,300]]}', 0.5, "漏检一个框"),
        ('{"boxes": [[0,0,100,100],[200,200,300,300]]}',
         '{"boxes": [[0,0,100,100]]}', 0.5, "多报一个框"),
        ('{"boxes": []}', '{"boxes": [[0,0,100,100]]}', 0.0, "预测为空"),
        ('{"boxes": [[0,0,100,100]]}', '{"boxes": []}', 0.0, "GT为空"),
        ("没有输出框", '{"boxes": [[0,0,100,100]]}', 0.0, "无 JSON 输出"),
        ('{"boxes": [[-10,-10,110,110]]}',
         '{"boxes": [[0,0,100,100]]}', None, "越界 clamp"),
        ('{"boxes": [[0,0,100,100],[200,200,300,300],[400,400,500,500]]}',
         '{"boxes": [[0,0,100,100],[200,200,300,300]]}', None, "3框中2个匹配"),
    ]
    for i, (sol, gt, exp, desc) in enumerate(cases_2):
        score = compute_score("humanref_cot", sol, gt)
        if exp is not None:
            status = "PASS" if abs(score - exp) < 1e-6 else "FAIL"
            if status == "PASS":
                passed += 1
            print(f"  [{status}] {i+1:2d}. {desc}: reward={score:.4f} (exp {exp:.4f})")
        else:
            print(f"  [INFO] {i+1:2d}. {desc}: reward={score:.4f}")

    # --- 第三类测试 (R = C×(0.15+0.65×R2+0.1×R3+0.1×R4)) ---
    # 这里只回归组合公式，固定 R4=1.0，避免把模型概率波动写成精确断言。
    # reward_model.py 的 __main__ 单独负责真实模型的单条/批量推理测试。
    _r4_score_summary_fn = lambda pred, gt: 1.0
    print("\n=== 第三类: spatial_consistency_bbox 组合奖励 ===\n")
    scb_think = "\n"
    gt_pos = '{"answer": "A", "summary": "consistent", "boxes": []}'
    # 负例 GT: bbox [0,0,100,100], move (100,50)
    # 框中心=(50,50), 方向=(100-50, 50-50)=(50,0) → 正右
    gt_neg = ('{"answer": "B", "summary": "inconsistent", '
              '"boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}')
    # 负例 GT（双关键词）: move and rotate (100, 50)
    gt_neg2 = ('{"answer": "B", "summary": "inconsistent", '
               '"boxes": [{"bbox": [0,0,100,100], "label": "move and rotate (100, 50)"}]}')
    # 负例 GT: move 无箭头 → pred 有 move 即可, dir_coef=1
    gt_neg_noarrow = ('{"answer": "B", "summary": "inconsistent", '
                      '"boxes": [{"bbox": [0,0,100,100], "label": "move"}]}')
    SCB_POS = "spatial_consistency_bbox_pos"
    SCB_NEG = "spatial_consistency_bbox_neg"
    cases_3 = [
        (SCB_NEG, scb_think + '{"answer": "A", "boxes": []}', gt_neg, 0.0, "C=0 门控拦截"),
        (SCB_POS, scb_think + '{"answer": "A", "boxes": []}', gt_pos, 1.0, "正例预测空=满分"),
        (SCB_POS, scb_think + '{"answer": "A", "boxes": [{"bbox":[0,0,10,10],"label":"background"}]}',
         gt_pos, 0.2, "正例预测非空=0.2"),
        # 负例完美匹配: R2=1, R3=1, R4=1 → R = 0.15+0.65+0.1+0.1 = 1.0
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg, 1.0, "负例完美匹配=1.0"),
        # 负例部分重叠: R2=1/7, R3: dir=0 → R3=0, R4=1
        #   R = 0.15 + 0.65/7 + 0 + 0.1 = 0.3429
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [50,50,150,150], "label": "move (100, 50)"}]}',
         gt_neg, 0.15 + 0.65/7 + 0.1, "负例部分重叠(方向垂直dir=0)"),
        # 负例完全不重叠: R2=0, R3=0, R4=1 → R = 0.15 + 0 + 0 + 0.1 = 0.25
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [500,500,600,600], "label": "move (100, 50)"}]}',
         gt_neg, 0.25, "负例不重叠(IoU=0拉零R3)"),
        # R3 说多了: pred 含 delete → kw=0 → R3=0, R4=1
        #   R = 0.15 + 0.65 + 0 + 0.1 = 0.9
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move and delete (100, 50)"}]}',
         gt_neg, 0.9, "R3说多了kw=0"),
        # R3 说少了: GT={move,rotate}, pred={move} → kw=1/2, R3=0.5, R4=1
        #   R = 0.15 + 0.65 + 0.05 + 0.1 = 0.95
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg2, 0.95, "R3说少了按召回给半分"),
        # 格式门控: pred 有 move 但无箭头 → R3=-1, R4=1
        #   R = 0.15 + 0.65 - 0.1 + 0.1 = 0.8
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move"}]}',
         gt_neg, 0.8, "格式门控:move无箭头→R3=-1"),
        # 格式门控: pred 无 move 但有箭头 → R3=-1, R4=1
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "rotate (100, 50)"}]}',
         gt_neg, 0.8, "格式门控:无move有箭头→R3=-1"),
        # 格式门控: pred 有 move 且多组向量 → R3=-1, R4=1
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50) (200, 50)"}]}',
         gt_neg, 0.8, "格式门控:多组向量→R3=-1"),
        # 方向反向: dir=-1, R3=-1, R4=1 → R = 0.15 + 0.65 - 0.1 + 0.1 = 0.8
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (0, 50)"}]}',
         gt_neg, 0.8, "方向反向dir=-1"),
        # GT move 无箭头, pred move 有箭头 → dir_coef=1, R3=1, R4=1
        #   R = 0.15 + 0.65 + 0.1 + 0.1 = 1.0
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg_noarrow, 1.0, "GT无箭头pred有箭头dir=1"),
        # 单词前缀匹配: pred "moved (100, 50)" 匹配 move → R3=1, R4=1
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "moved (100, 50)"}]}',
         gt_neg, 1.0, "前缀匹配:moved匹配move"),
        # background 关键词: GT 和 pred 都含 background → R3=1, R4=1
        (SCB_NEG, scb_think + '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "background"}]}',
         ('{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "background"}]}'),
         1.0, "background关键词匹配"),
    ]
    for i, (ds, sol, gt, exp, desc) in enumerate(cases_3):
        score = compute_score(ds, sol, gt)
        if exp is not None:
            status = "PASS" if abs(score - exp) < 1e-6 else "FAIL"
            if status == "PASS":
                passed += 1
            print(f"  [{status}] {i+1:2d}. {desc}: reward={score:.4f} (exp {exp:.4f})")
        else:
            print(f"  [INFO] {i+1:2d}. {desc}: reward={score:.4f}")
    # --- 路由测试 ---
    print("\n=== data_source 路由测试 ===\n")
    sd_gt_route = ('{"answer": "B", "summary": "inconsistent", '
                   '"boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}')
    route_cases = [
        ("spatialscore", think + good_json, '{"answer": "B"}', 1.0),
        ("spatial_consistency_pos", think + good_json, '{"answer": "B"}', 1.0),
        ("spatial_consistency_neg", think + good_json, '{"answer": "B"}', 1.0),
        ("vst_caption", think + good_json, '{"answer": "B"}', 1.0),
        ("viewspatial", think + good_json, '{"answer": "B"}', 1.0),
        ("vst_oor", think + good_json, '{"answer": "B"}', 1.0),
        ("spatialcorpus_vi", think + good_json, '{"answer": "B"}', 1.0),
        ("humanref_cot",
         '{"boxes": [[0,0,100,100]]}',
         '{"boxes": [[0,0,100,100]]}', 1.0),
        ("spatial_consistency_bbox_pos",
         scb_think + '{"answer": "A", "boxes": []}',
         gt_pos, 1.0),
        # R2=1, R3=1, R4=1 → R = 0.75+0.15+0.1 = 1.0
        ("spatial_detection",
         '{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         sd_gt_route, 1.0),
    ]
    for ds, sol, gt, exp in route_cases:
        score = compute_score(ds, sol, gt)
        if exp is not None:
            status = "PASS" if abs(score - exp) < 1e-6 else "FAIL"
            if status == "PASS":
                passed += 1
            print(f"  [{status}] {ds}: reward={score:.4f} (exp {exp:.4f})")
        else:
            print(f"  [INFO] {ds}: reward={score:.4f}")

    # --- 第四类测试 (R = 0.75×R2 + 0.15×R3 + 0.1×R4) ---
    print("\n=== 第四类: spatial_detection 组合奖励 ===\n")
    sd_gt = ('{"answer": "B", "summary": "inconsistent", '
             '"boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}')
    cases_4 = [
        # 完美匹配: R2=1, R3=1, R4=1 → R = 0.75+0.15+0.1 = 1.0
        ('{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         sd_gt, 1.0, "完美匹配=1.0"),
        # 部分重叠: R2=1/7, R3: dir=0 → R3=0, R4=1
        #   R = 0.75/7 + 0 + 0.1 = 0.2071
        ('{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [50,50,150,150], "label": "move (100, 50)"}]}',
         sd_gt, 0.75/7 + 0.1, "部分重叠(方向垂直dir=0)"),
        # 完全不重叠: R2=0, R3=0, R4=1 → R = 0 + 0 + 0.1 = 0.1
        ('{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [500,500,600,600], "label": "move (100, 50)"}]}',
         sd_gt, 0.1, "不重叠(IoU=0拉零R3)"),
        # 预测无框: R2=0, 无匹配 → R3=0, R4=1 → R = 0.1
        ('{"answer": "B", "summary": "inconsistent", "boxes": []}', sd_gt, 0.1, "预测无框(R4=0.1)"),
        # R3 说多了: pred 含 delete → kw=0 → R3=0, R4=1 → R = 0.75 + 0.1 = 0.85
        ('{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move and delete (100, 50)"}]}',
         sd_gt, 0.85, "R3说多了kw=0"),
        # 格式门控: pred move 无箭头 → R3=-1, R4=1 → R = 0.75 - 0.15 + 0.1 = 0.7
        ('{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move"}]}',
         sd_gt, 0.75 - 0.15 + 0.1, "格式门控:move无箭头→R3=-1"),
        # 方向反向: dir=-1, R3=-1, R4=1 → R = 0.75 - 0.15 + 0.1 = 0.7
        ('{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (0, 50)"}]}',
         sd_gt, 0.75 - 0.15 + 0.1, "方向反向dir=-1"),
    ]
    for i, (sol, gt, exp, desc) in enumerate(cases_4):
        score = compute_score("spatial_detection", sol, gt)
        if exp is not None:
            status = "PASS" if abs(score - exp) < 1e-6 else "FAIL"
            if status == "PASS":
                passed += 1
            print(f"  [{status}] {i+1:2d}. {desc}: reward={score:.4f} (exp {exp:.4f})")
        else:
            print(f"  [INFO] {i+1:2d}. {desc}: reward={score:.4f}")
    # --- P0 回归: _extract_last_json_obj 括号平衡扫描 ---
    print("\n=== P0 回归: think 含花括号 / 嵌套 / 并列 JSON ===\n")
    gt_neg = ('{"answer": "B", "summary": "inconsistent", '
              '"boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}')
    p0_cases = [
        # think 干净, 末尾单个 JSON → 正常解析
        ('<think>分析方位</think>\n{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg, 1.0, "think干净+嵌套boxes"),
        # think 内含花括号(模型在推理里写了 JSON 片段) → 旧贪婪正则会整体归零
        ('<think>先看 {"a": 1} 这个位置, 再看朝向</think>\n{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg, 1.0, "think含花括号(核心回归)"),
        # think 内含未闭合花括号(纯文本里的集合记号 {A,B})
        ('<think>候选集是 {A, B} 两个</think>\n{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg, 1.0, "think含未闭合花括号"),
        # think 内含引号包裹的花括号(JSON 字符串值里出现 {})
        ('<think>模型说 "{}" 表示空</think>\n{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg, 1.0, "think引号内花括号"),
        # 并列两个 JSON, 取最后一个含 answer 的
        ('{"answer": "A"} 然后 {"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg, 1.0, "并列JSON取最后"),
        # think 内含转义引号
        ('<think>她说 \\"hi {there}\\"</think>\n{"answer": "B", "summary": "inconsistent", "boxes": [{"bbox": [0,0,100,100], "label": "move (100, 50)"}]}',
         gt_neg, 1.0, "think含转义引号+花括号"),
    ]
    for i, (sol, gt, exp, desc) in enumerate(p0_cases):
        score = compute_score("spatial_consistency_bbox_neg", sol, gt)
        status = "PASS" if abs(score - exp) < 1e-6 else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] {i+1}. {desc}: reward={score:.4f} (exp {exp:.4f})")

    # --- 未知 data_source 必须报错（禁止静默兜底）---
    print("\n=== 未知 data_source 报错测试 ===\n")
    unknown_cases = [
        "spatial_consistency",          # 旧键名（已改名，不应再静默走第一类）
        "spatial_consistency_bbox",     # 旧键名
        "typo_source",                  # 拼写错误
        "",                             # 空值
    ]
    for ds in unknown_cases:
        try:
            score = compute_score(ds, think + good_json, '{"answer": "B"}')
            print(f"  [FAIL] {ds!r}: 未报错，返回 {score}（静默兜底！）")
        except ValueError as e:
            passed += 1
            print(f"  [PASS] {ds!r}: 正确抛出 ValueError")

    total = (len(cases_1)
             + sum(1 for _, _, e, _ in cases_2 if e is not None)
             + sum(1 for _, _, _, e, _ in cases_3 if e is not None)
             + sum(1 for _, _, e, _ in cases_4 if e is not None)
             + sum(1 for _, _, e, _ in route_cases if e is not None)
             + len(p0_cases)
             + len(unknown_cases))
    print(f"\n结果: {passed}/{total} 通过")
