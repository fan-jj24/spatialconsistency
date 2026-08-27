# -*- coding: utf-8 -*-
"""
读取 dataset.jsonl (负例) -> 调用 idealab 通道 Gemini 3.5 Flash ->
为每条样本生成 CoT + answer + summary(note翻译) + bbox description -> 写出 jsonl

使用 idealab OpenAI 兼容接口 (参考 check_idealab_model.py), 默认开启 thinking (level=high; gemini-3.5 无法关闭 thinking)。

用法:
  export IDEALAB_API_KEY=sk-xxxxx
  python generate_cot_negative_idealab.py dataset_2500.jsonl cot_output.jsonl --limit 20
  python generate_cot_negative_idealab.py dataset_2500.jsonl cot_output.jsonl --workers 8

特性: 断点续跑 / 失败重试 / JSON 容错解析 / 失败样本单独记录
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests

# =============================================================================
# idealab 接口配置 (参考 check_idealab_model.py)
# =============================================================================
API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
MODEL = "gemini-3.5-flash"
MAX_TOKENS = 8192
REQUEST_TIMEOUT = 300

# 瞬态错误码 (命中则重试)
_TRANSIENT_IDEALAB_CODES = ("MPE-001", "PL-002", "PL-001")
_TRANSIENT_HTTP_CODES = (408, 425, 429, 500, 502, 503, 504, 522, 524)


def _is_transient(err_text: str) -> bool:
    t = err_text.lower()
    for code in _TRANSIENT_IDEALAB_CODES:
        if code.lower() in t:
            return True
    for code in _TRANSIENT_HTTP_CODES:
        if str(code) in t:
            return True
    for kw in ("timeout", "timed out", "connection reset", "connection aborted",
               "remote disconnected", "temporarily unavailable", "service unavailable",
               "bad gateway", "gateway timeout", "rate limit", "too many requests",
               "模型提供方", "平台限流"):
        if kw in t:
            return True
    return False


def call_idealab(api_key: str, system_prompt: str, user_content: List[Dict],
                 max_retries: int = 3, thinking_level: str = "high") -> Dict:
    """
    调用 idealab OpenAI 兼容接口, 带重试。
    thinking_level: "low" / "medium" / "high" (gemini-3.5 无法关闭 thinking)
    返回 {"ok": True, "text": ...} 或 {"ok": False, "error": ...}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
    }
    # thinking 参数: Gemini 3 官方写法 (gemini-3.5 无法关闭 thinking, 默认 high)
    payload["thinking_config"] = {"thinking_level": thinking_level}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload,
                                 timeout=REQUEST_TIMEOUT)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if _is_transient(last_err) and attempt < max_retries - 1:
                time.sleep(min(30, 3 * (2 ** attempt)))
                continue
            return {"ok": False, "error": last_err}

        if resp.status_code != 200:
            body = resp.text[:500]
            last_err = f"HTTP {resp.status_code}: {body}"
            if _is_transient(last_err) and attempt < max_retries - 1:
                time.sleep(min(30, 3 * (2 ** attempt)))
                continue
            return {"ok": False, "error": last_err}

        try:
            data = resp.json()
        except Exception as e:
            last_err = f"response not JSON: {e}; body={resp.text[:300]}"
            return {"ok": False, "error": last_err}

        try:
            output = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            last_err = f"unexpected response shape: {str(data)[:300]}"
            return {"ok": False, "error": last_err}

        # content 有时是 list (多 part), 拼成字符串
        if isinstance(output, list):
            output = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in output
            )
        output = (output or "").strip()

        if not output:
            last_err = f"empty content; raw={str(data)[:300]}"
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"ok": False, "error": last_err}

        return {"ok": True, "text": output}

    return {"ok": False, "error": last_err}


# =============================================================================
# System Prompt (与 generate_cot_negative.py v2 一致)
# =============================================================================
SYSTEM_PROMPT = """# Task: Analyze spatial inconsistency between two video frames and produce CoT, answer, summary, and bbox descriptions

## Background
You are working with pairs of frames from AI-generated videos. Each pair has been annotated as spatially inconsistent — major elements have changed their spatial positions, appeared, disappeared, or been replaced. A human annotator has provided a note (in Chinese) describing the inconsistency, and a list of bounding boxes with modification types describing how to fix the second image.

## Input
The user message contains:
- Image A (original frame 1)
- Image B (original frame 2, which has spatial inconsistencies with A)
- A marked version of Image B with bounding boxes drawn on it (PRIVILEGED — for your understanding only, NEVER mention it in output)
- A note (Chinese, brief reason for the inconsistency)
- A list of objects, each with a bounding box and a modification type
- bg_global / bg_ambiguous flags

## CRITICAL: 3D direction convention
For "move3d" objects, the direction vector dir3d = [x, y, z] uses this convention:
- x+ = RIGHT, x- = LEFT
- y+ = UP, y- = DOWN
- z+ = INTO the screen (AWAY from the viewer, deeper into the scene), z- = OUT of the screen (TOWARD the viewer, closer)

You MUST read the sign of each component carefully:
- A POSITIVE z value means the object moves INTO the screen (away from viewer).
- A NEGATIVE z value means the object moves OUT of the screen (toward viewer).
Do NOT reverse the signs. Example: dir3d = [0.58, 0.02, -0.81] means move RIGHT (x+), slightly UP (y+), and OUT of the screen toward the viewer (z is negative).

## Modification types
Each object has a "type" field:
- "add": something needs to be added in this box region
- "delete": the content in this box needs to be deleted
- "replace": the content in this box needs to be replaced
- "move3d": the content needs to move in a 3D direction; dir3d = [x, y, z] with the convention above
- "move": the content needs to move to a 2D target point; target = {x, y} in image coordinates
- "rotate": the content needs to be rotated
- "bg_local": the local background in this box is inconsistent
- "custom": a custom text description (in the "text" field) describes the change
- "bg_global" (separate field): if true, the two images have large-scale background inconsistency

## What to produce

### 1. CoT (5-10 sentences, three parts)
- Part 1: FIRST, carefully read and describe both images in detail — the main elements, persons, objects, and background in Image A, then in Image B. Take your time to observe both images thoroughly before analyzing.
- Part 2: Analyze the inconsistencies STRICTLY according to the note and the bounding boxes provided. For each bounding box region, describe what is in the box and what the inconsistency is. If bg_global is true, describe the large-scale background inconsistency first, then each specific object change.
- Part 3: Conclusion.

IMPORTANT for Part 2:
- The note and the bounding boxes are the ground truth. Do NOT invent additional inconsistencies that are not in the note or bounding boxes.
- If something in the images seems to contradict the note or bounding boxes, attribute it to the camera viewpoint difference — the note and bounding boxes are correct. Do not overthink or add your own speculation.
- Do NOT describe changes to elements that are not mentioned in the note or bounding boxes.
- NEVER use the words "annotated", "annotation", or reference any "annotated image" / "third image" in your output. Refer to the bounding box data only as "bounding boxes" or "marked regions".

### 2. answer
Output exactly one of two strings: "consistent" or "inconsistent".
- If bg_ambiguous is true: output "consistent", and your CoT must end with "cannot be determined".
- Otherwise: output "inconsistent".

### 3. summary
Faithfully TRANSLATE the Chinese note into English. Do NOT summarize in your own words, do NOT add information, do NOT omit anything. If the note has multiple lines, join them into one sentence with commas. The translation must be faithful to the note.
If bg_ambiguous is true and the note is empty, output summary as an empty string "".

### 4. Bounding boxes (echo input, with natural language descriptions)
Output the same bounding boxes as the input, but replace each object's "type" with a natural language "description" field:

Rules for descriptions:
- Verbs must be in BASE FORM only — never use conjugated, tensed, or gerund forms. No "deleting", "moved", "is replaced", "was added", "needs moving". Use base forms like add, delete, replace, move, rotate. The sentence structure is flexible (e.g. "delete the man in brown robe" or "the man in brown robe to delete" are both fine), but every verb must stay in base form.
- ALWAYS include specific content — WHO or WHAT is being added/deleted/replaced/moved/rotated. Never write a bare "delete" or "move" without saying what.
- For "add": describe what should be added. Example: "add the missing man in black clothing"
- For "delete": describe what should be deleted. Example: "delete the man in brown robe and black hat"
- For "replace": describe what is replaced with what. Example: "replace the white-robed man with the yellow-robed bearded man"
- For "move3d": describe what moves and the direction in natural language, then append (x, y, z). Follow the 3D convention exactly — check each sign carefully. Example: dir3d = [0.58, 0.02, -0.81] means "move the man in white robe right, slightly up, and OUT of the screen toward the viewer (0.58, 0.02, -0.81)" — note z is negative so it is OUT of the screen, not into it.
- For "move": describe what moves and the target location in natural language, then append (x, y). Example: "move the man in white shirt to the lower right area (1310, 776)"
- For "rotate": describe what is rotated and how. Example: "rotate the elderly man in blue robe to face left"
- For "bg_local": describe the local background inconsistency. Example: "background changes from indoor curtains to wooden beam structure"
- For "custom": use the text field content, keeping all verbs in base form.
Keep the original box coordinates unchanged.

## Rules (strict)
- NEVER mention or reference the annotated image in any output field.
- The note and bounding boxes are ground truth — do not invent extra inconsistencies.
- Follow the 3D direction convention exactly (x+ right, y+ up, z+ into screen).
- Write CoT and summary in English.
- Bounding box coordinates must match the input exactly.

## Output format
Output strictly a single JSON object and nothing else:

{
  "line": <the input line number, as an integer>,
  "answer": "consistent" or "inconsistent",
  "cot": "<your 5-10 sentence chain-of-thought>",
  "summary": "<faithful English translation of the note>",
  "objects": [
    {
      "box": {"x": <int>, "y": <int>, "w": <int>, "h": <int>},
      "description": "<description with verbs in base form, with specific content>"
    }
  ]
}
"""


# =============================================================================
# 工具函数
# =============================================================================
def format_objects(objects):
    return json.dumps(objects, ensure_ascii=False, indent=2)


def extract_json(text):
    """从模型输出中容错提取 JSON 对象"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


BAD_VERB_PATTERNS = [
    r"\b(deleting|adding|replacing|moving|rotating)\b",
    r"\b(deleted|added|replaced|moved|rotated)\b",
    r"\b(is|are|was|were|be|been|being)\s+\w+ed\b",
    r"\b(needs?|should|must)\s+(be\s+)?\w+ing\b",
]


# 禁止在 cot/summary 中出现的词 (会泄漏特权标注图)
# 用词根匹配, 覆盖所有变体: annotated/annotation/annotations/annotator/annotators/annotate...
FORBIDDEN_TERMS = [
    "annotat",        # 覆盖 annotated / annotation / annotations / annotator / annotators / annotate / annotating
    "third image",    # 第三张图
    "labeled image",  # 标注图
    "marked image",   # 标记图
    "bounding box image",
]


# bg_ambiguous=true 且 note 为空时, 自动填入的默认 summary
DEFAULT_SUMMARY_CONSISTENT = (
    "The relative positions, orientations, and backgrounds of the characters "
    "in Figure A and Figure B are basically consistent"
)


def validate(obj, input_objects, bg_ambiguous, note):
    """轻量校验输出结构"""
    if not isinstance(obj, dict):
        return "not a dict"
    for field in ("answer", "cot", "summary", "objects"):
        if field not in obj:
            return f"missing field: {field}"
    ans = str(obj["answer"]).strip().lower()
    if ans not in ("consistent", "inconsistent"):
        return f"answer must be consistent/inconsistent, got {obj['answer']}"
    expected = "consistent" if bg_ambiguous else "inconsistent"
    if ans != expected:
        return f"answer should be '{expected}' (bg_ambiguous={bg_ambiguous}), got '{ans}'"
    if not obj["cot"].strip():
        return "empty cot"
    if bg_ambiguous and "cannot be determined" not in obj["cot"].lower():
        return "bg_ambiguous but cot missing 'cannot be determined'"
    # summary: bg_ambiguous=true 且 note 为空时, 自动填入默认句子
    if not obj["summary"].strip():
        if bg_ambiguous and not str(note).strip():
            obj["summary"] = DEFAULT_SUMMARY_CONSISTENT
        else:
            return "empty summary"
    out_objs = obj.get("objects", [])
    if len(out_objs) != len(input_objects):
        return f"object count mismatch: input={len(input_objects)} output={len(out_objs)}"
    for i, (inp, out) in enumerate(zip(input_objects, out_objs)):
        ib = inp.get("box", {})
        ob = out.get("box", {})
        for k in ("x", "y", "w", "h"):
            if ib.get(k) != ob.get(k):
                return f"box[{i}].{k} mismatch: input={ib.get(k)} output={ob.get(k)}"
        desc = out.get("description", "").strip()
        if not desc:
            return f"object[{i}] missing description"
        desc_lower = desc.lower()
        for pat in BAD_VERB_PATTERNS:
            m = re.search(pat, desc_lower)
            if m:
                return f"object[{i}] description has conjugated verb: '{m.group()}' in '{desc[:60]}'"
    # 严格检查 cot/summary 不得泄漏特权标注图 (annotation/annotator/annotated 等一律拦截)
    for field in ("cot", "summary"):
        text_lower = obj[field].lower()
        for term in FORBIDDEN_TERMS:
            if term in text_lower:
                idx = text_lower.find(term)
                snippet = obj[field][max(0, idx - 30):idx + 40]
                return f"{field} contains forbidden term '{term}': ...{snippet}..."
    return None


def process_one(api_key, item, max_retries=3, thinking_level="high"):
    """处理单条样本"""
    objects = item.get("objects", [])
    bg_ambiguous = item.get("bg_ambiguous", False)

    # 构建 OpenAI 格式的 user content (文本 + 图片)
    user_content = []

    # 文本部分
    text_part = f"""[note]
{item.get('note', '')}

[bg_global]
{item.get('bg_global', False)}

[bg_ambiguous]
{bg_ambiguous}

[objects]
{format_objects(objects)}

[3D direction convention reminder]
For move3d objects, dir3d = [x, y, z] means: x+ = RIGHT, y+ = UP, z+ = INTO the screen (away from viewer). Negative values mean the opposite: x- = LEFT, y- = DOWN, z- = OUT of the screen (toward viewer). Check each sign carefully before describing the direction.

Generate the CoT, answer, summary, and bbox descriptions for this sample. Output strictly in the JSON format specified in the system prompt."""
    user_content.append({"type": "text", "text": text_part})

    # 图片部分
    for url_key in ("image_a_url", "image_b_url", "annotated_b_url"):
        url = item.get(url_key, "")
        if url:
            user_content.append({"type": "image_url", "image_url": {"url": url}})

    result = call_idealab(api_key, SYSTEM_PROMPT, user_content,
                          max_retries=max_retries, thinking_level=thinking_level)
    if not result["ok"]:
        return False, result["error"], ""

    raw_text = result["text"]
    try:
        obj = extract_json(raw_text)
        obj["line"] = item.get("line", 0)
        err = validate(obj, objects, bg_ambiguous, item.get("note", ""))
        if err:
            return False, f"validate fail: {err}", raw_text
        return True, obj, raw_text
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return False, f"parse error: {e}", raw_text


def load_done_records(out_path):
    """断点续跑：读取已完成的 record_id"""
    done = set()
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["record_id"])
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return done


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="idealab 通道批量生成负例 CoT")
    parser.add_argument("input", help="输入 dataset jsonl")
    parser.add_argument("output", help="输出 cot_output jsonl")
    parser.add_argument("--fail-log", default=None, help="失败样本日志")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条")
    parser.add_argument("--workers", type=int, default=20, help="并发线程数")
    parser.add_argument("--thinking-level", default="high",
                        choices=["low", "medium", "high"],
                        help="Gemini thinking 等级 (默认 high; gemini-3.5 无法关闭 thinking)")
    args = parser.parse_args()

    thinking_level = args.thinking_level

    api_key = os.environ.get("IDEALAB_API_KEY") or ""
    if not api_key:
        print("ERROR: 请先设置环境变量 IDEALAB_API_KEY")
        print("  export IDEALAB_API_KEY=sk-xxxxxxxxxx")
        sys.exit(2)

    fail_log = args.fail_log or args.output + ".fail.jsonl"

    items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                obj = json.loads(line)
                obj["record_id"] = idx
                items.append(obj)
    print(f"[info] 输入 {len(items)} 条")

    done = load_done_records(args.output)
    if done:
        print(f"[info] 已完成 {len(done)} 条，跳过")
    todo = [it for it in items if it.get("record_id") not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"[info] 本次处理 {len(todo)} 条, 并发 {args.workers}")
    print(f"[info] 模型: {MODEL}, 端点: {API_URL}")
    if not todo:
        return

    write_lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}
    type_count = Counter()
    counter = [0]

    out_f = open(args.output, "a", encoding="utf-8")
    fail_f = open(fail_log, "a", encoding="utf-8")

    def handle(item):
        ok, obj_or_err, raw = process_one(api_key, item, thinking_level=thinking_level)
        with write_lock:
            counter[0] += 1
            if ok:
                stats["ok"] += 1
                for obj in item.get("objects", []):
                    type_count[obj.get("type", "unknown")] += 1
                out_obj = {
                    "record_id": item.get("record_id", 0),
                    "line": item.get("line", 0),
                    "dataset_id": item.get("dataset_id", ""),
                    "image_a_url": item.get("image_a_url", ""),
                    "image_b_url": item.get("image_b_url", ""),
                    "annotated_b_url": item.get("annotated_b_url", ""),
                    "note": item.get("note", ""),
                    "bg_global": item.get("bg_global", False),
                    "bg_ambiguous": item.get("bg_ambiguous", False),
                    "input_objects": item.get("objects", []),
                    "answer": obj_or_err["answer"],
                    "cot": obj_or_err["cot"],
                    "summary": obj_or_err["summary"],
                    "objects": obj_or_err["objects"],
                }
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()
            else:
                stats["fail"] += 1
                fail_f.write(json.dumps(
                    {"record_id": item.get("record_id", 0), "line": item.get("line", 0),
                     "error": obj_or_err, "raw": raw[:500]},
                    ensure_ascii=False) + "\n")
                fail_f.flush()
            if counter[0] % 20 == 0:
                print(f"[progress] {counter[0]}/{len(todo)} "
                      f"ok={stats['ok']} fail={stats['fail']}")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(handle, it) for it in todo]
            for fu in as_completed(futures):
                fu.result()
    finally:
        out_f.close()
        fail_f.close()

    print(f"\n===== 完成 =====")
    print(f"成功: {stats['ok']}  失败: {stats['fail']}")
    print(f"object type 分布: {dict(type_count)}")
    print(f"输出: {args.output}")
    if stats["fail"]:
        print(f"失败日志: {fail_log} (重跑同一命令会自动只补失败和未完成的)")


if __name__ == "__main__":
    main()

class OuterOSSHandle:
    def __init__(self, access_key_id, access_key_secret, endpoint, bucket_name):
        import oss2
        auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)

    def upload_file(self, oss_key, local_path):
        with open(local_path, "rb") as f:
            self.bucket.put_object(oss_key, f)

    def is_file_exist(self, oss_key):
        return self.bucket.object_exists(oss_key)

    def get_oss_url(self, oss_key, times=3600000000000):
        return self.bucket.sign_url("GET", oss_key, times)


def build_outer_handler_from_env():
    ak = os.environ.get("OUTER_OSS_ACCESS_KEY_ID", "")
    sk = os.environ.get("OUTER_OSS_ACCESS_KEY_SECRET", "")
    ep = os.environ.get("OUTER_OSS_ENDPOINT", "")
    bn = os.environ.get("OUTER_OSS_BUCKET_NAME", "")
    if not (ak and sk and ep and bn):
        return None
    return OuterOSSHandle(ak, sk, ep, bn)