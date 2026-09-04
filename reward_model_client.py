#!/usr/bin/env python3
"""R4/R5 共享动态批处理服务的同步客户端。

verl 的 ``compute_score`` 是同步函数，因此客户端也保持同步。任何连接、
超时、HTTP、JSON 或协议错误都会抛出异常，不重试、不返回降级分数。
"""

import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_REWARD_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 300.0


def _endpoint(path: str = "/score") -> str:
    base_url = os.environ.get("R4_REWARD_URL", DEFAULT_REWARD_URL).strip()
    if not base_url:
        raise ValueError("R4_REWARD_URL cannot be empty")
    return f"{base_url.rstrip('/')}{path}"


def _timeout_seconds() -> float:
    raw_value = os.environ.get(
        "R4_REWARD_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)
    )
    timeout = float(raw_value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "R4_REWARD_TIMEOUT_SECONDS must be a finite positive number, "
            f"got {raw_value!r}"
        )
    return timeout


def _post_score(path: str, payload_obj: dict, task_name: str) -> dict:
    payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
    request = Request(
        _endpoint(path),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=_timeout_seconds()) as response:
            status = response.status
            body = response.read()
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{task_name} reward service returned HTTP {exc.code}: {error_body}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", str(exc))
        raise ConnectionError(
            f"Cannot reach {task_name} reward service at {_endpoint(path)}: {reason}"
        ) from exc

    if status != 200:
        raise RuntimeError(
            f"{task_name} reward service returned unexpected HTTP {status}"
        )

    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{task_name} reward service returned invalid JSON"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            f"{task_name} reward service returned an invalid response: {result!r}"
        )
    return result


def score_summary(pred_summary: str, gt_summary: str) -> float:
    """请求中央 R4 服务并返回单条分数；任何服务异常都向上抛出。"""
    result = _post_score(
        "/score",
        {"pred_summary": pred_summary, "gt_summary": gt_summary},
        "R4",
    )

    if "score" not in result:
        raise RuntimeError(
            f"R4 reward service returned an invalid response: {result!r}"
        )
    score = float(result["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"R4 reward service returned an invalid score: {score!r}")
    return score


def score_conclusion(sentence: str, expected_stance: str) -> dict:
    """请求 R5 最后一句门控，返回 gate/冲突概率/U 概率。"""
    result = _post_score(
        "/score/r5",
        {"sentence": sentence, "expected_stance": expected_stance},
        "R5",
    )
    required = ("score", "conflict_probability", "unclear_probability")
    if any(key not in result for key in required):
        raise RuntimeError(
            f"R5 reward service returned an invalid response: {result!r}"
        )
    checked = {}
    for key in required:
        value = float(result[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"R5 reward service returned invalid {key}: {value!r}"
            )
        checked[key] = value
    return checked
