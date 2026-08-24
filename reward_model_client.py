#!/usr/bin/env python3
"""R4 动态批处理服务的同步客户端。

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


def _endpoint() -> str:
    base_url = os.environ.get("R4_REWARD_URL", DEFAULT_REWARD_URL).strip()
    if not base_url:
        raise ValueError("R4_REWARD_URL cannot be empty")
    return f"{base_url.rstrip('/')}/score"


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


def score_summary(pred_summary: str, gt_summary: str) -> float:
    """请求中央 R4 服务并返回单条分数；任何服务异常都向上抛出。"""
    payload = json.dumps(
        {"pred_summary": pred_summary, "gt_summary": gt_summary},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        _endpoint(),
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
            f"R4 reward service returned HTTP {exc.code}: {error_body}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", str(exc))
        raise ConnectionError(
            f"Cannot reach R4 reward service at {_endpoint()}: {reason}"
        ) from exc

    if status != 200:
        raise RuntimeError(f"R4 reward service returned unexpected HTTP {status}")

    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("R4 reward service returned invalid JSON") from exc

    if not isinstance(result, dict) or "score" not in result:
        raise RuntimeError(
            f"R4 reward service returned an invalid response: {result!r}"
        )
    score = float(result["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"R4 reward service returned an invalid score: {score!r}")
    return score
