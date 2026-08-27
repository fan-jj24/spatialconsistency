#!/usr/bin/env python3
"""R4 中央动态批处理 HTTP 服务。

多个 verl reward worker 仍逐条请求 ``POST /score``。服务把同时到达的
请求在很短的时间窗内合并，然后调用一次 ``reward_model.score_summaries``。

模型加载或任一批推理失败属于致命错误：失败会传给该批及队列中的所有
请求，服务随后以非零状态退出。服务不会重试，也不会返回降级分数。
"""

import argparse
from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from queue import Empty, Queue
import signal
import threading
import time
from typing import Callable, List, Optional, Tuple

LOGGER = logging.getLogger("r4_reward_server")
MAX_REQUEST_BYTES = 1024 * 1024


@dataclass
class PendingRequest:
    pair: Tuple[str, str]
    future: Future


class DynamicBatcher:
    """用最大批量和最大等待时间聚合并发的单条请求。"""

    def __init__(
        self,
        score_batch: Callable[[List[Tuple[str, str]]], List[float]],
        max_batch_size: int,
        max_wait_ms: float,
        on_fatal: Callable[[BaseException], None],
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not math.isfinite(max_wait_ms) or max_wait_ms < 0:
            raise ValueError("max_wait_ms must be finite and non-negative")
        self._score_batch = score_batch
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_ms / 1000.0
        self._on_fatal = on_fatal
        self._queue: Queue = Queue()
        self._fatal_error: Optional[BaseException] = None
        self._fatal_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="r4-dynamic-batcher", daemon=True
        )

    def start(self):
        self._thread.start()

    def submit(self, pred_summary: str, gt_summary: str) -> float:
        fatal_error = self.fatal_error
        if fatal_error is not None:
            raise RuntimeError("R4 batcher is in a fatal state") from fatal_error
        if self._stop.is_set():
            raise RuntimeError("R4 batcher is stopped")

        future = Future()
        self._queue.put(PendingRequest((pred_summary, gt_summary), future))
        return future.result()

    @property
    def fatal_error(self) -> Optional[BaseException]:
        with self._fatal_lock:
            return self._fatal_error

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except Empty:
                continue

            batch = [first]
            deadline = time.monotonic() + self._max_wait_seconds
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except Empty:
                    break

            try:
                scores = self._score_batch([request.pair for request in batch])
                if not isinstance(scores, (list, tuple)):
                    raise TypeError(
                        "score_summaries must return a list or tuple, "
                        f"got {type(scores).__name__}"
                    )
                if len(scores) != len(batch):
                    raise RuntimeError(
                        "score_summaries returned the wrong number of scores: "
                        f"expected {len(batch)}, got {len(scores)}"
                    )
                checked_scores = []
                for score in scores:
                    score = float(score)
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError(
                            f"score_summaries returned an invalid score: {score!r}"
                        )
                    checked_scores.append(score)
            except BaseException as exc:
                self._fail_fatally(exc, batch)
                return

            LOGGER.info("Scored R4 batch of %d request(s)", len(batch))
            for request, score in zip(batch, checked_scores):
                request.future.set_result(score)

    def _fail_fatally(self, exc: BaseException, active_batch: List[PendingRequest]):
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = exc
        self._stop.set()
        LOGGER.exception("Fatal R4 batch inference error", exc_info=exc)

        pending = list(active_batch)
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except Empty:
                break
        for request in pending:
            if not request.future.done():
                request.future.set_exception(exc)
        self._on_fatal(exc)


class ServerState:
    def __init__(self):
        self.batcher: Optional[DynamicBatcher] = None
        self.backend: Optional[str] = None
        self.ready = False
        self.fatal_error: Optional[BaseException] = None
        self.lock = threading.Lock()


class RewardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 1024

    def __init__(self, address, state: ServerState):
        self.state = state
        super().__init__(address, RewardRequestHandler)


class RewardRequestHandler(BaseHTTPRequestHandler):
    server: RewardHTTPServer

    def do_GET(self):
        if self.path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        with self.server.state.lock:
            ready = self.server.state.ready
            fatal_error = self.server.state.fatal_error
            backend = self.server.state.backend
        if fatal_error is not None:
            self._send_json(500, {"status": "fatal", "error": str(fatal_error)})
        elif ready:
            self._send_json(200, {"status": "ok", "backend": backend})
        else:
            self._send_json(503, {"status": "loading"})

    def do_POST(self):
        if self.path != "/score":
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = self._read_payload()
            pred_summary = payload.get("pred_summary")
            gt_summary = payload.get("gt_summary")
            if not isinstance(pred_summary, str) or not isinstance(gt_summary, str):
                raise ValueError("pred_summary and gt_summary must both be strings")

            with self.server.state.lock:
                batcher = self.server.state.batcher
                ready = self.server.state.ready
                fatal_error = self.server.state.fatal_error
            if fatal_error is not None:
                raise RuntimeError("R4 reward service is in a fatal state") from fatal_error
            if not ready or batcher is None:
                self._send_json(503, {"error": "R4 reward model is still loading"})
                return

            score = batcher.submit(pred_summary, gt_summary)
            self._send_json(200, {"score": score})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except BaseException as exc:
            LOGGER.exception("R4 score request failed", exc_info=exc)
            self._send_json(500, {"error": str(exc)})

    def _read_payload(self) -> dict:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        content_length = int(raw_length)
        if content_length < 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError(
                f"request body must be between 0 and {MAX_REQUEST_BYTES} bytes"
            )
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        LOGGER.debug("%s - %s", self.address_string(), fmt % args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--backend",
        choices=("auto", "vllm", "transformers"),
        default="auto",
        help="R4 后端；auto 优先 vLLM，不可用时回退 Transformers",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "奖励模型目录；默认使用 R4_MODEL_LOCAL_PATH，未配置时使用 "
            "R4_MODEL_NAME"
        ),
    )
    parser.add_argument(
        "--transformers-device",
        default="auto",
        help="Transformers 设备，如 auto、cuda、cuda:0 或 cpu",
    )
    # 每对 summary 对应一个四分类 prompt；默认一次送入后端 100 条。
    parser.add_argument("--max-batch-size", type=int, default=100)
    parser.add_argument("--max-wait-ms", type=float, default=20.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run_server(args) -> int:
    state = ServerState()
    server = RewardHTTPServer((args.host, args.port), state)
    server_thread = threading.Thread(
        target=server.serve_forever, name="r4-http-server", daemon=True
    )
    server_thread.start()

    shutdown_started = threading.Event()

    def shutdown():
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        server.shutdown()

    def fatal_shutdown(exc: BaseException):
        with state.lock:
            state.fatal_error = exc
            state.ready = False

        # 给正在返回 500 的 handler 一个很短的发送窗口，然后非零退出。
        def delayed_shutdown():
            time.sleep(0.25)
            shutdown()

        threading.Thread(target=delayed_shutdown, daemon=True).start()

    def handle_signal(signum, _frame):
        LOGGER.info("Received signal %s, shutting down", signum)
        threading.Thread(target=shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_signal)

    batcher = None
    model = None
    backend = None
    try:
        LOGGER.info("Loading R4 reward model before accepting score requests")
        # 延迟导入，确保模块导入和 DynamicBatcher 单元测试不要求任何
        # 模型依赖。auto 在 Windows 或 vLLM 无法导入时使用 Transformers。
        import reward_model as reward_model_module
        from reward_model_transformers import create_reward_model

        model, backend = create_reward_model(
            reward_model_module,
            model_path=getattr(args, "model_path", None),
            backend=getattr(args, "backend", "auto"),
            device=getattr(args, "transformers_device", "auto"),
        )
        LOGGER.info("Selected R4 backend: %s", backend)
        model.load()
        batcher = DynamicBatcher(
            model.score_summaries,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            on_fatal=fatal_shutdown,
        )
        batcher.start()
        with state.lock:
            state.batcher = batcher
            state.backend = backend
            state.ready = True
        LOGGER.info(
            "R4 reward service ready at http://%s:%d "
            "(backend=%s, max_batch_size=%d, max_wait_ms=%s)",
            args.host,
            args.port,
            backend,
            args.max_batch_size,
            args.max_wait_ms,
        )
        server_thread.join()
    except BaseException as exc:
        with state.lock:
            state.fatal_error = exc
            state.ready = False
        LOGGER.exception("R4 reward service failed", exc_info=exc)
        shutdown()
    finally:
        if batcher is not None:
            batcher.close()
        if model is not None:
            LOGGER.info("Shutting down R4 %s backend", backend or "model")
            model.close()
        server.server_close()
        server_thread.join(timeout=5)

    return 1 if state.fatal_error is not None else 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not 1 <= args.port <= 65535:
        raise ValueError(f"port must be in [1, 65535], got {args.port}")
    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
