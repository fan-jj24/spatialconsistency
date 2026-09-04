#!/usr/bin/env python3
"""R4 与 reasoning 门控共享模型的中央动态批处理 HTTP 服务。

多个 verl reward worker 仍逐条请求 ``POST /score``。服务把同时到达的
请求在很短的时间窗内合并。R4 和 reasoning gate 使用独立队列及 batch，
但调用同一个常驻 Qwen3.5-9B 模型，避免重复加载权重和异长 padding。
两个队列使用相同的 batch 上限和等待窗口；模型空闲时优先处理当前已
攒好请求数更多的 batch，请求数相同时处理更早等待的 batch。

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
from typing import Any, Callable, List, Optional, Tuple

LOGGER = logging.getLogger("r4_reward_server")
MAX_REQUEST_BYTES = 1024 * 1024


def _validate_unit_score(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"reward model returned an invalid score: {score!r}")
    return score


def _validate_reasoning_gate_result(value: Any) -> dict:
    supported_option = str(getattr(value, "supported_option", "")).upper()
    if supported_option not in {"A", "B", "U"}:
        raise ValueError(
            f"reward model returned an invalid supported option: {supported_option!r}"
        )
    result = {
        "supported_option": supported_option,
        "unclear_probability": _validate_unit_score(
            getattr(value, "unclear_probability", None)
        ),
    }
    return result


@dataclass
class PendingRequest:
    values: Tuple[str, ...]
    future: Future


class DynamicBatcher:
    """用最大批量和最大等待时间聚合并发的单条请求。"""

    def __init__(
        self,
        score_batch: Callable[[List[Tuple[str, ...]]], List[Any]],
        validate_result: Callable[[Any], Any],
        task_name: str,
        max_batch_size: int,
        max_wait_ms: float,
        on_fatal: Callable[[BaseException], None],
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not math.isfinite(max_wait_ms) or max_wait_ms < 0:
            raise ValueError("max_wait_ms must be finite and non-negative")
        self._score_batch = score_batch
        self._validate_result = validate_result
        self._task_name = task_name
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_ms / 1000.0
        self._on_fatal = on_fatal
        self._queue: Queue = Queue()
        self._fatal_error: Optional[BaseException] = None
        self._fatal_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"{task_name.lower()}-dynamic-batcher",
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def submit(self, *values: str) -> Any:
        if not values or not all(isinstance(value, str) for value in values):
            raise TypeError("batcher values must be strings")
        fatal_error = self.fatal_error
        if fatal_error is not None:
            raise RuntimeError(
                f"{self._task_name} batcher is in a fatal state"
            ) from fatal_error
        if self._stop.is_set():
            raise RuntimeError(f"{self._task_name} batcher is stopped")

        future = Future()
        self._queue.put(PendingRequest(tuple(values), future))
        return future.result()

    @property
    def fatal_error(self) -> Optional[BaseException]:
        with self._fatal_lock:
            return self._fatal_error

    @property
    def queued_count(self) -> int:
        """返回尚未并入当前 batch 的排队请求数。"""
        return self._queue.qsize()

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def abort(self, exc: BaseException):
        """使队列进入 fatal 状态，并唤醒尚未开始推理的请求。"""
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = exc
        self._stop.set()
        while True:
            try:
                request = self._queue.get_nowait()
            except Empty:
                break
            if not request.future.done():
                request.future.set_exception(exc)

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
                inference_started = time.monotonic()
                scores = self._score_batch([request.values for request in batch])
                inference_ms = (time.monotonic() - inference_started) * 1000.0
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
                checked_scores = [self._validate_result(score) for score in scores]
            except BaseException as exc:
                self._fail_fatally(exc, batch)
                return

            LOGGER.info(
                "Scored %s batch of %d request(s) in %.1f ms",
                self._task_name,
                len(batch),
                inference_ms,
            )
            for request, score in zip(batch, checked_scores):
                request.future.set_result(score)

    def _fail_fatally(self, exc: BaseException, active_batch: List[PendingRequest]):
        self.abort(exc)
        LOGGER.exception(
            "Fatal %s batch inference error", self._task_name, exc_info=exc
        )

        pending = list(active_batch)
        for request in pending:
            if not request.future.done():
                request.future.set_exception(exc)
        self._on_fatal(exc)


class QueueLengthInferenceScheduler:
    """串行化共享模型调用，优先执行请求数更多的已就绪 batch。"""

    def __init__(self):
        self._condition = threading.Condition()
        self._active = False
        # 值为 (batch 请求数, 到达序号, 后续队列长度读取函数)。每类只有
        # 一个 batcher 线程，因而同一任务同时最多有一个已攒好的 batch。
        self._waiting: dict[
            str, Optional[Tuple[int, int, Callable[[], int]]]
        ] = {
            "R4": None,
            "reasoning_gate": None,
        }
        self._arrival_sequence = 0

    def _may_start(self, task_name: str) -> bool:
        if self._active:
            return False
        candidates = [
            (name, waiting)
            for name, waiting in self._waiting.items()
            if waiting is not None
        ]
        if not candidates:
            return False
        # 当前总等待数 = 已攒好的 batch + 其后继续到达的排队请求。
        # 总等待数相同时，arrival_sequence 越小（等待越久）越优先。
        selected, _ = max(
            candidates,
            key=lambda item: (
                item[1][0] + item[1][2](),
                -item[1][1],
            ),
        )
        return selected == task_name

    def run(
        self,
        task_name: str,
        function: Callable,
        values: list,
        queued_count: Optional[Callable[[], int]] = None,
    ):
        if task_name not in self._waiting:
            raise ValueError(f"unknown inference task: {task_name!r}")
        with self._condition:
            if self._waiting[task_name] is not None:
                raise RuntimeError(f"{task_name} already has a waiting batch")
            self._arrival_sequence += 1
            self._waiting[task_name] = (
                len(values),
                self._arrival_sequence,
                queued_count or (lambda: 0),
            )
            try:
                self._condition.wait_for(lambda: self._may_start(task_name))
                self._active = True
            finally:
                self._waiting[task_name] = None
        try:
            return function(values)
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


class ServerState:
    def __init__(self):
        self.r4_batcher: Optional[DynamicBatcher] = None
        self.reasoning_gate_batcher: Optional[DynamicBatcher] = None
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
        if self.path not in {"/score", "/classify-option-support"}:
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = self._read_payload()
            with self.server.state.lock:
                ready = self.server.state.ready
                fatal_error = self.server.state.fatal_error
            if fatal_error is not None:
                raise RuntimeError(
                    "shared reward service is in a fatal state"
                ) from fatal_error
            if not ready:
                self._send_json(
                    503, {"error": "shared reward model is still loading"}
                )
                return

            if self.path == "/score":
                pred_summary = payload.get("pred_summary")
                gt_summary = payload.get("gt_summary")
                if not isinstance(pred_summary, str) or not isinstance(
                    gt_summary, str
                ):
                    raise ValueError(
                        "pred_summary and gt_summary must both be strings"
                    )
                batcher = self.server.state.r4_batcher
                if batcher is None:
                    raise RuntimeError("R4 batcher is unavailable")
                score = batcher.submit(pred_summary, gt_summary)
                self._send_json(200, {"score": score})
                return

            sentence = payload.get("sentence")
            option_a = payload.get("option_a")
            option_b = payload.get("option_b")
            if not all(
                isinstance(value, str) for value in (sentence, option_a, option_b)
            ):
                raise ValueError(
                    "sentence, option_a and option_b must all be strings"
                )
            batcher = self.server.state.reasoning_gate_batcher
            if batcher is None:
                raise RuntimeError("reasoning gate batcher is unavailable")
            result = batcher.submit(sentence, option_a, option_b)
            self._send_json(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except BaseException as exc:
            LOGGER.exception("reward model request failed", exc_info=exc)
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
    # R4/reasoning gate 分队列攒批，但共用相同的批量和等待设置。
    parser.add_argument("--max-batch-size", type=int, default=32)
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
            batchers = (state.r4_batcher, state.reasoning_gate_batcher)
        for shared_batcher in batchers:
            if shared_batcher is not None:
                shared_batcher.abort(exc)

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

    r4_batcher = None
    reasoning_gate_batcher = None
    model = None
    backend = None
    try:
        LOGGER.info("Loading shared R4/reasoning-gate model before accepting requests")
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
        LOGGER.info("Selected shared reward backend: %s", backend)
        model.load()
        inference_scheduler = QueueLengthInferenceScheduler()
        r4_batcher = DynamicBatcher(
            lambda pairs: inference_scheduler.run(
                "R4",
                model.score_summaries,
                pairs,
                lambda: r4_batcher.queued_count,
            ),
            validate_result=_validate_unit_score,
            task_name="R4",
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            on_fatal=fatal_shutdown,
        )
        reasoning_gate_batcher = DynamicBatcher(
            lambda items: inference_scheduler.run(
                "reasoning_gate",
                model.classify_option_support_batch,
                items,
                lambda: reasoning_gate_batcher.queued_count,
            ),
            validate_result=_validate_reasoning_gate_result,
            task_name="reasoning_gate",
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            on_fatal=fatal_shutdown,
        )
        r4_batcher.start()
        reasoning_gate_batcher.start()
        with state.lock:
            state.r4_batcher = r4_batcher
            state.reasoning_gate_batcher = reasoning_gate_batcher
            state.backend = backend
            state.ready = True
        LOGGER.info(
            "R4/reasoning-gate service ready at http://%s:%d "
            "(backend=%s, separate queues, batch=%d/%sms, "
            "larger ready queue first)",
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
        LOGGER.exception("shared reward service failed", exc_info=exc)
        shutdown()
    finally:
        if r4_batcher is not None:
            r4_batcher.close()
        if reasoning_gate_batcher is not None:
            reasoning_gate_batcher.close()
        if model is not None:
            LOGGER.info("Shutting down shared %s backend", backend or "model")
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
    if args.max_batch_size <= 0:
        raise ValueError("max-batch-size must be positive")
    if not math.isfinite(args.max_wait_ms) or args.max_wait_ms < 0:
        raise ValueError("max-wait-ms must be finite and non-negative")
    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
