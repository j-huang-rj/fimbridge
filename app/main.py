import asyncio
import datetime
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# ==========================
# CONFIG
# ==========================


load_dotenv()


# Helpers
def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# Base
UPSTREAM_BASE = _env_str("UPSTREAM_BASE", "http://127.0.0.1:11434")
LISTEN_HOST = _env_str("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = _env_int("LISTEN_PORT", 11435)

# Worker
WORKER_IDLE_TTL_S = _env_int("WORKER_IDLE_TTL_S", 300)
REAPER_INTERVAL_S = _env_int("REAPER_INTERVAL_S", 30)

# Adaptive Debounce
DEBOUNCE_BASE_MS = _env_int("DEBOUNCE_BASE_MS", 50)
DEBOUNCE_MIN_MS = _env_int("DEBOUNCE_MIN_MS", 20)
DEBOUNCE_MAX_MS = _env_int("DEBOUNCE_MAX_MS", 200)
DEBOUNCE_STEP_MS = _env_int("DEBOUNCE_STEP_MS", 20)
DEBOUNCE_EMA_ALPHA = _env_float("DEBOUNCE_EMA_ALPHA", 0.35)

LATENCY_HIGH_S = _env_float("LATENCY_HIGH_S", 0.6)
LATENCY_LOW_S = _env_float("LATENCY_LOW_S", 0.3)

# Upstream Timeouts
UPSTREAM_CONNECT_TIMEOUT = _env_float("UPSTREAM_CONNECT_TIMEOUT", 1.0)
UPSTREAM_WRITE_TIMEOUT = _env_float("UPSTREAM_WRITE_TIMEOUT", 5.0)
UPSTREAM_READ_TIMEOUT = _env_float("UPSTREAM_READ_TIMEOUT", 15.0)
UPSTREAM_POOL_TIMEOUT = _env_float("UPSTREAM_POOL_TIMEOUT", 5.0)

# Circuit Breaker
CB_THRESHOLD = _env_int("CB_THRESHOLD", 3)
CB_COOLDOWN_S = _env_float("CB_COOLDOWN_S", 5.0)
CB_HALF_OPEN_PROBES = _env_int("CB_HALF_OPEN_PROBES", 1)

# FIM Dialect Translation
FIM_SOURCE_PREFIX = _env_str("FIM_SOURCE_PREFIX", "<fim_prefix>")
FIM_SOURCE_SUFFIX = _env_str("FIM_SOURCE_SUFFIX", "<fim_suffix>")
FIM_SOURCE_MIDDLE = _env_str("FIM_SOURCE_MIDDLE", "<fim_middle>")

FIM_TARGET_PREFIX = _env_str("FIM_TARGET_PREFIX", "<|fim_prefix|>")
FIM_TARGET_SUFFIX = _env_str("FIM_TARGET_SUFFIX", "<|fim_suffix|>")
FIM_TARGET_MIDDLE = _env_str("FIM_TARGET_MIDDLE", "<|fim_middle|>")

# Stop Tokens
STOP_TOKENS = [
    "<|endoftext|>",
    "<|file_separator|>",
    "<|fim_pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<PRE>",
    "<SUF>",
    "<MID>",
    "[PREFIX]",
    "[SUFFIX]",
]


# ==========================
# Core
# ==========================


def _find_fim_delimiters_anchored(
    prompt: str,
    *,
    pre: str,
    suf: str,
    mid: str,
) -> tuple[int, int, int] | None:
    i1 = prompt.find(pre)
    if i1 < 0:
        return None

    stripped = prompt.rstrip()
    if not stripped.endswith(mid):
        return None

    i3 = stripped.rfind(mid)
    if i3 < 0:
        return None

    i2 = stripped.rfind(suf, i1 + len(pre), i3)
    if i2 < 0:
        return None

    return i1, i2, i3


def translate_fim(prompt: str) -> str:
    pos = _find_fim_delimiters_anchored(
        prompt,
        pre=FIM_SOURCE_PREFIX,
        suf=FIM_SOURCE_SUFFIX,
        mid=FIM_SOURCE_MIDDLE,
    )
    if pos is None:
        return prompt

    i1, i2, i3 = pos
    stripped = prompt.rstrip()
    trailing_ws = prompt[len(stripped) :]

    return (
        stripped[:i1]
        + FIM_TARGET_PREFIX
        + stripped[i1 + len(FIM_SOURCE_PREFIX) : i2]
        + FIM_TARGET_SUFFIX
        + stripped[i2 + len(FIM_SOURCE_SUFFIX) : i3]
        + FIM_TARGET_MIDDLE
        + stripped[i3 + len(FIM_SOURCE_MIDDLE) :]
        + trailing_ws
    )


def sanitize_generate_body(body: dict[str, Any]) -> None:
    opts = body.get("options")
    if not isinstance(opts, dict):
        opts = {}

    new_opts: dict[str, Any] = {}

    if "num_predict" in opts:
        new_opts["num_predict"] = opts["num_predict"]

    if "stop" in opts:
        new_opts["stop"] = opts["stop"]
    else:
        new_opts["stop"] = STOP_TOKENS

    if new_opts:
        body["options"] = new_opts
    else:
        body.pop("options", None)


def _empty_ollama_generate_response(model: str | None) -> bytes:
    created_at = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
    payload = {
        "model": model,
        "created_at": created_at,
        "response": "",
        "done": True,
        "done_reason": "canceled",
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _client_key(req: Request, body: dict[str, Any]) -> str:
    host = req.client.host if req.client else "unknown"
    model = body.get("model") or "unknown_model"
    return f"{host}::{model}"


def _model_key(body: dict[str, Any]) -> str:
    model = body.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip().lower()
    return "unknown_model"


# ==========================
# Circuit Breaker
# ==========================


@dataclass
class CircuitBreaker:
    state: str  # "closed" | "open" | "half_open"
    fail_count: int
    opened_at: float
    half_open_budget: int
    lock: asyncio.Lock

    @classmethod
    def create(cls) -> "CircuitBreaker":
        return cls(
            state="closed",
            fail_count=0,
            opened_at=0.0,
            half_open_budget=CB_HALF_OPEN_PROBES,
            lock=asyncio.Lock(),
        )

    async def allow(self) -> bool:
        """
        Return True if upstream call is allowed.
        - closed: allow
        - open: allow only after cooldown -> transition to half_open and allow limited probes
        - half_open: allow while budget > 0
        """

        now = time.monotonic()

        async with self.lock:
            if self.state == "closed":
                return True

            if self.state == "open":
                if (now - self.opened_at) >= CB_COOLDOWN_S:
                    self.state = "half_open"
                    self.half_open_budget = CB_HALF_OPEN_PROBES
                else:
                    return False

            # half_open
            if self.half_open_budget <= 0:
                return False
            self.half_open_budget -= 1
            return True

    async def on_success(self) -> None:
        async with self.lock:
            self.state = "closed"
            self.fail_count = 0
            self.opened_at = 0.0
            self.half_open_budget = CB_HALF_OPEN_PROBES

    async def on_failure(self) -> None:
        now = time.monotonic()
        async with self.lock:
            if self.state == "half_open":
                self.state = "open"
                self.fail_count = CB_THRESHOLD
                self.opened_at = now
                return

            self.fail_count += 1
            if self.fail_count >= CB_THRESHOLD:
                self.state = "open"
                self.opened_at = now


# ==========================
# COALESCING WORKER
# ==========================


@dataclass
class PendingRequest:
    req_id: int
    body: dict[str, Any]
    model: str | None
    model_key: str  # breaker key
    fut: asyncio.Future[Response]


@dataclass
class WorkerState:
    lock: asyncio.Lock
    wake: asyncio.Event
    next_id: int
    pending: PendingRequest | None
    task: asyncio.Task[None] | None
    last_seen: float

    # Adaptive debounce state
    debounce_ms: int
    ema_latency_s: float | None


def _now() -> float:
    return time.monotonic()


def _make_502_circuit_open(model: str | None) -> Response:
    payload = {
        "error": "upstream_circuit_open",
        "model": model,
        "retry_after_s": CB_COOLDOWN_S,
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        status_code=502,
        media_type="application/json",
    )


async def _upstream_post_generate(
    client: httpx.AsyncClient, body: dict[str, Any]
) -> httpx.Response:
    return await client.post(f"{UPSTREAM_BASE}/api/generate", json=body)


async def _worker_loop(app: FastAPI, key: str) -> None:
    st: WorkerState = app.state.workers[key]
    client: httpx.AsyncClient = app.state.http

    while True:
        await st.wake.wait()
        st.wake.clear()

        # Coalescing window with small debounce
        await asyncio.sleep(st.debounce_ms / 1000.0)

        # Claim the current pending request
        async with st.lock:
            pending = st.pending
            st.pending = None
            st.last_seen = _now()

        if pending is None:
            continue

        breaker: CircuitBreaker = app.state.breakers.setdefault(
            pending.model_key, CircuitBreaker.create()
        )

        # Circuit breaker gate
        allowed = await breaker.allow()
        if not allowed:
            resp = _make_502_circuit_open(pending.model)
            if not pending.fut.done():
                pending.fut.set_result(resp)
            continue

        # Execute upstream for the claimed request
        start = _now()
        try:
            r = await _upstream_post_generate(client, pending.body)

            latency = _now() - start

            # EMA latency
            if st.ema_latency_s is None:
                st.ema_latency_s = latency
            else:
                st.ema_latency_s = (
                    1.0 - DEBOUNCE_EMA_ALPHA
                ) * st.ema_latency_s + DEBOUNCE_EMA_ALPHA * latency

            # Breaker success / failure
            if r.status_code >= 500:
                await breaker.on_failure()
            else:
                await breaker.on_success()

            # Adaptive debounce using EMA
            ref_lat = st.ema_latency_s if st.ema_latency_s is not None else latency
            if ref_lat > LATENCY_HIGH_S:
                st.debounce_ms = min(DEBOUNCE_MAX_MS, st.debounce_ms + DEBOUNCE_STEP_MS)
            elif ref_lat < LATENCY_LOW_S:
                st.debounce_ms = max(DEBOUNCE_MIN_MS, st.debounce_ms - DEBOUNCE_STEP_MS)

            resp = Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )

        except (
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ):
            await breaker.on_failure()
            resp = Response(
                content=b'{"error":"upstream_timeout"}',
                status_code=502,
                media_type="application/json",
            )

        except httpx.RequestError:
            await breaker.on_failure()
            resp = Response(
                content=b'{"error":"upstream_request_error"}',
                status_code=502,
                media_type="application/json",
            )

        except asyncio.CancelledError:
            # Allow clean shutdown or reap
            raise

        except Exception:
            await breaker.on_failure()
            resp = Response(
                content=b'{"error":"upstream_unknown_error"}',
                status_code=502,
                media_type="application/json",
            )

        finally:
            async with st.lock:
                st.last_seen = _now()

        # Always complete the claimed future
        if not pending.fut.done():
            pending.fut.set_result(resp)


async def _reaper_loop(app: FastAPI) -> None:
    try:
        while True:
            await asyncio.sleep(REAPER_INTERVAL_S)
            now = _now()

            keys_to_drop: list[str] = []

            for key, st in list(app.state.workers.items()):
                async with st.lock:
                    idle = (now - st.last_seen) > WORKER_IDLE_TTL_S
                    has_pending = st.pending is not None
                    if idle and not has_pending:
                        keys_to_drop.append(key)

            for key in keys_to_drop:
                st = app.state.workers.get(key)
                if st is None:
                    continue

                if st.task is not None and not st.task.done():
                    st.task.cancel()

                app.state.workers.pop(key, None)

    except asyncio.CancelledError:
        return


# ==========================
# APP LIFESPAN
# ==========================


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        connect=UPSTREAM_CONNECT_TIMEOUT,
        read=UPSTREAM_READ_TIMEOUT,
        write=UPSTREAM_WRITE_TIMEOUT,
        pool=UPSTREAM_POOL_TIMEOUT,
    )

    client = httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=40,
        ),
    )

    app.state.http = client
    app.state.workers: dict[str, WorkerState] = {}
    app.state.breakers: dict[str, CircuitBreaker] = {}
    app.state.reaper_task = asyncio.create_task(_reaper_loop(app))

    try:
        yield

    finally:
        # Stop reaper
        if app.state.reaper_task is not None and not app.state.reaper_task.done():
            app.state.reaper_task.cancel()

        # Stop workers
        for st in app.state.workers.values():
            if st.task is not None and not st.task.done():
                st.task.cancel()

        await client.aclose()


app = FastAPI(lifespan=lifespan)


# ==========================
# ROUTES
# ==========================


@app.get("/api/tags")
async def tags(req: Request) -> Response:
    client: httpx.AsyncClient = app.state.http
    r = await client.get(f"{UPSTREAM_BASE}/api/tags", params=req.query_params)
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


@app.api_route("/api/show", methods=["GET", "POST"])
async def show(req: Request) -> Response:
    client: httpx.AsyncClient = app.state.http

    if req.method == "GET":
        r = await client.get(f"{UPSTREAM_BASE}/api/show", params=req.query_params)

    else:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
        r = await client.post(f"{UPSTREAM_BASE}/api/show", json=body)

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


@app.post("/api/generate")
async def generate(req: Request) -> Response:
    client: httpx.AsyncClient = app.state.http

    body = await req.json()
    if not isinstance(body, dict):
        body = {}

    prompt = body.get("prompt")
    if isinstance(prompt, str):
        body["prompt"] = translate_fim(prompt)

    sanitize_generate_body(body)

    # Stream passthrough after normalization
    if body.get("stream") is True:
        url = f"{UPSTREAM_BASE}/api/generate"

        async def gen():
            async with client.stream("POST", url, json=body) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    key = _client_key(req, body)
    model = body.get("model") if isinstance(body.get("model"), str) else None
    mkey = _model_key(body)

    # Get or create worker state
    st = app.state.workers.get(key)
    if st is None:
        st = WorkerState(
            lock=asyncio.Lock(),
            wake=asyncio.Event(),
            next_id=1,
            pending=None,
            task=None,
            last_seen=_now(),
            debounce_ms=max(DEBOUNCE_MIN_MS, min(DEBOUNCE_MAX_MS, DEBOUNCE_BASE_MS)),
            ema_latency_s=None,
        )
        app.state.workers[key] = st
        st.task = asyncio.create_task(_worker_loop(app, key))

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[Response] = loop.create_future()

    async with st.lock:
        st.last_seen = _now()

        req_id = st.next_id
        st.next_id += 1

        # Clean cancel previous
        if st.pending is not None and not st.pending.fut.done():
            st.pending.fut.set_result(
                Response(
                    content=_empty_ollama_generate_response(st.pending.model),
                    status_code=200,
                    media_type="application/json",
                )
            )

        st.pending = PendingRequest(
            req_id=req_id,
            body=body,
            model=model,
            model_key=mkey,
            fut=fut,
        )

        st.wake.set()

    return await fut


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    workers = []
    for key, st in app.state.workers.items():
        async with st.lock:
            workers.append(
                {
                    "key": key,
                    "debounce_ms": st.debounce_ms,
                    "ema_latency_s": st.ema_latency_s,
                    "has_pending": st.pending is not None,
                    "idle_s": max(0.0, _now() - st.last_seen),
                }
            )

    # breaker overview
    breakers = []
    for mk, cb in app.state.breakers.items():
        async with cb.lock:
            breakers.append(
                {
                    "model": mk,
                    "state": cb.state,
                    "fail_count": cb.fail_count,
                    "cooldown_s": CB_COOLDOWN_S,
                }
            )

    return {
        "status": "ok",
        "upstream": UPSTREAM_BASE,
        "timeouts": {
            "connect_s": UPSTREAM_CONNECT_TIMEOUT,
            "read_s": UPSTREAM_READ_TIMEOUT,
            "write_s": UPSTREAM_WRITE_TIMEOUT,
            "pool_s": UPSTREAM_POOL_TIMEOUT,
        },
        "debounce": {
            "base_ms": DEBOUNCE_BASE_MS,
            "min_ms": DEBOUNCE_MIN_MS,
            "max_ms": DEBOUNCE_MAX_MS,
            "step_ms": DEBOUNCE_STEP_MS,
            "ema_alpha": DEBOUNCE_EMA_ALPHA,
            "latency_high_s": LATENCY_HIGH_S,
            "latency_low_s": LATENCY_LOW_S,
        },
        "worker_ttl": {
            "idle_ttl_s": WORKER_IDLE_TTL_S,
            "reaper_interval_s": REAPER_INTERVAL_S,
        },
        "circuit_breaker": {
            "threshold": CB_THRESHOLD,
            "cooldown_s": CB_COOLDOWN_S,
            "half_open_probes": CB_HALF_OPEN_PROBES,
        },
        "workers": workers,
        "breakers": breakers,
    }


def serve():
    import os
    import subprocess

    cmd = [
        "fastapi",
        "run",
        "--entrypoint",
        "app.main:app",
        "--host",
        LISTEN_HOST,
        "--port",
        str(LISTEN_PORT),
    ]

    try:
        if os.name == "nt":
            subprocess.run(cmd, check=True)

        else:
            os.execvp(cmd[0], cmd)

    except KeyboardInterrupt:
        pass
