import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.main import (
    CB_COOLDOWN_S,
    CB_THRESHOLD,
    DEBOUNCE_MIN_MS,
    FIM_SOURCE_MIDDLE,
    FIM_SOURCE_PREFIX,
    FIM_SOURCE_SUFFIX,
    FIM_TARGET_MIDDLE,
    FIM_TARGET_PREFIX,
    FIM_TARGET_SUFFIX,
    STOP_TOKENS,
    UPSTREAM_BASE,
    CircuitBreaker,
    PendingRequest,
    WorkerState,
    _client_key,
    _empty_ollama_generate_response,
    _env_float,
    _env_int,
    _env_str,
    _find_fim_delimiters_anchored,
    _make_502_circuit_open,
    _model_key,
    sanitize_generate_body,
    translate_fim,
)

# ==========================
# ENV HELPERS
# ==========================


class TestEnvHelpers:
    def test_env_str_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = _env_str("NONEXISTENT_KEY", "default_value")
            assert result == "default_value"

    def test_env_str_from_env(self) -> None:
        with patch.dict("os.environ", {"MY_KEY": "my_value"}, clear=True):
            result = _env_str("MY_KEY", "default_value")
            assert result == "my_value"

    def test_env_int_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = _env_int("NONEXISTENT_KEY", 42)
            assert result == 42

    def test_env_int_from_env(self) -> None:
        with patch.dict("os.environ", {"MY_INT": "123"}, clear=True):
            result = _env_int("MY_INT", 42)
            assert result == 123

    def test_env_int_invalid_value(self) -> None:
        with patch.dict("os.environ", {"MY_INT": "not_a_number"}, clear=True):
            result = _env_int("MY_INT", 42)
            assert result == 42

    def test_env_float_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = _env_float("NONEXISTENT_KEY", 3.14)
            assert result == 3.14

    def test_env_float_from_env(self) -> None:
        with patch.dict("os.environ", {"MY_FLOAT": "2.718"}, clear=True):
            result = _env_float("MY_FLOAT", 3.14)
            assert result == pytest.approx(2.718)

    def test_env_float_invalid_value(self) -> None:
        with patch.dict("os.environ", {"MY_FLOAT": "not_a_float"}, clear=True):
            result = _env_float("MY_FLOAT", 3.14)
            assert result == pytest.approx(3.14)


# ==========================
# FIM TRANSLATION
# ==========================


class TestFindFimDelimiters:
    def test_valid_fim_prompt(self) -> None:
        prompt = f"{FIM_SOURCE_PREFIX}hello world{FIM_SOURCE_SUFFIX}more code{FIM_SOURCE_MIDDLE}"
        result = _find_fim_delimiters_anchored(
            prompt,
            pre=FIM_SOURCE_PREFIX,
            suf=FIM_SOURCE_SUFFIX,
            mid=FIM_SOURCE_MIDDLE,
        )
        assert result is not None
        i1, i2, i3 = result
        assert i1 == 0
        assert prompt[i1 : i1 + len(FIM_SOURCE_PREFIX)] == FIM_SOURCE_PREFIX
        assert prompt[i2 : i2 + len(FIM_SOURCE_SUFFIX)] == FIM_SOURCE_SUFFIX
        assert prompt[i3 : i3 + len(FIM_SOURCE_MIDDLE)] == FIM_SOURCE_MIDDLE

    def test_missing_prefix(self) -> None:
        prompt = f"hello world{FIM_SOURCE_SUFFIX}more code{FIM_SOURCE_MIDDLE}"
        result = _find_fim_delimiters_anchored(
            prompt,
            pre=FIM_SOURCE_PREFIX,
            suf=FIM_SOURCE_SUFFIX,
            mid=FIM_SOURCE_MIDDLE,
        )
        assert result is None

    def test_missing_suffix(self) -> None:
        prompt = f"{FIM_SOURCE_PREFIX}hello worldmore code{FIM_SOURCE_MIDDLE}"
        result = _find_fim_delimiters_anchored(
            prompt,
            pre=FIM_SOURCE_PREFIX,
            suf=FIM_SOURCE_SUFFIX,
            mid=FIM_SOURCE_MIDDLE,
        )
        assert result is None

    def test_missing_middle(self) -> None:
        prompt = f"{FIM_SOURCE_PREFIX}hello world{FIM_SOURCE_SUFFIX}more code"
        result = _find_fim_delimiters_anchored(
            prompt,
            pre=FIM_SOURCE_PREFIX,
            suf=FIM_SOURCE_SUFFIX,
            mid=FIM_SOURCE_MIDDLE,
        )
        assert result is None

    def test_middle_not_at_end(self) -> None:
        prompt = f"{FIM_SOURCE_PREFIX}hello{FIM_SOURCE_MIDDLE}world{FIM_SOURCE_SUFFIX}"
        result = _find_fim_delimiters_anchored(
            prompt,
            pre=FIM_SOURCE_PREFIX,
            suf=FIM_SOURCE_SUFFIX,
            mid=FIM_SOURCE_MIDDLE,
        )
        assert result is None

    def test_with_trailing_whitespace(self) -> None:
        prompt = (
            f"{FIM_SOURCE_PREFIX}hello{FIM_SOURCE_SUFFIX}world{FIM_SOURCE_MIDDLE}   \n"
        )
        result = _find_fim_delimiters_anchored(
            prompt,
            pre=FIM_SOURCE_PREFIX,
            suf=FIM_SOURCE_SUFFIX,
            mid=FIM_SOURCE_MIDDLE,
        )
        assert result is not None


class TestTranslateFim:
    def test_basic_translation(self) -> None:
        prompt = f"{FIM_SOURCE_PREFIX}prefix code{FIM_SOURCE_SUFFIX}suffix code{FIM_SOURCE_MIDDLE}"
        result = translate_fim(prompt)
        assert FIM_TARGET_PREFIX in result
        assert FIM_TARGET_SUFFIX in result
        assert FIM_TARGET_MIDDLE in result
        assert FIM_SOURCE_PREFIX not in result
        assert FIM_SOURCE_SUFFIX not in result
        assert FIM_SOURCE_MIDDLE not in result

    def test_no_translation_without_delimiters(self) -> None:
        prompt = "just regular code without FIM markers"
        result = translate_fim(prompt)
        assert result == prompt

    def test_preserves_content(self) -> None:
        prefix_content = "def hello():"
        suffix_content = "    pass"
        prompt = f"{FIM_SOURCE_PREFIX}{prefix_content}{FIM_SOURCE_SUFFIX}{suffix_content}{FIM_SOURCE_MIDDLE}"
        result = translate_fim(prompt)
        assert prefix_content in result
        assert suffix_content in result

    def test_preserves_trailing_whitespace(self) -> None:
        prompt = (
            f"{FIM_SOURCE_PREFIX}code{FIM_SOURCE_SUFFIX}more{FIM_SOURCE_MIDDLE}  \n  "
        )
        result = translate_fim(prompt)
        assert result.endswith("  \n  ")

    def test_empty_sections(self) -> None:
        prompt = f"{FIM_SOURCE_PREFIX}{FIM_SOURCE_SUFFIX}{FIM_SOURCE_MIDDLE}"
        result = translate_fim(prompt)
        expected = f"{FIM_TARGET_PREFIX}{FIM_TARGET_SUFFIX}{FIM_TARGET_MIDDLE}"
        assert result == expected


# ==========================
# SANITIZE GENERATE BODY
# ==========================


class TestSanitizeGenerateBody:
    def test_adds_default_stop_tokens(self) -> None:
        body: dict = {}
        sanitize_generate_body(body)
        assert body["options"]["stop"] == STOP_TOKENS

    def test_preserves_existing_stop_tokens(self) -> None:
        custom_stop = ["CUSTOM_STOP", "ANOTHER_STOP"]
        body = {"options": {"stop": custom_stop}}
        sanitize_generate_body(body)
        assert body["options"]["stop"] == custom_stop

    def test_preserves_num_predict(self) -> None:
        body = {"options": {"num_predict": 100}}
        sanitize_generate_body(body)
        assert body["options"]["num_predict"] == 100

    def test_removes_unrecognized_options(self) -> None:
        body = {"options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 50}}
        sanitize_generate_body(body)
        assert "temperature" not in body["options"]
        assert "top_p" not in body["options"]
        assert body["options"]["num_predict"] == 50
        assert body["options"]["stop"] == STOP_TOKENS

    def test_handles_non_dict_options(self) -> None:
        body = {"options": "invalid"}
        sanitize_generate_body(body)
        assert body["options"]["stop"] == STOP_TOKENS

    def test_handles_missing_options(self) -> None:
        body = {"model": "test-model"}
        sanitize_generate_body(body)
        assert body["options"]["stop"] == STOP_TOKENS

    def test_removes_options_if_empty(self) -> None:
        body = {"model": "test-model", "options": {}}
        # With default stop tokens, options won't be empty
        sanitize_generate_body(body)
        assert "options" in body
        assert body["options"]["stop"] == STOP_TOKENS


# ==========================
# CIRCUIT BREAKER
# ==========================


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self) -> None:
        cb = CircuitBreaker.create()
        assert cb.state == "closed"
        assert cb.fail_count == 0

    @pytest.mark.asyncio
    async def test_allows_when_closed(self) -> None:
        cb = CircuitBreaker.create()
        assert await cb.allow() is True

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker.create()
        for _ in range(CB_THRESHOLD):
            await cb.on_failure()
        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_blocks_when_open(self) -> None:
        cb = CircuitBreaker.create()
        # Force open state
        cb.state = "open"
        cb.opened_at = time.monotonic()
        assert await cb.allow() is False

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_cooldown(self) -> None:
        cb = CircuitBreaker.create()
        cb.state = "open"
        cb.opened_at = time.monotonic() - CB_COOLDOWN_S - 0.1
        assert await cb.allow() is True
        assert cb.state == "half_open"

    @pytest.mark.asyncio
    async def test_half_open_allows_limited_probes(self) -> None:
        cb = CircuitBreaker.create()
        cb.state = "half_open"
        cb.half_open_budget = 2

        assert await cb.allow() is True
        assert cb.half_open_budget == 1
        assert await cb.allow() is True
        assert cb.half_open_budget == 0
        assert await cb.allow() is False

    @pytest.mark.asyncio
    async def test_success_resets_to_closed(self) -> None:
        cb = CircuitBreaker.create()
        cb.state = "half_open"
        cb.fail_count = 3

        await cb.on_success()
        assert cb.state == "closed"
        assert cb.fail_count == 0

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self) -> None:
        cb = CircuitBreaker.create()
        cb.state = "half_open"

        await cb.on_failure()
        assert cb.state == "open"
        assert cb.fail_count == CB_THRESHOLD

    @pytest.mark.asyncio
    async def test_success_in_half_open_closes(self) -> None:
        cb = CircuitBreaker.create()
        cb.state = "half_open"

        await cb.on_success()
        assert cb.state == "closed"


# ==========================
# HELPER FUNCTIONS
# ==========================


class TestHelperFunctions:
    def test_empty_ollama_generate_response(self) -> None:
        result = _empty_ollama_generate_response("test-model")
        data = json.loads(result)
        assert data["model"] == "test-model"
        assert data["response"] == ""
        assert data["done"] is True
        assert data["done_reason"] == "canceled"
        assert "created_at" in data

    def test_empty_ollama_generate_response_none_model(self) -> None:
        result = _empty_ollama_generate_response(None)
        data = json.loads(result)
        assert data["model"] is None

    def test_make_502_circuit_open(self) -> None:
        from fastapi.responses import Response

        resp = _make_502_circuit_open("my-model")
        assert isinstance(resp, Response)
        assert resp.status_code == 502

        body = json.loads(resp.body)
        assert body["error"] == "upstream_circuit_open"
        assert body["model"] == "my-model"
        assert body["retry_after_s"] == CB_COOLDOWN_S

    def test_model_key_with_valid_model(self) -> None:
        body = {"model": "  MyModel  "}
        result = _model_key(body)
        assert result == "mymodel"

    def test_model_key_with_missing_model(self) -> None:
        body = {}
        result = _model_key(body)
        assert result == "unknown_model"

    def test_model_key_with_non_string_model(self) -> None:
        body = {"model": 123}
        result = _model_key(body)
        assert result == "unknown_model"

    def test_client_key_with_client_info(self) -> None:
        req = MagicMock()
        req.client = MagicMock()
        req.client.host = "192.168.1.1"
        body = {"model": "test-model"}

        result = _client_key(req, body)
        assert result == "192.168.1.1::test-model"

    def test_client_key_without_client(self) -> None:
        req = MagicMock()
        req.client = None
        body = {"model": "test-model"}

        result = _client_key(req, body)
        assert result == "unknown::test-model"


# ==========================
# DATA CLASSES
# ==========================


class TestDataClasses:
    @pytest.mark.asyncio
    async def test_pending_request(self) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        pr = PendingRequest(
            req_id=1,
            body={"model": "test"},
            model="test",
            model_key="test",
            fut=fut,
        )
        assert pr.req_id == 1
        assert pr.body == {"model": "test"}
        assert pr.model == "test"
        assert pr.model_key == "test"
        assert pr.fut == fut

    def test_worker_state(self) -> None:
        ws = WorkerState(
            lock=asyncio.Lock(),
            wake=asyncio.Event(),
            next_id=1,
            pending=None,
            task=None,
            last_seen=time.monotonic(),
            debounce_ms=DEBOUNCE_MIN_MS,
            ema_latency_s=None,
        )
        assert ws.next_id == 1
        assert ws.pending is None
        assert ws.debounce_ms == DEBOUNCE_MIN_MS


# ==========================
# API ENDPOINTS
# ==========================


class TestHealthzEndpoint:
    def test_healthz_returns_ok(self) -> None:
        """Test that healthz endpoint returns proper health info."""

        from app.main import app

        with TestClient(app) as client:
            response = client.get("/healthz")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "upstream" in data
            assert "timeouts" in data
            assert "debounce" in data
            assert "worker_ttl" in data
            assert "circuit_breaker" in data
            assert "workers" in data
            assert "breakers" in data


class TestTagsEndpoint:
    def test_tags_forwards_request(self) -> None:
        """Test that /api/tags forwards requests to upstream."""

        from app.main import app

        with respx.mock:
            respx.get(f"{UPSTREAM_BASE}/api/tags").mock(
                return_value=httpx.Response(200, json={"models": []})
            )

            with TestClient(app) as client:
                response = client.get("/api/tags?foo=bar")
                assert response.status_code == 200
                assert response.json() == {"models": []}


class TestShowEndpoint:
    def test_show_get(self) -> None:
        """Test GET /api/show endpoint."""

        from app.main import app

        with respx.mock:
            respx.get(f"{UPSTREAM_BASE}/api/show").mock(
                return_value=httpx.Response(200, json={"result": "ok"})
            )

            with TestClient(app) as client:
                response = client.get("/api/show?model=test")
                assert response.status_code == 200
                assert response.json() == {"result": "ok"}

    def test_show_post(self) -> None:
        """Test POST /api/show endpoint."""

        from app.main import app

        with respx.mock:
            respx.post(f"{UPSTREAM_BASE}/api/show").mock(
                return_value=httpx.Response(200, json={"result": "ok"})
            )

            with TestClient(app) as client:
                response = client.post("/api/show", json={"model": "test"})
                assert response.status_code == 200
                assert response.json() == {"result": "ok"}


class TestGenerateEndpoint:
    def test_generate_non_stream(self) -> None:
        """Test that generate handles request coalescing properly."""

        from app.main import app

        with respx.mock:
            respx.post(f"{UPSTREAM_BASE}/api/generate").mock(
                return_value=httpx.Response(
                    200, json={"response": "hello", "done": True}
                )
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={"model": "test-model", "prompt": "Hello"},
                )
                assert response.status_code == 200
                data = response.json()
                assert data["response"] == "hello"

    def test_generate_translates_fim(self) -> None:
        """Test that FIM prompts are translated correctly."""

        from app.main import app

        fim_prompt = (
            f"{FIM_SOURCE_PREFIX}hello{FIM_SOURCE_SUFFIX}suffix{FIM_SOURCE_MIDDLE}"
        )

        with respx.mock:
            # Capture the request to verify FIM translation
            route = respx.post(f"{UPSTREAM_BASE}/api/generate").mock(
                return_value=httpx.Response(
                    200, json={"response": "world", "done": True}
                )
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={"model": "test-model", "prompt": fim_prompt},
                )
                assert response.status_code == 200

                # Check that the posted body has translated FIM markers
                request = route.calls.last.request
                posted_body = json.loads(request.content)
                assert FIM_TARGET_PREFIX in posted_body["prompt"]
                assert FIM_TARGET_SUFFIX in posted_body["prompt"]
                assert FIM_TARGET_MIDDLE in posted_body["prompt"]

    def test_generate_adds_stop_tokens(self) -> None:
        """Test that stop tokens are added to the request."""

        from app.main import app

        with respx.mock:
            route = respx.post(f"{UPSTREAM_BASE}/api/generate").mock(
                return_value=httpx.Response(
                    200, json={"response": "test", "done": True}
                )
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={"model": "test-model", "prompt": "test"},
                )
                assert response.status_code == 200

                request = route.calls.last.request
                posted_body = json.loads(request.content)
                assert "options" in posted_body
                assert "stop" in posted_body["options"]

    def test_generate_stream_passthrough(self) -> None:
        """Test that streaming requests pass through correctly."""

        from app.main import app

        with respx.mock:
            # For streaming, mock with NDJSON response
            respx.post(f"{UPSTREAM_BASE}/api/generate").mock(
                return_value=httpx.Response(
                    200,
                    content=b'{"response":"test"}\n',
                    headers={"content-type": "application/x-ndjson"},
                )
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={"model": "test-model", "prompt": "test", "stream": True},
                )
                # Stream endpoint returns StreamingResponse
                assert response.status_code == 200

    def test_generate_upstream_timeout(self) -> None:
        """Test that upstream timeouts are handled gracefully."""

        from app.main import app

        with respx.mock:
            respx.post(f"{UPSTREAM_BASE}/api/generate").mock(
                side_effect=httpx.ReadTimeout("timeout")
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={"model": "test-model", "prompt": "test"},
                )
                assert response.status_code == 502
                data = response.json()
                assert data["error"] == "upstream_timeout"

    def test_generate_upstream_connection_error(self) -> None:
        """Test that upstream connection errors are handled gracefully."""

        from app.main import app

        with respx.mock:
            respx.post(f"{UPSTREAM_BASE}/api/generate").mock(
                side_effect=httpx.ConnectError("connection refused")
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={"model": "test-model", "prompt": "test"},
                )
                assert response.status_code == 502
                data = response.json()
                assert data["error"] == "upstream_request_error"

    def test_generate_upstream_500_error(self) -> None:
        """Test that upstream 500 errors are handled."""

        from app.main import app

        with respx.mock:
            respx.post(f"{UPSTREAM_BASE}/api/generate").mock(
                return_value=httpx.Response(500, json={"error": "internal error"})
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={"model": "test-model", "prompt": "test"},
                )
                assert response.status_code == 500

    def test_generate_with_custom_stop_tokens(self) -> None:
        """Test that custom stop tokens are preserved."""

        from app.main import app

        custom_stops = ["CUSTOM_STOP", "ANOTHER"]

        with respx.mock:
            route = respx.post(f"{UPSTREAM_BASE}/api/generate").mock(
                return_value=httpx.Response(
                    200, json={"response": "test", "done": True}
                )
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/generate",
                    json={
                        "model": "test-model",
                        "prompt": "test",
                        "options": {"stop": custom_stops},
                    },
                )
                assert response.status_code == 200

                request = route.calls.last.request
                posted_body = json.loads(request.content)
                assert posted_body["options"]["stop"] == custom_stops


# ==========================
# INTEGRATION TESTS
# ==========================


class TestIntegration:
    @pytest.mark.asyncio
    async def test_circuit_breaker_full_cycle(self) -> None:
        """Test a full circuit breaker cycle: closed -> open -> half_open -> closed."""

        cb = CircuitBreaker.create()

        # Start closed
        assert cb.state == "closed"
        assert await cb.allow() is True

        # Trigger failures to open
        for _ in range(CB_THRESHOLD):
            await cb.on_failure()
        assert cb.state == "open"

        # Should be blocked while open (before cooldown)
        assert await cb.allow() is False

        # Simulate cooldown passing
        cb.opened_at = time.monotonic() - CB_COOLDOWN_S - 0.1
        assert await cb.allow() is True
        assert cb.state == "half_open"

        # Success in half_open closes the circuit
        await cb.on_success()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_reopens_on_half_open_failure(self) -> None:
        """Test that circuit reopens on failure during half_open."""

        cb = CircuitBreaker.create()
        cb.state = "half_open"

        await cb.on_failure()
        assert cb.state == "open"

    def test_fim_translation_roundtrip_content(self) -> None:
        """Test that content is preserved during FIM translation."""

        prefix_code = "def hello():\n    "
        suffix_code = "\n    return True"
        prompt = f"{FIM_SOURCE_PREFIX}{prefix_code}{FIM_SOURCE_SUFFIX}{suffix_code}{FIM_SOURCE_MIDDLE}"

        result = translate_fim(prompt)

        assert prefix_code in result
        assert suffix_code in result
        assert FIM_TARGET_PREFIX in result
        assert FIM_TARGET_SUFFIX in result
        assert FIM_TARGET_MIDDLE in result
