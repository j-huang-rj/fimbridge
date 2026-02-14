# Fimbridge

A lightweight, adaptive gateway for **Ollama** that translates Fill-In-the-Middle (FIM) dialects and optimizes IDE inline code completion.

## ✨ Features

- 🔄 **FIM Dialect Translation** — Automatically translates between FIM marker formats (e.g., `<fim_prefix>` → `<|fim_prefix|>`)
- ⚡ **Request Coalescing** — Debounces rapid IDE completion requests to reduce upstream load
- 📊 **Adaptive Debounce** — Dynamically adjusts debounce timing based on upstream latency
- 🔒 **Circuit Breaker** — Protects against cascading failures with automatic recovery
- 🛑 **Stop Token Injection** — Automatically injects appropriate stop tokens for cleaner completions
- 🌊 **Streaming Support** — Pass-through streaming for real-time completions

## 🛠 Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Ollama server running locally (default: `http://127.0.0.1:11434`)

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/j-huang-rj/fimbridge.git
cd fimbridge
uv sync

# Run
uv run serve
```

Service starts at `http://localhost:11435`

## ⚙️ Configuration

Override defaults via environment variables in `.env`:

### Base

| Variable        | Default                  | Description         |
| --------------- | ------------------------ | ------------------- |
| `UPSTREAM_BASE` | `http://127.0.0.1:11434` | Ollama server URL   |
| `LISTEN_HOST`   | `127.0.0.1`              | Gateway listen host |
| `LISTEN_PORT`   | `11435`                  | Gateway listen port |

### Worker

| Variable            | Default | Description                     |
| ------------------- | ------- | ------------------------------- |
| `WORKER_IDLE_TTL_S` | `300`   | Worker idle timeout (seconds)   |
| `REAPER_INTERVAL_S` | `30`    | Reaper check interval (seconds) |

### Adaptive Debounce

| Variable             | Default | Description                      |
| -------------------- | ------- | -------------------------------- |
| `DEBOUNCE_BASE_MS`   | `50`    | Base debounce time (ms)          |
| `DEBOUNCE_MIN_MS`    | `20`    | Minimum debounce time (ms)       |
| `DEBOUNCE_MAX_MS`    | `200`   | Maximum debounce time (ms)       |
| `DEBOUNCE_STEP_MS`   | `20`    | Debounce adjustment step (ms)    |
| `DEBOUNCE_EMA_ALPHA` | `0.35`  | EMA smoothing factor for latency |
| `LATENCY_HIGH_S`     | `0.6`   | High latency threshold (seconds) |
| `LATENCY_LOW_S`      | `0.3`   | Low latency threshold (seconds)  |

### Upstream Timeouts

| Variable                   | Default | Description                  |
| -------------------------- | ------- | ---------------------------- |
| `UPSTREAM_CONNECT_TIMEOUT` | `1.0`   | Connection timeout (seconds) |
| `UPSTREAM_WRITE_TIMEOUT`   | `5.0`   | Write timeout (seconds)      |
| `UPSTREAM_READ_TIMEOUT`    | `15.0`  | Read timeout (seconds)       |
| `UPSTREAM_POOL_TIMEOUT`    | `5.0`   | Pool timeout (seconds)       |

### Circuit Breaker

| Variable              | Default | Description                         |
| --------------------- | ------- | ----------------------------------- |
| `CB_THRESHOLD`        | `3`     | Failure threshold to open circuit   |
| `CB_COOLDOWN_S`       | `5.0`   | Cooldown before half-open (seconds) |
| `CB_HALF_OPEN_PROBES` | `1`     | Probes allowed in half-open state   |

### FIM Dialect Translation

| Variable            | Default            | Description              |
| ------------------- | ------------------ | ------------------------ |
| `FIM_SOURCE_PREFIX` | `<fim_prefix>`     | Source FIM prefix marker |
| `FIM_SOURCE_SUFFIX` | `<fim_suffix>`     | Source FIM suffix marker |
| `FIM_SOURCE_MIDDLE` | `<fim_middle>`     | Source FIM middle marker |
| `FIM_TARGET_PREFIX` | `<\|fim_prefix\|>` | Target FIM prefix marker |
| `FIM_TARGET_SUFFIX` | `<\|fim_suffix\|>` | Target FIM suffix marker |
| `FIM_TARGET_MIDDLE` | `<\|fim_middle\|>` | Target FIM middle marker |

## 📡 API Endpoints

### Generate Completion (with FIM translation)

```bash
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder",
    "prompt": "<fim_prefix>def hello():<fim_suffix>    pass<fim_middle>"
  }'
```

### List Models

```bash
curl http://localhost:11435/api/tags
```

### Other Endpoints

| Endpoint    | Method    | Description                   |
| ----------- | --------- | ----------------------------- |
| `/api/show` | GET, POST | Show model details            |
| `/healthz`  | GET       | Health check with metrics     |
| `/docs`     | GET       | Interactive API documentation |

## 🧪 Development

```bash
# Run tests
uv run pytest tests/ -v

# Linting
uv run ruff check .
uv run ruff format .
```

## 📄 License

[MIT License](LICENSE)

---

**Maintainer**: Rui-Jie Huang ([@j-huang-rj](https://github.com/j-huang-rj))
