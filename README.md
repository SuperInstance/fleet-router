# fleet-router

Routes AI queries to the cheapest model that won't break. Uses empirically measured critical angles — the depth at which each model's accuracy drops from 100% to 0% — to pick the right model for each domain.

FastAPI service on port 8100. OpenAI-compatible endpoint included.

## Quick Start

```bash
pip install -e .
fleet-router --port 8100

# Or directly
uvicorn fleet_router.api:app --port 8100
```

## API

### POST /v1/completions — Route and execute

```bash
curl -X POST localhost:8100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "compute the Eisenstein integer norms for lattice vectors at 30° and 45°",
    "max_cost": 0.10
  }'
```

Response:

```json
{
  "answer": "...",
  "model": "seed-2.0-mini",
  "provider": "deepinfra",
  "temperature": 0.0,
  "domain_detected": "arithmetic",
  "cost": 0.003,
  "cost_per_1k": 0.05,
  "savings_vs_gpt4": "100%",
  "latency_ms": 1200,
  "routing": {
    "domain_detected": "arithmetic",
    "model": "seed-2.0-mini",
    "temperature": 0.0,
    "temp_mode": "pump",
    "critical_angle": "∞ (no cliff)",
    "cost_per_1k": 0.05,
    "reason": "seed-2.0-mini has infinite critical angle for arithmetic. No depth/magnitude cliff detected.",
    "tier": "pump"
  },
  "success": true
}
```

### POST /v1/route — Preview routing (no execution)

```bash
curl -X POST localhost:8100/v1/route \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "why does the Fourier transform preserve energy?"}'
```

Returns the routing decision without calling any model. Useful for debugging and cost estimation.

### POST /v1/chat/completions — OpenAI-compatible

Drop-in replacement. Just change `base_url`:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8100/v1",
    api_key="unused"  # fleet router handles keys
)

response = client.chat.completions.create(
    model="auto",  # let router decide
    messages=[
        {"role": "user", "content": "solve 2x + 3 = 11"}
    ]
)
```

Returns standard OpenAI response format with an extra `fleet_routing` field.

### GET /v1/models — List models

```json
{
  "models": [
    {
      "name": "seed-2.0-mini",
      "provider": "deepinfra",
      "tier": "pump",
      "cost_per_1k": 0.05,
      "accuracy": 0.895,
      "critical_angles": {
        "addition": "∞",
        "multiplication": "∞",
        "nesting": "∞",
        "magnitude": "∞",
        "coefficients": "4"
      },
      "temperature_modes": {"pump": 0.0, "strategist": 0.7}
    }
  ]
}
```

### GET /v1/savings — Cost report

Cumulative savings vs GPT-4 baseline.

### GET /health — Fleet health

## How Routing Works

### Critical Angles

A **critical angle** is the depth/difficulty at which a model's accuracy drops from 100% to 0%. Phase transitions are binary — not a gradual slope.

Example critical angle table:

| Model | Addition | Multiplication | Nesting | Syllogism | Coefficients |
|-------|----------|---------------|---------|-----------|-------------|
| seed-2.0-mini | ∞ | ∞ | ∞ | 4 | 4 |
| gemini-flash-lite | 25 | 9 | 5 | ∞ | 3 |
| hermes-70b | 10 | 5 | 3 | 3 | 2 |

∞ = no cliff detected at any tested depth.

### Routing Algorithm

```
1. Classify prompt → domain (keyword matching)
2. Look up routing table: domain → (model, temperature_mode, cost)
3. Check model's critical angle for that domain
4. If cost > max_cost, downgrade to cheapest safe option
5. Execute on selected model
```

### Domain Classification

Keyword-based classification into 6 domains:

| Domain | Keywords | Routed to |
|--------|----------|-----------|
| `arithmetic` | compute, calculate, solve, sum, multiply, norm | seed-2.0-mini (T=0.0) |
| `reasoning` | why, explain, analyze, syllogism, deduce | gemini-flash-lite (T=0.0) |
| `code` | code, function, rust, python, refactor | glm-5-turbo (T=0.3) |
| `design` | design, architect, plan, creative, brainstorm | seed-2.0-mini (T=0.7) |
| `analysis` | measure, benchmark, profile, metric, trend | gemini-flash-lite (T=0.0) |
| `general` | (fallback) | seed-2.0-mini (T=0.0) |

### Model Tiers

| Tier | Temperature | Use case |
|------|------------|----------|
| `pump` | 0.0 | Structured tasks: arithmetic, extraction |
| `scalpel` | 0.0 | Precise reasoning: syllogisms, analysis |
| `strategist` | 0.7 | Creative: design, planning |
| `diagnostic` | 0.0 | Wrong but informative (debugging) |
| `heavy` | varies | Expensive models for novel problems only |

## Providers

| Provider | Models | Auth |
|----------|--------|------|
| DeepInfra | seed-2.0-mini, gemini-flash-lite, hermes-70b | `~/.openclaw/workspace/.credentials/deepinfra-api-key.txt` |
| z.ai | glm-5-turbo, glm-5.1 | Hardcoded in `providers.py` |
| Groq | llama-8b, llama-70b, llama-4-scout, qwen3-32b | `~/.openclaw/workspace/.credentials/groq-api-key.txt` |

Provider adapters are in `providers.py`. Each returns a unified `CompletionResult` with content, reasoning_content, cost, and latency.

## Configuration

```python
# Override routing in angles.py
ROUTING_TABLE = {
    Domain.ARITHMETIC: ("seed-2.0-mini", "pump", 0.05),
    Domain.REASONING: ("gemini-flash-lite", "scalpel", 0.002),
    # ...
}
```

All routing logic is in `angles.py` — a single file you can edit without touching the API or provider code.

## License

MIT
