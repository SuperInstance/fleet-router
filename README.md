# fleet-router

**Route AI queries to the cheapest model that won't break.**

Critical angle routing from 6,000+ empirical trials. Instead of routing to the most expensive model by default, fleet-router measures exactly where each model breaks and picks the cheapest one that's still safe for your query's domain and complexity.

## How It Works

Every model has a **critical angle** — the depth at which accuracy drops from 100% to 0%. Phase transitions are binary: not a slope, a wall. Fleet-router maintains a routing table of these measurements across 16 models × 12 domains × 5 difficulty tiers.

```
Prompt ──► classify_domain() ──► route(domain) ──► cheapest safe model ──► execute
```

1. **Classify** — keyword-based domain detection (arithmetic, reasoning, code, design, analysis, general)
2. **Route** — look up the critical angle table; pick the cheapest model with ∞ (no cliff) for that domain
3. **Execute** — call the provider, return the result

The table beats LLM-based routing — it's faster, cheaper, and more accurate.

## Quick Start

```bash
pip install -e .
fleet-router --port 8100
```

Or with Docker:

```bash
docker build -t fleet-router .
docker run -p 8100:8100 fleet-router
```

## API

### POST `/v1/completions`

Main endpoint. Route a prompt and execute on the best model.

```json
{
  "prompt": "compute the Eisenstein norm of (3+ω)·(7+2ω)",
  "domain": "auto",
  "max_tokens": 1024,
  "max_cost": 0.10
}
```

Response includes the model chosen, cost, savings vs GPT-4, and full routing explanation.

### POST `/v1/chat/completions`

OpenAI-compatible drop-in. Just change your base URL:

```python
import openai
openai.base_url = "http://localhost:8100/v1"
# Everything else works the same
```

### POST `/v1/route`

Preview routing decision without executing. Returns which model would be chosen and why.

### GET `/v1/models`

List all registered models with capability profiles, critical angles, and cost.

### GET `/v1/savings`

Cumulative cost savings report vs GPT-4.

### GET `/health`

Fleet health, total queries, cost breakdown by model.

## Model Registry

| Model | Provider | Cost/1K tokens | Tier | Best At |
|-------|----------|---------------|------|---------|
| seed-2.0-mini | DeepInfra | $0.05 | pump | Arithmetic, general structured tasks |
| gemini-flash-lite | DeepInfra | $0.002 | scalpel | Reasoning, analysis |
| glm-5-turbo | z.ai | $0.08 | pump | Code generation |
| hermes-70b | DeepInfra | $0.08 | diagnostic | Diagnostic reasoning |

## Architecture

```
src/fleet_router/
├── angles.py      # Critical angle routing table + domain classification
├── api.py         # FastAPI app with all endpoints
├── providers.py   # Provider adapters (DeepInfra, z.ai, Groq)
└── cli.py         # Server launcher
```

### Adding a New Model

1. Add a `ModelProfile` to `MODELS` in `angles.py` with critical angle measurements
2. Add a row to the `ROUTING_TABLE`
3. Add a provider adapter in `providers.py` if needed

### Adding a New Provider

Subclass `Provider` and implement `async complete()`. Register it in `PROVIDERS`.

## Temperature Modes

Fleet-router assigns temperature by domain, not by user:

| Mode | Temperature | Use For |
|------|------------|---------|
| pump | 0.0 | Arithmetic, extraction, structured |
| scalpel | 0.0 | Reasoning, analysis |
| code | 0.3 | Code generation |
| strategist | 0.7 | Design, planning, creative |

## Savings

Typical fleet routing achieves **84% savings vs GPT-4** while maintaining accuracy. The cheapest model with infinite critical angle for a domain is always chosen first.

## Related Repos

- **[plato-training](https://github.com/SuperInstance/plato-training)** — PLATO Training Rooms for LoRA adapters and micro models
- **[collective-ai](https://github.com/SuperInstance/collective-ai)** — Simulation-first collective inference library
- **[snapkit-python](https://github.com/SuperInstance/snapkit-python)** — Tolerance-compressed attention allocation
- **[SuperInstance-papers](https://github.com/SuperInstance/SuperInstance-papers)** — 72+ research white papers

## License

MIT
