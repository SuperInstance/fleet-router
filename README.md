# Fleet Router

> Route AI queries to the cheapest model that won't break.

## What It Does

Takes any prompt, classifies the domain, and routes to the cheapest model that empirical testing has proven won't fail for that type of query.

**84% cost reduction vs GPT-4** with equal or better accuracy on structured tasks.

## Quick Start

```bash
# Install
pip install -e .

# Start the server
fleet-router --port 8100

# Or directly
python -m fleet_router.cli
```

## API

```bash
# Route and execute a query
curl -X POST http://localhost:8100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the Eisenstein norm of 3+2ω?"}'

# Preview routing (no execution)
curl -X POST http://localhost:8100/v1/route \
  -d '{"prompt": "Design a rate limiter system"}'

# List models
curl http://localhost:8100/v1/models

# Health check
curl http://localhost:8100/health
```

## How Routing Works

Every query goes through:

1. **Domain classification** — keyword matching (arithmetic, reasoning, code, design, analysis, general)
2. **Critical angle lookup** — each model has measured phase transition depths per domain
3. **Cheapest safe model** — pick the lowest-cost model with ∞ or high critical angle for the detected domain
4. **Temperature selection** — T=0.0 for structured tasks, T=0.7 for creative/strategy

### Critical Angle Table

| Model | Addition | Multiplication | Syllogism | Analogy | Cost/1K |
|-------|---------|---------------|-----------|---------|---------|
| seed-2.0-mini | ∞ | ∞ | 4 | 2 | $0.05 |
| gemini-flash-lite | 25 | 9 | ∞ | ∞ | $0.002 |
| hermes-70b | 10 | 5 | 3 | 3 | $0.08 |
| glm-5-turbo | — | — | — | — | $0.08 |

∞ = no accuracy cliff detected through maximum test depth.

## Evidence

Based on 6,000+ experimental trials across 30+ models:
- F19: Phase transitions are binary (100% → 0% in one step)
- F21: 84% cost reduction via critical angle routing
- F24: Models have non-overlapping infinite domains
- F25: Temperature is the mode switch (T=0.0 pump, T=0.7 strategist)

Full findings: SuperInstance/casting-call

## Models

| Role | Model | Temp | Accuracy | Cost |
|------|-------|------|----------|------|
| Pump | seed-2.0-mini | 0.0 | 89.5% | $0.05/1K |
| Scalpel | gemini-flash-lite | 0.0 | 82.5% | $0.002/1K |
| Strategist | seed-2.0-mini | 0.7 | 8/8 design | $0.05/1K |
| Diagnostic | hermes-70b | 0.0 | 65% | $0.08/1K |
| Code | glm-5-turbo | 0.3 | ~70% | $0.08/1K |

## Architecture

```
POST /v1/completions
       │
       ▼
  classify_domain()  →  route(domain)  →  provider.complete()
       │                      │                    │
       ▼                      ▼                    ▼
   arithmetic         seed-mini T=0.0       DeepInfra API
   reasoning          gemini-lite T=0.0     DeepInfra API
   design             seed-mini T=0.7       DeepInfra API
   code               glm-5-turbo T=0.3     z.ai API
   general            seed-mini T=0.0       DeepInfra API
```
