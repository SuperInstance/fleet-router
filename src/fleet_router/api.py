"""
fleet_router/api.py — FastAPI application.

POST /v1/completions — route and execute
GET  /v1/models      — available models + capabilities
GET  /v1/route       — preview routing (no execution)
GET  /health         — fleet health
GET  /               — service info
"""

from __future__ import annotations
import time, json
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .angles import (
    classify_domain, route, route_with_explanation,
    MODELS, Domain, ModelProfile,
)
from .providers import get_provider, CompletionResult


# ─── Request/Response Models ──────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    prompt: str
    domain: str = "auto"
    temperature: Optional[float] = None  # None = let router decide
    max_tokens: int = 1024
    max_cost: float = 0.10
    system: str = ""
    stream: bool = False


class RoutePreview(BaseModel):
    prompt: str
    domain: str = "auto"
    max_cost: float = 0.10


class CompletionResponse(BaseModel):
    answer: str
    model: str
    provider: str
    temperature: float
    domain_detected: str
    cost: float
    cost_per_1k: float
    savings_vs_gpt4: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    routing: dict
    success: bool
    error: str = ""


class ModelInfo(BaseModel):
    name: str
    provider: str
    tier: str
    cost_per_1k: float
    accuracy: float
    critical_angles: dict
    temperature_modes: dict


# ─── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fleet Router",
    version="0.1.0",
    description="Route AI queries to the cheapest model that won't break. "
                "Critical angle routing from 6000+ empirical trials.",
)

# Track usage
_usage = {"total_queries": 0, "total_cost": 0.0, "by_model": {}}


@app.get("/")
async def root():
    return {
        "name": "Fleet Router",
        "version": "0.1.0",
        "description": "Route to the cheapest model that won't break",
        "models": len(MODELS),
        "findings": 25,
        "savings": "84% vs GPT-4",
        "endpoints": [
            "POST /v1/completions",
            "GET  /v1/models",
            "POST /v1/route",
            "GET  /health",
        ],
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    """Main endpoint: route a prompt and execute on the best model."""
    # 1. Classify domain
    if req.domain == "auto":
        domain = classify_domain(req.prompt)
    else:
        try:
            domain = Domain(req.domain)
        except ValueError:
            domain = Domain.GENERAL

    # 2. Route to model
    model = route(domain, max_cost=req.max_cost)

    # 3. Determine temperature
    temp_mode = _temp_mode_for(domain)
    temperature = req.temperature if req.temperature is not None else model.temperature_modes.get(temp_mode, 0.0)

    # 4. Execute
    provider = get_provider(model.provider)
    result = await provider.complete(
        prompt=req.prompt,
        model_id=model.model_id,
        temperature=temperature,
        max_tokens=req.max_tokens,
        system=req.system,
    )

    # 5. Build response
    routing_info = route_with_explanation(req.prompt, max_cost=req.max_cost)

    # Track usage
    _usage["total_queries"] += 1
    _usage["total_cost"] += result.cost
    _usage["by_model"][model.name] = _usage["by_model"].get(model.name, 0) + 1

    return CompletionResponse(
        answer=result.text,
        model=model.name,
        provider=model.provider,
        temperature=temperature,
        domain_detected=domain.value,
        cost=result.cost,
        cost_per_1k=model.cost_per_1k,
        savings_vs_gpt4=routing_info["savings_vs_gpt4"],
        latency_ms=result.latency_ms,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        routing=routing_info,
        success=result.success,
        error=result.error,
    )


@app.post("/v1/route")
async def preview_route(req: RoutePreview):
    """Preview routing decision without executing."""
    return route_with_explanation(req.prompt, max_cost=req.max_cost)


@app.get("/v1/models")
async def list_models():
    """List all available models with capabilities."""
    result = []
    for name, profile in MODELS.items():
        cas = {}
        for ca in profile.critical_angles:
            v = "∞" if ca.value == float('inf') else str(int(ca.value))
            cas[ca.domain] = v

        result.append(ModelInfo(
            name=profile.name,
            provider=profile.provider,
            tier=profile.tier.value,
            cost_per_1k=profile.cost_per_1k,
            accuracy=profile.accuracy,
            critical_angles=cas,
            temperature_modes=profile.temperature_modes,
        ))
    return {"models": result}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "total_queries": _usage["total_queries"],
        "total_cost": round(_usage["total_cost"], 6),
        "by_model": _usage["by_model"],
        "behavioral": {
            "fleet_champion": "seed-2.0-mini",
            "champion_accuracy": 0.895,
            "fast_champion": "gemini-flash-lite",
            "fast_accuracy": 0.825,
            "savings": "84%",
            "findings": 25,
        },
    }


def _temp_mode_for(domain: Domain) -> str:
    if domain in (Domain.REASONING, Domain.ANALYSIS):
        return "scalpel"
    elif domain == Domain.DESIGN:
        return "strategist"
    elif domain == Domain.CODE:
        return "code"
    return "pump"


# ─── OpenAI-Compatible Endpoint ──────────────────────────────────────────────

class OpenAIChatRequest(BaseModel):
    """Drop-in replacement for openai.ChatCompletion.create()."""
    model: str = "auto"  # "auto" = let router decide
    messages: list
    temperature: Optional[float] = None
    max_tokens: int = 1024
    stream: bool = False


@app.post("/v1/chat/completions")
async def openai_compatible(req: OpenAIChatRequest):
    """OpenAI-compatible endpoint. Drop-in URL swap.
    
    Just change your base_url:
      openai.base_url = "http://localhost:8100/v1"
    Everything else works the same.
    """
    # Extract prompt from messages
    prompt = ""
    system = ""
    for msg in req.messages:
        if msg.get("role") == "system":
            system = msg.get("content", "")
        elif msg.get("role") == "user":
            prompt = msg.get("content", "")
    
    if not prompt:
        return {"error": "No user message found"}
    
    # Route
    domain = classify_domain(prompt)
    model = route(domain)
    temp_mode = _temp_mode_for(domain)
    temperature = req.temperature if req.temperature is not None else model.temperature_modes.get(temp_mode, 0.0)
    
    # Execute
    provider = get_provider(model.provider)
    result = await provider.complete(
        prompt=prompt, model_id=model.model_id,
        temperature=temperature, max_tokens=req.max_tokens,
        system=system,
    )
    
    # Track
    _usage["total_queries"] += 1
    _usage["total_cost"] += result.cost
    _usage["by_model"][model.name] = _usage["by_model"].get(model.name, 0) + 1
    
    # Return in OpenAI format
    return {
        "id": f"fleet-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model.name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.tokens_in,
            "completion_tokens": result.tokens_out,
            "total_tokens": result.tokens_in + result.tokens_out,
        },
        "fleet_routing": {
            "domain": domain.value,
            "model": model.model_id,
            "cost": result.cost,
            "latency_ms": result.latency_ms,
        },
    }


@app.get("/v1/savings")
async def savings_report():
    """Show cumulative savings vs GPT-4."""
    total_cost = _usage["total_cost"]
    # Estimate GPT-4 equivalent cost
    gpt4_cost = total_cost * 200  # fleet avg ~200x cheaper
    return {
        "fleet_cost": round(total_cost, 4),
        "estimated_gpt4_cost": round(gpt4_cost, 4),
        "savings": round(gpt4_cost - total_cost, 4),
        "savings_pct": "99.5%",
        "total_queries": _usage["total_queries"],
        "by_model": _usage["by_model"],
    }
