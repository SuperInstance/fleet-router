"""
fleet_router/angles.py — Critical angle routing table.

The critical angle map IS the IP. Anyone can call seed-mini or gemini-lite.
Nobody else has measured exactly where each model breaks across
16 models × 12 domains × 5 difficulty tiers.

This module encodes those measurements as a routing table.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Domain(Enum):
    ARITHMETIC = "arithmetic"
    REASONING = "reasoning"
    CODE = "code"
    DESIGN = "design"
    ANALYSIS = "analysis"
    GENERAL = "general"


class ModelTier(Enum):
    PUMP = "pump"           # T=0.0, fast, cheap, accurate on structured tasks
    SCALPEL = "scalpel"     # T=0.0, precise on reasoning domains
    STRATEGIST = "strategist"  # T=0.7, creative/planning
    DIAGNOSTIC = "diagnostic"  # Wrong but informative
    HEAVY = "heavy"         # Opus-level, expensive, novel only


@dataclass
class CriticalAngle:
    """The depth at which accuracy drops from 100% to 0%.

    Phase transitions are BINARY (F19). Not a slope — a wall.
    ∞ means no cliff detected through maximum test depth.
    """
    model: str
    domain: str
    value: float  # ∞ = float('inf')
    confidence: str = "HIGH"  # HIGH/MEDIUM/LOW
    source: str = "fm-experiments"
    measured_date: str = "2026-05-15"


@dataclass
class ModelProfile:
    """Complete capability profile for a model."""
    name: str
    provider: str  # "deepinfra", "zai", "groq", "anthropic", "local"
    model_id: str  # API model identifier
    cost_per_1k: float  # USD per 1K tokens (input+output average)
    max_tokens: int
    supports_streaming: bool = True
    is_thinking: bool = False  # reasoning_content vs content
    temperature_modes: dict = field(default_factory=dict)
    critical_angles: list[CriticalAngle] = field(default_factory=list)
    tier: ModelTier = ModelTier.PUMP

    @property
    def accuracy(self) -> float:
        """Overall accuracy from champion atlas."""
        accuracies = {
            "seed-2.0-mini": 0.895,
            "gemini-flash-lite": 0.825,
            "hermes-70b": 0.65,
            "glm-5-turbo": 0.70,  # estimated
        }
        return accuracies.get(self.name, 0.5)

    def critical_angle_for(self, domain: str) -> float:
        """Get critical angle for a specific domain."""
        for ca in self.critical_angles:
            if ca.domain == domain:
                return ca.value
        return 0.0  # Unknown = assume no capability


# ─── Model Registry ──────────────────────────────────────────────────────────

MODELS: dict[str, ModelProfile] = {
    "seed-2.0-mini": ModelProfile(
        name="seed-2.0-mini",
        provider="deepinfra",
        model_id="ByteDance/Seed-2.0-mini",
        cost_per_1k=0.05,
        max_tokens=8192,
        tier=ModelTier.PUMP,
        temperature_modes={
            "pump": 0.0,      # Arithmetic, extraction, structured
            "strategist": 0.7, # Design, planning, creative
        },
        critical_angles=[
            CriticalAngle("seed-2.0-mini", "addition", float('inf')),
            CriticalAngle("seed-2.0-mini", "multiplication", float('inf')),
            CriticalAngle("seed-2.0-mini", "nesting", float('inf')),
            CriticalAngle("seed-2.0-mini", "magnitude", float('inf')),
            CriticalAngle("seed-2.0-mini", "coefficients", 4),
            CriticalAngle("seed-2.0-mini", "syllogism", 4),
            CriticalAngle("seed-2.0-mini", "analogy", 2),
        ],
    ),
    "gemini-flash-lite": ModelProfile(
        name="gemini-flash-lite",
        provider="deepinfra",
        model_id="google/gemini-3.1-flash-lite",
        cost_per_1k=0.002,
        max_tokens=8192,
        tier=ModelTier.SCALPEL,
        is_thinking=False,
        temperature_modes={"scalpel": 0.0},
        critical_angles=[
            CriticalAngle("gemini-flash-lite", "addition", 25),
            CriticalAngle("gemini-flash-lite", "multiplication", 9),
            CriticalAngle("gemini-flash-lite", "nesting", 5),
            CriticalAngle("gemini-flash-lite", "syllogism", float('inf')),
            CriticalAngle("gemini-flash-lite", "analogy", float('inf')),
            CriticalAngle("gemini-flash-lite", "coefficients", 3),
        ],
    ),
    "hermes-70b": ModelProfile(
        name="hermes-70b",
        provider="deepinfra",
        model_id="NousResearch/Hermes-3-Llama-3.1-405B",
        cost_per_1k=0.08,
        max_tokens=4096,
        tier=ModelTier.DIAGNOSTIC,
        temperature_modes={"diagnostic": 0.0},
        critical_angles=[
            CriticalAngle("hermes-70b", "addition", 10),
            CriticalAngle("hermes-70b", "multiplication", 5),
            CriticalAngle("hermes-70b", "nesting", 3),
            CriticalAngle("hermes-70b", "syllogism", 3),
            CriticalAngle("hermes-70b", "coefficients", 2),
        ],
    ),
    "glm-5-turbo": ModelProfile(
        name="glm-5-turbo",
        provider="zai",
        model_id="glm-5-turbo",
        cost_per_1k=0.08,
        max_tokens=4096,
        tier=ModelTier.PUMP,
        is_thinking=False,
        temperature_modes={"code": 0.3, "general": 0.0},
        critical_angles=[
            CriticalAngle("glm-5-turbo", "code", float('inf'), confidence="MEDIUM"),
        ],
    ),
}


# ─── Routing Logic ────────────────────────────────────────────────────────────

DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.ARITHMETIC: [
        "compute", "calculate", "what is", "solve", "sum", "product",
        "multiply", "divide", "square", "norm", "eisenstein", "drift",
        "snap", "integer", "factorial", "lcm", "gcd",
    ],
    Domain.REASONING: [
        "why", "explain", "analyze", "compare", "evaluate",
        "syllogism", "inference", "deduce", "conclude",
    ],
    Domain.CODE: [
        "code", "function", "class", "implement", "refactor",
        "bug", "compile", "rust", "python", "test",
    ],
    Domain.DESIGN: [
        "design", "architect", "plan", "strategy", "recommend",
        "suggest", "propose", "creative", "brainstorm",
    ],
    Domain.ANALYSIS: [
        "analyze", "measure", "benchmark", "profile", "metric",
        "statistic", "trend", "pattern", "correlation",
    ],
}


def classify_domain(prompt: str) -> Domain:
    """Classify a prompt into a routing domain.

    Uses keyword matching (v1). Hardcoded table beats LLM classification
    (Opus's insight: the table is more accurate AND cheaper).
    """
    prompt_lower = prompt.lower()

    # Score each domain by keyword hits
    scores: dict[Domain, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return Domain.GENERAL

    return max(scores, key=scores.get)


def route(domain: Domain, complexity: str = "medium",
          max_cost: float = 0.10) -> ModelProfile:
    """Route a query to the cheapest model that won't break.

    The core routing algorithm:
    1. Classify domain
    2. Find models with ∞ critical angle for that domain
    3. Among those, pick the cheapest
    4. If no ∞ model, pick highest CA within budget
    5. If nothing fits, escalate

    Returns the ModelProfile to use.
    """
    # Direct mapping from domain to best model
    ROUTING_TABLE: dict[Domain, tuple[str, str, float]] = {
        # (model_name, temperature_mode, cost_per_1k)
        Domain.ARITHMETIC: ("seed-2.0-mini", "pump", 0.05),
        Domain.REASONING: ("gemini-flash-lite", "scalpel", 0.002),
        Domain.CODE: ("glm-5-turbo", "code", 0.08),
        Domain.DESIGN: ("seed-2.0-mini", "strategist", 0.05),
        Domain.ANALYSIS: ("gemini-flash-lite", "scalpel", 0.002),
        Domain.GENERAL: ("seed-2.0-mini", "pump", 0.05),
    }

    model_name, temp_mode, cost = ROUTING_TABLE.get(domain, ROUTING_TABLE[Domain.GENERAL])

    # Check budget
    if cost > max_cost:
        # Downgrade to cheapest safe option
        model_name, temp_mode, cost = ("gemini-flash-lite", "scalpel", 0.002)

    model = MODELS[model_name]
    return model


def route_with_explanation(prompt: str, max_cost: float = 0.10) -> dict:
    """Full routing decision with explanation. For the demo page and API response."""
    domain = classify_domain(prompt)
    model = route(domain, max_cost=max_cost)

    temp_mode = "pump"
    if domain == Domain.REASONING or domain == Domain.ANALYSIS:
        temp_mode = "scalpel"
    elif domain == Domain.DESIGN:
        temp_mode = "strategist"
    elif domain == Domain.CODE:
        temp_mode = "code"

    temperature = model.temperature_modes.get(temp_mode, 0.0)

    # Build explanation
    ca = model.critical_angle_for(domain.value if isinstance(domain, Domain) else domain)
    ca_str = "∞ (no cliff)" if ca == float('inf') else str(ca)

    explanation = {
        "domain_detected": domain.value,
        "model": model.name,
        "provider": model.provider,
        "temperature": temperature,
        "temp_mode": temp_mode,
        "critical_angle": ca_str,
        "cost_per_1k": model.cost_per_1k,
        "savings_vs_gpt4": f"{round((1 - model.cost_per_1k / 30) * 100)}%",
        "reason": _routing_reason(model, domain, ca),
        "tier": model.tier.value,
    }

    return explanation


def _routing_reason(model: ModelProfile, domain: Domain, ca: float) -> str:
    """Generate human-readable routing reason."""
    if ca == float('inf'):
        return f"{model.name} has infinite critical angle for {domain.value}. No depth/magnitude cliff detected."
    elif ca >= 10:
        return f"{model.name} has CA={ca:.0f} for {domain.value}. Safe for most practical queries."
    elif ca >= 5:
        return f"{model.name} has CA={ca:.0f} for {domain.value}. Safe for moderate complexity."
    else:
        return f"{model.name} has CA={ca:.0f} for {domain.value}. Use with caution on complex queries."
