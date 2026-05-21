"""Tests for fleet-router."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fleet_router.angles import (
    Domain, ModelTier, CriticalAngle, ModelProfile,
    MODELS, classify_domain, route, route_with_explanation,
)
from fleet_router.providers import CompletionResult, calculate_savings


# ─── Domain Classification ────────────────────────────────────────────────────

def test_classify_arithmetic():
    assert classify_domain("compute 2+2") == Domain.ARITHMETIC
    assert classify_domain("What is 3 * 5?") == Domain.ARITHMETIC
    assert classify_domain("calculate the sum") == Domain.ARITHMETIC


def test_classify_reasoning():
    assert classify_domain("explain why this works") == Domain.REASONING
    assert classify_domain("analyze the results") == Domain.REASONING


def test_classify_code():
    assert classify_domain("write a python function") == Domain.CODE
    assert classify_domain("refactor the class") == Domain.CODE


def test_classify_design():
    assert classify_domain("design the system") == Domain.DESIGN
    assert classify_domain("recommend an architecture") == Domain.DESIGN


def test_classify_general():
    assert classify_domain("hello there") == Domain.GENERAL
    assert classify_domain("") == Domain.GENERAL


# ─── Model Registry ───────────────────────────────────────────────────────────

def test_models_exist():
    assert len(MODELS) >= 3
    assert "seed-2.0-mini" in MODELS
    assert "gemini-flash-lite" in MODELS


def test_model_profile_fields():
    model = MODELS["seed-2.0-mini"]
    assert model.provider == "deepinfra"
    assert model.cost_per_1k > 0
    assert model.max_tokens > 0
    assert isinstance(model.tier, ModelTier)


def test_model_accuracy():
    model = MODELS["seed-2.0-mini"]
    assert 0.0 <= model.accuracy <= 1.0


def test_critical_angle_for():
    model = MODELS["seed-2.0-mini"]
    ca = model.critical_angle_for("addition")
    assert ca == float("inf")

    ca2 = model.critical_angle_for("coefficients")
    assert ca2 == 4


def test_critical_angle_unknown_domain():
    model = MODELS["seed-2.0-mini"]
    ca = model.critical_angle_for("unknown_domain_xyz")
    assert ca == 0.0


# ─── Routing ──────────────────────────────────────────────────────────────────

def test_route_arithmetic():
    model = route(Domain.ARITHMETIC)
    assert isinstance(model, ModelProfile)
    assert model.name == "seed-2.0-mini"


def test_route_reasoning():
    model = route(Domain.REASONING)
    assert model.name == "gemini-flash-lite"


def test_route_code():
    model = route(Domain.CODE)
    assert model.name == "glm-5-turbo"


def test_route_general():
    model = route(Domain.GENERAL)
    assert isinstance(model, ModelProfile)


def test_route_with_budget():
    model = route(Domain.ARITHMETIC, max_cost=0.001)
    # Should fallback to cheapest
    assert model.cost_per_1k <= 0.01


# ─── Route with Explanation ───────────────────────────────────────────────────

def test_route_with_explanation():
    result = route_with_explanation("compute 2+2")
    assert "domain_detected" in result
    assert "model" in result
    assert "temperature" in result
    assert "reason" in result
    assert result["domain_detected"] == "arithmetic"


def test_route_explanation_unknown():
    result = route_with_explanation("hello world")
    assert "model" in result
    assert result["domain_detected"] == "general"


# ─── CriticalAngle ────────────────────────────────────────────────────────────

def test_critical_angle_dataclass():
    ca = CriticalAngle("test", "addition", 10.0, "HIGH", "test-src", "2026-01-01")
    assert ca.model == "test"
    assert ca.value == 10.0
    assert ca.confidence == "HIGH"


# ─── CompletionResult ─────────────────────────────────────────────────────────

def test_completion_result_text():
    cr = CompletionResult(content="hello", reasoning_content="thinking...")
    assert cr.text == "hello"


def test_completion_result_fallback():
    cr = CompletionResult(content="", reasoning_content="thinking...")
    assert cr.text == "thinking..."


# ─── Cost Savings ─────────────────────────────────────────────────────────────

def test_calculate_savings():
    queries = [
        {"domain": "arithmetic", "model": "seed-2.0-mini", "tokens": 100, "cost": 0.05},
        {"domain": "reasoning", "model": "gemini-lite", "tokens": 200, "cost": 0.002},
    ]
    result = calculate_savings(queries)
    assert result["total_queries"] == 2
    assert result["gpt4_cost"] > result["fleet_cost"]
    assert result["savings"] > 0
    assert result["savings_pct"] > 50


def test_calculate_savings_empty():
    result = calculate_savings([])
    assert result["total_queries"] == 0
    assert result["savings"] == 0


# ─── Import Tests ─────────────────────────────────────────────────────────────

def test_import_init():
    import fleet_router
    assert hasattr(fleet_router, "__version__")
