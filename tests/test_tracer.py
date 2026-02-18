"""Tests for FORGE cognitive tracer — observability for agent decisions."""
import pytest
from core.types import CognitivePhase, CognitiveStep
from core.tracer import CognitiveTracer, Span


class TestSpan:
    def test_duration_calculation(self):
        span = Span(name="test")
        import time
        time.sleep(0.01)
        span.end()
        assert span.duration_ms > 0

    def test_to_dict(self):
        span = Span(name="test_span", phase=CognitivePhase.ORIENT)
        span.add_event("observed", {"entities": 3})
        span.end()
        d = span.to_dict()
        assert d["name"] == "test_span"
        assert d["phase"] == "ORIENT"
        assert len(d["events"]) == 1


class TestCognitiveTracer:
    def test_cycle_context_manager(self):
        tracer = CognitiveTracer()
        with tracer.cycle("test_cycle") as span:
            assert span.name == "test_cycle"
        assert tracer.cycle_count == 1

    def test_phase_nesting(self):
        tracer = CognitiveTracer()
        with tracer.cycle("cycle") as cycle:
            with tracer.phase(CognitivePhase.ORIENT) as orient:
                orient.token_cost = 50
            with tracer.phase(CognitivePhase.PLAN) as plan:
                plan.token_cost = 200
        summary = tracer.get_cycle_summary()
        assert summary["total_token_cost"] == 250
        assert "ORIENT" in summary["phases"]
        assert "PLAN" in summary["phases"]

    def test_record_step(self):
        tracer = CognitiveTracer()
        step = CognitiveStep(
            phase=CognitivePhase.ACT,
            action_taken="search",
            token_cost=100,
            success=True,
        )
        with tracer.cycle("cycle"):
            span = tracer.record_step(step)
        assert span.token_cost == 100

    def test_learning_curve(self):
        tracer = CognitiveTracer()
        for i in range(3):
            with tracer.cycle(f"cycle_{i}"):
                pass
        curve = tracer.get_learning_curve()
        assert len(curve) == 3
        assert curve[0]["iteration"] == 0

    def test_cost_attribution(self):
        tracer = CognitiveTracer()
        with tracer.cycle("cycle"):
            with tracer.phase(CognitivePhase.PLAN) as p:
                p.token_cost = 300
            with tracer.phase(CognitivePhase.LEARN) as l:
                l.token_cost = 200
        costs = tracer.get_cost_attribution()
        assert costs.get("PLAN", 0) == 300
        assert costs.get("LEARN", 0) == 200
