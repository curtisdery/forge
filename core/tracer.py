"""
FORGE Cognitive Tracer — OpenTelemetry-style tracing for cognitive steps.

Every decision the agent makes produces a trace. This enables:
- Debugging: Why did the agent do X instead of Y?
- Cost attribution: Where are the tokens being spent?
- Performance analysis: Which phases are bottlenecks?
- Learning analysis: Is the agent getting better over time?

Traces form trees: a cognitive cycle is the root span, each phase
is a child span, and nested reasoning creates deeper children.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .types import CognitivePhase, CognitiveStep


@dataclass
class Span:
    """A single span in the trace tree — one unit of cognitive work."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    parent_id: str | None = None
    name: str = ""
    phase: CognitivePhase | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    token_cost: int = 0
    status: str = "ok"  # ok, error, gated

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def end(self, status: str = "ok") -> None:
        self.end_time = time.time()
        self.status = status

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "name": self.name,
            "phase": self.phase.name if self.phase else None,
            "duration_ms": round(self.duration_ms, 2),
            "token_cost": self.token_cost,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "children": self.children,
        }


class CognitiveTracer:
    """Trace tree builder for cognitive cycles.

    Usage:
        tracer = CognitiveTracer()
        with tracer.cycle("process_task") as cycle:
            with tracer.phase(CognitivePhase.ORIENT) as orient:
                orient.add_event("observed_state", {"entities": 3})
            with tracer.phase(CognitivePhase.PLAN) as plan:
                plan.token_cost = 500
        summary = tracer.get_cycle_summary()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._spans: dict[str, Span] = {}
        self._active_cycle: Span | None = None
        self._active_phase: Span | None = None
        self._completed_cycles: list[Span] = []

    def cycle(self, name: str) -> _CycleContext:
        """Start a new cognitive cycle trace."""
        return _CycleContext(self, name)

    def phase(self, phase: CognitivePhase) -> _PhaseContext:
        """Start a new phase span within the active cycle."""
        return _PhaseContext(self, phase)

    def record_step(self, step: CognitiveStep) -> Span:
        """Record a CognitiveStep as a span."""
        span = Span(
            name=f"step_{step.phase.name.lower()}",
            phase=step.phase,
            attributes={
                "action": step.action_taken,
                "reasoning": step.reasoning[:200] if step.reasoning else "",
                "success": step.success,
                "goal_id": step.goal_id,
            },
            token_cost=step.token_cost,
            status="ok" if step.success is not False else "error",
        )
        if step.was_gated:
            span.status = "gated"
        span.end_time = span.start_time + (step.duration_ms / 1000)
        self._spans[span.id] = span

        if self._active_phase:
            self._active_phase.children.append(span.id)
            span.parent_id = self._active_phase.id
        elif self._active_cycle:
            self._active_cycle.children.append(span.id)
            span.parent_id = self._active_cycle.id

        return span

    def get_cycle_summary(self, cycle_id: str | None = None) -> dict[str, Any]:
        """Summarize a cognitive cycle: phases, durations, costs, decisions."""
        if cycle_id:
            cycle = self._spans.get(cycle_id)
        elif self._completed_cycles:
            cycle = self._completed_cycles[-1]
        elif self._active_cycle:
            cycle = self._active_cycle
        else:
            return {"error": "No cycles recorded"}

        if cycle is None:
            return {"error": f"Cycle {cycle_id} not found"}

        phases = {}
        total_tokens = 0
        gated_count = 0

        for child_id in cycle.children:
            child = self._spans.get(child_id)
            if child:
                phase_name = child.phase.name if child.phase else child.name
                phases[phase_name] = {
                    "duration_ms": round(child.duration_ms, 2),
                    "token_cost": child.token_cost,
                    "status": child.status,
                    "events": len(child.events),
                }
                total_tokens += child.token_cost
                if child.status == "gated":
                    gated_count += 1

        return {
            "cycle_id": cycle.id,
            "name": cycle.name,
            "duration_ms": round(cycle.duration_ms, 2),
            "total_token_cost": total_tokens,
            "phases": phases,
            "gated_actions": gated_count,
            "status": cycle.status,
        }

    def get_all_summaries(self) -> list[dict[str, Any]]:
        """Summaries of all completed cycles."""
        return [self.get_cycle_summary(c.id) for c in self._completed_cycles]

    def get_cost_attribution(self) -> dict[str, int]:
        """Where are the tokens being spent? Breakdown by phase."""
        costs: dict[str, int] = {}
        for span in self._spans.values():
            if span.phase:
                key = span.phase.name
                costs[key] = costs.get(key, 0) + span.token_cost
        return costs

    def get_learning_curve(self) -> list[dict[str, Any]]:
        """Track performance over time for learning analysis."""
        curve = []
        for i, cycle in enumerate(self._completed_cycles):
            summary = self.get_cycle_summary(cycle.id)
            curve.append({
                "iteration": i,
                "duration_ms": summary["duration_ms"],
                "token_cost": summary["total_token_cost"],
                "gated": summary["gated_actions"],
                "status": summary["status"],
            })
        return curve

    @property
    def cycle_count(self) -> int:
        return len(self._completed_cycles)

    # ─── Internal state management ─────────────────────────────

    def _start_cycle(self, name: str) -> Span:
        span = Span(name=name)
        self._spans[span.id] = span
        self._active_cycle = span
        return span

    def _end_cycle(self, status: str = "ok") -> None:
        if self._active_cycle:
            self._active_cycle.end(status)
            self._completed_cycles.append(self._active_cycle)
            self._active_cycle = None

    def _start_phase(self, phase: CognitivePhase) -> Span:
        span = Span(
            name=phase.name.lower(),
            phase=phase,
            parent_id=self._active_cycle.id if self._active_cycle else None,
        )
        self._spans[span.id] = span
        if self._active_cycle:
            self._active_cycle.children.append(span.id)
        self._active_phase = span
        return span

    def _end_phase(self, status: str = "ok") -> None:
        if self._active_phase:
            self._active_phase.end(status)
            self._active_phase = None


class _CycleContext:
    """Context manager for cognitive cycle tracing."""
    def __init__(self, tracer: CognitiveTracer, name: str):
        self._tracer = tracer
        self._name = name
        self._span: Span | None = None

    def __enter__(self) -> Span:
        self._span = self._tracer._start_cycle(self._name)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "error" if exc_type else "ok"
        self._tracer._end_cycle(status)


class _PhaseContext:
    """Context manager for cognitive phase tracing."""
    def __init__(self, tracer: CognitiveTracer, phase: CognitivePhase):
        self._tracer = tracer
        self._phase = phase
        self._span: Span | None = None

    def __enter__(self) -> Span:
        self._span = self._tracer._start_phase(self._phase)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "error" if exc_type else "ok"
        self._tracer._end_phase(status)
