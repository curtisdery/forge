"""
FORGE Cognitive Agent — The integration layer.

Eight-phase cognitive cycle: ORIENT → RECALL → PLAN → GATE → ACT → OBSERVE → LEARN → CHECK

This is composition, not monolith. The agent wires together:
- Memory system (three tiers of learning)
- Goal stack (protected objectives)
- World model (causal understanding + simulation)
- LLM provider (reasoning engine)
- Tracer (observability)

The agent is the conductor. The subsystems are the orchestra.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .types import (
    CognitivePhase, CognitiveStep, UncertaintyEstimate,
    WorldState, AgentConfig, ActionEffect, ReversibilityClass,
    GoalStatus,
)
from .memory import MemorySystem
from .goal_stack import GoalStack
from .world_model import WorldModel, CausalGraph
from .llm import LLMProvider, MockLLMProvider
from .tracer import CognitiveTracer


@dataclass
class CycleResult:
    """Output of one cognitive cycle."""
    action_taken: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    steps: list[CognitiveStep] = field(default_factory=list)
    was_gated: bool = False
    gate_reason: str = ""
    drift_detected: bool = False
    goal_completed: bool = False
    tokens_used: int = 0


class CognitiveAgent:
    """A cognitive agent with genuine architectural primitives for agency.

    Not a chatbot with tools. An engineering system with:
    - World understanding (causal graphs, simulation)
    - Self-knowledge (empirical capability tracking)
    - Memory that learns (working → episodic → procedural)
    - Goals that persist (outside the context window)
    - Uncertainty that gates (irreversible actions require confidence)
    - Every decision traced (for debugging and learning analysis)
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        config: AgentConfig | None = None,
        tools: dict[str, Any] | None = None,
    ):
        self.config = config or AgentConfig()
        self.llm = llm or MockLLMProvider()
        self.memory = MemorySystem(self.config)
        self.goals = GoalStack(self.config)
        self.world_model = WorldModel(CausalGraph(), self.config)
        self.tracer = CognitiveTracer(enabled=self.config.trace_enabled)

        # Tools the agent can use
        self._tools: dict[str, Any] = tools or {}

        # Running state
        self._cycle_count: int = 0
        self._recent_alignments: list[float] = []
        self._capability_profiles: dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    async def run_cycle(self, observation: str | None = None) -> CycleResult:
        """Execute one full cognitive cycle.

        ORIENT → RECALL → PLAN → GATE → ACT → OBSERVE → LEARN → CHECK

        Each phase produces a CognitiveStep. The cycle can short-circuit
        at GATE (uncertainty too high) or CHECK (drift detected).
        """
        self._cycle_count += 1
        result = CycleResult()

        with self.tracer.cycle(f"cycle_{self._cycle_count}") as cycle_span:

            # ─── ORIENT: Perceive and contextualize ─────────────
            with self.tracer.phase(CognitivePhase.ORIENT) as orient_span:
                orient_step = await self._orient(observation)
                result.steps.append(orient_step)
                self.tracer.record_step(orient_step)

            # ─── RECALL: Retrieve relevant memories ─────────────
            with self.tracer.phase(CognitivePhase.RECALL) as recall_span:
                recall_step = await self._recall(orient_step)
                result.steps.append(recall_step)
                self.tracer.record_step(recall_step)

            # ─── PLAN: Generate candidate actions ───────────────
            with self.tracer.phase(CognitivePhase.PLAN) as plan_span:
                plan_step = await self._plan(orient_step, recall_step)
                result.steps.append(plan_step)
                result.tokens_used += plan_step.token_cost
                self.tracer.record_step(plan_step)

            # ─── GATE: Uncertainty check ────────────────────────
            with self.tracer.phase(CognitivePhase.GATE) as gate_span:
                gate_step = await self._gate(plan_step)
                result.steps.append(gate_step)
                self.tracer.record_step(gate_step)

                if gate_step.success is False:
                    # Gated — cannot proceed
                    result.was_gated = True
                    result.gate_reason = gate_step.reasoning
                    gate_span.status = "gated"
                    return result

            # ─── ACT: Execute the chosen action ─────────────────
            with self.tracer.phase(CognitivePhase.ACT) as act_span:
                act_step = await self._act(plan_step)
                result.steps.append(act_step)
                result.action_taken = act_step.action_taken
                result.result = act_step.output_state
                self.tracer.record_step(act_step)

            # ─── OBSERVE: Capture and compare to prediction ─────
            with self.tracer.phase(CognitivePhase.OBSERVE) as obs_span:
                observe_step = await self._observe(plan_step, act_step)
                result.steps.append(observe_step)
                self.tracer.record_step(observe_step)

            # ─── LEARN: Update memories ─────────────────────────
            with self.tracer.phase(CognitivePhase.LEARN) as learn_span:
                learn_step = await self._learn(act_step, observe_step)
                result.steps.append(learn_step)
                result.tokens_used += learn_step.token_cost
                self.tracer.record_step(learn_step)

            # ─── CHECK: Goal alignment ──────────────────────────
            with self.tracer.phase(CognitivePhase.CHECK) as check_span:
                check_step = await self._check(act_step)
                result.steps.append(check_step)
                result.drift_detected = check_step.output_state.get("drift", False)
                result.goal_completed = check_step.output_state.get("goal_completed", False)
                self.tracer.record_step(check_step)

        return result

    async def set_goal(self, description: str) -> None:
        """Set the agent's primary goal."""
        self.goals.set_primary(description)

    async def add_tool(self, name: str, executor: Any,
                       effect: ActionEffect | None = None) -> None:
        """Register a tool the agent can use."""
        self._tools[name] = executor
        if effect:
            self.world_model.graph.register_action(effect)

    # ═══════════════════════════════════════════════════════════
    # COGNITIVE PHASES — each returns a CognitiveStep
    # ═══════════════════════════════════════════════════════════

    async def _orient(self, observation: str | None) -> CognitiveStep:
        """Phase 1: Perceive and contextualize the current situation."""
        start = time.time()

        # Get current focus goal
        focus = self.goals.get_focus()
        goal_desc = focus.description if focus else "No active goal"

        # Build situation context
        context = {
            "observation": observation or "No new observation",
            "active_goal": goal_desc,
            "working_memory_items": self.memory.working.size,
            "available_tools": list(self._tools.keys()),
            "cycle_number": self._cycle_count,
        }

        step = CognitiveStep(
            phase=CognitivePhase.ORIENT,
            input_state={"observation": observation},
            output_state=context,
            reasoning=f"Cycle {self._cycle_count}: Oriented on goal '{goal_desc}'",
            goal_id=focus.id if focus else None,
            duration_ms=(time.time() - start) * 1000,
        )
        return step

    async def _recall(self, orient_step: CognitiveStep) -> CognitiveStep:
        """Phase 2: Retrieve relevant memories and learned strategies."""
        start = time.time()

        # Build recall tags from context (clean, alpha-only)
        tags = []
        goal = self.goals.get_focus()
        if goal:
            words = [w.strip(":.!?,") for w in goal.description.lower().split()[:5]]
            tags.extend(w for w in words if w.isalpha())

        # Retrieve across all tiers
        memories = self.memory.retrieve_all(tags=tags, top_k=3)

        # Extract applicable strategies
        strategies = [
            m.content for m in memories.get("procedural", [])
            if isinstance(m.content, dict)
        ]

        step = CognitiveStep(
            phase=CognitivePhase.RECALL,
            input_state={"tags": tags},
            output_state={
                "working_items": len(memories.get("working", [])),
                "episodic_items": len(memories.get("episodic", [])),
                "strategies": strategies,
            },
            reasoning=f"Retrieved {sum(len(v) for v in memories.values())} memories, {len(strategies)} applicable strategies",
            goal_id=goal.id if goal else None,
            duration_ms=(time.time() - start) * 1000,
        )
        return step

    async def _plan(self, orient_step: CognitiveStep, recall_step: CognitiveStep) -> CognitiveStep:
        """Phase 3: Generate candidate action sequences using LLM."""
        start = time.time()

        goal = self.goals.get_focus()
        goal_desc = goal.description if goal else "Complete the current task"
        strategies = recall_step.output_state.get("strategies", [])

        context = {
            "situation": orient_step.output_state,
            "learned_strategies": strategies,
            "available_tools": list(self._tools.keys()),
        }

        # Ask LLM to plan
        actions = await self.llm.plan(goal_desc, context)
        elapsed = time.time() - start

        # Estimate confidence based on strategy availability
        confidence = 0.6
        if strategies:
            confidence = min(0.9, confidence + 0.1 * len(strategies))

        step = CognitiveStep(
            phase=CognitivePhase.PLAN,
            input_state=context,
            output_state={
                "planned_actions": actions,
                "confidence": confidence,
                "used_strategies": len(strategies),
            },
            action_taken=actions[0] if actions else None,
            reasoning=f"Planned {len(actions)} actions (confidence: {confidence:.2f})",
            uncertainty=UncertaintyEstimate(
                confidence=confidence,
                evidence_weight=0.3 + (0.1 * len(strategies)),
            ),
            goal_id=goal.id if goal else None,
            duration_ms=elapsed * 1000,
            token_cost=100,  # Estimated
        )
        return step

    async def _gate(self, plan_step: CognitiveStep) -> CognitiveStep:
        """Phase 4: Uncertainty gate — can we proceed safely?

        This is architectural, not a suggestion. If confidence is too low
        for the action's reversibility class, the agent CANNOT proceed.
        """
        start = time.time()

        action = plan_step.action_taken
        uncertainty = plan_step.uncertainty or UncertaintyEstimate(confidence=0.5)

        # Check if action should be gated
        should_block = False
        reason = ""

        if action:
            should_block = self.world_model.should_gate(action, uncertainty)
            if should_block:
                rev = self.world_model.graph.get_reversibility(action)
                reason = (
                    f"Action '{action}' blocked: confidence {uncertainty.confidence:.2f} "
                    f"is below threshold for {rev.name} action"
                )

        step = CognitiveStep(
            phase=CognitivePhase.GATE,
            input_state={
                "action": action,
                "confidence": uncertainty.confidence,
            },
            output_state={"gated": should_block, "reason": reason},
            reasoning=reason if should_block else f"Gate passed: confidence {uncertainty.confidence:.2f} sufficient",
            uncertainty=uncertainty,
            success=not should_block,  # False = gated (blocked)
            duration_ms=(time.time() - start) * 1000,
        )
        return step

    async def _act(self, plan_step: CognitiveStep) -> CognitiveStep:
        """Phase 5: Execute the chosen action."""
        start = time.time()

        action = plan_step.action_taken
        result = {}

        if action and action in self._tools:
            try:
                tool = self._tools[action]
                if callable(tool):
                    result = await tool() if _is_coroutine(tool) else tool()
                    if not isinstance(result, dict):
                        result = {"output": result}
                result["success"] = True
            except Exception as e:
                result = {"success": False, "error": str(e)}
        elif action:
            # Action not found in tools — simulated execution
            result = {"success": True, "simulated": True, "action": action}
        else:
            result = {"success": False, "error": "No action planned"}

        # Store action result in working memory
        self.memory.working.store(
            content={"action": action, "result": result},
            salience=0.7,
            tags=["action_result", action or "none"],
        )

        step = CognitiveStep(
            phase=CognitivePhase.ACT,
            input_state={"planned_action": action},
            output_state=result,
            action_taken=action,
            reasoning=f"Executed '{action}': {'success' if result.get('success') else 'failed'}",
            success=result.get("success", False),
            duration_ms=(time.time() - start) * 1000,
        )
        return step

    async def _observe(self, plan_step: CognitiveStep, act_step: CognitiveStep) -> CognitiveStep:
        """Phase 6: Capture result and compare to prediction."""
        start = time.time()

        predicted = plan_step.output_state.get("planned_actions", [])
        actual = act_step.output_state
        action_succeeded = actual.get("success", False)

        # Prediction accuracy — did reality match our model?
        prediction_correct = action_succeeded  # Simplified; real version compares state

        step = CognitiveStep(
            phase=CognitivePhase.OBSERVE,
            input_state={
                "predicted": predicted,
                "actual": actual,
            },
            output_state={
                "prediction_correct": prediction_correct,
                "action_succeeded": action_succeeded,
                "surprise": not prediction_correct,
            },
            reasoning=f"Observation: prediction {'correct' if prediction_correct else 'incorrect'}, action {'succeeded' if action_succeeded else 'failed'}",
            success=action_succeeded,
            duration_ms=(time.time() - start) * 1000,
        )
        return step

    async def _learn(self, act_step: CognitiveStep, observe_step: CognitiveStep) -> CognitiveStep:
        """Phase 7: Update memories based on outcome.

        This is where learning happens:
        1. Store episode (what happened)
        2. Reinforce/penalize used strategies
        3. Attempt consolidation if threshold met
        """
        start = time.time()
        token_cost = 0

        succeeded = observe_step.output_state.get("action_succeeded", False)
        goal = self.goals.get_focus()

        # Build clean tags (alphanumeric only)
        tags = ["episode", act_step.action_taken or "none"]
        if goal:
            words = [w.strip(":.!?,") for w in goal.description.lower().split()[:5]]
            tags.extend(w for w in words if w.isalpha())

        # Store episode in working memory, then flush to episodic
        # flush_working_to_episodic increments the consolidation counter
        episode = {
            "action": act_step.action_taken,
            "result": act_step.output_state,
            "succeeded": succeeded,
            "goal": goal.description if goal else None,
            "cycle": self._cycle_count,
        }
        self.memory.working.store(
            content=episode,
            salience=0.6 if succeeded else 0.8,
            tags=tags,
        )
        self.memory.flush_working_to_episodic(task_tags=tags)

        # Reinforce applicable strategies
        strategies_used = self.memory.procedural.retrieve(tags=tags, top_k=2)
        for s in strategies_used:
            self.memory.procedural.reinforce(s.id, succeeded)

        # Attempt consolidation if enough episodes
        consolidated = None
        if self.memory.should_consolidate():
            recent = self.memory.episodic.get_recent(n=self.memory._consolidation_threshold)
            episodes_data = [m.content for m in recent if isinstance(m.content, dict)]
            if episodes_data:
                extracted = await self.llm.extract_strategy(episodes_data)
                consolidated = self.memory.consolidate(
                    extracted_strategy=extracted,
                    tags=tags,
                )
                token_cost = 200  # Estimated cost of strategy extraction

        step = CognitiveStep(
            phase=CognitivePhase.LEARN,
            input_state={"episode": episode},
            output_state={
                "episode_stored": True,
                "strategies_reinforced": len(strategies_used),
                "consolidated": consolidated is not None,
            },
            reasoning=f"Stored episode, reinforced {len(strategies_used)} strategies" +
                      (", consolidated new strategy" if consolidated else ""),
            success=True,
            duration_ms=(time.time() - start) * 1000,
            token_cost=token_cost,
        )
        return step

    async def _check(self, act_step: CognitiveStep) -> CognitiveStep:
        """Phase 8: Verify goal alignment — did we drift?"""
        start = time.time()

        goal = self.goals.get_focus()
        goal_completed = False
        drift = False

        if goal and act_step.action_taken:
            # Check alignment
            alignment = self.goals.check_alignment(act_step.action_taken)
            self._recent_alignments.append(alignment)

            # Keep last 10 alignment scores
            if len(self._recent_alignments) > 10:
                self._recent_alignments = self._recent_alignments[-10:]

            drift = self.goals.detect_drift(self._recent_alignments)

            # Check if goal is completed (via LLM evaluation or simple check)
            if act_step.output_state.get("success"):
                score = await self.llm.evaluate(act_step.output_state, goal.description)
                if score >= 0.9:
                    self.goals.complete(goal.id)
                    goal_completed = True

        step = CognitiveStep(
            phase=CognitivePhase.CHECK,
            input_state={"action": act_step.action_taken},
            output_state={
                "drift": drift,
                "goal_completed": goal_completed,
                "alignment_history": self._recent_alignments[-5:],
            },
            reasoning=f"Alignment check: {'DRIFT DETECTED' if drift else 'on track'}" +
                      (", goal completed!" if goal_completed else ""),
            success=not drift,
            duration_ms=(time.time() - start) * 1000,
        )
        return step

    # ═══════════════════════════════════════════════════════════
    # INTROSPECTION
    # ═══════════════════════════════════════════════════════════

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def get_performance_summary(self) -> dict[str, Any]:
        """How is the agent doing? Empirical self-knowledge."""
        return {
            "cycles_completed": self._cycle_count,
            "procedural_strategies": self.memory.procedural.size,
            "episodic_memories": self.memory.episodic.size,
            "strongest_strategies": [
                {"content": s.content, "strength": s.strength}
                for s in self.memory.procedural.get_strongest(3)
            ],
            "recent_alignment": (
                sum(self._recent_alignments) / len(self._recent_alignments)
                if self._recent_alignments else None
            ),
            "trace_summary": self.tracer.get_cost_attribution(),
        }


def _is_coroutine(fn) -> bool:
    """Check if a function is a coroutine."""
    import inspect
    return inspect.iscoroutinefunction(fn)
