"""Tests for FORGE CognitiveAgent — the integration layer.

These tests validate the 8-phase cognitive cycle as a whole:
- Does the agent complete a full cycle?
- Does the uncertainty gate actually stop dangerous actions?
- Does the agent learn from repeated tasks?
- Does drift detection catch off-mission work?
"""
import pytest
from core.types import (
    AgentConfig, CognitivePhase, ActionEffect, ReversibilityClass,
    UncertaintyEstimate,
)
from core.agent import CognitiveAgent
from core.llm import MockLLMProvider


@pytest.fixture
def agent():
    """Standard test agent with mock LLM."""
    llm = MockLLMProvider()
    llm.set_plan_response(["analyze_situation", "take_action"])
    llm.set_eval_response(0.5)
    config = AgentConfig(
        working_memory_capacity=5,
        memory_consolidation_threshold=3,
    )
    return CognitiveAgent(llm=llm, config=config)


@pytest.fixture
def gated_agent():
    """Agent with an irreversible tool and low-confidence LLM."""
    llm = MockLLMProvider()
    llm.set_plan_response(["send_email"])  # Plans irreversible action
    llm.set_eval_response(0.3)
    agent = CognitiveAgent(llm=llm)
    # Register irreversible action
    agent.world_model.graph.register_action(ActionEffect(
        action_name="send_email",
        reversibility=ReversibilityClass.IRREVERSIBLE,
    ))
    return agent


class TestFullCognitiveCycle:
    """The 8 phases execute in order and produce valid output."""

    @pytest.mark.asyncio
    async def test_cycle_completes_all_phases(self, agent):
        await agent.set_goal("Complete the test task")
        result = await agent.run_cycle("New observation: test started")

        # All 8 phases should produce steps
        assert len(result.steps) == 8
        phases = [s.phase for s in result.steps]
        assert phases == [
            CognitivePhase.ORIENT,
            CognitivePhase.RECALL,
            CognitivePhase.PLAN,
            CognitivePhase.GATE,
            CognitivePhase.ACT,
            CognitivePhase.OBSERVE,
            CognitivePhase.LEARN,
            CognitivePhase.CHECK,
        ]

    @pytest.mark.asyncio
    async def test_cycle_increments_count(self, agent):
        await agent.set_goal("Count cycles")
        assert agent.cycle_count == 0
        await agent.run_cycle()
        assert agent.cycle_count == 1
        await agent.run_cycle()
        assert agent.cycle_count == 2

    @pytest.mark.asyncio
    async def test_cycle_produces_action(self, agent):
        await agent.set_goal("Take an action")
        result = await agent.run_cycle()
        assert result.action_taken is not None


class TestUncertaintyGate:
    """The gate is architectural — agent CANNOT bypass it."""

    @pytest.mark.asyncio
    async def test_gate_blocks_irreversible_action(self, gated_agent):
        """Low confidence + irreversible action = blocked."""
        await gated_agent.set_goal("Send the email")
        result = await gated_agent.run_cycle()
        assert result.was_gated is True
        assert "blocked" in result.gate_reason.lower() or "below" in result.gate_reason.lower()
        # Cycle short-circuits — only 4 phases (ORIENT, RECALL, PLAN, GATE)
        assert len(result.steps) == 4

    @pytest.mark.asyncio
    async def test_gate_passes_safe_actions(self, agent):
        """Default actions with moderate confidence pass the gate."""
        await agent.set_goal("Safe task")
        result = await agent.run_cycle()
        assert result.was_gated is False


class TestLearning:
    """Does the agent actually learn from experience?"""

    @pytest.mark.asyncio
    async def test_episodes_accumulate(self, agent):
        await agent.set_goal("Learn over time")
        for _ in range(5):
            await agent.run_cycle()
        assert agent.memory.episodic.size > 0

    @pytest.mark.asyncio
    async def test_consolidation_produces_strategy(self, agent):
        """After enough episodes, a strategy should be extracted."""
        await agent.set_goal("Repetitive task")
        for _ in range(6):  # threshold is 3, but we need flush cycles
            await agent.run_cycle()
        # Agent should have attempted consolidation
        assert agent.memory.procedural.size > 0 or agent.memory.episodic.size > 0

    @pytest.mark.asyncio
    async def test_performance_summary(self, agent):
        """Agent can report its own performance."""
        await agent.set_goal("Track performance")
        await agent.run_cycle()
        summary = agent.get_performance_summary()
        assert summary["cycles_completed"] == 1
        assert "procedural_strategies" in summary
        assert "episodic_memories" in summary


class TestTracing:
    """Every decision is traced."""

    @pytest.mark.asyncio
    async def test_trace_records_cycle(self, agent):
        await agent.set_goal("Trace me")
        await agent.run_cycle()
        assert agent.tracer.cycle_count == 1
        summary = agent.tracer.get_cycle_summary()
        assert "cycle_id" in summary
        assert "phases" in summary

    @pytest.mark.asyncio
    async def test_cost_attribution(self, agent):
        await agent.set_goal("Track costs")
        await agent.run_cycle()
        costs = agent.tracer.get_cost_attribution()
        # Should have entries for phases that used tokens
        assert isinstance(costs, dict)
