"""Tests for FORGE world model — causal understanding and mental simulation.

Cognitive behavior being tested:
- Agent understands what actions DO, not just how to call them
- Simulation predicts futures before committing
- Irreversible actions are gated by uncertainty
"""
import pytest
from core.types import (
    ActionEffect, ReversibilityClass, UncertaintyEstimate, WorldState,
)
from core.world_model import WorldModel, CausalGraph, CausalEdge, SimulationBranch


class TestCausalGraph:
    """The agent's model of cause and effect."""

    def test_register_and_retrieve_action(self):
        cg = CausalGraph()
        effect = ActionEffect(
            action_name="send_email",
            reversibility=ReversibilityClass.IRREVERSIBLE,
            affected_entities=("recipient",),
            failure_modes=("bounce", "spam_filter"),
        )
        cg.register_action(effect)
        retrieved = cg.get_effects("send_email")
        assert retrieved is not None
        assert retrieved.reversibility == ReversibilityClass.IRREVERSIBLE

    def test_predict_with_matching_edge(self):
        """Given an action and state, predict the next state."""
        cg = CausalGraph()
        cg.add_edge(CausalEdge(
            action="open_file",
            preconditions={"file": {"exists": True}},
            effects={"file": {"status": "open"}},
            probability=0.95,
        ))
        state = WorldState()
        state.update_entity("file", {"exists": True, "status": "closed"})

        outcomes = cg.predict("open_file", state)
        assert len(outcomes) == 1
        next_state, prob = outcomes[0]
        assert next_state.get_entity("file")["status"] == "open"
        assert prob == 0.95

    def test_predict_unknown_action(self):
        """Unknown action-state pair returns low confidence."""
        cg = CausalGraph()
        state = WorldState()
        outcomes = cg.predict("unknown_action", state)
        assert len(outcomes) == 1
        _, prob = outcomes[0]
        assert prob == 0.5  # Low confidence

    def test_available_actions_from_state(self):
        cg = CausalGraph()
        cg.add_edge(CausalEdge(
            action="save",
            preconditions={"doc": {"modified": True}},
            effects={"doc": {"modified": False}},
        ))
        cg.add_edge(CausalEdge(
            action="delete",
            preconditions={"doc": {"exists": True}},
            effects={"doc": {"exists": False}},
        ))
        state = WorldState()
        state.update_entity("doc", {"exists": True, "modified": True})
        available = cg.get_available_actions(state)
        assert "save" in available
        assert "delete" in available


class TestWorldModel:
    """Mental simulation — think before you act."""

    def _build_model(self) -> WorldModel:
        cg = CausalGraph()
        cg.register_action(ActionEffect(
            action_name="safe_read",
            reversibility=ReversibilityClass.FULLY_REVERSIBLE,
            cost=0.01,
        ))
        cg.register_action(ActionEffect(
            action_name="risky_delete",
            reversibility=ReversibilityClass.IRREVERSIBLE,
            cost=0.5,
        ))
        cg.add_edge(CausalEdge(
            action="safe_read",
            preconditions={},
            effects={"data": {"loaded": True}},
            probability=0.99,
        ))
        cg.add_edge(CausalEdge(
            action="risky_delete",
            preconditions={},
            effects={"data": {"deleted": True}},
            probability=0.95,
        ))
        return WorldModel(cg)

    def test_simulate_action_sequence(self):
        model = self._build_model()
        branch = model.simulate(["safe_read", "risky_delete"])
        assert len(branch.actions) == 2
        assert branch.cumulative_probability > 0
        assert branch.cumulative_cost > 0

    def test_branch_and_bound_ranks_by_utility(self):
        """Multiple plans simulated and ranked — best one first."""
        model = self._build_model()
        candidates = [
            ["safe_read"],          # Low cost, high probability
            ["risky_delete"],       # High cost
            ["safe_read", "risky_delete"],  # Mixed
        ]
        ranked = model.branch_and_bound(candidates)
        # safe_read only should rank highest (lowest cost, high prob)
        assert ranked[0].actions == ["safe_read"]

    def test_gate_blocks_irreversible_with_low_confidence(self):
        """Uncertainty gate BLOCKS irreversible actions when confidence is low."""
        model = self._build_model()
        low_conf = UncertaintyEstimate(confidence=0.4)
        assert model.should_gate("risky_delete", low_conf) is True

    def test_gate_passes_reversible_with_low_confidence(self):
        """Reversible actions pass even with low confidence."""
        model = self._build_model()
        low_conf = UncertaintyEstimate(confidence=0.4)
        assert model.should_gate("safe_read", low_conf) is False

    def test_safest_action(self):
        """Agent can pick the safest option from candidates."""
        model = self._build_model()
        safest = model.get_safest_action(["safe_read", "risky_delete"])
        assert safest == "safe_read"
