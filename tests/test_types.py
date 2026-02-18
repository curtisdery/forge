"""Tests for FORGE atomic primitives — types are the foundation."""
import pytest
from core.types import (
    UncertaintyEstimate, ReversibilityClass, CapabilityProfile,
    ActionEffect, Goal, GoalStatus, GoalPriority, Memory, MemoryTier,
    CognitiveStep, CognitivePhase, WorldState, AgentConfig,
)


class TestUncertaintyEstimate:
    """The uncertainty gate is architectural, not a suggestion."""

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            UncertaintyEstimate(confidence=1.5)
        with pytest.raises(ValueError):
            UncertaintyEstimate(confidence=-0.1)

    def test_should_gate_irreversible_low_confidence(self):
        """Irreversible actions with low confidence MUST be blocked."""
        est = UncertaintyEstimate(confidence=0.5)
        assert est.should_gate(ReversibilityClass.IRREVERSIBLE) is True

    def test_should_not_gate_reversible_low_confidence(self):
        """Reversible actions pass even with moderate confidence."""
        est = UncertaintyEstimate(confidence=0.4)
        assert est.should_gate(ReversibilityClass.FULLY_REVERSIBLE) is False

    def test_should_not_gate_irreversible_high_confidence(self):
        """Irreversible actions pass with high confidence."""
        est = UncertaintyEstimate(confidence=0.9)
        assert est.should_gate(ReversibilityClass.IRREVERSIBLE) is False

    def test_reliability_requires_all_three(self):
        """Reliable = high confidence + low cal error + real evidence."""
        reliable = UncertaintyEstimate(confidence=0.8, calibration_error=0.05, evidence_weight=0.5)
        assert reliable.is_reliable is True

        no_evidence = UncertaintyEstimate(confidence=0.8, calibration_error=0.05, evidence_weight=0.1)
        assert no_evidence.is_reliable is False


class TestCapabilityProfile:
    """Self-knowledge must come from data, not self-assessment."""

    def test_update_tracks_success_rate(self):
        profile = CapabilityProfile(domain="math", success_rate=0.5, attempts=10)
        updated = profile.update(succeeded=True, stated_confidence=0.7)
        assert updated.success_rate > 0.5
        assert updated.attempts == 11

    def test_calibration_error_converges(self):
        """If agent says 0.9 confident but fails, calibration error grows."""
        profile = CapabilityProfile(domain="test", success_rate=0.5, attempts=5, calibration_error=0.0)
        updated = profile.update(succeeded=False, stated_confidence=0.9)
        assert updated.calibration_error > 0  # Overconfidence detected


class TestGoal:
    def test_critical_goal_flag(self):
        g = Goal(priority=GoalPriority.CRITICAL)
        assert g.is_critical is True

    def test_terminal_states(self):
        g = Goal(status=GoalStatus.COMPLETED)
        assert g.is_terminal is True
        assert g.is_actionable is False


class TestWorldState:
    def test_entity_update(self):
        ws = WorldState()
        ws.update_entity("file_1", {"status": "open", "size": 100})
        assert ws.get_entity("file_1")["status"] == "open"
        ws.update_entity("file_1", {"status": "closed"})
        assert ws.get_entity("file_1")["status"] == "closed"
        assert ws.get_entity("file_1")["size"] == 100  # Preserved
