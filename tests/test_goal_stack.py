"""Tests for FORGE goal stack — goals persist outside the context window.

Cognitive behavior being tested:
- Primary goal cannot be silently overridden
- Dependencies block progress correctly
- Drift detection catches wandering
- Integrity checking finds structural problems
"""
import pytest
from core.types import GoalPriority, GoalStatus
from core.goal_stack import (
    GoalStack, GoalProtectionError, GoalNotFoundError,
    GoalDepthError, GoalIncompleteError,
)


class TestPrimaryGoalProtection:
    """The primary goal is sacred. Cannot be silently replaced."""

    def test_set_primary(self):
        gs = GoalStack()
        goal = gs.set_primary("Deploy the application")
        assert goal.is_critical
        assert goal.priority == GoalPriority.CRITICAL
        assert gs.get_primary() is goal

    def test_cannot_override_active_primary(self):
        """Attempting to set a new primary while one is active raises."""
        gs = GoalStack()
        gs.set_primary("Goal A")
        with pytest.raises(GoalProtectionError):
            gs.set_primary("Goal B")

    def test_can_replace_completed_primary(self):
        """After completing primary, can set a new one."""
        gs = GoalStack()
        g1 = gs.set_primary("Goal A")
        gs.complete(g1.id)
        g2 = gs.set_primary("Goal B")
        assert gs.get_primary() is g2


class TestGoalDecomposition:
    """Goals decompose into subgoals with dependencies."""

    def test_add_subgoal(self):
        gs = GoalStack()
        parent = gs.set_primary("Build the system")
        child = gs.add_subgoal(parent.id, "Write the tests")
        assert child.parent_id == parent.id
        assert child.id in parent.subgoals

    def test_dependency_blocking(self):
        """Subgoal B blocked until subgoal A completes."""
        gs = GoalStack()
        primary = gs.set_primary("Build system")
        a = gs.add_subgoal(primary.id, "Write code")
        b = gs.add_subgoal(primary.id, "Run tests", depends_on=[a.id])
        assert b.status == GoalStatus.BLOCKED

        gs.complete(a.id)
        assert b.status == GoalStatus.PENDING  # Unblocked

    def test_depth_limit(self):
        """Cannot nest goals deeper than max_depth."""
        from core.types import AgentConfig
        gs = GoalStack(config=AgentConfig(max_goal_depth=2))
        g1 = gs.set_primary("Level 0")
        g2 = gs.add_subgoal(g1.id, "Level 1")
        g3 = gs.add_subgoal(g2.id, "Level 2")
        with pytest.raises(GoalDepthError):
            gs.add_subgoal(g3.id, "Level 3 — too deep")

    def test_primary_requires_subgoal_completion(self):
        """Cannot complete primary if subgoals remain."""
        gs = GoalStack()
        primary = gs.set_primary("Build system")
        gs.add_subgoal(primary.id, "Write tests")
        with pytest.raises(GoalIncompleteError):
            gs.complete(primary.id)


class TestFocusManagement:
    """The agent should always know what to work on NOW."""

    def test_focus_on_primary_initially(self):
        gs = GoalStack()
        primary = gs.set_primary("Do the thing")
        assert gs.get_focus() is primary

    def test_focus_shifts_to_leaf(self):
        """Focus should go to the deepest actionable leaf goal."""
        gs = GoalStack()
        primary = gs.set_primary("Build app")
        sub = gs.add_subgoal(primary.id, "Write module A")
        focus = gs.get_focus()
        assert focus.id == sub.id  # Leaf gets focus

    def test_focus_advances_after_completion(self):
        """After completing focus goal, focus moves to next."""
        gs = GoalStack()
        primary = gs.set_primary("Build app")
        a = gs.add_subgoal(primary.id, "Step A")
        b = gs.add_subgoal(primary.id, "Step B", depends_on=[a.id])
        assert gs.get_focus().id == a.id

        gs.complete(a.id)
        assert gs.get_focus().id == b.id  # Focus advanced


class TestDriftDetection:
    """Catch when the agent wanders off-mission."""

    def test_no_drift_on_aligned_actions(self):
        gs = GoalStack()
        gs.set_primary("optimize database queries")
        scores = [
            gs.check_alignment("optimize database queries for speed"),
            gs.check_alignment("database queries need optimization"),
        ]
        assert gs.detect_drift(scores) is False

    def test_drift_on_unrelated_actions(self):
        gs = GoalStack()
        gs.set_primary("optimize database queries")
        scores = [
            gs.check_alignment("rewrite the CSS stylesheet"),
            gs.check_alignment("update the logo image"),
            gs.check_alignment("change button colors"),
        ]
        assert gs.detect_drift(scores) is True


class TestIntegrity:
    """Goal stack structural health checks."""

    def test_clean_stack_has_no_problems(self):
        gs = GoalStack()
        primary = gs.set_primary("Goal")
        gs.add_subgoal(primary.id, "Sub")
        assert gs.verify_integrity() == []

    def test_missing_parent_detected(self):
        """Orphaned subgoal should be flagged."""
        gs = GoalStack()
        primary = gs.set_primary("Goal")
        sub = gs.add_subgoal(primary.id, "Sub")
        # Simulate corruption: remove parent from store
        del gs._goals[primary.id]
        problems = gs.verify_integrity()
        assert len(problems) > 0
