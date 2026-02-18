"""
FORGE Goal Stack — Goals exist outside the context window.

Goals are protected, structured data — not prompts that can be overridden.
The primary goal cannot be silently replaced. Every step is checked for
alignment. Drift detection catches when the agent wanders off-mission.
"""
from __future__ import annotations

from typing import Any

from .types import Goal, GoalStatus, GoalPriority, AgentConfig


class GoalStack:
    """Hierarchical goal management with protection and drift detection.

    The stack maintains:
    - A protected primary goal (CRITICAL priority, cannot be overridden)
    - Hierarchical decomposition (goals have subgoals)
    - Sequential dependencies (goal B waits for goal A)
    - Automatic focus management (what should we work on NOW?)
    - Drift detection (are we still aligned with the primary goal?)
    """

    def __init__(self, config: AgentConfig | None = None):
        cfg = config or AgentConfig()
        self._goals: dict[str, Goal] = {}
        self._primary_id: str | None = None
        self._focus_id: str | None = None
        self.drift_threshold = cfg.drift_threshold
        self.max_depth = cfg.max_goal_depth

    def set_primary(self, description: str, metadata: dict[str, Any] | None = None) -> Goal:
        """Set the primary goal. Protected from override. Only one allowed."""
        if self._primary_id is not None:
            existing = self._goals.get(self._primary_id)
            if existing and existing.is_actionable:
                raise GoalProtectionError(
                    f"Primary goal already active: '{existing.description}'. "
                    "Complete or abandon it before setting a new one."
                )

        goal = Goal(
            description=description,
            status=GoalStatus.ACTIVE,
            priority=GoalPriority.CRITICAL,
            metadata=metadata or {},
        )
        self._goals[goal.id] = goal
        self._primary_id = goal.id
        self._focus_id = goal.id
        return goal

    def add_subgoal(self, parent_id: str, description: str,
                    priority: GoalPriority = GoalPriority.MEDIUM,
                    depends_on: list[str] | None = None) -> Goal:
        """Decompose a goal into a subgoal."""
        parent = self._goals.get(parent_id)
        if parent is None:
            raise GoalNotFoundError(f"Parent goal {parent_id} not found")

        # Check depth
        depth = self._get_depth(parent_id)
        if depth >= self.max_depth:
            raise GoalDepthError(f"Max goal depth ({self.max_depth}) exceeded")

        goal = Goal(
            description=description,
            status=GoalStatus.PENDING,
            priority=priority,
            parent_id=parent_id,
            depends_on=depends_on or [],
        )

        # Check dependencies are valid
        for dep_id in goal.depends_on:
            if dep_id not in self._goals:
                raise GoalNotFoundError(f"Dependency {dep_id} not found")

        self._goals[goal.id] = goal
        parent.subgoals.append(goal.id)
        self._update_blocked_status()
        return goal

    def complete(self, goal_id: str) -> Goal:
        """Mark a goal as completed. Unblocks dependents."""
        goal = self._get_goal(goal_id)
        if goal.is_critical and goal.id == self._primary_id:
            # Primary goal completion — check all subgoals are done
            incomplete = [
                sg_id for sg_id in goal.subgoals
                if sg_id in self._goals and not self._goals[sg_id].is_terminal
            ]
            if incomplete:
                raise GoalIncompleteError(
                    f"Cannot complete primary goal: {len(incomplete)} subgoals remain"
                )
        import time
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = time.time()
        self._update_blocked_status()
        self._advance_focus()
        return goal

    def fail(self, goal_id: str, reason: str = "") -> Goal:
        """Mark a goal as failed."""
        goal = self._get_goal(goal_id)
        goal.status = GoalStatus.FAILED
        goal.metadata["failure_reason"] = reason
        self._update_blocked_status()
        self._advance_focus()
        return goal

    def get_focus(self) -> Goal | None:
        """What should the agent work on RIGHT NOW?

        Returns the highest-priority actionable leaf goal.
        Always re-evaluates to pick the deepest actionable leaf.
        """
        self._advance_focus()
        if self._focus_id:
            return self._goals.get(self._focus_id)
        return None

    def check_alignment(self, action_description: str) -> float:
        """Check if a proposed action aligns with the primary goal.

        Returns alignment score 0-1. Below drift_threshold triggers alert.
        This is a simple keyword overlap heuristic — real version uses LLM.
        """
        if not self._primary_id:
            return 1.0  # No primary goal = anything goes

        primary = self._goals[self._primary_id]
        focus = self._goals.get(self._focus_id) if self._focus_id else None

        # Simple heuristic: word overlap between action and goal descriptions
        action_words = set(action_description.lower().split())
        goal_words = set(primary.description.lower().split())
        if focus:
            goal_words |= set(focus.description.lower().split())

        if not goal_words:
            return 1.0

        overlap = len(action_words & goal_words)
        total = len(action_words | goal_words)
        return overlap / total if total > 0 else 0.0

    def detect_drift(self, recent_alignment_scores: list[float]) -> bool:
        """Are recent actions drifting away from the goal?

        Returns True if the average alignment of recent actions
        falls below the drift threshold.
        """
        if not recent_alignment_scores:
            return False
        avg = sum(recent_alignment_scores) / len(recent_alignment_scores)
        return avg < self.drift_threshold

    def get_primary(self) -> Goal | None:
        """Get the primary goal."""
        if self._primary_id:
            return self._goals.get(self._primary_id)
        return None

    def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_all(self) -> list[Goal]:
        """All goals, ordered by priority then creation time."""
        return sorted(
            self._goals.values(),
            key=lambda g: (-g.priority.value, g.created_at),
        )

    def get_actionable(self) -> list[Goal]:
        """All goals that can be worked on right now."""
        self._update_blocked_status()
        return [g for g in self._goals.values() if g.is_actionable]

    def verify_integrity(self) -> list[str]:
        """Check the goal stack for structural issues. Returns list of problems."""
        problems = []

        # Primary goal check
        if self._primary_id and self._primary_id not in self._goals:
            problems.append(f"Primary goal {self._primary_id} missing from goal store")

        # Orphan check
        for gid, goal in self._goals.items():
            if goal.parent_id and goal.parent_id not in self._goals:
                problems.append(f"Goal {gid} references missing parent {goal.parent_id}")
            for dep in goal.depends_on:
                if dep not in self._goals:
                    problems.append(f"Goal {gid} depends on missing goal {dep}")
            for sg in goal.subgoals:
                if sg not in self._goals:
                    problems.append(f"Goal {gid} lists missing subgoal {sg}")

        # Cycle detection (BFS)
        for gid in self._goals:
            visited = set()
            queue = [gid]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    problems.append(f"Cycle detected involving goal {gid}")
                    break
                visited.add(current)
                g = self._goals.get(current)
                if g:
                    queue.extend(g.depends_on)

        return problems

    # ─── Internal ──────────────────────────────────────────────

    def _get_goal(self, goal_id: str) -> Goal:
        goal = self._goals.get(goal_id)
        if goal is None:
            raise GoalNotFoundError(f"Goal {goal_id} not found")
        return goal

    def _get_depth(self, goal_id: str) -> int:
        """How deep in the hierarchy is this goal?"""
        depth = 0
        current = self._goals.get(goal_id)
        while current and current.parent_id:
            depth += 1
            current = self._goals.get(current.parent_id)
        return depth

    def _update_blocked_status(self) -> None:
        """Update BLOCKED status based on dependency completion."""
        for goal in self._goals.values():
            if goal.is_terminal:
                continue
            all_deps_met = all(
                self._goals.get(dep_id) and self._goals[dep_id].status == GoalStatus.COMPLETED
                for dep_id in goal.depends_on
            )
            if goal.depends_on and not all_deps_met:
                if goal.status != GoalStatus.BLOCKED:
                    goal.status = GoalStatus.BLOCKED
            elif goal.status == GoalStatus.BLOCKED:
                goal.status = GoalStatus.PENDING

    def _advance_focus(self) -> None:
        """Move focus to the next best actionable goal."""
        actionable = self.get_actionable()
        if not actionable:
            self._focus_id = None
            return

        # Prefer: leaf goals (no uncompleted subgoals) > deeper > higher priority
        def score(g: Goal) -> tuple:
            has_active_subgoals = any(
                sg_id in self._goals and self._goals[sg_id].is_actionable
                for sg_id in g.subgoals
            )
            depth = self._get_depth(g.id)
            return (not has_active_subgoals, depth, g.priority.value)

        best = max(actionable, key=score)
        self._focus_id = best.id

    @property
    def size(self) -> int:
        return len(self._goals)


# ─── Exceptions ──────────────────────────────────────────────

class GoalProtectionError(Exception):
    """Attempted to override a protected primary goal."""

class GoalNotFoundError(Exception):
    """Referenced a goal that doesn't exist."""

class GoalDepthError(Exception):
    """Exceeded maximum goal nesting depth."""

class GoalIncompleteError(Exception):
    """Attempted to complete a goal with unfinished subgoals."""
