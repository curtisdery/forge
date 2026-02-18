"""
FORGE World Model — Causal graphs and mental simulation.

Agents must understand what their actions DO to reality. Not just that
send_email is a function, but that it's irreversible, affects a human,
and has social consequences. This module implements:

1. CausalGraph — directed graph of action → effect relationships
2. WorldModel — simulation engine for branching futures

Before acting, agents simulate multiple possible outcomes and select
the best path. Not chain-of-thought (serial). Branch-and-bound against
the world model, evaluating outcomes before committing.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .types import (
    ActionEffect, ReversibilityClass, UncertaintyEstimate,
    WorldState, AgentConfig,
)


@dataclass
class CausalEdge:
    """A cause-effect relationship: action X in state S produces effect E."""
    action: str
    preconditions: dict[str, Any]       # State conditions that must hold
    effects: dict[str, Any]             # State changes produced
    probability: float = 1.0            # How likely is this outcome?
    reversibility: ReversibilityClass = ReversibilityClass.FULLY_REVERSIBLE


@dataclass
class SimulationBranch:
    """One possible future — a sequence of actions and predicted states."""
    actions: list[str] = field(default_factory=list)
    states: list[WorldState] = field(default_factory=list)
    cumulative_probability: float = 1.0
    cumulative_cost: float = 0.0
    utility: float = 0.0  # How good is this future?

    @property
    def is_viable(self) -> bool:
        return self.cumulative_probability > 0.1

    def append_step(self, action: str, state: WorldState,
                    probability: float, cost: float) -> None:
        self.actions.append(action)
        self.states.append(state)
        self.cumulative_probability *= probability
        self.cumulative_cost += cost


class CausalGraph:
    """Directed graph of action → effect relationships.

    Every tool/action the agent can take is registered here with its
    causal effects, preconditions, reversibility class, and failure modes.
    The graph answers: "If I do X in state S, what happens?"
    """

    def __init__(self):
        self._actions: dict[str, ActionEffect] = {}
        self._edges: list[CausalEdge] = []

    def register_action(self, effect: ActionEffect) -> None:
        """Register an action and its effects in the causal graph."""
        self._actions[effect.action_name] = effect

    def add_edge(self, edge: CausalEdge) -> None:
        """Add a cause-effect relationship."""
        self._edges.append(edge)

    def get_effects(self, action: str) -> ActionEffect | None:
        """What does this action DO?"""
        return self._actions.get(action)

    def predict(self, action: str, state: WorldState) -> list[tuple[WorldState, float]]:
        """Given action in state, predict possible next states with probabilities.

        Returns list of (next_state, probability) tuples.
        Multiple outcomes are possible (e.g., API call might succeed or fail).
        """
        results = []
        matching_edges = [
            e for e in self._edges
            if e.action == action and self._preconditions_met(e.preconditions, state)
        ]

        if not matching_edges:
            # Unknown action-state pair — return current state with low confidence
            return [(copy.deepcopy(state), 0.5)]

        for edge in matching_edges:
            next_state = copy.deepcopy(state)
            for entity_id, props in edge.effects.items():
                next_state.update_entity(entity_id, props)
            next_state.observations.append(f"Action '{action}' executed")
            results.append((next_state, edge.probability))

        return results

    def get_reversibility(self, action: str) -> ReversibilityClass:
        """How reversible is this action?"""
        effect = self._actions.get(action)
        if effect:
            return effect.reversibility
        return ReversibilityClass.PARTIALLY_REVERSIBLE  # Unknown = cautious default

    def get_available_actions(self, state: WorldState) -> list[str]:
        """What actions can be taken from this state?"""
        available = []
        for edge in self._edges:
            if self._preconditions_met(edge.preconditions, state):
                if edge.action not in available:
                    available.append(edge.action)
        return available

    def _preconditions_met(self, preconditions: dict[str, Any],
                           state: WorldState) -> bool:
        """Check if all preconditions hold in the given state."""
        for entity_id, required_props in preconditions.items():
            entity = state.get_entity(entity_id)
            if entity is None:
                return False
            for key, value in required_props.items():
                if entity.get(key) != value:
                    return False
        return True

    @property
    def action_count(self) -> int:
        return len(self._actions)

    @property
    def edge_count(self) -> int:
        return len(self._edges)


class WorldModel:
    """Mental simulation engine — branch-and-bound over possible futures.

    Before committing to an irreversible action, the agent simulates
    multiple paths and evaluates outcomes. This is the "think before
    you act" primitive implemented as architecture, not prompting.
    """

    def __init__(self, causal_graph: CausalGraph, config: AgentConfig | None = None):
        cfg = config or AgentConfig()
        self.graph = causal_graph
        self.max_branches = cfg.max_simulation_branches
        self.max_depth = cfg.simulation_depth
        self._current_state: WorldState = WorldState()

    @property
    def current_state(self) -> WorldState:
        return self._current_state

    def update_state(self, state: WorldState) -> None:
        """Update the world model with new observations."""
        self._current_state = state

    def simulate(self, action_sequence: list[str],
                 start_state: WorldState | None = None) -> SimulationBranch:
        """Simulate a specific action sequence. Returns the resulting branch."""
        state = copy.deepcopy(start_state or self._current_state)
        branch = SimulationBranch(states=[copy.deepcopy(state)])

        for action in action_sequence:
            outcomes = self.graph.predict(action, state)
            if not outcomes:
                break
            # Take the most likely outcome for deterministic simulation
            best_state, prob = max(outcomes, key=lambda x: x[1])
            effect = self.graph.get_effects(action)
            cost = effect.cost if effect else 0.0
            branch.append_step(action, best_state, prob, cost)
            state = best_state

        return branch

    def branch_and_bound(self, candidate_actions: list[list[str]],
                         utility_fn: callable | None = None,
                         start_state: WorldState | None = None) -> list[SimulationBranch]:
        """Simulate multiple action sequences and rank by utility.

        This is the core "mental simulation" primitive. The agent generates
        candidate plans, simulates each against the world model, and picks
        the best one BEFORE acting.

        utility_fn: (branch) -> float. Higher is better. Default: probability * -cost.
        """
        if utility_fn is None:
            utility_fn = lambda b: b.cumulative_probability - b.cumulative_cost

        branches = []
        for actions in candidate_actions[:self.max_branches]:
            branch = self.simulate(actions, start_state)
            branch.utility = utility_fn(branch)
            branches.append(branch)

        branches.sort(key=lambda b: b.utility, reverse=True)
        return branches

    def should_gate(self, action: str, confidence: UncertaintyEstimate) -> bool:
        """Should the uncertainty gate block this action?

        Combines the action's reversibility class with the agent's confidence.
        Irreversible actions with low confidence are BLOCKED architecturally.
        """
        reversibility = self.graph.get_reversibility(action)
        return confidence.should_gate(reversibility)

    def get_safest_action(self, candidates: list[str],
                          state: WorldState | None = None) -> str | None:
        """From a list of candidate actions, return the safest (most reversible)."""
        state = state or self._current_state
        scored = []
        for action in candidates:
            rev = self.graph.get_reversibility(action)
            rev_score = {
                ReversibilityClass.FULLY_REVERSIBLE: 3,
                ReversibilityClass.PARTIALLY_REVERSIBLE: 2,
                ReversibilityClass.IRREVERSIBLE: 1,
            }.get(rev, 0)
            scored.append((action, rev_score))

        if not scored:
            return None
        return max(scored, key=lambda x: x[1])[0]
