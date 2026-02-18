"""
FORGE Atomic Primitives — the periodic table of cognitive architecture.

Every type here is fundamental. No convenience wrappers. Each represents
an irreducible concept in agent cognition. All other modules import from here.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


# ═══════════════════════════════════════════════════════════════════
# ENUMERATIONS — Finite state spaces
# ═══════════════════════════════════════════════════════════════════

class CognitivePhase(Enum):
    """The 8 phases of the cognitive cycle."""
    ORIENT = auto()    # Perceive and contextualize the current situation
    RECALL = auto()    # Retrieve relevant memories and learned strategies
    PLAN = auto()      # Generate candidate action sequences
    GATE = auto()      # Uncertainty check — can we proceed safely?
    ACT = auto()       # Execute the chosen action
    OBSERVE = auto()   # Capture the result and compare to prediction
    LEARN = auto()     # Update memories based on outcome
    CHECK = auto()     # Verify goal alignment — did we drift?


class GoalStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    BLOCKED = auto()   # Waiting on dependency
    COMPLETED = auto()
    FAILED = auto()
    ABANDONED = auto()


class GoalPriority(Enum):
    CRITICAL = 4   # Primary goal — protected from override
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class ReversibilityClass(Enum):
    """How reversible is an action? Gates escalate scrutiny for irreversible actions."""
    FULLY_REVERSIBLE = auto()   # Can undo completely (e.g., set a variable)
    PARTIALLY_REVERSIBLE = auto()  # Can undo with effort (e.g., edit a file)
    IRREVERSIBLE = auto()       # Cannot undo (e.g., send an email, delete data)


class MemoryTier(Enum):
    WORKING = auto()    # Active scratchpad, capacity-limited
    EPISODIC = auto()   # Compressed past experiences, subject to decay
    PROCEDURAL = auto() # Generalized strategies, reinforced by success


# ═══════════════════════════════════════════════════════════════════
# VALUE OBJECTS — Immutable data carriers
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class UncertaintyEstimate:
    """Calibrated confidence for a claim or decision.

    confidence: 0.0-1.0, where 0.5 = coin flip, 1.0 = certain
    calibration_error: running delta between stated and actual accuracy
    evidence_weight: how much data supports this estimate (0 = pure guess)
    """
    confidence: float
    calibration_error: float = 0.0
    evidence_weight: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be [0,1], got {self.confidence}")

    @property
    def is_reliable(self) -> bool:
        """High confidence AND low calibration error AND non-trivial evidence."""
        return self.confidence >= 0.7 and self.calibration_error < 0.15 and self.evidence_weight > 0.3

    def should_gate(self, reversibility: ReversibilityClass) -> bool:
        """Should the uncertainty gate block this action?

        Irreversible actions require higher confidence. This is architectural,
        not a suggestion — the agent cannot bypass this check.
        """
        thresholds = {
            ReversibilityClass.FULLY_REVERSIBLE: 0.3,
            ReversibilityClass.PARTIALLY_REVERSIBLE: 0.6,
            ReversibilityClass.IRREVERSIBLE: 0.85,
        }
        return self.confidence < thresholds[reversibility]


@dataclass(frozen=True)
class ActionEffect:
    """What an action DOES to the world — causal, not syntactic.

    Not "send_email is a function" but "send_email is irreversible,
    affects a human, has social consequences depending on timing/content."
    """
    action_name: str
    reversibility: ReversibilityClass
    affected_entities: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    estimated_duration_ms: float = 0.0
    cost: float = 0.0  # Resource cost (API calls, compute, money)


@dataclass(frozen=True)
class CapabilityProfile:
    """Empirical self-knowledge for a domain.

    Not self-assessment ("I'm good at math") but measured reality
    ("I solved 73% of math problems correctly, with 0.12 calibration error").
    """
    domain: str
    success_rate: float          # Actual fraction of successes
    attempts: int = 0
    calibration_error: float = 0.0  # |stated_confidence - actual_accuracy|
    avg_latency_ms: float = 0.0

    def update(self, succeeded: bool, stated_confidence: float) -> CapabilityProfile:
        """Return new profile incorporating this outcome."""
        new_attempts = self.attempts + 1
        new_rate = (self.success_rate * self.attempts + (1.0 if succeeded else 0.0)) / new_attempts
        # Exponential moving average for calibration error
        actual = 1.0 if succeeded else 0.0
        new_cal = self.calibration_error * 0.9 + abs(stated_confidence - actual) * 0.1
        return CapabilityProfile(
            domain=self.domain,
            success_rate=new_rate,
            attempts=new_attempts,
            calibration_error=new_cal,
            avg_latency_ms=self.avg_latency_ms,
        )


# ═══════════════════════════════════════════════════════════════════
# ENTITIES — Mutable, identity-bearing objects
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Goal:
    """A persistent goal that exists outside the context window.

    Goals are structured data, not prompts. They have explicit status,
    dependencies, and alignment checks. The primary goal is protected
    from silent override.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    parent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    subgoals: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    alignment_score: float = 1.0  # How aligned is current work with this goal?
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED)

    @property
    def is_actionable(self) -> bool:
        return self.status in (GoalStatus.PENDING, GoalStatus.ACTIVE)

    @property
    def is_critical(self) -> bool:
        return self.priority == GoalPriority.CRITICAL


@dataclass
class Memory:
    """A unit of stored experience.

    Memories have salience (how important), strength (how well remembered),
    access count (for decay), and content (the actual data).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tier: MemoryTier = MemoryTier.WORKING
    content: Any = None
    salience: float = 0.5          # 0-1, how important/relevant
    strength: float = 1.0          # 0-1, how well retained (decays)
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    source_episode: str | None = None  # For procedural memories: which episode taught this

    def access(self) -> None:
        """Record an access, refreshing strength."""
        self.access_count += 1
        self.last_accessed = time.time()
        # Accessing a memory strengthens it (testing effect)
        self.strength = min(1.0, self.strength + 0.1)


@dataclass
class WorldState:
    """Observable state of the world at a point in time.

    Not a god-view — only what the agent can observe. Includes
    entities, their properties, and known relationships.
    """
    timestamp: float = field(default_factory=time.time)
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)  # (subject, predicate, object)
    observations: list[str] = field(default_factory=list)

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        return self.entities.get(entity_id)

    def update_entity(self, entity_id: str, properties: dict[str, Any]) -> None:
        if entity_id not in self.entities:
            self.entities[entity_id] = {}
        self.entities[entity_id].update(properties)


@dataclass
class CognitiveStep:
    """A single step in the cognitive cycle — the atom of agent reasoning.

    Every action the agent takes produces a CognitiveStep. This is the
    fundamental unit of tracing, debugging, and learning.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    phase: CognitivePhase = CognitivePhase.ORIENT
    input_state: dict[str, Any] = field(default_factory=dict)
    output_state: dict[str, Any] = field(default_factory=dict)
    action_taken: str | None = None
    reasoning: str = ""
    uncertainty: UncertaintyEstimate | None = None
    goal_id: str | None = None          # Which goal drove this step
    parent_step_id: str | None = None   # For nested reasoning
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    token_cost: int = 0
    success: bool | None = None         # None = not yet evaluated

    @property
    def was_gated(self) -> bool:
        """Was this step blocked by the uncertainty gate?"""
        return self.phase == CognitivePhase.GATE and self.success is False


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — Single source of truth, no magic numbers
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """All tunable parameters in one place."""
    # Memory
    working_memory_capacity: int = 7        # Miller's number
    episodic_decay_rate: float = 0.1        # Ebbinghaus-inspired forgetting
    procedural_reinforcement: float = 0.2   # How much success strengthens a strategy
    procedural_penalty: float = 0.1         # How much failure weakens a strategy
    memory_consolidation_threshold: int = 5  # Episodes before attempting consolidation

    # Goals
    drift_threshold: float = 0.3            # Alignment below this triggers drift alert
    max_goal_depth: int = 5                 # Maximum subgoal nesting

    # Uncertainty Gate
    gate_confidence_reversible: float = 0.3
    gate_confidence_partial: float = 0.6
    gate_confidence_irreversible: float = 0.85

    # World Model
    max_simulation_branches: int = 5
    simulation_depth: int = 3

    # LLM
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 0.3

    # Tracing
    trace_enabled: bool = True
    trace_token_costs: bool = True


# ═══════════════════════════════════════════════════════════════════
# PROTOCOLS — Depend on interfaces, not implementations
# ═══════════════════════════════════════════════════════════════════

@runtime_checkable
class ToolExecutor(Protocol):
    """Any system that can execute actions in the world."""
    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]: ...
    def get_effects(self, action: str) -> ActionEffect | None: ...


@runtime_checkable
class ReasoningEngine(Protocol):
    """The thinking core — currently an LLM, but could be anything."""
    async def reason(self, prompt: str, context: dict[str, Any]) -> str: ...
    async def extract_strategy(self, episodes: list[dict]) -> str: ...
