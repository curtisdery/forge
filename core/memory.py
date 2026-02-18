"""
FORGE Memory System — Three-tier architecture for genuine skill acquisition.

After 10,000 tasks, an agent should be better than after task 1.
This requires three distinct memory tiers with different characteristics:

Working Memory: Active scratchpad. Capacity-limited (Miller's 7±2).
    Salience-based eviction. Gone when the task ends.

Episodic Memory: Compressed experiences with Ebbinghaus forgetting curves.
    Deduplication prevents redundant storage. Accessed memories resist decay.

Procedural Memory: Generalized strategies extracted from episodes.
    Reinforced by success, weakened by failure. Transfer across tasks.
    THIS is where learning lives.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .types import Memory, MemoryTier, AgentConfig


class WorkingMemory:
    """Capacity-limited active scratchpad. Salience-based eviction.

    Mimics human working memory constraints: you can hold ~7 items.
    When full, the least salient item is evicted. This forces the agent
    to prioritize what matters — a feature, not a bug.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._items: list[Memory] = []

    def store(self, content: Any, salience: float = 0.5, tags: list[str] | None = None) -> Memory:
        """Store an item. Evicts lowest-salience item if at capacity."""
        mem = Memory(
            tier=MemoryTier.WORKING,
            content=content,
            salience=salience,
            tags=tags or [],
        )
        if len(self._items) >= self.capacity:
            self._evict_lowest()
        self._items.append(mem)
        return mem

    def retrieve(self, tags: list[str] | None = None, top_k: int = 3) -> list[Memory]:
        """Retrieve items by tag relevance, sorted by salience."""
        if tags:
            scored = []
            for m in self._items:
                overlap = len(set(tags) & set(m.tags))
                if overlap > 0:
                    scored.append((m, m.salience * (1 + overlap)))
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [m for m, _ in scored[:top_k]]
        else:
            results = sorted(self._items, key=lambda m: m.salience, reverse=True)[:top_k]

        for m in results:
            m.access()
        return results

    def clear(self) -> list[Memory]:
        """Flush working memory. Returns evicted items for potential episodic storage."""
        evicted = self._items[:]
        self._items = []
        return evicted

    def _evict_lowest(self) -> Memory | None:
        if not self._items:
            return None
        lowest = min(self._items, key=lambda m: m.salience)
        self._items.remove(lowest)
        return lowest

    @property
    def items(self) -> list[Memory]:
        return self._items[:]

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def is_full(self) -> bool:
        return len(self._items) >= self.capacity


class EpisodicMemory:
    """Compressed past experiences with Ebbinghaus-inspired forgetting.

    Memories decay over time following a power law. Accessing a memory
    resets its decay clock (spacing effect). Duplicate or near-duplicate
    episodes are merged rather than stored redundantly.
    """

    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
        self._episodes: list[Memory] = []

    def store(self, content: Any, salience: float = 0.5, tags: list[str] | None = None,
              source_episode: str | None = None) -> Memory:
        """Store an episode. Deduplicates if similar content exists."""
        # Dedup: if content matches an existing episode, strengthen it instead
        for existing in self._episodes:
            if self._is_duplicate(existing.content, content):
                existing.strength = min(1.0, existing.strength + 0.2)
                existing.access()
                existing.salience = max(existing.salience, salience)
                return existing

        mem = Memory(
            tier=MemoryTier.EPISODIC,
            content=content,
            salience=salience,
            tags=tags or [],
            source_episode=source_episode,
        )
        self._episodes.append(mem)
        return mem

    def retrieve(self, tags: list[str] | None = None, top_k: int = 5,
                 min_strength: float = 0.1) -> list[Memory]:
        """Retrieve episodes by relevance, accounting for decay."""
        self._apply_decay()

        candidates = [m for m in self._episodes if m.strength >= min_strength]

        if tags:
            scored = []
            for m in candidates:
                overlap = len(set(tags) & set(m.tags))
                score = m.salience * m.strength * (1 + overlap)
                scored.append((m, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [m for m, _ in scored[:top_k]]
        else:
            candidates.sort(key=lambda m: m.salience * m.strength, reverse=True)
            results = candidates[:top_k]

        for m in results:
            m.access()
        return results

    def get_recent(self, n: int = 10) -> list[Memory]:
        """Get the N most recent episodes (for consolidation)."""
        return sorted(self._episodes, key=lambda m: m.created_at, reverse=True)[:n]

    def _apply_decay(self) -> None:
        """Ebbinghaus forgetting curve: strength = e^(-decay_rate * time_since_access)."""
        now = time.time()
        for m in self._episodes:
            elapsed_hours = (now - m.last_accessed) / 3600.0
            # Retention factor based on access count (spacing effect)
            retention_bonus = math.log1p(m.access_count) * 0.3
            decay = math.exp(-self.decay_rate * elapsed_hours) + retention_bonus
            m.strength = max(0.0, min(1.0, m.strength * min(1.0, decay)))

    def _is_duplicate(self, existing: Any, new: Any) -> bool:
        """Content-based dedup. Exact match for now; semantic similarity later."""
        if type(existing) != type(new):
            return False
        if isinstance(existing, dict) and isinstance(new, dict):
            # Same keys and values = duplicate
            return existing == new
        return existing == new

    def prune(self, min_strength: float = 0.05) -> int:
        """Remove memories below strength threshold. Returns count removed."""
        before = len(self._episodes)
        self._episodes = [m for m in self._episodes if m.strength >= min_strength]
        return before - len(self._episodes)

    @property
    def episodes(self) -> list[Memory]:
        return self._episodes[:]

    @property
    def size(self) -> int:
        return len(self._episodes)


class ProceduralMemory:
    """Generalized strategies that transfer across tasks.

    This is WHERE LEARNING LIVES. Strategies are extracted from episodic
    patterns (multiple similar episodes → one general rule). They're
    reinforced when they lead to success, weakened on failure.

    A strategy is: "When [situation pattern], do [action sequence], because [rationale]."
    """

    def __init__(self, reinforcement: float = 0.2, penalty: float = 0.1):
        self.reinforcement = reinforcement
        self.penalty = penalty
        self._strategies: list[Memory] = []

    def store(self, strategy: dict[str, Any], salience: float = 0.5,
              tags: list[str] | None = None, source_episode: str | None = None) -> Memory:
        """Store a new strategy. Merges if a similar one exists."""
        # Check for existing strategy with same pattern
        for existing in self._strategies:
            if isinstance(existing.content, dict) and isinstance(strategy, dict):
                if existing.content.get("pattern") == strategy.get("pattern"):
                    # Merge: keep the one with higher success rate
                    existing.strength = min(1.0, existing.strength + 0.15)
                    existing.access()
                    return existing

        mem = Memory(
            tier=MemoryTier.PROCEDURAL,
            content=strategy,
            salience=salience,
            strength=0.5,  # New strategies start at 50% — must prove themselves
            tags=tags or [],
            source_episode=source_episode,
        )
        self._strategies.append(mem)
        return mem

    def retrieve(self, tags: list[str] | None = None, top_k: int = 3,
                 min_strength: float = 0.2) -> list[Memory]:
        """Retrieve applicable strategies for current situation."""
        candidates = [s for s in self._strategies if s.strength >= min_strength]

        if tags:
            scored = []
            for s in candidates:
                overlap = len(set(tags) & set(s.tags))
                if overlap > 0 or not tags:
                    score = s.strength * s.salience * (1 + overlap)
                    scored.append((s, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [s for s, _ in scored[:top_k]]
        else:
            candidates.sort(key=lambda s: s.strength * s.salience, reverse=True)
            results = candidates[:top_k]

        for s in results:
            s.access()
        return results

    def reinforce(self, strategy_id: str, success: bool) -> None:
        """Update strategy strength based on outcome. The learning signal."""
        for s in self._strategies:
            if s.id == strategy_id:
                if success:
                    s.strength = min(1.0, s.strength + self.reinforcement)
                    s.salience = min(1.0, s.salience + 0.05)
                else:
                    s.strength = max(0.0, s.strength - self.penalty)
                return

    @property
    def strategies(self) -> list[Memory]:
        return self._strategies[:]

    @property
    def size(self) -> int:
        return len(self._strategies)

    def get_strongest(self, n: int = 5) -> list[Memory]:
        """Return the N strongest strategies — the agent's best-learned skills."""
        return sorted(self._strategies, key=lambda s: s.strength, reverse=True)[:n]


class MemorySystem:
    """Unified interface to all three memory tiers.

    Handles cross-tier operations: consolidation (working → episodic → procedural),
    retrieval across tiers, and memory lifecycle management.
    """

    def __init__(self, config: AgentConfig | None = None):
        cfg = config or AgentConfig()
        self.working = WorkingMemory(capacity=cfg.working_memory_capacity)
        self.episodic = EpisodicMemory(decay_rate=cfg.episodic_decay_rate)
        self.procedural = ProceduralMemory(
            reinforcement=cfg.procedural_reinforcement,
            penalty=cfg.procedural_penalty,
        )
        self._consolidation_threshold = cfg.memory_consolidation_threshold
        self._episodes_since_consolidation = 0

    def flush_working_to_episodic(self, task_tags: list[str] | None = None) -> list[Memory]:
        """End of task: move working memory items to episodic storage."""
        working_items = self.working.clear()
        episodic_items = []
        for item in working_items:
            tags = list(set((item.tags or []) + (task_tags or [])))
            ep = self.episodic.store(
                content=item.content,
                salience=item.salience,
                tags=tags,
            )
            episodic_items.append(ep)
        self._episodes_since_consolidation += 1
        return episodic_items

    def should_consolidate(self) -> bool:
        """Have we accumulated enough episodes to attempt strategy extraction?"""
        return self._episodes_since_consolidation >= self._consolidation_threshold

    def consolidate(self, extracted_strategy: dict[str, Any] | None = None,
                    tags: list[str] | None = None) -> Memory | None:
        """Promote a pattern from episodic to procedural memory.

        In a real system, the LLM extracts strategies from recent episodes.
        Here we accept the extracted strategy as input.
        """
        if extracted_strategy is None:
            return None

        mem = self.procedural.store(
            strategy=extracted_strategy,
            salience=0.6,
            tags=tags or [],
        )
        self._episodes_since_consolidation = 0
        return mem

    def retrieve_all(self, tags: list[str] | None = None, top_k: int = 5) -> dict[str, list[Memory]]:
        """Retrieve relevant memories from all tiers."""
        return {
            "working": self.working.retrieve(tags=tags, top_k=top_k),
            "episodic": self.episodic.retrieve(tags=tags, top_k=top_k),
            "procedural": self.procedural.retrieve(tags=tags, top_k=top_k),
        }
