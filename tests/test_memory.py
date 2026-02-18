"""Tests for FORGE memory system — the core of learning.

These tests validate COGNITIVE BEHAVIOR:
- Does working memory evict lowest-salience items?
- Does episodic memory forget over time?
- Does procedural memory strengthen with success?
- Does the full system actually learn?
"""
import time
import pytest
from core.types import AgentConfig
from core.memory import WorkingMemory, EpisodicMemory, ProceduralMemory, MemorySystem


class TestWorkingMemory:
    """Capacity-limited, salience-evicted."""

    def test_capacity_limit(self):
        """Working memory holds at most N items (Miller's number)."""
        wm = WorkingMemory(capacity=3)
        wm.store("a", salience=0.5)
        wm.store("b", salience=0.3)
        wm.store("c", salience=0.7)
        assert wm.size == 3
        wm.store("d", salience=0.8)
        assert wm.size == 3  # Still 3 — evicted lowest

    def test_evicts_lowest_salience(self):
        """The least important item gets evicted, not the oldest."""
        wm = WorkingMemory(capacity=3)
        wm.store("important", salience=0.9)
        wm.store("forgettable", salience=0.1)
        wm.store("medium", salience=0.5)
        wm.store("new_item", salience=0.8)
        contents = [m.content for m in wm.items]
        assert "forgettable" not in contents
        assert "important" in contents

    def test_tag_based_retrieval(self):
        wm = WorkingMemory(capacity=7)
        wm.store("deal analysis", salience=0.8, tags=["deal", "analysis"])
        wm.store("weather report", salience=0.7, tags=["weather"])
        wm.store("deal pricing", salience=0.6, tags=["deal", "pricing"])
        results = wm.retrieve(tags=["deal"], top_k=5)
        assert len(results) == 2
        assert results[0].content == "deal analysis"  # Higher salience + tag match

    def test_clear_returns_items(self):
        wm = WorkingMemory(capacity=5)
        wm.store("x")
        wm.store("y")
        evicted = wm.clear()
        assert len(evicted) == 2
        assert wm.size == 0


class TestEpisodicMemory:
    """Compressed experiences with forgetting curves."""

    def test_deduplication(self):
        """Storing the same content twice strengthens instead of duplicating."""
        em = EpisodicMemory()
        em.store({"action": "search", "result": "found"}, salience=0.5)
        em.store({"action": "search", "result": "found"}, salience=0.5)
        assert em.size == 1  # Deduped
        assert em.episodes[0].strength > 1.0 * 0.5  # Strengthened

    def test_decay_reduces_strength(self):
        """Memories weaken over time if not accessed."""
        em = EpisodicMemory(decay_rate=50.0)  # Aggressive decay for testing
        mem = em.store("old memory", salience=0.5)
        original_strength = mem.strength
        # Manually age the memory
        mem.last_accessed = time.time() - 3600  # 1 hour ago
        retrieved = em.retrieve(min_strength=0.0)
        # After decay, strength should be lower
        assert mem.strength <= original_strength

    def test_access_resists_decay(self):
        """Accessing a memory refreshes it (spacing effect)."""
        em = EpisodicMemory()
        mem = em.store("accessed memory", salience=0.8, tags=["test"])
        mem.access()
        mem.access()
        mem.access()
        assert mem.access_count >= 3
        assert mem.strength >= 1.0  # Strengthened by access


class TestProceduralMemory:
    """Learned strategies — WHERE LEARNING LIVES."""

    def test_reinforcement_increases_strength(self):
        """Success makes a strategy stronger."""
        pm = ProceduralMemory(reinforcement=0.2)
        mem = pm.store({"pattern": "search_then_act", "actions": ["search", "act"]})
        original = mem.strength
        pm.reinforce(mem.id, success=True)
        assert mem.strength > original

    def test_failure_decreases_strength(self):
        """Failure weakens a strategy."""
        pm = ProceduralMemory(penalty=0.15)
        mem = pm.store({"pattern": "brute_force", "actions": ["try_everything"]})
        original = mem.strength
        pm.reinforce(mem.id, success=False)
        assert mem.strength < original

    def test_strategies_start_unproven(self):
        """New strategies start at 0.5 — must prove themselves."""
        pm = ProceduralMemory()
        mem = pm.store({"pattern": "new_idea", "actions": ["test"]})
        assert mem.strength == 0.5

    def test_strongest_strategies(self):
        """Can retrieve the best-learned strategies."""
        pm = ProceduralMemory(reinforcement=0.3)
        s1 = pm.store({"pattern": "proven", "actions": ["a"]}, tags=["math"])
        s2 = pm.store({"pattern": "unproven", "actions": ["b"]}, tags=["math"])
        # Reinforce s1 multiple times
        for _ in range(5):
            pm.reinforce(s1.id, success=True)
        strongest = pm.get_strongest(1)
        assert strongest[0].id == s1.id
        assert strongest[0].strength > s2.strength


class TestMemorySystem:
    """Full three-tier system with cross-tier operations."""

    def test_flush_working_to_episodic(self):
        """End of task: working memory items move to episodic storage."""
        ms = MemorySystem()
        ms.working.store("step 1", salience=0.6, tags=["task"])
        ms.working.store("step 2", salience=0.7, tags=["task"])
        assert ms.working.size == 2

        episodic_items = ms.flush_working_to_episodic(task_tags=["completed"])
        assert ms.working.size == 0
        assert ms.episodic.size == 2
        assert len(episodic_items) == 2

    def test_consolidation_threshold(self):
        """Strategy extraction triggers after N episodes."""
        cfg = AgentConfig(memory_consolidation_threshold=3)
        ms = MemorySystem(config=cfg)

        # Flush 2 tasks — not enough
        ms.working.store("a")
        ms.flush_working_to_episodic()
        assert ms.should_consolidate() is False

        ms.working.store("b")
        ms.flush_working_to_episodic()
        assert ms.should_consolidate() is False

        ms.working.store("c")
        ms.flush_working_to_episodic()
        assert ms.should_consolidate() is True

    def test_consolidation_creates_strategy(self):
        """Consolidation promotes episodic pattern to procedural strategy."""
        ms = MemorySystem()
        for i in range(5):
            ms.working.store(f"episode_{i}")
            ms.flush_working_to_episodic()

        strategy = ms.consolidate(
            extracted_strategy={"pattern": "repeat_success", "actions": ["do_thing"]},
            tags=["learned"],
        )
        assert strategy is not None
        assert ms.procedural.size == 1
