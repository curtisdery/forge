"""Tests for FORGE learning benchmark — THE acid test.

Does procedural memory actually improve performance across task iterations?
If yes — architecture works. If no — redesign memory consolidation.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.learning_benchmark import run_benchmark


class TestLearningBenchmark:
    """The core promise: agents get better with experience."""

    @pytest.mark.asyncio
    async def test_agent_learns_from_repeated_tasks(self):
        """After 20 task variations, late performance > early performance."""
        summary = await run_benchmark()

        # Core assertion: the agent learned
        assert summary["learned"] is True, (
            f"Agent failed to learn. Early: {summary['early_accuracy_1_5']:.0%}, "
            f"Late: {summary['late_accuracy_15_20']:.0%}"
        )

    @pytest.mark.asyncio
    async def test_strategies_accumulate(self):
        """The agent should build up procedural strategies over time."""
        summary = await run_benchmark()
        assert summary["final_procedural_strategies"] > 0, "No strategies were learned"

    @pytest.mark.asyncio
    async def test_late_accuracy_above_threshold(self):
        """By iterations 15-20, accuracy should be meaningfully above chance."""
        summary = await run_benchmark()
        assert summary["late_accuracy_15_20"] >= 0.5, (
            f"Late accuracy {summary['late_accuracy_15_20']:.0%} is too low"
        )
