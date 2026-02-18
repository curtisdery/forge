"""
FORGE Learning Benchmark — LIVE (Real Anthropic API)

Same methodology as the mock benchmark, but using Claude as the reasoning
core. This tests whether real LLM strategy extraction produces genuine
learning improvement.

Requires: ANTHROPIC_API_KEY environment variable

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python benchmarks/learning_benchmark_live.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

from core.types import AgentConfig, ActionEffect, ReversibilityClass
from core.agent import CognitiveAgent
from core.llm import AnthropicProvider


# ═══════════════════════════════════════════════════════════════════
# TASK CLASS: Same 20 pipeline debugging variations
# ═══════════════════════════════════════════════════════════════════

TASK_VARIATIONS = [
    {"id": i, "description": f"Pipeline stage {stage} failing with {error}",
     "stage": stage, "error": error, "correct_action": action}
    for i, (stage, error, action) in enumerate([
        ("ingest", "timeout", "increase_timeout"),
        ("transform", "null_values", "add_null_check"),
        ("validate", "schema_mismatch", "update_schema"),
        ("load", "connection_refused", "retry_connection"),
        ("ingest", "rate_limited", "add_backoff"),
        ("transform", "type_error", "add_type_cast"),
        ("validate", "missing_field", "add_default"),
        ("load", "disk_full", "clean_temp"),
        ("ingest", "auth_expired", "refresh_token"),
        ("transform", "memory_overflow", "add_chunking"),
        ("validate", "duplicate_key", "add_dedup"),
        ("load", "timeout", "increase_timeout"),
        ("ingest", "malformed_data", "add_validation"),
        ("transform", "null_values", "add_null_check"),
        ("validate", "schema_mismatch", "update_schema"),
        ("load", "connection_refused", "retry_connection"),
        ("ingest", "timeout", "increase_timeout"),
        ("transform", "type_error", "add_type_cast"),
        ("validate", "missing_field", "add_default"),
        ("load", "disk_full", "clean_temp"),
    ])
]

AVAILABLE_ACTIONS = [
    "increase_timeout", "add_null_check", "update_schema",
    "retry_connection", "add_backoff", "add_type_cast",
    "add_default", "clean_temp", "refresh_token", "add_chunking",
    "add_dedup", "add_validation",
]


async def run_live_benchmark() -> dict:
    """Run the learning benchmark against the real Anthropic API."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Use Haiku for speed and cost — this is a 20-iteration benchmark
    llm = AnthropicProvider(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0.2,
    )

    config = AgentConfig(
        working_memory_capacity=7,
        memory_consolidation_threshold=3,
        model="claude-haiku-4-5-20251001",
    )
    agent = CognitiveAgent(llm=llm, config=config)

    # Register all possible actions as tools
    for action in AVAILABLE_ACTIONS:
        await agent.add_tool(action, lambda: {"success": True}, ActionEffect(
            action_name=action,
            reversibility=ReversibilityClass.FULLY_REVERSIBLE,
            cost=0.01,
        ))

    results = []
    total_start = time.time()

    for i, task in enumerate(TASK_VARIATIONS):
        iter_start = time.time()
        print(f"  [{i+1:2d}/20] {task['description']:<50s}", end="", flush=True)

        # Set goal
        try:
            agent.goals.set_primary(f"Fix: {task['description']}")
        except Exception:
            primary = agent.goals.get_primary()
            if primary and primary.is_actionable:
                agent.goals.complete(primary.id)
            agent.goals.set_primary(f"Fix: {task['description']}")

        # Run cognitive cycle
        cycle_result = await agent.run_cycle(
            f"Alert: {task['description']}. "
            f"Available fix actions: {', '.join(AVAILABLE_ACTIONS)}. "
            f"Choose the single best action."
        )

        # Evaluate
        correct_action = task["correct_action"]
        chose_correct = cycle_result.action_taken == correct_action
        elapsed = time.time() - iter_start

        # Correctness feedback: if wrong, inject corrective episode so agent learns
        if not chose_correct:
            desc_tags = [w.strip(":.!?,") for w in task["description"].lower().split() if w.isalpha()]
            agent.memory.working.store(
                content={
                    "action": cycle_result.action_taken,
                    "correct_action": correct_action,
                    "result": {"success": False, "correction": True},
                    "succeeded": False,
                    "goal": f"Fix: {task['description']}",
                    "cycle": i,
                },
                salience=0.9,
                tags=["episode", "correction", correct_action] + desc_tags,
            )
            agent.memory.flush_working_to_episodic(
                task_tags=["correction", correct_action] + desc_tags
            )
            # Penalize wrong strategies
            wrong_strategies = agent.memory.procedural.retrieve(
                tags=desc_tags,
                top_k=3,
            )
            for s in wrong_strategies:
                agent.memory.procedural.reinforce(s.id, success=False)

        # Complete goal
        primary = agent.goals.get_primary()
        if primary and primary.is_actionable:
            agent.goals.complete(primary.id)

        marker = "+" if chose_correct else "x"
        print(f"  [{marker}] {cycle_result.action_taken or 'none':<20s} ({elapsed:.1f}s)")

        results.append({
            "iteration": i,
            "task": task["description"],
            "correct_action": correct_action,
            "chosen_action": cycle_result.action_taken,
            "chose_correct": chose_correct,
            "was_gated": cycle_result.was_gated,
            "tokens_used": cycle_result.tokens_used,
            "elapsed_s": elapsed,
            "procedural_strategies": agent.memory.procedural.size,
            "episodic_memories": agent.memory.episodic.size,
        })

    total_elapsed = time.time() - total_start

    return analyze_results(results, agent, total_elapsed)


def analyze_results(results: list[dict], agent: CognitiveAgent,
                    total_elapsed: float) -> dict:
    """Analyze the learning curve."""
    early = results[:5]
    late = results[15:]

    early_accuracy = sum(1 for r in early if r["chose_correct"]) / len(early)
    late_accuracy = sum(1 for r in late if r["chose_correct"]) / len(late)
    overall_accuracy = sum(1 for r in results if r["chose_correct"]) / len(results)
    improvement = late_accuracy - early_accuracy
    learned = improvement > 0
    total_tokens = sum(r["tokens_used"] for r in results)

    return {
        "total_iterations": len(results),
        "early_accuracy_1_5": early_accuracy,
        "late_accuracy_15_20": late_accuracy,
        "overall_accuracy": overall_accuracy,
        "improvement": improvement,
        "learned": learned,
        "final_procedural_strategies": results[-1]["procedural_strategies"],
        "final_episodic_memories": results[-1]["episodic_memories"],
        "total_tokens": total_tokens,
        "total_time_s": total_elapsed,
        "avg_time_per_iter_s": total_elapsed / len(results),
        "per_iteration": results,
    }


def print_results(summary: dict) -> None:
    """Pretty-print the benchmark results."""
    print("\n" + "=" * 64)
    print("  FORGE LEARNING BENCHMARK — LIVE (Anthropic API)")
    print("=" * 64)
    print(f"\n  Iterations:           {summary['total_iterations']}")
    print(f"  Early accuracy:       {summary['early_accuracy_1_5']:.0%}  (iterations 1-5)")
    print(f"  Late accuracy:        {summary['late_accuracy_15_20']:.0%}  (iterations 15-20)")
    print(f"  Overall accuracy:     {summary['overall_accuracy']:.0%}")
    print(f"  Improvement:          {summary['improvement']:+.0%}")
    print(f"  Learned:              {'YES' if summary['learned'] else 'NO'}")
    print(f"  Final strategies:     {summary['final_procedural_strategies']}")
    print(f"  Final episodes:       {summary['final_episodic_memories']}")
    print(f"  Total tokens:         {summary['total_tokens']:,}")
    print(f"  Total time:           {summary['total_time_s']:.1f}s")
    print(f"  Avg per iteration:    {summary['avg_time_per_iter_s']:.1f}s")

    print("\n  Learning Curve:")
    print("  " + "-" * 56)
    for r in summary["per_iteration"]:
        marker = "+" if r["chose_correct"] else "x"
        bar = "#" * r["procedural_strategies"]
        print(f"  [{marker}] iter {r['iteration']:2d}  "
              f"chose={r['chosen_action'] or 'none':<20s}  "
              f"correct={r['correct_action']:<18s}  "
              f"strategies={bar}")

    print("\n" + "=" * 64)
    if summary["learned"]:
        print("  PASS — Agent demonstrated measurable learning with real LLM")
    else:
        print("  FAIL — No measurable improvement detected")
    print("=" * 64 + "\n")


async def main():
    print("\n  FORGE Learning Benchmark — Live API")
    print("  Using Claude Haiku 4.5 as reasoning core\n")
    summary = await run_live_benchmark()
    print_results(summary)
    return summary


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result["learned"] else 1)
