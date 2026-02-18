"""
FORGE LLM Integration — The engine, not the car.

LLMs are the reasoning core. The cognitive architecture is the value.
A mediocre LLM inside a great cognitive architecture outperforms a
brilliant LLM in a naked ReAct loop.

This module provides:
- LLMProvider protocol (any LLM can plug in)
- AnthropicProvider (Claude integration)
- MockLLMProvider (deterministic testing without API calls)
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """The reasoning engine interface. Depend on this, not implementations."""

    async def reason(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        """Generate reasoning/response for the given prompt and context."""
        ...

    async def extract_strategy(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract a generalizable strategy from a set of episodes.

        This is the consolidation step: episodic → procedural memory.
        The LLM looks at multiple similar experiences and finds the pattern.
        Returns: {pattern, actions, rationale, confidence}
        """
        ...

    async def plan(self, goal: str, context: dict[str, Any] | None = None) -> list[str]:
        """Generate a plan (list of action names) to achieve a goal."""
        ...

    async def evaluate(self, state: dict[str, Any], goal: str) -> float:
        """Evaluate how close a state is to achieving a goal. Returns 0-1."""
        ...


class AnthropicProvider:
    """Claude-powered reasoning engine.

    Wraps the Anthropic API for use as FORGE's reasoning core.
    All prompts go through here — the cognitive agent never calls
    the API directly.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929",
                 max_tokens: int = 4096, temperature: float = 0.3):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic()
            except ImportError:
                raise RuntimeError("anthropic package required. Install: pip install anthropic")
        return self._client

    async def reason(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        client = self._get_client()
        system = self._build_system_prompt(context)
        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def extract_strategy(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = (
            "Analyze these episodes and extract a generalizable strategy.\n\n"
            f"Episodes:\n{json.dumps(episodes, indent=2)}\n\n"
            "Return a JSON object with:\n"
            '- "pattern": situation pattern this strategy applies to\n'
            '- "actions": recommended action sequence\n'
            '- "rationale": why this works\n'
            '- "confidence": 0-1 how confident in this generalization\n'
            "Return ONLY the JSON object, no other text."
        )
        text = await self.reason(prompt)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {"pattern": "unknown", "actions": [], "rationale": text, "confidence": 0.3}

    async def plan(self, goal: str, context: dict[str, Any] | None = None) -> list[str]:
        ctx_str = json.dumps(context, indent=2) if context else "None"
        prompt = (
            f"Goal: {goal}\n"
            f"Context: {ctx_str}\n\n"
            "Generate a plan as a JSON array of action names to achieve this goal.\n"
            "Return ONLY the JSON array, no other text."
        )
        text = await self.reason(prompt)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return []

    async def evaluate(self, state: dict[str, Any], goal: str) -> float:
        prompt = (
            f"Goal: {goal}\n"
            f"Current state: {json.dumps(state, indent=2)}\n\n"
            "On a scale of 0.0 to 1.0, how close is the current state to achieving the goal?\n"
            "Return ONLY a number between 0.0 and 1.0."
        )
        text = await self.reason(prompt)
        try:
            return max(0.0, min(1.0, float(text.strip())))
        except ValueError:
            return 0.5

    def _build_system_prompt(self, context: dict[str, Any] | None) -> str:
        base = (
            "You are the reasoning core of a cognitive agent architecture called FORGE. "
            "You think precisely, avoid unnecessary hedging, and produce structured outputs. "
            "When asked for JSON, return ONLY valid JSON — no markdown, no explanation."
        )
        if context:
            base += f"\n\nCurrent context:\n{json.dumps(context, indent=2)}"
        return base


class MockLLMProvider:
    """Deterministic LLM provider for testing.

    Returns predictable outputs based on input patterns.
    No API calls, no cost, fully reproducible.
    """

    def __init__(self):
        self.call_log: list[dict[str, Any]] = []
        self._reason_responses: dict[str, str] = {}
        self._strategy_response: dict[str, Any] | None = None
        self._plan_response: list[str] | None = None
        self._eval_response: float = 0.5

    def set_reason_response(self, pattern: str, response: str) -> None:
        """Set a canned response for prompts matching a pattern."""
        self._reason_responses[pattern] = response

    def set_strategy_response(self, strategy: dict[str, Any]) -> None:
        self._strategy_response = strategy

    def set_plan_response(self, plan: list[str]) -> None:
        self._plan_response = plan

    def set_eval_response(self, score: float) -> None:
        self._eval_response = score

    async def reason(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        self.call_log.append({"method": "reason", "prompt": prompt, "context": context})
        for pattern, response in self._reason_responses.items():
            if pattern.lower() in prompt.lower():
                return response
        return "I'll analyze the situation and take appropriate action."

    async def extract_strategy(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        self.call_log.append({"method": "extract_strategy", "episodes": episodes})
        if self._strategy_response:
            return self._strategy_response
        return {
            "pattern": "task_completion",
            "actions": ["analyze", "execute", "verify"],
            "rationale": f"Extracted from {len(episodes)} episodes",
            "confidence": 0.7,
        }

    async def plan(self, goal: str, context: dict[str, Any] | None = None) -> list[str]:
        self.call_log.append({"method": "plan", "goal": goal, "context": context})
        if self._plan_response:
            return self._plan_response
        return ["analyze_situation", "take_action", "verify_result"]

    async def evaluate(self, state: dict[str, Any], goal: str) -> float:
        self.call_log.append({"method": "evaluate", "state": state, "goal": goal})
        return self._eval_response
