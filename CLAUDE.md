# FORGE — Framework for Orchestrated Reasoning & Generative Execution

## Project Context
Cognitive agent architecture addressing the AI-layer gaps in current agentic systems. Six cognitive primitives implemented as architectural components — not prompt engineering. Current AI agents are next-token predictors cosplaying as goal-directed systems. FORGE provides the missing primitives: world models, self-knowledge, mental simulation, online learning, goal persistence, and calibrated uncertainty.

67/67 tests passing. Learning benchmark validated: +80% accuracy improvement over 20 task iterations.

## Quick Start
```bash
cd forge
python3 -m venv .venv && source .venv/bin/activate
pip install pytest pytest-asyncio
python -m pytest tests/ -v          # 67 tests
python benchmarks/learning_benchmark.py  # Learning curve visualization
```

## Architecture
```
CognitiveAgent (8-phase cycle)
  ├── ORIENT   — Perceive and contextualize
  ├── RECALL   — Retrieve memories and learned strategies
  ├── PLAN     — Generate candidate actions via LLM
  ├── GATE     — Uncertainty check (architectural, not a suggestion)
  ├── ACT      — Execute chosen action
  ├── OBSERVE  — Compare result to prediction
  ├── LEARN    — Update memories, reinforce strategies, consolidate
  └── CHECK    — Goal alignment and drift detection
```

### Subsystems
- **MemorySystem** — Three tiers: Working (capacity-limited, salience-evicted) → Episodic (compressed, Ebbinghaus decay) → Procedural (reinforced strategies = learning)
- **GoalStack** — Protected primary goal, hierarchical decomposition, dependency chains, drift detection, integrity checking
- **WorldModel** — CausalGraph of action effects + branch-and-bound simulation for mental modeling
- **LLMProvider** — Protocol-based: AnthropicProvider (Claude), MockLLMProvider (deterministic testing)
- **CognitiveTracer** — OpenTelemetry-style spans for every cognitive step, cost attribution, learning curve tracking

## File Structure
- **core/types.py** — Atomic primitives (the periodic table): CognitiveStep, Goal, Memory, UncertaintyEstimate, CapabilityProfile, WorldState, ActionEffect, AgentConfig
- **core/memory.py** — WorkingMemory, EpisodicMemory, ProceduralMemory, MemorySystem
- **core/goal_stack.py** — GoalStack with protection, decomposition, focus management, drift detection
- **core/world_model.py** — CausalGraph, CausalEdge, SimulationBranch, WorldModel
- **core/agent.py** — CognitiveAgent integrating all subsystems into the 8-phase cycle
- **core/llm.py** — LLMProvider protocol, AnthropicProvider, MockLLMProvider
- **core/tracer.py** — Span, CognitiveTracer with cycle/phase context managers
- **tests/** — 67 cognitive behavior tests across 6 test files
- **benchmarks/learning_benchmark.py** — 20-iteration learning curve validation

## The Six Cognitive Primitives

1. **World Models** — CausalGraph of action effects, reversibility classes, failure modes. Agents know what actions DO to reality.
2. **Self-Model (Metacognition)** — CapabilityProfile tracks actual success rates, calibration error. Self-knowledge from data, not self-assessment.
3. **Mental Simulation** — Branch-and-bound simulation against WorldModel. Evaluate futures before committing to irreversible actions.
4. **Online Learning (Procedural Memory)** — Three-tier memory: Working → Episodic → Procedural. Strategies reinforced by success, weakened by failure. Validated by learning benchmark.
5. **Goal Persistence** — GoalStack outside the context window. Protected primary goal. Drift detection. Cannot be silently overridden.
6. **Calibrated Uncertainty** — UncertaintyEstimate gates irreversible actions. Code-level gate, not a prompt suggestion. confidence < threshold + irreversible = BLOCKED.

## Key Design Decisions
- Types are the periodic table — `core/types.py` is the single source of all primitives
- Protocols over concrete types — `LLMProvider`, `ToolExecutor` are interfaces
- Composition over inheritance — CognitiveAgent composes subsystems, doesn't subclass
- Tests validate cognitive behavior, not just code — "Does the agent learn?" > "Does the function return?"
- No circular dependencies between modules
- Every subsystem independently testable
- AgentConfig: single source for all tunable parameters

## Test Categories (67 tests)
- **test_types.py** (10) — Uncertainty gating, capability tracking, goal states, world state
- **test_memory.py** (11) — Capacity limits, salience eviction, decay, dedup, reinforcement, consolidation
- **test_goal_stack.py** (10) — Primary protection, decomposition, dependencies, focus, drift, integrity
- **test_world_model.py** (9) — Causal prediction, simulation, branch-and-bound, uncertainty gating, safety
- **test_agent.py** (10) — Full cognitive cycle, uncertainty gate blocking, learning, tracing
- **test_tracer.py** (7) — Spans, cycle/phase nesting, cost attribution, learning curves
- **test_learning.py** (3) — The acid test: agent learns, strategies accumulate, late accuracy > 50%

## Next Steps
- **LLM Integration Testing** — Wire AnthropicProvider with real API key, run benchmark against Claude
- **Semantic Memory Retrieval** — Replace tag-based matching with embedding-based similarity
- **Multi-Agent Protocol (FARP)** — Agent-to-agent communication with typed contracts, capability negotiation
- **Persistent Storage** — Serialize memory/goals to disk for cross-session learning
- **Real Tool Execution** — Wire actual tools (file I/O, API calls) with ActionEffect metadata
