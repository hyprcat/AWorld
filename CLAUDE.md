# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWorld (Agent World) is a Python framework for building intelligent agents with rich environments. It provides four core capabilities: Environment Access (MCP-based), Agent Construction, Experience Retrieval (Trajectories), and Model Training.

## Common Commands

```bash
# Installation
pip install -e .

# Run tests
pytest tests/
pytest tests/memory/              # Specific module
pytest tests/test_state_manager.py  # Single file

# Lint (syntax errors only)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# CLI commands
aworld web --port 8000    # Web UI server
aworld api --port 8000    # API server
```

## Environment Variables

```bash
export LLM_MODEL_NAME="gpt-4"           # Model identifier
export LLM_API_KEY="your-key"           # API credentials
export LLM_BASE_URL="https://api.openai.com/v1"  # LLM endpoint
export INVITATION_CODE=""               # Hosted environment access token
export AWORLD_EXTRA=""                  # Module selection during installation
```

## Architecture

### Core Execution Flow
```
Task → Agent → Sandbox (MCP tools) → LLM → Response → Trajectory
```

### Key Modules

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `aworld/runner.py` | Main public API (Runners class) | `sync_run()`, `run()`, `step()`, `evaluate()` |
| `aworld/agents/llm_agent.py` | Agent implementations | `Agent`, `LLMAgent` |
| `aworld/core/agent/swarm.py` | Multi-agent coordination | `Swarm`, `TeamSwarm` |
| `aworld/runners/` | Execution engines | `CallDrivenRunner`, `EventRunner`, `StateManager` |
| `aworld/sandbox/` | Environment + MCP tools | `Sandbox`, `MCPClient` |
| `aworld/core/task.py` | Work unit definition | `Task`, `TaskResponse` |
| `aworld/core/context/` | Agent state management | `Context` |
| `aworld/memory/` | Short/long-term storage | `MemoryFactory`, `MemoryItem` |
| `aworld/models/llm.py` | LLM provider integration | `get_llm_model()`, `acall_llm_model()` |
| `aworld/dataset/` | Trajectory data pipeline | `TrajectoryDataset` |
| `train/` | Training orchestration | `AgentTrainer` |

### Design Patterns

- **Factory Pattern**: `AgentFactory`, `MemoryFactory`, `ToolFactory`, `HookFactory`
- **Event-Driven**: Message bus with event types: AGENT, TOOL, MEMORY, TASK
- **Async-First**: All runners are async (sync wrappers available via `Runners.sync_run()`)
- **MCP Integration**: Model Context Protocol for tool access via `Sandbox`

### Agent Construction Example

```python
from aworld.agents.llm_agent import Agent
from aworld.runner import Runners

agent = Agent(
    name="MyAgent",
    system_prompt="You are helpful.",
    mcp_config={...}  # MCP servers config
)

result = Runners.sync_run(input="Your task", agent=agent)
```

### Single-Step Execution (for training)

```python
from aworld.core.task import Task
from aworld.config import TaskConfig, TaskRunMode

task = Task(
    input="...",
    agent=agent,
    conf=TaskConfig(resp_carry_context=True, run_mode=TaskRunMode.INTERACTIVE)
)

is_finished, observation, response = await Runners.step(task)
```

## Configuration

- `aworld/config/conf.py` - Configuration classes (AgentConfig, TaskConfig, ModelConfig)
- `aworld/requirements.txt` - Dependencies organized by section markers `######### [section] #########`
- Requirements sections: `framework` (required), `optional` (installed but errors ignored)

## Testing

Tests use pytest with pytest-asyncio. Test utilities in `tests/base_test.py` provide:
- `init_agent()` helper for test agent creation
- Mock LLM setup pointing to localhost:1234 by default
