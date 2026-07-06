from __future__ import annotations

from pathlib import Path

from rs_agent.agent.events import EventBus
from rs_agent.agent.planner import DeterministicPlanner
from rs_agent.agent.runtime import AgentRuntime
from rs_agent.memory.service import MemoryService
from rs_agent.rag.retriever import LocalRagRetriever
from rs_agent.storage.json_store import JsonFileStore
from rs_agent.tools.defaults import build_default_registry
from rs_agent.tools.executor import ToolExecutor


def build_runtime(data_root: str | Path = ".rs_agent_data") -> AgentRuntime:
    store = JsonFileStore(data_root)
    registry = build_default_registry()
    executor = ToolExecutor(registry, store)
    return AgentRuntime(
        store=store,
        planner=DeterministicPlanner(),
        rag=LocalRagRetriever(),
        memory=MemoryService(store),
        executor=executor,
        events=EventBus(store),
    )

