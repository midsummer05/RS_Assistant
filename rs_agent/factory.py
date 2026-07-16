from __future__ import annotations

from pathlib import Path

from rs_agent.agent.events import EventBus
from rs_agent.agent.planner import DeterministicPlanner
from rs_agent.agent.runtime import AgentRuntime
from rs_agent.agent.llm_planner import LLMAgentPlanner
from rs_agent.memory.service import MemoryService
from rs_agent.rag.retriever import LocalRagRetriever
from rs_agent.storage.json_store import JsonFileStore
from rs_agent.tools.defaults import build_default_registry
from rs_agent.tools.executor import ToolExecutor
from rs_agent.models.gateway import build_chat_model_from_env


def build_runtime(data_root: str | Path = ".rs_agent_data") -> AgentRuntime:
    store = JsonFileStore(data_root)
    registry = build_default_registry()
    executor = ToolExecutor(registry, store)
    model = build_chat_model_from_env()
    planner = DeterministicPlanner()
    agent_planner = LLMAgentPlanner(model, registry) if model else None
    return AgentRuntime(
        store=store,
        planner=planner,
        agent_planner=agent_planner,
        rag=LocalRagRetriever(knowledge_store=store.knowledge_memory),
        memory=MemoryService(store),
        executor=executor,
        events=EventBus(store),
    )
