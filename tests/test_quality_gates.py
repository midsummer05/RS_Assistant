import numpy as np

from rs_agent.agent.events import EventBus
from rs_agent.agent.planner import DeterministicPlanner
from rs_agent.agent.runtime import AgentRuntime
from rs_agent.agent.state import Plan, StepState
from rs_agent.memory.service import MemoryService
from rs_agent.rag.retriever import LocalRagRetriever
from rs_agent.storage.json_store import JsonFileStore
from rs_agent.tools.executor import ToolExecutor
from rs_agent.tools.registry import ToolRegistry
from rs_agent.tools.schemas import ToolResult, ToolSpec


class FailingGatePlanner(DeterministicPlanner):
    def generate_plan(self, state, context, memories):
        return Plan(
            task_type="change_detection",
            steps=[
                StepState(
                    step_id="gate",
                    name="Failing quality gate",
                    tool_name="quality.fake_gate",
                    quality_gate="fake_quality",
                )
            ],
        )


def test_failed_quality_gate_creates_human_interrupt(tmp_path):
    def fake_gate(context, params):
        return ToolResult(
            tool_name="quality.fake_gate",
            outputs={
                "passed": False,
                "issues": [{"name": "test", "passed": False}],
                "recommendation": "人工检查输入。",
            },
        )

    store = JsonFileStore(tmp_path)
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="quality.fake_gate", description="test gate"),
        fake_gate,
    )
    runtime = AgentRuntime(
        store=store,
        planner=FailingGatePlanner(),
        rag=LocalRagRetriever(),
        memory=MemoryService(store),
        executor=ToolExecutor(registry, store),
        events=EventBus(store),
    )
    state = runtime.create_task(
        user_goal="测试质量门",
        image_t1="demo://image_t1",
        image_t2="demo://image_t2",
        auto_confirm=True,
    )

    assert state.status == "waiting_human"
    assert state.interrupts[-1].type == "quality_gate_review"
    assert state.interrupts[-1].payload["quality_gate"] == "fake_quality"
    event_types = [event.event_type for event in runtime.list_events(state.task_id)]
    assert "QualityGateFailed" in event_types
