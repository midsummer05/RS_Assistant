from rs_agent.factory import build_runtime
from rs_agent.agent.llm_planner import LLMAgentPlanner, LLMPlanDecision
from rs_agent.agent.planner import DeterministicPlanner
from rs_agent.agent.runtime import AgentRuntime
from rs_agent.agent.events import EventBus
from rs_agent.memory.service import MemoryService
from rs_agent.rag.retriever import LocalRagRetriever
from rs_agent.storage.json_store import JsonFileStore
from rs_agent.tools.defaults import build_default_registry
from rs_agent.tools.executor import ToolExecutor


class FakeChatModel:
    model_name = "fake-structured-model"

    def generate_json(self, system_prompt, user_prompt, response_model):
        assert response_model is LLMPlanDecision
        return LLMPlanDecision(
            task_type="change_detection",
            confidence=0.9,
            assumptions=["输入由元数据工具检查"],
            risks=["季节差异可能造成假变化"],
            step_ids=[
                "s01_metadata",
                "s02_input_quality",
                "s03_align_pair",
                "s04_alignment_quality",
                "s05_ndbi_t1",
                "s06_ndbi_t2",
                "s07_detect_change",
                "s08_result_quality",
                "s09_filter_small_regions",
                "s10_raster_to_vector",
                "s11_area_statistics",
                "s12_quicklook",
                "s13_report",
            ],
            parameter_overrides={"s07_detect_change": {"threshold": 0.22}},
            reasoning_summary="选择完整变化检测闭环。",
        )


class FlakyChatModel(FakeChatModel):
    def __init__(self):
        self.calls = 0

    def generate_json(self, system_prompt, user_prompt, response_model):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary model failure")
        return super().generate_json(system_prompt, user_prompt, response_model)


def test_execution_budget_pauses_and_resumes_from_checkpoint(tmp_path):
    runtime = build_runtime(tmp_path)
    state = runtime.create_task(
        user_goal="分析两期影像变化并生成报告",
        image_t1="demo://image_t1",
        image_t2="demo://image_t2",
        auto_confirm=True,
        execution_budget=3,
    )

    assert state.status == "paused"
    assert len([step for step in state.plan if step.status == "succeeded"]) == 3

    while state.status == "paused":
        state = runtime.run_task(state.task_id)

    assert state.status == "succeeded"
    event_types = [event.event_type for event in runtime.list_events(state.task_id)]
    assert "TaskPaused" in event_types
    assert "TaskResumed" in event_types


def test_llm_agent_plans_with_allowlisted_steps(tmp_path):
    store = JsonFileStore(tmp_path)
    registry = build_default_registry()
    runtime = AgentRuntime(
        store=store,
        planner=DeterministicPlanner(),
        agent_planner=LLMAgentPlanner(FakeChatModel(), registry),
        rag=LocalRagRetriever(),
        memory=MemoryService(store),
        executor=ToolExecutor(registry, store),
        events=EventBus(store),
    )
    state = runtime.create_task(
        user_goal="自主规划并完成两期建设用地变化分析",
        image_t1="demo://image_t1",
        image_t2="demo://image_t2",
        auto_confirm=False,
        agent_mode="agent",
    )

    assert state.status == "waiting_human"
    assert state.working_memory["llm_planning"]["model"] == "fake-structured-model"
    detect_step = next(step for step in state.plan if step.step_id == "s07_detect_change")
    assert detect_step.params["threshold"] == 0.22


def test_failed_llm_planning_can_be_retried(tmp_path):
    store = JsonFileStore(tmp_path)
    registry = build_default_registry()
    runtime = AgentRuntime(
        store=store,
        planner=DeterministicPlanner(),
        agent_planner=LLMAgentPlanner(FlakyChatModel(), registry),
        rag=LocalRagRetriever(),
        memory=MemoryService(store),
        executor=ToolExecutor(registry, store),
        events=EventBus(store),
    )
    failed = runtime.create_task(
        user_goal="规划变化检测任务",
        image_t1="demo://image_t1",
        image_t2="demo://image_t2",
        auto_confirm=False,
        agent_mode="agent",
    )

    assert failed.status == "failed"
    assert failed.working_memory["last_failure"]["stage"] == "planning"

    retried = runtime.retry_task(failed.task_id)

    assert retried.status == "waiting_human"
    event_types = [event.event_type for event in runtime.list_events(retried.task_id)]
    assert "LLMCallFailed" in event_types
    assert "TaskRetryRequested" in event_types
    assert "LLMCallSucceeded" in event_types
