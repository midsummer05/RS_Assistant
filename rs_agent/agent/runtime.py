from __future__ import annotations

from typing import Any, Dict, List, Optional

from rs_agent.agent.events import EventBus
from rs_agent.agent.planner import DeterministicPlanner
from rs_agent.agent.state import Interrupt, TaskState, utc_now
from rs_agent.memory.service import MemoryService
from rs_agent.rag.retriever import LocalRagRetriever
from rs_agent.storage.json_store import JsonFileStore
from rs_agent.tools.executor import ToolExecutor
from rs_agent.tools.schemas import ToolError, ToolResult


class AgentRuntime:
    def __init__(
        self,
        store: JsonFileStore,
        planner: DeterministicPlanner,
        rag: LocalRagRetriever,
        memory: MemoryService,
        executor: ToolExecutor,
        events: EventBus,
    ) -> None:
        self.store = store
        self.planner = planner
        self.rag = rag
        self.memory = memory
        self.executor = executor
        self.events = events

    def create_task(
        self,
        user_goal: str,
        image_t1: str,
        image_t2: str,
        user_id: str = "local_user",
        project_id: Optional[str] = "default_project",
        auto_confirm: bool = False,
    ) -> TaskState:
        state = TaskState(
            user_id=user_id,
            project_id=project_id,
            title=self._title_from_goal(user_goal),
            user_goal=user_goal,
            constraints={
                "inputs": {
                    "image_t1": image_t1,
                    "image_t2": image_t2,
                },
                "requires_human_review": not auto_confirm,
            },
        )
        self.store.save_task(state)
        self.events.emit(
            state.task_id,
            "TaskCreated",
            {"user_goal": user_goal, "inputs": state.constraints["inputs"]},
        )
        return self.run_task(state.task_id)

    def run_task(self, task_id: str) -> TaskState:
        state = self.store.load_latest_checkpoint(task_id)

        while not state.is_finished:
            self.store.save_checkpoint(state, f"status_{state.status}")

            if state.status == "created":
                state = self.planner.understand(state)
                state.status = "planning"
                self.events.emit(
                    task_id,
                    "UserGoalUnderstood",
                    {"task_type": state.task_type, "goal": state.user_goal},
                )
                self.store.save_task(state)
                continue

            if state.status == "planning":
                context = self.rag.retrieve_for_planning(state)
                memories = self.memory.retrieve_relevant(state)
                state.retrieved_context = context
                state.working_memory["retrieved_memory_count"] = len(memories)
                self.events.emit(
                    task_id,
                    "ContextRetrieved",
                    {
                        "chunks": [chunk.model_dump(mode="json") for chunk in context],
                        "memory_count": len(memories),
                    },
                )
                plan = self.planner.generate_plan(state, context, memories)
                state.attach_plan(plan)
                self.events.emit(
                    task_id,
                    "PlanGenerated",
                    {
                        "plan_id": plan.plan_id,
                        "steps": [step.model_dump(mode="json") for step in plan.steps],
                        "human_checkpoints": plan.human_checkpoints,
                    },
                )
                if state.constraints.get("requires_human_review", True):
                    interrupt = Interrupt(
                        type="plan_review",
                        reason="请确认结构化变化检测计划后再执行工具链。",
                        payload={"plan": plan.model_dump(mode="json")},
                    )
                    state.add_interrupt(interrupt)
                    state.status = "waiting_human"
                    self.events.emit(
                        task_id,
                        "HumanInterruptCreated",
                        {"interrupt_id": interrupt.interrupt_id, "type": interrupt.type},
                    )
                    self.store.save_checkpoint(state, "waiting_plan_review")
                    self.store.save_task(state)
                    return state
                state.status = "running"
                self.store.save_task(state)
                continue

            if state.status == "waiting_human":
                self.store.save_task(state)
                return state

            if state.status == "running":
                step = state.next_pending_step()
                if step is None:
                    state.status = "finalizing"
                    self.store.save_task(state)
                    continue
                if step.tool_name == "report.generate_markdown" and "quality" not in state.working_memory:
                    state.working_memory["quality"] = self.evaluate_quality(state)

                try:
                    self._execute_step(state, step.step_id)
                except ToolError as exc:
                    state.status = "failed"
                    self.events.emit(
                        task_id,
                        "ToolCallFailed",
                        {"tool_name": exc.tool_name, "code": exc.code, "message": exc.message},
                    )
                    self.store.save_checkpoint(state, f"failed_{step.step_id}")
                    self.store.save_task(state)
                    return state
                continue

            if state.status == "finalizing":
                state.working_memory.setdefault("quality", self.evaluate_quality(state))
                memory = self.memory.write_from_task(state)
                self.events.emit(
                    task_id,
                    "MemoryWritten",
                    {"memory_id": memory.memory_id, "memory_type": memory.memory_type},
                )
                state.status = "succeeded"
                self.events.emit(
                    task_id,
                    "TaskFinalized",
                    {"status": state.status, "artifact_count": len(state.artifacts)},
                )
                self.store.save_checkpoint(state, "succeeded")
                self.store.save_task(state)
                return state

        self.store.save_task(state)
        return state

    def approve_interrupt(self, task_id: str, interrupt_id: str) -> TaskState:
        state = self.store.load_task(task_id)
        interrupt = state.get_open_interrupt(interrupt_id)
        if not interrupt:
            raise KeyError(f"Open interrupt not found: {interrupt_id}")
        interrupt.status = "approved"
        interrupt.resolved_at = utc_now()
        state.status = "running"
        self.events.emit(task_id, "HumanInterruptApproved", {"interrupt_id": interrupt_id})
        self.store.save_checkpoint(state, "plan_approved")
        self.store.save_task(state)
        return self.run_task(task_id)

    def submit_feedback(
        self,
        task_id: str,
        rating: int,
        comment: str,
        accepted: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskState:
        state = self.store.load_task(task_id)
        feedback = {
            "rating": rating,
            "comment": comment,
            "accepted": accepted,
            "metadata": metadata or {},
            "created_at": utc_now().isoformat(),
        }
        state.user_feedback.append(feedback)
        memory = self.memory.write_feedback(state, rating, comment, accepted, metadata)
        self.events.emit(
            task_id,
            "FeedbackReceived",
            {"rating": rating, "accepted": accepted, "memory_id": memory.memory_id},
        )
        self.store.save_task(state)
        return state

    def list_events(self, task_id: str):
        return self.store.list_events(task_id)

    def _execute_step(self, state: TaskState, step_id: str) -> ToolResult:
        step = next(step for step in state.plan if step.step_id == step_id)
        step.status = "running"
        step.started_at = utc_now()
        self.events.emit(
            state.task_id,
            "ToolCallStarted",
            {"step_id": step.step_id, "tool_name": step.tool_name, "params": step.params},
        )
        self.store.save_checkpoint(state, f"before_{step.step_id}")
        result = self.executor.execute(state, step)
        step.status = "succeeded"
        step.finished_at = utc_now()
        for artifact in result.artifacts:
            state.add_artifact(artifact)
            step.output_refs.append(artifact.artifact_id)
        state.working_memory[step.step_id] = result.outputs
        self._promote_step_outputs(state, step.step_id, result.outputs)
        self.events.emit(
            state.task_id,
            "ToolCallSucceeded",
            {
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "outputs": result.outputs,
                "artifact_ids": [artifact.artifact_id for artifact in result.artifacts],
                "metrics": result.metrics,
            },
        )
        self.store.save_checkpoint(state, f"after_{step.step_id}")
        self.store.save_task(state)
        return result

    def _promote_step_outputs(self, state: TaskState, step_id: str, outputs: Dict[str, Any]) -> None:
        if step_id == "s01_metadata":
            state.working_memory["metadata_t1"] = outputs.get("metadata_t1")
            state.working_memory["metadata_t2"] = outputs.get("metadata_t2")
        if step_id == "s08_area_statistics":
            state.working_memory["area_statistics"] = outputs

    def evaluate_quality(self, state: TaskState) -> Dict[str, Any]:
        metadata_t1 = state.working_memory.get("metadata_t1") or {}
        metadata_t2 = state.working_memory.get("metadata_t2") or {}
        area_summary = state.working_memory.get("area_statistics", {}).get("summary", {})
        checks: List[Dict[str, Any]] = []
        same_crs = metadata_t1.get("crs") == metadata_t2.get("crs")
        checks.append({"name": "metadata_crs_consistency", "passed": same_crs, "details": "两期 CRS 一致" if same_crs else "两期 CRS 不一致"})
        has_vector = "change_vector" in state.artifact_refs
        checks.append({"name": "vector_output_exists", "passed": has_vector, "details": "已生成变化矢量"})
        has_report = "markdown_report" in state.artifact_refs or any(step.tool_name == "report.generate_markdown" for step in state.plan)
        checks.append({"name": "report_planned", "passed": has_report, "details": "已规划或生成报告"})
        area_non_negative = float(area_summary.get("area_m2", 0.0) or 0.0) >= 0
        checks.append({"name": "area_non_negative", "passed": area_non_negative, "details": "面积统计非负"})
        passed_count = sum(1 for check in checks if check["passed"])
        score = round(passed_count / len(checks), 2)
        return {
            "passed": score >= 0.75,
            "score": score,
            "checks": checks,
            "issues": [check for check in checks if not check["passed"]],
            "details": "MVP 自动质检完成，重点检查元数据一致性、矢量产物、报告和面积统计。",
        }

    def _title_from_goal(self, user_goal: str) -> str:
        return user_goal[:40] or "遥感变化检测任务"

