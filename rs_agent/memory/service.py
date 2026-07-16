from __future__ import annotations

from typing import Any, Dict, List

from rs_agent.agent.state import MemoryRecord, TaskState
from rs_agent.storage.json_store import JsonFileStore


class MemoryService:
    def __init__(self, store: JsonFileStore) -> None:
        self.store = store

    def retrieve_relevant(self, state: TaskState, limit: int = 5) -> List[MemoryRecord]:
        tags = [state.task_type or "change_detection"]
        return self.store.search_memories(
            query=state.user_goal,
            user_id=state.user_id,
            project_id=state.project_id,
            tags=tags,
            limit=limit,
        )

    def write_from_task(self, state: TaskState) -> MemoryRecord:
        area_summary = state.working_memory.get("area_statistics", {})
        quality = state.working_memory.get("quality", {})
        content = (
            f"任务 {state.task_id} 完成 {state.task_type}。"
            f"质量评分：{quality.get('score', 'unknown')}；"
            f"面积统计：{area_summary.get('summary', area_summary)}。"
        )
        memory = MemoryRecord(
            user_id=state.user_id,
            project_id=state.project_id,
            memory_type="task_summary",
            title="变化检测任务结果摘要",
            content=content,
            confidence=0.82,
            scope="project",
            importance=0.65,
            source_task_id=state.task_id,
            tags=[state.task_type or "change_detection", "task_summary"],
            metadata={
                "artifact_refs": state.artifact_refs,
                "quality": quality,
                "input_quality": state.working_memory.get("input_quality"),
                "alignment_quality": state.working_memory.get("alignment_quality"),
                "change_result_quality": state.working_memory.get("change_result_quality"),
                "model_id": self._model_id(state),
            },
        )
        return self.store.save_memory(memory)

    def write_feedback(
        self,
        state: TaskState,
        rating: int,
        comment: str,
        accepted: bool | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> MemoryRecord:
        content = f"用户反馈评分 {rating}/5。"
        if accepted is not None:
            content += f" 结果验收：{'通过' if accepted else '未通过'}。"
        if comment:
            content += f" 反馈内容：{comment}"
        memory = MemoryRecord(
            user_id=state.user_id,
            project_id=state.project_id,
            memory_type="result_feedback",
            title="用户对变化检测结果的反馈",
            content=content,
            confidence=1.0,
            scope="project",
            importance=0.9 if accepted is False else 0.75,
            source_task_id=state.task_id,
            tags=[state.task_type or "change_detection", "feedback"],
            metadata=metadata or {},
        )
        return self.store.save_memory(memory)

    def write_human_decision(
        self,
        state: TaskState,
        decision_type: str,
        reason: str,
        payload: Dict[str, Any],
    ) -> MemoryRecord:
        memory = MemoryRecord(
            user_id=state.user_id,
            project_id=state.project_id,
            memory_type="human_decision",
            title=f"人工决策：{decision_type}",
            content=f"任务 {state.task_id} 中，用户批准继续执行。原因：{reason}",
            confidence=1.0,
            source_task_id=state.task_id,
            tags=[state.task_type or "change_detection", "human_decision", decision_type],
            metadata=payload,
            scope="project",
            importance=0.9,
        )
        return self.store.save_memory(memory)

    def _model_id(self, state: TaskState) -> str | None:
        if "change_raster" not in state.artifact_refs:
            return None
        return state.artifact_by_alias("change_raster").metadata.get("model_id")
