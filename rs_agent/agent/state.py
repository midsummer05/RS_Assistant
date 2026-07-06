from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


StepStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "skipped",
    "waiting_human",
]
TaskStatus = Literal[
    "created",
    "planning",
    "waiting_human",
    "running",
    "finalizing",
    "succeeded",
    "failed",
    "cancelled",
]
ArtifactType = Literal["raster", "vector", "table", "image", "report", "model", "log"]


class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    type: ArtifactType
    uri: str
    alias: Optional[str] = None
    crs: Optional[str] = None
    bbox: Optional[List[float]] = None
    time_range: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class StepState(BaseModel):
    step_id: str
    name: str
    status: StepStatus = "pending"
    tool_name: Optional[str] = None
    input_refs: List[str] = Field(default_factory=list)
    output_refs: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    expected_outputs: List[str] = Field(default_factory=list)
    quality_gate: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class Plan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    task_type: str
    confidence: float = 0.0
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    steps: List[StepState] = Field(default_factory=list)
    human_checkpoints: List[str] = Field(default_factory=list)
    estimated_resources: Dict[str, Any] = Field(default_factory=dict)
    retrieved_context_refs: List[str] = Field(default_factory=list)


class Interrupt(BaseModel):
    interrupt_id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    reason: str
    status: Literal["open", "approved", "revised", "rejected"] = "open"
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: Optional[datetime] = None


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeChunk(BaseModel):
    chunk_id: str
    title: str
    content: str
    source_type: str
    task_tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    project_id: Optional[str] = None
    memory_type: str
    title: str
    content: str
    confidence: float = 1.0
    source_task_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AssetRecord(BaseModel):
    asset_id: str = Field(default_factory=lambda: str(uuid4()))
    uri: str
    sensor: Optional[str] = None
    acquired_at: Optional[str] = None
    cloud_cover: Optional[float] = None
    crs: Optional[str] = None
    resolution: Optional[float] = None
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TaskState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    project_id: Optional[str] = None
    title: Optional[str] = None
    user_goal: str
    task_type: Optional[str] = None
    status: TaskStatus = "created"
    constraints: Dict[str, Any] = Field(default_factory=dict)
    plan: List[StepState] = Field(default_factory=list)
    plan_summary: Optional[Plan] = None
    artifacts: Dict[str, Artifact] = Field(default_factory=dict)
    artifact_refs: Dict[str, str] = Field(default_factory=dict)
    working_memory: Dict[str, Any] = Field(default_factory=dict)
    retrieved_context: List[KnowledgeChunk] = Field(default_factory=list)
    interrupts: List[Interrupt] = Field(default_factory=list)
    user_feedback: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def is_finished(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}

    def touch(self) -> None:
        self.updated_at = utc_now()

    def next_pending_step(self) -> Optional[StepState]:
        for step in self.plan:
            if step.status == "pending":
                return step
        return None

    def attach_plan(self, plan: Plan) -> None:
        self.task_type = plan.task_type
        self.plan = plan.steps
        self.plan_summary = plan
        self.touch()

    def add_interrupt(self, interrupt: Interrupt) -> None:
        self.interrupts.append(interrupt)
        self.touch()

    def get_open_interrupt(self, interrupt_id: str) -> Optional[Interrupt]:
        for interrupt in self.interrupts:
            if interrupt.interrupt_id == interrupt_id and interrupt.status == "open":
                return interrupt
        return None

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts[artifact.artifact_id] = artifact
        if artifact.alias:
            self.artifact_refs[artifact.alias] = artifact.artifact_id
        self.touch()

    def artifact_by_alias(self, alias: str) -> Artifact:
        artifact_id = self.artifact_refs[alias]
        return self.artifacts[artifact_id]

