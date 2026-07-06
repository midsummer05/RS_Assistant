from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from rs_agent.agent.state import Artifact, TaskState
from rs_agent.storage.json_store import JsonFileStore


class ToolSpec(BaseModel):
    name: str
    version: str = "0.1.0"
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    resource_profile: Dict[str, Any] = Field(default_factory=dict)
    safety_level: str = "normal"


class ToolResult(BaseModel):
    ok: bool = True
    tool_name: str
    tool_version: str = "0.1.0"
    outputs: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Artifact] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


class ToolError(RuntimeError):
    def __init__(self, tool_name: str, message: str, code: str = "tool_error") -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.code = code
        self.message = message


@dataclass
class ToolContext:
    task_state: TaskState
    store: JsonFileStore

    def artifact_path(self, subdir: str, filename: str):
        return self.store.artifact_path(self.task_state.task_id, subdir, filename)


ToolCallable = Callable[[ToolContext, Dict[str, Any]], ToolResult]

