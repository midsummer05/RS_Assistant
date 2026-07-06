from __future__ import annotations

import time
from typing import Any, Dict

from rs_agent.agent.state import Artifact, StepState, TaskState
from rs_agent.storage.json_store import JsonFileStore
from rs_agent.tools.registry import ToolRegistry
from rs_agent.tools.schemas import ToolContext, ToolError, ToolResult


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, store: JsonFileStore) -> None:
        self.registry = registry
        self.store = store

    def execute(self, state: TaskState, step: StepState) -> ToolResult:
        if not step.tool_name:
            raise ToolError("unknown", f"Step has no tool_name: {step.step_id}", "missing_tool")
        spec, func = self.registry.get(step.tool_name)
        params = self.resolve_params(state, step.params)
        started = time.perf_counter()
        result = func(ToolContext(task_state=state, store=self.store), params)
        result.tool_name = spec.name
        result.tool_version = spec.version
        result.metrics.setdefault("duration_sec", round(time.perf_counter() - started, 4))
        if not result.ok:
            message = result.error.get("message", "tool failed") if result.error else "tool failed"
            raise ToolError(spec.name, message)
        return result

    def resolve_params(self, state: TaskState, value: Any) -> Any:
        if isinstance(value, str):
            return self._resolve_string(state, value)
        if isinstance(value, list):
            return [self.resolve_params(state, item) for item in value]
        if isinstance(value, dict):
            return {key: self.resolve_params(state, item) for key, item in value.items()}
        return value

    def _resolve_string(self, state: TaskState, value: str) -> Any:
        if value.startswith("$inputs."):
            key = value.split(".", 1)[1]
            return state.constraints.get("inputs", {}).get(key)
        if value.startswith("$artifacts."):
            parts = value.split(".")
            if len(parts) < 3:
                raise KeyError(f"Invalid artifact reference: {value}")
            alias = parts[1]
            field = parts[2]
            artifact: Artifact = state.artifact_by_alias(alias)
            return getattr(artifact, field)
        if value.startswith("$memory."):
            key = value.split(".", 1)[1]
            return state.working_memory.get(key)
        return value

