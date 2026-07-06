from __future__ import annotations

from typing import Dict, Iterable, Tuple

from rs_agent.tools.schemas import ToolCallable, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tuple[ToolSpec, ToolCallable]] = {}

    def register(self, spec: ToolSpec, func: ToolCallable) -> None:
        self._tools[spec.name] = (spec, func)

    def get(self, name: str) -> Tuple[ToolSpec, ToolCallable]:
        if name not in self._tools:
            raise KeyError(f"Tool is not registered: {name}")
        return self._tools[name]

    def specs(self) -> Iterable[ToolSpec]:
        return [item[0] for item in self._tools.values()]

