from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from rs_agent.agent.state import Artifact
from rs_agent.tools.raster.io import checksum_file
from rs_agent.tools.schemas import ToolContext, ToolResult


def generate_markdown(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    state = context.task_state
    output_alias = params.get("output_alias", "markdown_report")
    area_summary = state.working_memory.get("area_statistics", {}).get("summary", {})
    quality = state.working_memory.get("quality", {})
    citations = [
        f"- `{chunk.chunk_id}`：{chunk.title}（{chunk.metadata.get('citation', chunk.source_type)}）"
        for chunk in state.retrieved_context
    ]
    artifact_lines = [
        f"- `{artifact.alias or artifact.artifact_id}` ({artifact.type}): `{artifact.uri}`"
        for artifact in state.artifacts.values()
    ]
    step_lines = [
        f"- {step.step_id} {step.name}: {step.status}"
        for step in state.plan
    ]
    text = f"""# 遥感变化检测报告

## 任务目标

{state.user_goal}

## 方法摘要

系统按最小闭环执行：读取元数据、双时相对齐、计算 NDBI、使用规则型建设用地扩张模型进行变化检测、过滤小图斑、矢量化、统计面积并生成预览图。

## 面积统计

- 变化类型：{area_summary.get('class', 'built_up_expansion')}
- 图斑数量：{area_summary.get('polygon_count', 0)}
- 面积：{area_summary.get('area_m2', 0)} 平方米 / {area_summary.get('area_ha', 0)} 公顷

## 质量检查

- 是否通过：{quality.get('passed', 'pending')}
- 评分：{quality.get('score', 'pending')}
- 说明：{quality.get('details', '报告生成时质量信息由运行时写入。')}

## 执行步骤

{chr(10).join(step_lines)}

## 产物清单

{chr(10).join(artifact_lines)}

## 检索依据

{chr(10).join(citations)}

## 限制说明

当前 MVP 使用轻量规则模型和本地文件存储，适合验证闭环；生产环境应接入严格的重投影/重采样、云掩膜、模型服务、空间数据库和人工质检流程。
"""
    path = context.artifact_path("outputs", f"{output_alias}.md")
    Path(path).write_text(text, encoding="utf-8")
    artifact = Artifact(
        artifact_id=f"art_{uuid4().hex[:12]}",
        type="report",
        alias=output_alias,
        uri=str(Path(path).resolve()),
        checksum=checksum_file(path),
        metadata={"format": "markdown", "task_id": state.task_id},
    )
    return ToolResult(
        tool_name="report.generate_markdown",
        outputs={"artifact_id": artifact.artifact_id, "uri": artifact.uri},
        artifacts=[artifact],
        logs=["generated markdown report"],
    )

