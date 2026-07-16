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
    input_quality = state.working_memory.get("input_quality", {})
    alignment_quality = state.working_memory.get("alignment_quality", {})
    result_quality = state.working_memory.get("change_result_quality", {})
    change_artifact = None
    if "change_raster" in state.artifact_refs:
        change_artifact = state.artifact_by_alias("change_raster")
    model_id = change_artifact.metadata.get("model_id") if change_artifact else "unknown"
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

系统按长程变化检测流程执行：读取元数据、输入适用性诊断、双时相对齐、配准质量检查、特征构建、模型推理、结果合理性检查、后处理、矢量化、统计和预览。

- 变化检测模型：{model_id}
- 输入质量门：{input_quality.get('passed', 'pending')}
- 配准质量门：{alignment_quality.get('passed', 'pending')}
- 配准相关性：{alignment_quality.get('correlation', 'pending')}
- 结果质量门：{result_quality.get('passed', 'pending')}
- 原始变化比例：{result_quality.get('changed_ratio', 'pending')}

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

当前系统已加入质量门和 BIT 深度模型，但 `raster.align_pair` 仍采用公共尺寸裁剪策略。生产环境仍应接入严格的重投影、亚像素配准、云/云影掩膜、分布外检测和抽样精度验证。
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
