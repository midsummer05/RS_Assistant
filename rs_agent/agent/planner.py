from __future__ import annotations

from typing import List

from rs_agent.agent.state import KnowledgeChunk, MemoryRecord, Plan, StepState, TaskState


class DeterministicPlanner:
    """Rule-based MVP planner for the first recommended closed loop."""

    def understand(self, state: TaskState) -> TaskState:
        text = state.user_goal.lower()
        if "变化" in text or "change" in text:
            state.task_type = "change_detection"
        else:
            state.task_type = "change_detection"
            state.working_memory["task_type_note"] = "MVP 当前默认进入变化检测闭环。"
        return state

    def generate_plan(
        self,
        state: TaskState,
        context: List[KnowledgeChunk],
        memories: List[MemoryRecord],
    ) -> Plan:
        inputs = state.constraints.get("inputs", {})
        missing = []
        if not inputs.get("image_t1"):
            missing.append("image_t1")
        if not inputs.get("image_t2"):
            missing.append("image_t2")

        threshold = self._threshold_from_context(context)
        min_area_m2 = self._min_area_from_context(context)
        use_bit = self._use_bit_model(state)
        context_refs = [chunk.chunk_id for chunk in context]
        memory_notes = [memory.content for memory in memories]

        steps = [
            StepState(
                step_id="s01_metadata",
                name="读取两期影像元数据",
                tool_name="raster.inspect_metadata",
                params={"raster_uris": ["$inputs.image_t1", "$inputs.image_t2"]},
                expected_outputs=["metadata_t1", "metadata_t2"],
                quality_gate="metadata_valid",
            ),
            StepState(
                step_id="s02_input_quality",
                name="评估双时相输入适用性",
                tool_name="quality.assess_change_inputs",
                params={
                    "raster_t1": "$inputs.image_t1",
                    "raster_t2": "$inputs.image_t2",
                    "minimum_bands": 3,
                    "minimum_finite_ratio": 0.95,
                },
                expected_outputs=["passed", "checks", "recommendation"],
                quality_gate="input_suitability",
            ),
            StepState(
                step_id="s03_align_pair",
                name="统一空间参考、分辨率和像元网格",
                tool_name="raster.align_pair",
                params={
                    "raster_t1": "$inputs.image_t1",
                    "raster_t2": "$inputs.image_t2",
                    "resampling": "bilinear",
                },
                expected_outputs=["aligned_t1", "aligned_t2"],
                quality_gate="pair_aligned",
            ),
            StepState(
                step_id="s04_alignment_quality",
                name="检查双时相配准质量",
                tool_name="quality.assess_alignment",
                params={
                    "raster_t1": "$artifacts.aligned_t1.uri",
                    "raster_t2": "$artifacts.aligned_t2.uri",
                    "minimum_correlation": 0.35,
                },
                expected_outputs=["passed", "correlation", "checks"],
                quality_gate="alignment_quality",
            ),
            StepState(
                step_id="s05_ndbi_t1",
                name="计算第一期 NDBI 指数",
                tool_name="raster.calculate_index",
                params={
                    "raster_uri": "$artifacts.aligned_t1.uri",
                    "index_name": "NDBI",
                    "band_mapping": "auto",
                    "output_alias": "ndbi_t1",
                },
                expected_outputs=["ndbi_t1"],
            ),
            StepState(
                step_id="s06_ndbi_t2",
                name="计算第二期 NDBI 指数",
                tool_name="raster.calculate_index",
                params={
                    "raster_uri": "$artifacts.aligned_t2.uri",
                    "index_name": "NDBI",
                    "band_mapping": "auto",
                    "output_alias": "ndbi_t2",
                },
                expected_outputs=["ndbi_t2"],
            ),
            StepState(
                step_id="s07_detect_change",
                name="使用 BIT 深度模型执行变化检测" if use_bit else "执行建设用地扩张变化检测",
                tool_name="ml.bit_change_detection" if use_bit else "ml.detect_change",
                params=(
                    {
                        "raster_t1": "$artifacts.aligned_t1.uri",
                        "raster_t2": "$artifacts.aligned_t2.uri",
                        "tile_size": 256,
                        "overlap": 32,
                        "batch_size": 4,
                        "device": "auto",
                        "output_alias": "change_raster",
                    }
                    if use_bit
                    else {
                        "raster_t1": "$artifacts.aligned_t1.uri",
                        "raster_t2": "$artifacts.aligned_t2.uri",
                        "feature_t1": "$artifacts.ndbi_t1.uri",
                        "feature_t2": "$artifacts.ndbi_t2.uri",
                        "model_id": "builtup_ndbi_delta_v0",
                        "threshold": threshold,
                        "target": "built_up_expansion",
                        "output_alias": "change_raster",
                    }
                ),
                expected_outputs=["change_raster"],
                quality_gate="change_mask_reasonable",
            ),
            StepState(
                step_id="s08_result_quality",
                name="评估变化结果合理性",
                tool_name="quality.assess_change_result",
                params={
                    "change_raster": "$artifacts.change_raster.uri",
                    "minimum_changed_ratio": 0.0001,
                    "maximum_changed_ratio": 0.6,
                    "maximum_components": 100000,
                },
                expected_outputs=["passed", "changed_ratio", "component_count", "checks"],
                quality_gate="change_result_quality",
            ),
            StepState(
                step_id="s09_filter_small_regions",
                name="过滤小图斑",
                tool_name="post.filter_small_regions",
                params={
                    "raster_uri": "$artifacts.change_raster.uri",
                    "min_area_m2": min_area_m2,
                    "output_alias": "change_filtered",
                },
                expected_outputs=["change_filtered"],
            ),
            StepState(
                step_id="s10_raster_to_vector",
                name="变化栅格转矢量图斑",
                tool_name="post.raster_to_vector",
                params={
                    "raster_uri": "$artifacts.change_filtered.uri",
                    "output_alias": "change_vector",
                },
                expected_outputs=["change_vector"],
            ),
            StepState(
                step_id="s11_area_statistics",
                name="统计变化面积",
                tool_name="post.area_statistics",
                params={
                    "vector_uri": "$artifacts.change_vector.uri",
                    "raster_uri": "$artifacts.change_filtered.uri",
                    "output_alias": "area_statistics",
                },
                expected_outputs=["area_statistics"],
            ),
            StepState(
                step_id="s12_quicklook",
                name="生成变化结果预览图",
                tool_name="viz.change_overlay",
                params={
                    "base_raster": "$artifacts.aligned_t2.uri",
                    "change_raster": "$artifacts.change_filtered.uri",
                    "output_alias": "change_preview",
                },
                expected_outputs=["change_preview"],
            ),
            StepState(
                step_id="s13_report",
                name="生成 Markdown 报告",
                tool_name="report.generate_markdown",
                params={"output_alias": "markdown_report"},
                expected_outputs=["markdown_report"],
                quality_gate="report_has_artifacts",
            ),
        ]

        assumptions = [
            "输入为两期可读取的多光谱遥感影像或 demo 影像。",
            "元数据必须由 raster.inspect_metadata 工具读取，计划不直接猜测 CRS、分辨率或波段。",
            "若两期尺寸不一致，MVP align_pair 会裁剪到公共尺寸；生产阶段应替换为严格重投影和重采样。",
        ]
        if memory_notes:
            assumptions.append("已检索到项目记忆，规划时参考历史反馈和参数经验。")

        return Plan(
            task_type="change_detection",
            confidence=0.86 if not missing else 0.62,
            missing_information=missing,
            assumptions=assumptions,
            risks=[
                "高云量、季节差异或配准误差可能造成假变化。",
                (
                    "BIT_LEVIR 主要面向建筑物变化，跨传感器、跨区域使用前需要验证泛化精度。"
                    if use_bit
                    else "当前 MVP 使用规则型 NDBI 差分模型，复杂场景需要后续接入深度模型或人工样本。"
                ),
            ],
            steps=steps,
            human_checkpoints=[
                "after_plan",
                "on_input_quality_failure",
                "on_alignment_quality_failure",
                "on_change_result_quality_failure",
            ],
            estimated_resources={
                "time_minutes": 5 if use_bit else 2,
                "gpu_required": use_bit,
                "storage_gb": 0.1,
            },
            retrieved_context_refs=context_refs,
        )

    def _threshold_from_context(self, context: List[KnowledgeChunk]) -> float:
        joined = "\n".join(chunk.content for chunk in context)
        if "0.18" in joined:
            return 0.18
        return 0.2

    def _min_area_from_context(self, context: List[KnowledgeChunk]) -> float:
        joined = "\n".join(chunk.content for chunk in context)
        if "100" in joined and "平方米" in joined:
            return 100.0
        return 120.0

    def _use_bit_model(self, state: TaskState) -> bool:
        goal = state.user_goal.lower()
        return any(
            keyword in goal
            for keyword in ["bit", "transformer", "深度模型", "深度变化检测"]
        )
