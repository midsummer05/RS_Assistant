from __future__ import annotations

import re
from typing import Any, Iterable, List

from rs_agent.agent.state import KnowledgeChunk, TaskState


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    english = set(re.findall(r"[a-z0-9_\-]+", lowered))
    chinese_terms = {
        term
        for term in [
            "变化检测",
            "建设用地",
            "扩张",
            "对齐",
            "配准",
            "指数",
            "模型",
            "后处理",
            "面积",
            "统计",
            "报告",
            "图斑",
            "sentinel",
            "ndbi",
            "ndvi",
            "ndwi",
            "bit",
            "质量门",
            "相关性",
        ]
        if term in lowered
    }
    return english | chinese_terms


class LocalRagRetriever:
    """Tiny retrieval service with built-in remote-sensing knowledge.

    The implementation deliberately keeps chunk metadata and scores so the MVP
    exercises the RAG path even before a vector database is introduced.
    """

    def __init__(
        self,
        chunks: Iterable[KnowledgeChunk] | None = None,
        knowledge_store: Any | None = None,
    ) -> None:
        self.chunks = list(chunks or self._default_chunks())
        self.knowledge_store = knowledge_store
        if self.knowledge_store is not None:
            self.knowledge_store.upsert_chunks(self.chunks)

    def retrieve_for_planning(self, state: TaskState, limit: int = 4) -> List[KnowledgeChunk]:
        query = " ".join(
            [
                state.user_goal,
                state.task_type or "",
                " ".join(state.constraints.get("inputs", {}).values()),
            ]
        )
        if self.knowledge_store is not None:
            return self.knowledge_store.search_knowledge(
                query=query,
                limit=limit,
                task_tags=[state.task_type or "change_detection"],
            )
        query_tokens = _tokens(query)
        scored: List[KnowledgeChunk] = []
        for chunk in self.chunks:
            doc_tokens = _tokens(" ".join([chunk.title, chunk.content, " ".join(chunk.task_tags)]))
            tag_bonus = 1.5 if "change_detection" in chunk.task_tags else 0.0
            score = len(query_tokens.intersection(doc_tokens)) + tag_bonus
            if score <= 0:
                continue
            scored_chunk = chunk.model_copy(update={"score": float(score)})
            scored.append(scored_chunk)
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def search(self, query: str, limit: int = 5) -> List[KnowledgeChunk]:
        if self.knowledge_store is not None:
            return self.knowledge_store.search_knowledge(query=query, limit=limit)
        dummy = TaskState(user_id="search", user_goal=query)
        return self.retrieve_for_planning(dummy, limit=limit)

    def _default_chunks(self) -> List[KnowledgeChunk]:
        return [
            KnowledgeChunk(
                chunk_id="change_detection_sop#core_flow",
                title="双时相变化检测 SOP",
                source_type="project_sop",
                task_tags=["change_detection", "preprocessing", "quality"],
                metadata={"version": "2026-07", "citation": "internal://sop/change_detection"},
                content=(
                    "双时相变化检测必须先读取两期影像元数据，不得猜测 CRS、分辨率、"
                    "波段数量和空间范围。若 CRS、分辨率或像元网格不一致，应先执行"
                    "对齐或重采样，再构建指数或特征。结果需经过小斑块过滤、矢量化、"
                    "面积统计和报告说明。"
                ),
            ),
            KnowledgeChunk(
                chunk_id="bit_levir_model_card#v1",
                title="BIT_LEVIR 深度变化检测模型卡",
                source_type="model_card",
                task_tags=["change_detection", "built_up", "model_selection", "bit"],
                metadata={
                    "model_id": "BIT_LEVIR",
                    "architecture": "Bitemporal Image Transformer",
                    "citation": "internal://model-card/BIT_LEVIR",
                },
                content=(
                    "BIT_LEVIR 使用共享 ResNet18、语义 Token 和 Transformer 对双时相 RGB "
                    "影像进行建筑物变化检测。输入必须完成精确配准，推荐 256 像素切片并使用"
                    "重叠融合。模型主要在 LEVIR-CD 上训练，跨传感器和跨区域应用必须进行"
                    "结果质量检查和人工抽样验证。"
                ),
            ),
            KnowledgeChunk(
                chunk_id="change_quality_gates#v1",
                title="变化检测质量门规范",
                source_type="project_sop",
                task_tags=["change_detection", "quality", "human_review"],
                metadata={"citation": "internal://sop/change-quality-gates"},
                content=(
                    "长程变化检测至少设置三个质量门：输入适用性、配准质量和结果合理性。"
                    "输入质量检查 CRS、分辨率、尺寸、波段和有效像元；配准质量检查网格和"
                    "双时相相关性；结果质量检查变化比例、连通图斑数量和二值掩膜。任一"
                    "质量门失败时必须暂停任务，保存 checkpoint 并请求人工复核。"
                ),
            ),
            KnowledgeChunk(
                chunk_id="builtup_change_model_card#v0",
                title="建设用地扩张轻量模型卡",
                source_type="model_card",
                task_tags=["change_detection", "built_up", "model_selection"],
                metadata={
                    "model_id": "builtup_ndbi_delta_v0",
                    "sensor": "Sentinel-2",
                    "version": "0.1.0",
                    "citation": "internal://model-card/builtup_ndbi_delta_v0",
                },
                content=(
                    "builtup_ndbi_delta_v0 是 MVP 阶段的规则型变化检测模型，适用于"
                    "Sentinel-2 或具备 NIR/SWIR 波段的多光谱影像。默认使用 NDBI 差值"
                    "识别建设用地扩张，推荐阈值 0.18，最小图斑面积 100 平方米。"
                    "高云量、季节差异和配准误差会显著增加误检。"
                ),
            ),
            KnowledgeChunk(
                chunk_id="tool_raster_align_pair#schema",
                title="工具说明：raster.align_pair",
                source_type="tool_doc",
                task_tags=["change_detection", "tool", "preprocessing"],
                metadata={"tool_name": "raster.align_pair", "citation": "internal://tools/raster.align_pair"},
                content=(
                    "raster.align_pair 接收 raster_t1、raster_t2 和 resampling 参数，"
                    "输出 aligned_t1、aligned_t2。变化检测前应确认两期影像具有"
                    "一致的宽高、CRS、分辨率和空间范围。"
                ),
            ),
            KnowledgeChunk(
                chunk_id="index_ndbi#definition",
                title="遥感指数：NDBI",
                source_type="remote_sensing_basics",
                task_tags=["change_detection", "index_calculation", "built_up"],
                metadata={"citation": "internal://knowledge/index/ndbi"},
                content=(
                    "NDBI 通常使用 SWIR 与 NIR 计算：(SWIR - NIR) / (SWIR + NIR)。"
                    "建设用地往往呈现较高 NDBI，植被区域可结合 NDVI 抑制误检。"
                ),
            ),
        ]
