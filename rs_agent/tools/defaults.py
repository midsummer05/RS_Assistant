from __future__ import annotations

from rs_agent.tools.ml.change_detection import detect_change
from rs_agent.tools.ml.bit_change_detection import bit_change_detection
from rs_agent.tools.postprocess.morphology import filter_small_regions
from rs_agent.tools.postprocess.statistics import area_statistics
from rs_agent.tools.postprocess.vectorize import raster_to_vector
from rs_agent.tools.raster.indices import calculate_index
from rs_agent.tools.raster.metadata import inspect_metadata
from rs_agent.tools.raster.preprocess import align_pair
from rs_agent.tools.registry import ToolRegistry
from rs_agent.tools.report.markdown import generate_markdown
from rs_agent.tools.schemas import ToolSpec
from rs_agent.tools.visualization.quicklook import change_overlay
from rs_agent.tools.quality.change_detection import (
    assess_alignment,
    assess_change_inputs,
    assess_change_result,
)


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="raster.inspect_metadata", description="读取影像元数据，防止 Agent 猜测数据属性。"),
        inspect_metadata,
    )
    registry.register(
        ToolSpec(name="raster.align_pair", description="将双时相影像对齐到公共尺寸和元数据框架。"),
        align_pair,
    )
    registry.register(
        ToolSpec(
            name="quality.assess_change_inputs",
            description="检查双时相影像的尺寸、CRS、分辨率、波段和有效像元是否适合变化检测。",
            safety_level="quality_gate",
        ),
        assess_change_inputs,
    )
    registry.register(
        ToolSpec(
            name="quality.assess_alignment",
            description="使用网格一致性和影像相关性评估双时相配准质量。",
            safety_level="quality_gate",
        ),
        assess_alignment,
    )
    registry.register(
        ToolSpec(name="raster.calculate_index", description="计算 NDVI、NDWI、NDBI 等遥感指数。"),
        calculate_index,
    )
    registry.register(
        ToolSpec(name="ml.detect_change", description="基于双时相特征差值执行变化检测。"),
        detect_change,
    )
    registry.register(
        ToolSpec(
            name="ml.bit_change_detection",
            version="1.0.0",
            description="使用 BIT_LEVIR 预训练 Transformer 对已配准双时相 RGB/多光谱影像执行分块变化检测。",
            input_schema={
                "type": "object",
                "required": ["raster_t1", "raster_t2"],
                "properties": {
                    "raster_t1": {"type": "string"},
                    "raster_t2": {"type": "string"},
                    "tile_size": {"type": "integer", "default": 256},
                    "overlap": {"type": "integer", "default": 32},
                    "batch_size": {"type": "integer", "default": 4},
                    "device": {"type": "string", "default": "auto"},
                    "output_alias": {"type": "string", "default": "change_raster"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string"},
                    "changed_pixels": {"type": "integer"},
                    "changed_ratio": {"type": "number"},
                    "model_id": {"type": "string"},
                },
            },
            resource_profile={"gpu_optional": True, "memory_gb": 4},
            safety_level="model_inference",
        ),
        bit_change_detection,
    )
    registry.register(
        ToolSpec(
            name="quality.assess_change_result",
            description="检查变化比例、连通图斑数量和二值掩膜合理性。",
            safety_level="quality_gate",
        ),
        assess_change_result,
    )
    registry.register(
        ToolSpec(name="post.filter_small_regions", description="过滤小于指定面积的变化图斑。"),
        filter_small_regions,
    )
    registry.register(
        ToolSpec(name="post.raster_to_vector", description="将二值变化栅格转换为 GeoJSON 图斑。"),
        raster_to_vector,
    )
    registry.register(
        ToolSpec(name="post.area_statistics", description="统计变化图斑面积。"),
        area_statistics,
    )
    registry.register(
        ToolSpec(name="viz.change_overlay", description="生成变化区域叠加预览图。"),
        change_overlay,
    )
    registry.register(
        ToolSpec(name="report.generate_markdown", description="生成 Markdown 任务报告。"),
        generate_markdown,
    )
    return registry
