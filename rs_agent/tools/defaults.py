from __future__ import annotations

from rs_agent.tools.ml.change_detection import detect_change
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
        ToolSpec(name="raster.calculate_index", description="计算 NDVI、NDWI、NDBI 等遥感指数。"),
        calculate_index,
    )
    registry.register(
        ToolSpec(name="ml.detect_change", description="基于双时相特征差值执行变化检测。"),
        detect_change,
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

