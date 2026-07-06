from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from rs_agent.agent.state import Artifact
from rs_agent.tools.postprocess.components import filter_components
from rs_agent.tools.raster.io import checksum_file, load_raster, preferred_raster_suffix, save_raster
from rs_agent.tools.schemas import ToolContext, ToolResult


def filter_small_regions(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    raster_uri = params["raster_uri"]
    output_alias = params.get("output_alias", "change_filtered")
    min_area_m2 = float(params.get("min_area_m2", 100.0))
    data, metadata = load_raster(raster_uri)
    resolution = metadata.get("resolution", [1.0, 1.0])
    pixel_area = float(resolution[0]) * float(resolution[1]) if isinstance(resolution, list) else float(resolution) ** 2
    min_pixels = max(1, int(round(min_area_m2 / max(pixel_area, 1e-6))))
    filtered = filter_components(data, min_pixels)

    out_metadata = {
        **metadata,
        "source_uri": raster_uri,
        "min_area_m2": min_area_m2,
        "min_pixels": min_pixels,
        "remaining_pixels": int(filtered.sum()),
    }
    path = context.artifact_path("outputs", f"{output_alias}{preferred_raster_suffix(out_metadata)}")
    uri = save_raster(path, filtered, out_metadata)
    artifact = Artifact(
        artifact_id=f"art_{uuid4().hex[:12]}",
        type="raster",
        alias=output_alias,
        uri=uri,
        crs=metadata.get("crs"),
        bbox=metadata.get("bbox"),
        checksum=checksum_file(uri),
        metadata=out_metadata,
    )
    return ToolResult(
        tool_name="post.filter_small_regions",
        outputs={
            "artifact_id": artifact.artifact_id,
            "min_pixels": min_pixels,
            "remaining_pixels": int(filtered.sum()),
        },
        artifacts=[artifact],
        logs=[f"filtered regions smaller than {min_area_m2} m2"],
    )
