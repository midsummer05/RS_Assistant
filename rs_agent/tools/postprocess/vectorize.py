from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np

from rs_agent.agent.state import Artifact
from rs_agent.tools.postprocess.components import component_stats
from rs_agent.tools.raster.io import checksum_file, load_raster
from rs_agent.tools.schemas import ToolContext, ToolResult


def raster_to_vector(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    raster_uri = params["raster_uri"]
    output_alias = params.get("output_alias", "change_vector")
    data, metadata = load_raster(raster_uri)
    mask = np.asarray(data).astype(bool)
    components = component_stats(mask)
    features: List[Dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        coordinates = _component_polygon(component, mask.shape, metadata.get("bbox"))
        area_m2 = component["pixel_count"] * _pixel_area(metadata)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": index,
                    "class": "built_up_expansion",
                    "pixel_count": component["pixel_count"],
                    "area_m2": round(area_m2, 2),
                    "area_ha": round(area_m2 / 10000.0, 4),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                },
            }
        )

    collection = {
        "type": "FeatureCollection",
        "name": output_alias,
        "crs": metadata.get("crs"),
        "source_raster": raster_uri,
        "features": features,
    }
    path = context.artifact_path("outputs", f"{output_alias}.geojson")
    Path(path).write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact = Artifact(
        artifact_id=f"art_{uuid4().hex[:12]}",
        type="vector",
        alias=output_alias,
        uri=str(Path(path).resolve()),
        crs=metadata.get("crs"),
        bbox=metadata.get("bbox"),
        checksum=checksum_file(path),
        metadata={
            "source_uri": raster_uri,
            "feature_count": len(features),
            "total_area_m2": round(sum(feature["properties"]["area_m2"] for feature in features), 2),
        },
    )
    return ToolResult(
        tool_name="post.raster_to_vector",
        outputs={
            "artifact_id": artifact.artifact_id,
            "feature_count": len(features),
            "total_area_m2": artifact.metadata["total_area_m2"],
        },
        artifacts=[artifact],
        logs=[f"vectorized {len(features)} change polygon(s)"],
    )


def _component_polygon(component: Dict[str, int], shape, bbox):
    height, width = shape[:2]
    if not bbox:
        bbox = [0.0, 0.0, float(width), float(height)]
    xmin, ymin, xmax, ymax = [float(item) for item in bbox]
    min_col = component["min_col"]
    max_col = component["max_col"] + 1
    min_row = component["min_row"]
    max_row = component["max_row"] + 1
    x0 = xmin + min_col / width * (xmax - xmin)
    x1 = xmin + max_col / width * (xmax - xmin)
    y_top = ymax - min_row / height * (ymax - ymin)
    y_bottom = ymax - max_row / height * (ymax - ymin)
    return [
        [x0, y_bottom],
        [x1, y_bottom],
        [x1, y_top],
        [x0, y_top],
        [x0, y_bottom],
    ]


def _pixel_area(metadata: Dict[str, Any]) -> float:
    resolution = metadata.get("resolution", [1.0, 1.0])
    if isinstance(resolution, list):
        return float(resolution[0]) * float(resolution[1])
    return float(resolution) ** 2

