from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

import numpy as np

from rs_agent.agent.state import Artifact
from rs_agent.tools.raster.io import band_index, checksum_file, load_raster, preferred_raster_suffix, raster_stats, save_raster
from rs_agent.tools.schemas import ToolContext, ToolResult


def calculate_index(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    raster_uri = params["raster_uri"]
    index_name = params["index_name"].upper()
    output_alias = params.get("output_alias", index_name.lower())
    data, metadata = load_raster(raster_uri)
    data = _ensure_3d(data).astype("float32")

    if index_name == "NDBI":
        first = data[:, :, band_index(metadata, ["B11", "SWIR", "SWIR1"], min(4, data.shape[2] - 1))]
        second = data[:, :, band_index(metadata, ["B08", "NIR"], min(3, data.shape[2] - 1))]
    elif index_name == "NDVI":
        first = data[:, :, band_index(metadata, ["B08", "NIR"], min(3, data.shape[2] - 1))]
        second = data[:, :, band_index(metadata, ["B04", "RED"], min(2, data.shape[2] - 1))]
    elif index_name == "NDWI":
        first = data[:, :, band_index(metadata, ["B03", "GREEN"], min(1, data.shape[2] - 1))]
        second = data[:, :, band_index(metadata, ["B08", "NIR"], min(3, data.shape[2] - 1))]
    else:
        raise ValueError(f"Unsupported index: {index_name}")

    index = (first - second) / (first + second + 1e-6)
    index = np.clip(index, -1, 1).astype("float32")
    out_metadata = {
        **metadata,
        "source_uri": raster_uri,
        "index_name": index_name,
        "band_names": [index_name],
        "count": 1,
        "stats": raster_stats(index),
    }
    path = context.artifact_path("intermediate", f"{output_alias}{preferred_raster_suffix(out_metadata)}")
    uri = save_raster(path, index, out_metadata)
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
        tool_name="raster.calculate_index",
        outputs={"artifact_id": artifact.artifact_id, "stats": raster_stats(index)},
        artifacts=[artifact],
        logs=[f"calculated {index_name}"],
    )


def _ensure_3d(data):
    if data.ndim == 2:
        return data[:, :, None]
    return data
