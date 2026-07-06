from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

import numpy as np

from rs_agent.agent.state import Artifact
from rs_agent.tools.raster.io import checksum_file, load_raster, preferred_raster_suffix, save_raster
from rs_agent.tools.schemas import ToolContext, ToolResult


def align_pair(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    raster_t1 = params["raster_t1"]
    raster_t2 = params["raster_t2"]
    data_t1, meta_t1 = load_raster(raster_t1)
    data_t2, meta_t2 = load_raster(raster_t2)

    height = min(data_t1.shape[0], data_t2.shape[0])
    width = min(data_t1.shape[1], data_t2.shape[1])
    bands = min(data_t1.shape[2] if data_t1.ndim == 3 else 1, data_t2.shape[2] if data_t2.ndim == 3 else 1)
    aligned_t1 = _ensure_3d(data_t1)[:height, :width, :bands]
    aligned_t2 = _ensure_3d(data_t2)[:height, :width, :bands]

    common_metadata = dict(meta_t1)
    common_metadata.update(
        {
            "width": int(width),
            "height": int(height),
            "count": int(bands),
            "aligned_from": [raster_t1, raster_t2],
            "resampling": params.get("resampling", "bilinear"),
            "alignment_strategy": "crop_to_common_extent",
        }
    )
    metadata_t1 = {**common_metadata, "source_uri": raster_t1}
    metadata_t2 = {**common_metadata, **meta_t2, "source_uri": raster_t2}
    metadata_t2.update(
        {
            "width": int(width),
            "height": int(height),
            "count": int(bands),
            "aligned_from": [raster_t1, raster_t2],
            "resampling": params.get("resampling", "bilinear"),
            "alignment_strategy": "crop_to_common_extent",
        }
    )
    suffix_t1 = preferred_raster_suffix(metadata_t1)
    suffix_t2 = preferred_raster_suffix(metadata_t2)
    path_t1 = context.artifact_path("intermediate", f"aligned_t1{suffix_t1}")
    path_t2 = context.artifact_path("intermediate", f"aligned_t2{suffix_t2}")
    uri_t1 = save_raster(path_t1, aligned_t1.astype(np.float32), metadata_t1)
    uri_t2 = save_raster(path_t2, aligned_t2.astype(np.float32), metadata_t2)

    artifacts = [
        Artifact(
            artifact_id=f"art_{uuid4().hex[:12]}",
            type="raster",
            alias="aligned_t1",
            uri=uri_t1,
            crs=metadata_t1.get("crs"),
            bbox=metadata_t1.get("bbox"),
            checksum=checksum_file(uri_t1),
            metadata=metadata_t1,
        ),
        Artifact(
            artifact_id=f"art_{uuid4().hex[:12]}",
            type="raster",
            alias="aligned_t2",
            uri=uri_t2,
            crs=metadata_t2.get("crs"),
            bbox=metadata_t2.get("bbox"),
            checksum=checksum_file(uri_t2),
            metadata=metadata_t2,
        ),
    ]
    return ToolResult(
        tool_name="raster.align_pair",
        outputs={
            "aligned_shape": [int(height), int(width), int(bands)],
            "aligned_t1": artifacts[0].artifact_id,
            "aligned_t2": artifacts[1].artifact_id,
        },
        artifacts=artifacts,
        logs=["aligned pair by common dimensions"],
    )


def _ensure_3d(data):
    if data.ndim == 2:
        return data[:, :, None]
    return data
