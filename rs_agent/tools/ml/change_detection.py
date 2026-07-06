from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

import numpy as np

from rs_agent.agent.state import Artifact
from rs_agent.tools.raster.io import checksum_file, load_raster, preferred_raster_suffix, raster_stats, save_raster
from rs_agent.tools.schemas import ToolContext, ToolResult


def detect_change(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    threshold = float(params.get("threshold", 0.18))
    output_alias = params.get("output_alias", "change_raster")
    feature_t1, meta_t1 = load_raster(params["feature_t1"])
    feature_t2, meta_t2 = load_raster(params["feature_t2"])
    diff = np.asarray(feature_t2, dtype="float32") - np.asarray(feature_t1, dtype="float32")
    change = (diff > threshold).astype("uint8")

    changed_pixels = int(change.sum())
    total_pixels = int(change.size)
    out_metadata = {
        **meta_t2,
        "source_features": [params["feature_t1"], params["feature_t2"]],
        "model_id": params.get("model_id", "builtup_ndbi_delta_v0"),
        "target": params.get("target", "change"),
        "threshold": threshold,
        "changed_pixels": changed_pixels,
        "changed_ratio": changed_pixels / max(total_pixels, 1),
        "band_names": ["change"],
        "count": 1,
        "stats": raster_stats(change),
    }
    path = context.artifact_path("outputs", f"{output_alias}{preferred_raster_suffix(out_metadata)}")
    uri = save_raster(path, change, out_metadata)
    artifact = Artifact(
        artifact_id=f"art_{uuid4().hex[:12]}",
        type="raster",
        alias=output_alias,
        uri=uri,
        crs=meta_t2.get("crs"),
        bbox=meta_t2.get("bbox"),
        checksum=checksum_file(uri),
        metadata=out_metadata,
    )
    return ToolResult(
        tool_name="ml.detect_change",
        outputs={
            "artifact_id": artifact.artifact_id,
            "changed_pixels": changed_pixels,
            "changed_ratio": out_metadata["changed_ratio"],
            "threshold": threshold,
        },
        artifacts=[artifact],
        logs=[f"detected change with threshold={threshold}"],
    )
