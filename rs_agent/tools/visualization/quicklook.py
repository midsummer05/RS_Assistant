from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import numpy as np
from PIL import Image

from rs_agent.agent.state import Artifact
from rs_agent.tools.raster.io import band_index, checksum_file, load_raster
from rs_agent.tools.schemas import ToolContext, ToolResult


def change_overlay(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    base_uri = params["base_raster"]
    change_uri = params["change_raster"]
    output_alias = params.get("output_alias", "change_preview")
    base, base_meta = load_raster(base_uri)
    change, _ = load_raster(change_uri)
    rgb = _rgb(base, base_meta)
    mask = np.asarray(change).astype(bool)
    overlay = rgb.copy()
    overlay[mask] = (0.72 * overlay[mask] + np.array([255, 30, 30]) * 0.28).astype("uint8")

    path = context.artifact_path("outputs", f"{output_alias}.png")
    Image.fromarray(overlay).save(path)
    artifact = Artifact(
        artifact_id=f"art_{uuid4().hex[:12]}",
        type="image",
        alias=output_alias,
        uri=str(Path(path).resolve()),
        crs=base_meta.get("crs"),
        bbox=base_meta.get("bbox"),
        checksum=checksum_file(path),
        metadata={
            "base_raster": base_uri,
            "change_raster": change_uri,
            "changed_pixels": int(mask.sum()),
        },
    )
    return ToolResult(
        tool_name="viz.change_overlay",
        outputs={"artifact_id": artifact.artifact_id, "changed_pixels": int(mask.sum())},
        artifacts=[artifact],
        logs=["generated change overlay preview"],
    )


def _rgb(data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    if data.ndim == 2:
        stretched = _stretch(data)
        return np.dstack([stretched, stretched, stretched])
    red = data[:, :, band_index(metadata, ["B04", "RED"], min(2, data.shape[2] - 1))]
    green = data[:, :, band_index(metadata, ["B03", "GREEN"], min(1, data.shape[2] - 1))]
    blue = data[:, :, band_index(metadata, ["B02", "BLUE"], 0)]
    return np.dstack([_stretch(red), _stretch(green), _stretch(blue)])


def _stretch(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel, dtype="float32")
    p2, p98 = np.percentile(channel, [2, 98])
    if p98 <= p2:
        return np.zeros(channel.shape, dtype="uint8")
    out = (channel - p2) / (p98 - p2)
    return (np.clip(out, 0, 1) * 255).astype("uint8")
