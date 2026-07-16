from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import numpy as np

from rs_agent.agent.state import Artifact
from rs_agent.models.BIT_CD import BITChangeDetector, BITModelConfig
from rs_agent.tools.raster.io import (
    band_index,
    checksum_file,
    load_raster,
    preferred_raster_suffix,
    raster_stats,
    save_raster,
)
from rs_agent.tools.schemas import ToolContext, ToolError, ToolResult


def bit_change_detection(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    raster_t1 = params["raster_t1"]
    raster_t2 = params["raster_t2"]
    output_alias = params.get("output_alias", "change_raster")
    data_t1, meta_t1 = load_raster(raster_t1)
    data_t2, meta_t2 = load_raster(raster_t2)
    if data_t1.shape[:2] != data_t2.shape[:2]:
        raise ToolError(
            "ml.bit_change_detection",
            "BIT requires aligned inputs with identical height and width.",
            "input_not_aligned",
        )
    try:
        rgb_t1, rgb_mapping_t1 = _select_rgb(data_t1, meta_t1)
        rgb_t2, rgb_mapping_t2 = _select_rgb(data_t2, meta_t2)
        checkpoint = Path(
            params.get(
                "checkpoint_path",
                Path(__file__).resolve().parents[2]
                / "models"
                / "BIT_CD"
                / "checkpoints"
                / "BIT_LEVIR"
                / "best_ckpt.pt",
            )
        )
        config = BITModelConfig(
            checkpoint_path=checkpoint,
            network_name=params.get(
                "network_name", "base_transformer_pos_s4_dd8_dedim8"
            ),
            tile_size=int(params.get("tile_size", 256)),
            overlap=int(params.get("overlap", 32)),
            batch_size=int(params.get("batch_size", 4)),
            device=params.get("device", "auto"),
        )
        change = BITChangeDetector(config).predict(rgb_t1, rgb_t2)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise ToolError("ml.bit_change_detection", str(exc), "bit_inference_failed") from exc

    changed_pixels = int(change.sum())
    total_pixels = int(change.size)
    metadata = {
        **meta_t2,
        "source_rasters": [raster_t1, raster_t2],
        "model_id": "BIT_LEVIR",
        "model_family": "Bitemporal Image Transformer",
        "network_name": config.network_name,
        "checkpoint_path": str(config.checkpoint_path.resolve()),
        "checkpoint_checksum": checksum_file(config.checkpoint_path),
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "device": str(config.device),
        "rgb_mapping_t1": rgb_mapping_t1,
        "rgb_mapping_t2": rgb_mapping_t2,
        "changed_pixels": changed_pixels,
        "changed_ratio": changed_pixels / max(total_pixels, 1),
        "band_names": ["change"],
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "stats": raster_stats(change),
    }
    path = context.artifact_path(
        "outputs", f"{output_alias}{preferred_raster_suffix(metadata)}"
    )
    uri = save_raster(path, change, metadata)
    artifact = Artifact(
        artifact_id=f"art_{uuid4().hex[:12]}",
        type="raster",
        alias=output_alias,
        uri=uri,
        crs=metadata.get("crs"),
        bbox=metadata.get("bbox"),
        checksum=checksum_file(uri),
        metadata=metadata,
    )
    return ToolResult(
        tool_name="ml.bit_change_detection",
        outputs={
            "artifact_id": artifact.artifact_id,
            "changed_pixels": changed_pixels,
            "changed_ratio": metadata["changed_ratio"],
            "model_id": metadata["model_id"],
        },
        artifacts=[artifact],
        logs=[
            f"BIT inference completed with tile_size={config.tile_size}, overlap={config.overlap}"
        ],
        metrics={"tile_size": config.tile_size, "overlap": config.overlap},
    )


def _select_rgb(data: np.ndarray, metadata: Dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    array = data[:, :, None] if data.ndim == 2 else data
    if array.shape[2] < 3:
        raise ValueError("BIT requires at least three input bands.")
    band_names = metadata.get("band_names", [])
    red = band_index(metadata, ["B04", "RED", "R"], min(2, array.shape[2] - 1))
    green = band_index(metadata, ["B03", "GREEN", "G"], min(1, array.shape[2] - 1))
    blue = band_index(metadata, ["B02", "BLUE", "B"], 0)
    indexes = [red, green, blue]
    names = [
        str(band_names[index]) if index < len(band_names) else f"band_{index + 1}"
        for index in indexes
    ]
    return np.asarray(array[:, :, indexes]), names

