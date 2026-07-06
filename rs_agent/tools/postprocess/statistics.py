from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from rs_agent.agent.state import Artifact
from rs_agent.tools.raster.io import checksum_file
from rs_agent.tools.schemas import ToolContext, ToolResult


def area_statistics(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    vector_uri = params["vector_uri"]
    output_alias = params.get("output_alias", "area_statistics")
    payload = json.loads(Path(vector_uri).read_text(encoding="utf-8"))
    features = payload.get("features", [])
    total_area_m2 = sum(float(feature.get("properties", {}).get("area_m2", 0.0)) for feature in features)
    total_area_ha = total_area_m2 / 10000.0

    path = context.artifact_path("outputs", f"{output_alias}.csv")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "polygon_count", "area_m2", "area_ha"])
        writer.writeheader()
        writer.writerow(
            {
                "class": "built_up_expansion",
                "polygon_count": len(features),
                "area_m2": round(total_area_m2, 2),
                "area_ha": round(total_area_ha, 4),
            }
        )

    summary = {
        "class": "built_up_expansion",
        "polygon_count": len(features),
        "area_m2": round(total_area_m2, 2),
        "area_ha": round(total_area_ha, 4),
    }
    artifact = Artifact(
        artifact_id=f"art_{uuid4().hex[:12]}",
        type="table",
        alias=output_alias,
        uri=str(Path(path).resolve()),
        checksum=checksum_file(path),
        metadata={"source_vector": vector_uri, "summary": summary},
    )
    return ToolResult(
        tool_name="post.area_statistics",
        outputs={"artifact_id": artifact.artifact_id, "summary": summary},
        artifacts=[artifact],
        logs=["wrote area statistics table"],
    )

