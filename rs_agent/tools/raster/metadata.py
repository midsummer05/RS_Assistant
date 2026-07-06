from __future__ import annotations

from typing import Any, Dict, List

from rs_agent.tools.raster.io import checksum_file, load_raster, raster_stats
from rs_agent.tools.schemas import ToolContext, ToolResult


def inspect_metadata(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    raster_uris: List[str] = params["raster_uris"]
    metadata_items = []
    for index, uri in enumerate(raster_uris, start=1):
        data, metadata = load_raster(uri)
        item = dict(metadata)
        item.update(
            {
                "uri": uri,
                "shape": list(data.shape),
                "stats": raster_stats(data),
            }
        )
        if not uri.startswith("demo://"):
            item["checksum"] = checksum_file(uri.replace("file://", ""))
        metadata_items.append(item)

    outputs = {
        "metadata": metadata_items,
        "metadata_t1": metadata_items[0] if metadata_items else None,
        "metadata_t2": metadata_items[1] if len(metadata_items) > 1 else None,
    }
    return ToolResult(
        tool_name="raster.inspect_metadata",
        outputs=outputs,
        logs=[f"read metadata for {len(metadata_items)} raster(s)"],
    )

