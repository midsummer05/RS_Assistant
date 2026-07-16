from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rs_agent.tools.postprocess.components import component_stats
from rs_agent.tools.raster.io import load_raster
from rs_agent.tools.schemas import ToolContext, ToolResult


def assess_change_inputs(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    data_t1, meta_t1 = load_raster(params["raster_t1"])
    data_t2, meta_t2 = load_raster(params["raster_t2"])
    shape_match = data_t1.shape[:2] == data_t2.shape[:2]
    crs_match = meta_t1.get("crs") == meta_t2.get("crs")
    resolution_match = meta_t1.get("resolution") == meta_t2.get("resolution")
    band_count = min(_band_count(data_t1), _band_count(data_t2))
    enough_bands = band_count >= int(params.get("minimum_bands", 3))
    finite_ratio_t1 = _finite_ratio(data_t1)
    finite_ratio_t2 = _finite_ratio(data_t2)
    minimum_finite_ratio = float(params.get("minimum_finite_ratio", 0.95))
    checks = [
        _check("shape_match", shape_match, f"{data_t1.shape[:2]} vs {data_t2.shape[:2]}"),
        _check("crs_match", crs_match, f"{meta_t1.get('crs')} vs {meta_t2.get('crs')}"),
        _check(
            "resolution_match",
            resolution_match,
            f"{meta_t1.get('resolution')} vs {meta_t2.get('resolution')}",
        ),
        _check("minimum_bands", enough_bands, f"common bands={band_count}"),
        _check(
            "finite_ratio_t1",
            finite_ratio_t1 >= minimum_finite_ratio,
            f"{finite_ratio_t1:.4f}",
        ),
        _check(
            "finite_ratio_t2",
            finite_ratio_t2 >= minimum_finite_ratio,
            f"{finite_ratio_t2:.4f}",
        ),
    ]
    passed = all(item["passed"] for item in checks)
    return ToolResult(
        tool_name="quality.assess_change_inputs",
        outputs={
            "passed": passed,
            "checks": checks,
            "issues": [item for item in checks if not item["passed"]],
            "recommendation": (
                "输入满足变化检测基本要求。"
                if passed
                else "先完成严格重投影、重采样、无效像元处理或补充波段。"
            ),
        },
        logs=["assessed change detection input suitability"],
    )


def assess_alignment(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    data_t1, meta_t1 = load_raster(params["raster_t1"])
    data_t2, meta_t2 = load_raster(params["raster_t2"])
    gray_t1 = _gray(data_t1)
    gray_t2 = _gray(data_t2)
    sample_step = max(1, int(max(gray_t1.shape) / 512))
    sample_t1 = gray_t1[::sample_step, ::sample_step].ravel()
    sample_t2 = gray_t2[::sample_step, ::sample_step].ravel()
    valid = np.isfinite(sample_t1) & np.isfinite(sample_t2)
    if valid.sum() < 16:
        correlation = 0.0
    elif float(np.std(sample_t1[valid])) < 1e-8 or float(np.std(sample_t2[valid])) < 1e-8:
        correlation = 1.0 if np.allclose(sample_t1[valid], sample_t2[valid]) else 0.0
    else:
        correlation = float(np.corrcoef(sample_t1[valid], sample_t2[valid])[0, 1])
    minimum_correlation = float(params.get("minimum_correlation", 0.35))
    checks = [
        _check("shape_match", data_t1.shape[:2] == data_t2.shape[:2], str(data_t1.shape[:2])),
        _check("crs_match", meta_t1.get("crs") == meta_t2.get("crs"), str(meta_t2.get("crs"))),
        _check(
            "grid_correlation",
            correlation >= minimum_correlation,
            f"{correlation:.4f} >= {minimum_correlation:.4f}",
        ),
    ]
    passed = all(item["passed"] for item in checks)
    return ToolResult(
        tool_name="quality.assess_alignment",
        outputs={
            "passed": passed,
            "correlation": correlation,
            "checks": checks,
            "issues": [item for item in checks if not item["passed"]],
            "recommendation": (
                "对齐质量满足自动执行条件。"
                if passed
                else "建议人工检查控制点并采用严格影像配准后重试。"
            ),
        },
        logs=[f"alignment correlation={correlation:.4f}"],
    )


def assess_change_result(context: ToolContext, params: Dict[str, Any]) -> ToolResult:
    data, metadata = load_raster(params["change_raster"])
    mask = np.asarray(data).astype(bool)
    changed_ratio = float(mask.mean()) if mask.size else 0.0
    components = component_stats(mask)
    min_ratio = float(params.get("minimum_changed_ratio", 0.0001))
    max_ratio = float(params.get("maximum_changed_ratio", 0.6))
    max_components = int(params.get("maximum_components", 100000))
    checks = [
        _check(
            "changed_ratio_range",
            min_ratio <= changed_ratio <= max_ratio,
            f"{min_ratio:.4f} <= {changed_ratio:.4f} <= {max_ratio:.4f}",
        ),
        _check(
            "component_count",
            len(components) <= max_components,
            f"{len(components)} <= {max_components}",
        ),
        _check("binary_mask", _is_binary(data), f"dtype={data.dtype}"),
    ]
    passed = all(item["passed"] for item in checks)
    return ToolResult(
        tool_name="quality.assess_change_result",
        outputs={
            "passed": passed,
            "changed_ratio": changed_ratio,
            "component_count": len(components),
            "checks": checks,
            "issues": [item for item in checks if not item["passed"]],
            "recommendation": (
                "变化结果通过自动合理性检查。"
                if passed
                else "建议检查配准、季节差异、云影并调整模型或阈值后重试。"
            ),
            "source_metadata": {
                "model_id": metadata.get("model_id"),
                "threshold": metadata.get("threshold"),
            },
        },
        logs=[f"change ratio={changed_ratio:.4f}, components={len(components)}"],
    )


def _band_count(data: np.ndarray) -> int:
    return int(data.shape[2]) if data.ndim == 3 else 1


def _finite_ratio(data: np.ndarray) -> float:
    return float(np.isfinite(data).sum() / max(data.size, 1))


def _gray(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    return np.mean(array[:, :, : min(3, array.shape[2])], axis=2) if array.ndim == 3 else array


def _is_binary(data: np.ndarray) -> bool:
    values = np.unique(np.asarray(data))
    return bool(np.all(np.isin(values, [0, 1, 255])))


def _check(name: str, passed: bool, details: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}

