from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

_rasterio_package_dir = Path(sys.prefix) / "Lib" / "site-packages" / "rasterio"
_proj_data_dir = _rasterio_package_dir / "proj_data"
_gdal_data_dir = _rasterio_package_dir / "gdal_data"
if _proj_data_dir.exists():
    os.environ["PROJ_LIB"] = str(_proj_data_dir)
if _gdal_data_dir.exists():
    os.environ["GDAL_DATA"] = str(_gdal_data_dir)

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:  # pragma: no cover - optional dependency fallback
    rasterio = None
    Affine = None


def checksum_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checksum_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raster(uri: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if uri.startswith("demo://"):
        return _demo_raster(uri)
    path = Path(uri.replace("file://", ""))
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {uri}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        data = payload["data"]
        metadata_text = str(payload["metadata"]) if "metadata" in payload else "{}"
        metadata = json.loads(metadata_text)
        return data, metadata
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        return data, _metadata_for_array(data, source_uri=str(path))
    if suffix in {".tif", ".tiff", ".cog"}:
        return _load_geotiff(path)

    with Image.open(path) as image:
        data = np.asarray(image.convert("RGB")).astype("float32") / 255.0
    return data, _metadata_for_array(data, source_uri=str(path))


def save_raster(path: str | Path, data: np.ndarray, metadata: Dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tif", ".tiff"}:
        return _save_geotiff(path, data, metadata)
    metadata = dict(metadata)
    metadata.setdefault("width", int(data.shape[1]))
    metadata.setdefault("height", int(data.shape[0]))
    metadata.setdefault("dtype", str(data.dtype))
    metadata.setdefault("count", int(data.shape[2]) if data.ndim == 3 else 1)
    np.savez_compressed(path, data=data, metadata=json.dumps(metadata, ensure_ascii=False))
    return str(path.resolve())


def preferred_raster_suffix(metadata: Dict[str, Any]) -> str:
    if rasterio is None:
        return ".npz"
    crs = metadata.get("crs")
    transform = metadata.get("transform")
    if crs and crs != "unknown" and transform:
        return ".tif"
    return ".npz"


def raster_stats(data: np.ndarray) -> Dict[str, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def band_index(metadata: Dict[str, Any], names: list[str], fallback: int) -> int:
    band_names = [str(item).upper() for item in metadata.get("band_names", [])]
    for name in names:
        if name.upper() in band_names:
            return band_names.index(name.upper())
    return fallback


def _metadata_for_array(data: np.ndarray, source_uri: str) -> Dict[str, Any]:
    count = int(data.shape[2]) if data.ndim == 3 else 1
    default_names = ["B02", "B03", "B04", "B08", "B11"][:count]
    return {
        "source_uri": source_uri,
        "width": int(data.shape[1]),
        "height": int(data.shape[0]),
        "count": count,
        "dtype": str(data.dtype),
        "crs": "unknown",
        "bbox": [0.0, 0.0, float(data.shape[1]), float(data.shape[0])],
        "resolution": [1.0, 1.0],
        "band_names": default_names,
        "sensor": "unknown",
        "nodata": None,
    }


def _load_geotiff(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if rasterio is None:
        raise RuntimeError("rasterio is required to read GeoTIFF/COG files")
    with rasterio.open(path) as dataset:
        raw = dataset.read()
        if raw.shape[0] == 1:
            data = raw[0]
        else:
            data = np.transpose(raw, (1, 2, 0))
        descriptions = [item for item in dataset.descriptions if item]
        if descriptions:
            band_names = list(descriptions)
        else:
            band_names = _default_band_names(dataset.count)
        metadata = {
            "source_uri": str(path),
            "width": int(dataset.width),
            "height": int(dataset.height),
            "count": int(dataset.count),
            "dtype": str(dataset.dtypes[0]),
            "crs": dataset.crs.to_string() if dataset.crs else "unknown",
            "transform": list(dataset.transform)[:6],
            "bbox": [float(dataset.bounds.left), float(dataset.bounds.bottom), float(dataset.bounds.right), float(dataset.bounds.top)],
            "resolution": [abs(float(dataset.res[0])), abs(float(dataset.res[1]))],
            "band_names": band_names,
            "sensor": dataset.tags().get("sensor", "unknown"),
            "acquired_at": dataset.tags().get("acquired_at"),
            "cloud_cover": _safe_float(dataset.tags().get("cloud_cover")),
            "processing_level": dataset.tags().get("processing_level"),
            "nodata": dataset.nodata,
        }
    return data, metadata


def _save_geotiff(path: Path, data: np.ndarray, metadata: Dict[str, Any]) -> str:
    if rasterio is None or Affine is None:
        raise RuntimeError("rasterio is required to write GeoTIFF files")
    array = np.asarray(data)
    if array.ndim == 2:
        write_data = array[None, :, :]
    elif array.ndim == 3:
        write_data = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported raster shape for GeoTIFF: {array.shape}")

    transform_value = metadata.get("transform")
    if transform_value:
        transform = Affine(*transform_value)
    else:
        resolution = metadata.get("resolution", [1.0, 1.0])
        bbox = metadata.get("bbox", [0.0, 0.0, float(array.shape[1]), float(array.shape[0])])
        transform = Affine(float(resolution[0]), 0.0, float(bbox[0]), 0.0, -float(resolution[1]), float(bbox[3]))

    profile = {
        "driver": "GTiff",
        "height": int(write_data.shape[1]),
        "width": int(write_data.shape[2]),
        "count": int(write_data.shape[0]),
        "dtype": str(write_data.dtype),
        "crs": metadata.get("crs") if metadata.get("crs") != "unknown" else None,
        "transform": transform,
        "compress": "deflate",
    }
    nodata = metadata.get("nodata")
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(write_data)
        for index, band_name in enumerate(metadata.get("band_names", []), start=1):
            if index <= write_data.shape[0]:
                dataset.set_band_description(index, str(band_name))
        tags = {
            key: str(value)
            for key, value in {
                "sensor": metadata.get("sensor"),
                "acquired_at": metadata.get("acquired_at"),
                "cloud_cover": metadata.get("cloud_cover"),
                "processing_level": metadata.get("processing_level"),
                "source_uri": metadata.get("source_uri"),
                "index_name": metadata.get("index_name"),
                "model_id": metadata.get("model_id"),
            }.items()
            if value is not None
        }
        if tags:
            dataset.update_tags(**tags)
    return str(path.resolve())


def _default_band_names(count: int) -> list[str]:
    default_names = ["B02", "B03", "B04", "B08", "B11"]
    if count <= len(default_names):
        return default_names[:count]
    return [f"B{index:02d}" for index in range(1, count + 1)]


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _demo_raster(uri: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(42)
    height, width, bands = 128, 128, 5
    data = np.zeros((height, width, bands), dtype="float32")
    data[:, :, 0] = 0.12 + rng.normal(0, 0.01, (height, width))  # Blue
    data[:, :, 1] = 0.18 + rng.normal(0, 0.01, (height, width))  # Green
    data[:, :, 2] = 0.20 + rng.normal(0, 0.01, (height, width))  # Red
    data[:, :, 3] = 0.55 + rng.normal(0, 0.02, (height, width))  # NIR
    data[:, :, 4] = 0.22 + rng.normal(0, 0.02, (height, width))  # SWIR

    _paint_builtup(data, 24, 26, 48, 52)
    if uri.endswith("image_t2"):
        _paint_builtup(data, 56, 45, 86, 78)
        _paint_builtup(data, 72, 82, 98, 104)

    data = np.clip(data, 0, 1)
    metadata = {
        "source_uri": uri,
        "width": width,
        "height": height,
        "count": bands,
        "dtype": "float32",
        "crs": "EPSG:32650",
        "bbox": [120.10, 30.10, 120.22, 30.22],
        "resolution": [10.0, 10.0],
        "band_names": ["B02", "B03", "B04", "B08", "B11"],
        "sensor": "Sentinel-2",
        "acquired_at": "2022-07-01" if uri.endswith("image_t1") else "2025-07-01",
        "cloud_cover": 2.0,
        "processing_level": "demo_L2A",
        "nodata": None,
    }
    return data, metadata


def _paint_builtup(data: np.ndarray, row0: int, col0: int, row1: int, col1: int) -> None:
    data[row0:row1, col0:col1, 0] = 0.28
    data[row0:row1, col0:col1, 1] = 0.30
    data[row0:row1, col0:col1, 2] = 0.34
    data[row0:row1, col0:col1, 3] = 0.28
    data[row0:row1, col0:col1, 4] = 0.68
