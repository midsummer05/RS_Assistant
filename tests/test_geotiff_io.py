import numpy as np
import rasterio
from rasterio.transform import from_origin

from rs_agent.tools.raster.io import load_raster, preferred_raster_suffix, save_raster


def test_load_and_save_geotiff_with_metadata(tmp_path):
    source = tmp_path / "source.tif"
    transform = from_origin(120.0, 31.0, 10.0, 10.0)
    data = np.zeros((5, 8, 9), dtype="float32")
    data[0] = 0.1
    data[1] = 0.2
    data[2] = 0.3
    data[3] = 0.5
    data[4] = 0.7

    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=8,
        width=9,
        count=5,
        dtype="float32",
        crs="EPSG:32650",
        transform=transform,
    ) as dataset:
        dataset.write(data)
        for index, name in enumerate(["B02", "B03", "B04", "B08", "B11"], start=1):
            dataset.set_band_description(index, name)
        dataset.update_tags(sensor="Sentinel-2", acquired_at="2025-07-01", cloud_cover="3.5")

    loaded, metadata = load_raster(str(source))

    assert loaded.shape == (8, 9, 5)
    assert metadata["crs"] == "EPSG:32650"
    assert metadata["band_names"] == ["B02", "B03", "B04", "B08", "B11"]
    assert metadata["sensor"] == "Sentinel-2"
    assert metadata["cloud_cover"] == 3.5
    assert preferred_raster_suffix(metadata) == ".tif"

    target = tmp_path / "target.tif"
    saved_uri = save_raster(target, loaded[:, :, 0], {**metadata, "band_names": ["B02"], "count": 1})
    saved, saved_metadata = load_raster(saved_uri)

    assert saved.shape == (8, 9)
    assert saved_metadata["crs"] == "EPSG:32650"
    assert saved_metadata["band_names"] == ["B02"]

