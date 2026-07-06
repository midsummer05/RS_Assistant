import os
import sys
from pathlib import Path


def pytest_configure():
    rasterio_dir = Path(sys.prefix) / "Lib" / "site-packages" / "rasterio"
    proj_data = rasterio_dir / "proj_data"
    gdal_data = rasterio_dir / "gdal_data"
    if proj_data.exists():
        os.environ["PROJ_LIB"] = str(proj_data)
    if gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
