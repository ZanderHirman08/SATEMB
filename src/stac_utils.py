"""STAC search, signing, and chip-grid helpers for the Front Range imagery pipeline.

Used by notebooks/01_fetch_and_chip.ipynb (Sentinel-2 L2A) and
notebooks/03_analyze_and_export.ipynb (ESA WorldCover, for validation).
"""

from __future__ import annotations

import numpy as np
import planetary_computer
import pystac_client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Boulder -> Denver, foothills -> plains, includes Boulder Reservoir and the
# Dec-2021 Marshall Fire burn scar near Superior/Louisville, CO.
FRONT_RANGE_BBOX = [-105.55, 39.55, -104.75, 40.25]

# Sentinel-2 bands Clay v1.5 was trained on for the "sentinel-2-l2a" platform,
# in the order clay_embed.py expects. B01/B09/B10 are dropped (60m atmospheric
# bands with no surface-reflectance value for this task).
S2_BANDS = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
]

CHIP_SIZE_PX = 224  # matches Clay v1.5's trained positional embedding grid
GSD_M = 10          # Sentinel-2 10m bands


def open_catalog() -> pystac_client.Client:
    """Open the Planetary Computer STAC catalog with auto-signing enabled."""
    return pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )


def search_sentinel2(
    catalog: pystac_client.Client,
    bbox: list[float] = FRONT_RANGE_BBOX,
    datetime_range: str = "2024-06-01/2024-09-15",
    max_cloud_cover: float = 10.0,
):
    """Search Sentinel-2 L2A scenes over the AOI, filtered by cloud cover.

    Returns a pystac ItemCollection. Re-run search_and_sign() right before
    reading pixels elsewhere -- signed asset URLs expire after ~1 hour.
    """
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = search.item_collection()
    if len(items) == 0:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for bbox={bbox}, "
            f"datetime={datetime_range}, cloud_cover<{max_cloud_cover}. "
            "Try widening the date range or cloud-cover threshold."
        )
    return items


def search_worldcover(catalog: pystac_client.Client, bbox: list[float] = FRONT_RANGE_BBOX):
    """Search the ESA WorldCover 10m land-cover collection over the AOI.

    Only 2020/2021 vintages exist. Used purely as an external validation
    signal in notebook 03, not as model input.
    """
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox)
    items = search.item_collection()
    if len(items) == 0:
        raise RuntimeError(f"No ESA WorldCover tiles found for bbox={bbox}.")
    return items


def make_pixel_chip_grid(height: int, width: int, chip_size_px: int = CHIP_SIZE_PX):
    """Build a regular grid of non-overlapping pixel windows over a (height, width)
    raster, dropping any partial chip at the bottom/right edge.

    Returns a list of dicts: {id, row, col, y_slice, x_slice} where the slices
    index directly into the loaded xarray/numpy raster.
    """
    n_rows = height // chip_size_px
    n_cols = width // chip_size_px

    chips = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0, y1 = row * chip_size_px, (row + 1) * chip_size_px
            x0, x1 = col * chip_size_px, (col + 1) * chip_size_px
            chips.append(
                {
                    "id": f"r{row:03d}c{col:03d}",
                    "row": row,
                    "col": col,
                    "y_slice": slice(y0, y1),
                    "x_slice": slice(x0, x1),
                }
            )
    return chips


def pixel_window_to_lonlat_bounds(x_coords, y_coords, chip, raster_crs) -> tuple[float, float, float, float]:
    """Convert a chip's pixel window into a (minlon, minlat, maxlon, maxlat) box
    in EPSG:4326, for writing GeoJSON that MapLibre can render.

    x_coords / y_coords: the raster's 1-D projected-CRS coordinate arrays
    (e.g. ds.x.values, ds.y.values from the odc-stac-loaded dataset).
    """
    from pyproj import Transformer

    xs = x_coords[chip["x_slice"]]
    ys = y_coords[chip["y_slice"]]
    px_x = abs(xs[1] - xs[0]) if len(xs) > 1 else 10.0
    px_y = abs(ys[1] - ys[0]) if len(ys) > 1 else 10.0

    minx, maxx = xs.min() - px_x / 2, xs.max() + px_x / 2
    miny, maxy = ys.min() - px_y / 2, ys.max() + px_y / 2

    transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = transformer.transform(minx, miny)
    lon1, lat1 = transformer.transform(maxx, maxy)
    return (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))


def bounds_centroid(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    """(minlon, minlat, maxlon, maxlat) -> (lat, lon) centroid, for Clay's
    location conditioning in clay_embed.make_time_latlon_tensors.
    """
    minlon, minlat, maxlon, maxlat = bounds
    return (minlat + maxlat) / 2, (minlon + maxlon) / 2
