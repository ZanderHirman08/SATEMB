"""STAC search, signing, and chip-grid helpers for the Front Range imagery pipeline.

Used by notebooks/01_fetch_and_chip.ipynb (Sentinel-2 L2A),
notebooks/03_analyze_and_export.ipynb (ESA WorldCover, for validation),
notebooks/04_elevation_correlation.ipynb (USGS 3DEP elevation, for validation),
notebooks/05_fire_before_after.ipynb (pre/post Marshall Fire comparison),
notebooks/06_seasonal_stability.ipynb (summer-vs-winter embedding comparison),
and notebooks/07_cross_region_seattle.ipynb (Front Range vs. Seattle check).
"""

from __future__ import annotations

import numpy as np
import planetary_computer
import pystac_client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Boulder -> Denver, foothills -> plains, includes Boulder Reservoir and the
# Dec-2021 Marshall Fire burn scar near Superior/Louisville, CO.
FRONT_RANGE_BBOX = [-105.55, 39.55, -104.75, 40.25]

# Tight box around the Marshall Fire's actual burned extent (Marshall Mesa
# through Superior to Louisville, CO), for notebook 05's before/after study --
# small on purpose so a GPU embedding pass over 3 dates is fast and cheap.
MARSHALL_FIRE_BBOX = [-105.25, 39.89, -105.02, 40.02]

# Seattle / Puget Sound, WA, for notebook 07's cross-region generalization
# check -- deliberately a different biome from the Front Range (marine
# climate, year-round evergreen forest, real open water at scale via Puget
# Sound and Lake Washington, a denser urban core). UTM zone 10N here, not
# 13N -- see SEATTLE_CRS below.
SEATTLE_BBOX = [-122.55, 47.45, -122.05, 47.75]
SEATTLE_CRS = "EPSG:32610"

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


def search_single_scene(
    catalog: pystac_client.Client,
    bbox: list[float],
    datetime_range: str,
    max_cloud_cover: float = 20.0,
    label: str = "",
):
    """Return the single least-cloudy Sentinel-2 scene in a date range.

    Unlike search_sentinel2() (used for the main study's multi-tile median
    composite), notebook 05's before/after comparison wants one specific,
    identifiable date per period rather than a blended composite -- blending
    would blur exactly the snow/no-snow, burned/unburned distinction the
    comparison depends on.
    """
    items = search_sentinel2(catalog, bbox=bbox, datetime_range=datetime_range, max_cloud_cover=max_cloud_cover)
    best = min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))
    tag = f"[{label}] " if label else ""
    print(f"{tag}{best.id}  date={best.datetime.date()}  cloud_cover={best.properties.get('eo:cloud_cover'):.1f}%")
    return best


# Sentinel-2 Scene Classification Layer codes for cloud shadow, cloud
# (medium/high probability), thin cirrus, and snow/ice -- see the SCL
# definition in the Sentinel-2 User Handbook (European Space Agency, 2015).
SCL_BAD_CLASSES = {3, 8, 9, 10, 11}


def select_clearest_scene(
    catalog: pystac_client.Client,
    bbox: list[float],
    datetime_range: str,
    crs: str = "EPSG:32613",
    resolution: float = 10,
    max_candidates: int = 6,
    label: str = "",
):
    """Pick the scene in a date range with the lowest actual cloud+snow pixel
    fraction *over this specific AOI*, checked directly via the SCL
    classification band -- not trusted from STAC's eo:cloud_cover metadata,
    which is computed over the whole scene (not this AOI) and says nothing
    about snow. Snow in particular matters for notebook 05: winter Front
    Range imagery routinely has patchy snow that eo:cloud_cover never flags
    at all.

    More expensive than search_single_scene() (reads real pixels for several
    candidates, not just metadata), but the AOI here is small enough that
    this stays fast.
    """
    import odc.stac

    items = search_sentinel2(catalog, bbox=bbox, datetime_range=datetime_range, max_cloud_cover=80.0)
    candidates = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))[:max_candidates]

    tag = f"[{label}] " if label else ""
    best_item, best_frac = None, 1.0
    for item in candidates:
        ds = odc.stac.load([item], bands=["SCL"], bbox=bbox, crs=crs, resolution=resolution, resampling="nearest")
        scl_var = ds["SCL"]
        scl = scl_var.isel(time=0).values if "time" in scl_var.dims else scl_var.values
        bad_frac = float(np.isin(scl, list(SCL_BAD_CLASSES)).mean())
        print(
            f"  {tag}candidate {item.id}  date={item.datetime.date()}  "
            f"scene cloud_cover={item.properties.get('eo:cloud_cover'):.1f}%  "
            f"AOI cloud+snow fraction={bad_frac * 100:.1f}%"
        )
        if bad_frac < best_frac:
            best_item, best_frac = item, bad_frac

    print(f"{tag}selected {best_item.id}  date={best_item.datetime.date()}  AOI cloud+snow fraction={best_frac * 100:.1f}%")
    return best_item


def select_least_cloudy_per_tile(items):
    """Group items by Sentinel-2 MGRS tile and return the least-cloudy item
    from each tile, so a multi-tile AOI gets full coverage instead of
    multiple scenes from one tile while another tile goes unpicked (the same
    fix applied to notebook 01's original AOI coverage bug). Snow is left in
    deliberately where relevant (e.g. notebook 06's seasonal comparison) --
    unlike select_clearest_scene(), this does not filter it out.
    """
    from collections import defaultdict

    by_tile = defaultdict(list)
    for it in items:
        tile = it.properties.get("s2:mgrs_tile", "unknown")
        by_tile[tile].append(it)

    selected = []
    for tile, tile_items in sorted(by_tile.items()):
        best = min(tile_items, key=lambda it: it.properties.get("eo:cloud_cover", 100))
        selected.append(best)
        print(f"  tile {tile}: {best.id}  cloud_cover={best.properties.get('eo:cloud_cover'):.1f}%  date={best.datetime.date()}")
    return selected


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


def search_dem(catalog: pystac_client.Client, bbox: list[float] = FRONT_RANGE_BBOX, prefer_gsd: float = 10.0):
    """Search the USGS 3DEP Seamless elevation collection over the AOI.

    3DEP covers CONUS at either 10m or 30m ground sample distance (gsd)
    depending on region -- prefer the higher-resolution 10m tiles where
    available (matches the 10m Sentinel-2 grid), falling back to whatever
    is actually returned if no 10m coverage exists for this AOI. The
    elevation values (asset "data") are in meters.
    """
    search = catalog.search(collections=["3dep-seamless"], bbox=bbox)
    items = list(search.item_collection())
    if len(items) == 0:
        raise RuntimeError(f"No 3DEP elevation tiles found for bbox={bbox}.")

    preferred = [it for it in items if it.properties.get("gsd") == prefer_gsd]
    if preferred:
        return preferred
    print(
        f"No {prefer_gsd}m 3DEP tiles found for this AOI; falling back to all "
        f"{len(items)} available tile(s) (mixed/lower resolution)."
    )
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
