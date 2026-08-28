"""PCA/UMAP/clustering and GeoJSON + binary export helpers for notebook 03.

The web map (docs/app.js) expects exactly two generated files:

  docs/data/chips.geojson
    FeatureCollection of chip polygons. Each feature's properties must include:
      id            (str, matches row order in embeddings.bin)
      pca_color     (str, "#rrggbb" -- PCA-projected embedding as a color)
      cluster       (int, KMeans/HDBSCAN cluster id, -1 for noise)
      ndvi          (float, mean NDVI of the chip, for the info panel)

  docs/data/embeddings.bin
    Raw little-endian float32, shape (N, D) flattened row-major, where N is
    the number of features in chips.geojson (same order) and D is the
    embedding dimensionality (1024 for Clay v1.5). app.js reads this as a
    Float32Array and reshapes it implicitly via stride D for client-side
    cosine-similarity nearest-neighbor search.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def pca_to_hex_colors(embeddings: np.ndarray, seed: int = 0) -> list[str]:
    """Project embeddings to 3 components and map to #rrggbb hex strings.

    This is the core "semantic map" trick: patches whose embeddings are
    similar get similar colors, with zero supervision -- whatever visual
    structure shows up (does it track land cover? terrain? something else?)
    is a direct, honest readout of what the model actually encodes.
    """
    pca = PCA(n_components=3, random_state=seed)
    projected = pca.fit_transform(embeddings)
    scaled = MinMaxScaler().fit_transform(projected)
    rgb = (scaled * 255).astype(np.uint8)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb], pca.explained_variance_ratio_


def cluster_embeddings(embeddings: np.ndarray, method: str = "kmeans", n_clusters: int = 8, seed: int = 0):
    """Cluster embeddings for the map's "by cluster" color mode.

    KMeans is the default (deterministic cluster count, easy to reason
    about); HDBSCAN is offered as an alternative that can surface an
    unknown number of clusters and flag outlier chips as noise (-1).
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(embeddings)
    elif method == "hdbscan":
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return labels


def nearest_neighbors(embeddings: np.ndarray, query_idx: int, k: int = 10) -> list[int]:
    """Cosine-similarity nearest neighbors of embeddings[query_idx] (excluding itself).

    Mirrors the client-side search in docs/app.js -- useful for sanity-checking
    a handful of chips server-side in notebook 03 before trusting the map.
    """
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = norm @ norm[query_idx]
    order = np.argsort(-sims)
    order = order[order != query_idx]
    return order[:k].tolist()


def export_chips_geojson(chips_meta: list[dict], pca_colors: list[str], clusters: np.ndarray,
                          ndvi: np.ndarray, out_path: str) -> None:
    """Write the FeatureCollection consumed by docs/app.js."""
    features = []
    for meta, color, cluster, ndvi_val in zip(chips_meta, pca_colors, clusters, ndvi):
        minx, miny, maxx, maxy = meta["bounds"]
        polygon = [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": polygon},
                "properties": {
                    "id": meta["id"],
                    "pca_color": color,
                    "cluster": int(cluster),
                    "ndvi": float(ndvi_val),
                },
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"Wrote {len(features)} chip features to {out_path}")


def export_embeddings_bin(embeddings: np.ndarray, out_path: str) -> None:
    """Write embeddings as raw little-endian float32 for docs/app.js to fetch()."""
    embeddings.astype("<f4").tofile(out_path)
    n, d = embeddings.shape
    print(f"Wrote {n}x{d} float32 embeddings ({embeddings.nbytes / 1e6:.1f} MB) to {out_path}")
