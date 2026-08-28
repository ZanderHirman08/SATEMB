# Front Range Satellite Embeddings

**Live map:** _fill in after enabling GitHub Pages —_ `https://zanderhirman08.github.io/SATEMB/`

An unsupervised, embedding-only map of the Colorado Front Range (Boulder → Denver), built to
answer a specific question: **what does a satellite-imagery foundation model's embedding vector
actually contain?** Not "can I use it," but "what's really in there" — is it mostly vegetation?
Built-up-ness? Something texture-based that doesn't reduce to a spectral index at all?

> **TODO:** once notebook 03 has run, embed `outputs/figures/pca_semantic_map.png` here — every
> chip colored by PCA-projecting its 1024-dim [Clay v1.5](https://github.com/Clay-foundation/model)
> embedding down to RGB, with zero supervision. Whatever structure shows up is a direct readout
> of what the model learned, e.g. `![PCA semantic map](outputs/figures/pca_semantic_map.png)`.

## Why this AOI

The bounding box (`stac_utils.FRONT_RANGE_BBOX`) deliberately packs in a lot of different kinds
of ground truth within one Sentinel-2 mosaic: Denver's street grid, Boulder's suburban fringe,
irrigated farmland out on the plains, forested foothills climbing toward the Rockies, Boulder
Reservoir, and the scar of the December 2021 Marshall Fire near Superior/Louisville. If an
embedding space is capturing anything real, these should end up visually and numerically distinct.

## Method

1. **[`01_fetch_and_chip.ipynb`](notebooks/01_fetch_and_chip.ipynb)** — search
   [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) for low-cloud
   Sentinel-2 L2A scenes over the AOI, composite the least-cloudy few, reproject to UTM 13N, and
   cut into a grid of 224×224px (2.24km) chips.
2. **[`02_generate_embeddings.ipynb`](notebooks/02_generate_embeddings.ipynb)** — run every chip
   through [Clay v1.5](https://github.com/Clay-foundation/model), a metadata-conditioned
   foundation model: it takes not just pixels but the physical wavelength of each band and a
   rough time/location, and returns a 1024-dim embedding per chip.
3. **[`03_analyze_and_export.ipynb`](notebooks/03_analyze_and_export.ipynb)** — the actual
   analysis:
   - PCA-project embeddings to RGB (the map's default color mode)
   - correlate PCA components against hand-computed NDVI / NDBI / NDWI (interpretable band math)
   - cluster the embeddings (KMeans) and check agreement against
     [ESA WorldCover](https://esa-worldcover.org/en) 10m land cover — an external, human-labeled
     signal the model never saw
   - pick a handful of chips (forest, urban, water, the Marshall Fire scar) and inspect their
     nearest neighbors in embedding space against true-color thumbnails
   - export the static `docs/data/chips.geojson` + `docs/data/embeddings.bin` the live map reads
4. **[`docs/`](docs/)** — a static [MapLibre GL JS](https://maplibre.org/) map, no backend: toggle
   between PCA-projection and cluster coloring, click any chip to compute its 10 nearest
   neighbors by cosine similarity, entirely client-side, against the real 1024-dim vectors.

## What Clay's embeddings actually encode

_TODO after running notebook 03 — replace this section with your actual numbers._

- Top-3 PCA components explained **_XX%_** of embedding variance.
- Correlation of PCA components with NDVI / NDBI / NDWI: **_paste the table notebook 03 prints_**.
- Adjusted Rand Index between KMeans clusters and ESA WorldCover classes: **_XX_** (compare
  against a shuffled-label baseline, not against 1.0 — unsupervised clusters won't map 1:1 onto
  WorldCover's categories even when they're picking up real structure).
- Nearest-neighbor sanity checks: **_did the forest/urban/water/burn-scar queries return
  visually-similar neighbors? Paste a thumbnail grid from `outputs/figures/neighbors_*.png`._**

## Running it yourself

This repo splits cleanly into "author locally, run in Colab": everything here is already
written, but generating real embeddings needs a GPU, which is why the actual execution happens
in Google Colab rather than on this machine.

1. Push this repo to GitHub (see below).
2. Open each notebook directly from GitHub in Colab — no upload needed:
   ```
   https://colab.research.google.com/github/ZanderHirman08/SATEMB/blob/main/notebooks/01_fetch_and_chip.ipynb
   ```
   (swap the filename for `02_generate_embeddings.ipynb`, then `03_analyze_and_export.ipynb`)
3. In each notebook's first cell, set `REPO_URL` to your repo's clone URL.
4. Runtime → Change runtime type → **T4 GPU**.
5. Run 01 → 02 → 03, top to bottom, in order. Notebook 03's last cell writes
   `docs/data/chips.geojson`, `docs/data/embeddings.bin`, and `outputs/figures/*.png`.
6. Download those files from Colab's file browser and commit them back to the repo (or clone the
   repo inside Colab with a GitHub token in Colab Secrets and push directly from there).
7. Enable GitHub Pages: repo Settings → Pages → Deploy from branch → `main` / `/docs`.

## Repo layout

```
notebooks/    the actual pipeline (run in Colab, in order 01 -> 02 -> 03)
src/          helper modules the notebooks import (STAC search, Clay wrapper, PCA/export)
docs/         the static site (GitHub Pages source) + the two generated data files
outputs/      static figures for this README, written by notebook 03
```

## Credits

Imagery: [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) (ESA/Copernicus)
via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/). Land cover:
[ESA WorldCover](https://esa-worldcover.org/en). Model:
[Clay Foundation Model](https://github.com/Clay-foundation/model) v1.5. Map:
[MapLibre GL JS](https://maplibre.org/) + [CARTO Basemaps](https://carto.com/basemaps).
