# Front Range Satellite Embeddings

**Live map:** [zanderhirman08.github.io/SATEMB](https://zanderhirman08.github.io/SATEMB/)

An unsupervised, embedding-only map of the Colorado Front Range (Boulder → Denver), built to
answer a specific question: **what does a satellite-imagery foundation model's embedding vector
actually contain?** Not "can I use it," but "what's really in there" — is it mostly vegetation?
Built-up-ness? Something texture-based that doesn't reduce to a spectral index at all?

![PCA semantic map](docs/figures/pca_semantic_map.png)
_Every chip colored by PCA-projecting its 1024-dim [Clay v1.5](https://github.com/Clay-foundation/model)
embedding down to RGB, with zero supervision — no labels, no land-cover data, nothing but the
raw embedding vectors. The black wedge in the upper-left is outside both Sentinel-2 tiles used
for the mosaic (a real coverage gap, not a rendering issue)._

## Why this AOI

The bounding box (`stac_utils.FRONT_RANGE_BBOX`) deliberately packs in a lot of different kinds
of ground truth within one Sentinel-2 mosaic: Denver's street grid, Boulder's suburban fringe,
irrigated farmland out on the plains, forested foothills climbing toward the Rockies, Boulder
Reservoir, and the scar of the December 2021 Marshall Fire near Superior/Louisville. If an
embedding space is capturing anything real, these should end up visually and numerically distinct.

## Method

1. **[`01_fetch_and_chip.ipynb`](notebooks/01_fetch_and_chip.ipynb)** — search
   [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) for low-cloud
   Sentinel-2 L2A scenes over the AOI (the bbox straddles two MGRS tiles, so it takes the
   least-cloudy scene from each), reproject to UTM 13N, and cut into a grid of 224×224px
   (2.24km) chips — 725 of them survive a nodata filter.
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
   between PCA-projection, cluster, and continuous-similarity-heatmap coloring; click any chip to
   compute its nearest neighbors, entirely client-side, against the real 1024-dim vectors.
5. **[`04_elevation_correlation.ipynb`](notebooks/04_elevation_correlation.ipynb)** — a lighter,
   CPU-only follow-up: fetches USGS 3DEP elevation over the AOI and checks whether any PCA
   component (extended past the top 3 examined in notebook 03) or the embedding clusters
   correlate with terrain, which Clay never saw as an input. Reads directly from the committed
   `docs/data/` files, so it needs no GPU and no Colab Drive round-trip.
6. **[`05_fire_before_after.ipynb`](notebooks/05_fire_before_after.ipynb)** — a proper before/after
   test of the inconclusive Marshall Fire result from notebook 03: fetches pre-fire, immediate
   post-fire, and 2024 long-term single dates over a tight box around the actual burn area,
   computes dNBR (an independent burn-severity index) per chip, embeds all three dates with Clay,
   and checks whether embedding movement correlates with actual burn severity — and whether that
   correlation fades by 2024, which would explain notebook 03's null result as "signal faded,"
   not "Clay never saw it." Small, self-contained AOI (~50 chips), no Drive round-trip needed.
7. **[`06_seasonal_stability.ipynb`](notebooks/06_seasonal_stability.ipynb)** — fetches a winter
   mosaic over the *same* 725-chip AOI and grid as the main study, embeds it with Clay, and
   measures how far each chip's embedding moves between summer and winter — which clusters/land
   cover are seasonally stable vs. volatile — plus a targeted ablation testing notebook 05's open
   question directly: re-embeds a sample of chips with identical pixels but a swapped date, to
   separate how much embedding movement comes from real seasonal content vs. Clay's own
   date-conditioning input. Reuses `docs/data/embeddings.bin` as the summer baseline directly.
8. **[`analysis/`](analysis/)** — small follow-up experiments that run entirely against the
   committed `docs/data/` files, no Colab needed at all (e.g. `embedding_arithmetic.js`, a
   word2vec-style vector-arithmetic test — see the [Analysis Log](https://zanderhirman08.github.io/SATEMB/log.html)).

## What Clay's embeddings actually encode

**PCA is dominated by one axis.** The top-3 components explain 82.3% of embedding variance, but
almost all of that is PC1 alone (59.4%); PC2 adds 14.2%, PC3 a further 8.6%. Most of what
distinguishes these chips from each other, embedding-wise, lives on a single dimension.

**That dominant axis tracks human development, not "vegetation" cleanly.**

| index | PC1 | PC2 | PC3 |
|---|---|---|---|
| NDVI | −0.55 | −0.05 | −0.14 |
| NDBI | 0.61 | 0.66 | −0.14 |
| NDWI | 0.34 | −0.02 | 0.24 |

PC1 is negatively correlated with vegetation and positively with built-up-ness *and* water —
read literally, it's closer to an "anything-that-isn't-bare-dry-land" axis than a pure
vegetation axis. PC2 is even more strongly tied to NDBI (0.66) but essentially uncorrelated with
NDVI/NDWI, suggesting it's picking up a *different* flavor of built-up-ness — density or texture,
maybe — orthogonal to how green a chip is. PC3 correlates weakly with everything, which likely
means it's capturing something a simple band-ratio index can't describe: texture, spatial
context, or a trace of Clay's time/location conditioning.

**Update, after checking against elevation (USGS 3DEP, not something Clay was ever given):**
PC1 correlates with raw elevation at **r = −0.82** — stronger than either spectral index above.
That reframes it: PC1 may be more fundamentally a *terrain* axis than a "development" axis, with
vegetation and built-up-ness riding along as downstream correlates, since on this landscape
elevation, land cover, and development are all confounded (mountains are forested and
undeveloped, plains are dry and built up). Extending the PCA check to 10 components also caught
two things the original top-3 check couldn't: PC2 tracks *slope* (r = 0.48) more than elevation,
plausibly a "development-on-foothill-terrain" signal distinct from PC1's plains-vs-mountains
split; and PC4, only 6.2% of variance, turns out to be a *cleaner* vegetation axis (r = −0.73
with NDVI) than dominant PC1 ever was. Full write-up in the
[Analysis Log](https://zanderhirman08.github.io/SATEMB/log.html).

![Mean elevation per embedding cluster](docs/figures/cluster_elevation.png)
_Clusters were built with zero elevation information, yet split cleanly: the three
forest-dominant clusters sit at 2,241–2,742m, the five built-up/cropland/grassland-dominant
clusters sit at 1,533–1,772m — independent confirmation the clusters track real geography._

**Clusters agree with real land cover, unevenly.** KMeans-8 over the raw 1024-dim embeddings
scores an Adjusted Rand Index of **0.275** against ESA WorldCover (0 = chance, 1 = perfect;
matched all 725/725 chips). That's a real, well-above-chance signal, not a coincidence — and
the breakdown explains why it isn't higher: one cluster is *almost entirely* "Built-up" chips,
another is *almost entirely* "Tree cover," but "Cropland" and "Grassland" bleed across several
clusters rather than each getting a clean cluster of their own.

![Cluster vs. WorldCover](docs/figures/cluster_vs_worldcover.png)

That's a sensible failure mode, not a broken pipeline: cropland and grassland can look nearly
identical spectrally in a single snapshot (their spectral difference is mostly seasonal/temporal
— irrigation timing, crop calendar — which one Sentinel-2 date can't capture), while "built-up"
and "forest" are visually and spectrally distinct enough to separate cleanly even from a single
date.

![UMAP colored by cluster and NDVI](docs/figures/umap_scatter.png)

The embedding space itself is well-organized: UMAP shows discrete, well-separated blobs (left
panel), and coloring the same layout by NDVI (right panel) produces a clean, continuous
gradient across the manifold rather than random speckle — the geometry of the embedding space
lines up with a real physical quantity it was never told about.

**Nearest neighbors mostly make sense — with one honest miss.** Querying the highest-NDVI,
highest-NDBI, and highest-NDWI chips each returns neighbors that are visually the same kind of
place (forest near forest, dense urban near dense urban, water near water). The one query aimed
at the Marshall Fire burn scar (Superior/Louisville) returned suburban-edge/grassland neighbors
that don't look distinctly "burned":

![Marshall Fire burn scar neighbors](docs/figures/neighbors_marshall_fire_burn_scar.png)

Worth stating plainly rather than glossing over: by the 2024 imagery used here, the scar is
~2.5 years regrown, and a chip centroid nearest the fire's location isn't guaranteed to land on
the most heavily burned pixels. Whether that's "Clay doesn't retain a burn signature this long"
or "this particular chip missed the scar" is exactly the kind of question this method surfaces
but can't answer by itself — a real limitation of a single 2.24km-chip, single-date snapshot, not
a marketing footnote.

**Follow-up: that open question got a real answer, and it's a clean negative.**
[`05_fire_before_after.ipynb`](notebooks/05_fire_before_after.ipynb) did the proper before/after
test — real pre-fire and post-fire dates, dNBR (an independent burn-severity index) as ground
truth, embedding shift measured directly. After two failed attempts at picking clean dates
(season mismatch, then undetected snow — see the [Analysis Log](https://zanderhirman08.github.io/SATEMB/log.html)
for the full debugging story), the properly controlled result: embedding shift shows **no
correlation** with actual burn severity, immediately after the fire (r = −0.06) or by 2024
(r = −0.01). Not "the signal faded" — a whole-chip mean-pooled embedding doesn't appear to track
fire severity at all in this sample, most likely because averaging a 2.24km chip's patch tokens
dilutes a burn signal that only covers part of the chip.

**Second follow-up: is that a metadata artifact, or real content change?** The fire study left
one thing unresolved — chips with essentially no burn severity still showed a large baseline
embedding shift, and it wasn't clear how much of that was genuine seasonal content vs. Clay's own
time-conditioning input reacting to the date label alone.
[`06_seasonal_stability.ipynb`](notebooks/06_seasonal_stability.ipynb) tests this directly: embed
a chip's real pixels with its real date, then the *same pixels* again with a fake date, and
compare. The answer is decisive — swapping only the date moves the embedding by **4%** of what a
real summer-vs-winter comparison does (0.007 vs. 0.170 average shift, n=30). Embedding shift
really is driven by pixel content, not metadata. The same notebook, run across all 725 chips,
also found forest clusters are ~2–3× more seasonally volatile than grassland (elevation and
NDVI both correlate with shift, r=0.55 and r=0.42) — a clean snow-line effect, and further
confirmation the clusters track real physical geography rather than an artifact of clustering.

## Running it yourself

This repo splits cleanly into "author locally, run in Colab": everything here is already
written, but generating real embeddings needs a GPU, which is why the actual execution happens
in Google Colab rather than on this machine.

1. Push this repo to GitHub (see below).
2. Open each notebook directly from GitHub in Colab — no upload needed, one tab per notebook:
   ```
   https://colab.research.google.com/github/ZanderHirman08/SATEMB/blob/main/notebooks/01_fetch_and_chip.ipynb
   ```
   (swap the filename for `02_generate_embeddings.ipynb`, then `03_analyze_and_export.ipynb`)
3. `REPO_URL` in each notebook's first cell is already set to this repo.
4. Runtime → Change runtime type → **T4 GPU**, for each notebook.
5. Run 01 → 02 → 03, top to bottom, in order. **Each notebook you open this way gets its own
   fresh Colab VM**, so they hand off intermediate files (`chip_pixels.npy`, `embeddings.npy`,
   etc.) through Google Drive rather than local disk — expect a one-time Drive-access prompt per
   notebook. Notebook 03's last cell writes the real `docs/data/chips.geojson`,
   `docs/data/embeddings.bin`, and `docs/figures/*.png` into the local repo checkout on its VM.
6. Commit and push those files from inside Colab (a GitHub personal access token as a Colab
   Secret works well — see the token-based `git push` snippet in this repo's history/commits for
   the exact cell), since downloading and re-uploading a binary `.bin` file by hand is error-prone.
7. Enable GitHub Pages: repo Settings → Pages → Deploy from branch → `main` / `/docs`.
8. Optional, CPU-only: run `04_elevation_correlation.ipynb` afterward for the terrain-correlation
   follow-up. It reads directly from the just-committed `docs/data/` files, so no GPU or Drive
   round-trip is needed — a plain (non-GPU) Colab runtime is enough.
9. Optional, needs GPU: run `05_fire_before_after.ipynb` for the Marshall Fire before/after test.
   It's fully self-contained (own small AOI, own three STAC fetches) and doesn't touch
   `docs/data/chips.geojson` — just commit the new `docs/figures/fire_*.png` files it writes.
10. Optional, needs GPU: run `06_seasonal_stability.ipynb` for the summer-vs-winter comparison. It
    reuses `docs/data/embeddings.bin` as the summer baseline directly (no Drive round-trip) and
    adds a new `seasonal_shift` property onto `docs/data/chips.geojson` — commit that plus the new
    `docs/figures/seasonal_*.png` files.

## Repo layout

```
notebooks/    the actual pipeline (run in Colab, in order 01 -> 02 -> 03)
src/          helper modules the notebooks import (STAC search, Clay wrapper, PCA/export)
docs/         the static site (GitHub Pages source): map, generated data, and figures
analysis/     lightweight follow-up experiments that run locally against docs/data/
              (no Colab needed -- the full embeddings are already committed there)
```

## Credits

Imagery: [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) (ESA/Copernicus)
via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/). Land cover:
[ESA WorldCover](https://esa-worldcover.org/en). Model:
[Clay Foundation Model](https://github.com/Clay-foundation/model) v1.5. Map:
[MapLibre GL JS](https://maplibre.org/) + [Esri World Imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9).
