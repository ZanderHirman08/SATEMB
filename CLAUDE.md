# Front Range Satellite Embeddings (SATEMB)

## What this is
A portfolio project understanding what a satellite-imagery foundation model (Clay v1.5)
actually encodes, using the Boulder/Denver Front Range, Colorado as the primary study area.
GitHub: https://github.com/ZanderHirman08/SATEMB — Live map: https://zanderhirman08.github.io/SATEMB/
— Analysis Log: https://zanderhirman08.github.io/SATEMB/log.html — Paper:
https://zanderhirman08.github.io/SATEMB/paper.html

**Status as of 2026-08-29: the six originally planned exploratory experiments are all
complete** (embedding arithmetic, similarity heatmap, elevation correlation, Marshall Fire
before/after, seasonal stability, cross-region generalization to Seattle), plus a growing set of
unplanned follow-ups chasing open questions those six raised:
- Notebook 08 (fire, patch-level retest): **done** — dilution ruled out, an unexplained inverse
  correlation found instead. Its SCL-based follow-up diagnostic (snow/cloud-shadow contamination
  as the cause) is also **done**: mixed result, not a clean resolution — the direct max-shift-vs-
  typical-patch comparison actually ran backwards, but modest significant correlations (r=-0.33
  w/ dNBR, r=0.36 w/ shift) plus one visually unambiguous example (a lake freezing over between
  the two dates) confirm non-fire seasonal content is part of the story without fully explaining
  the original r=-0.55.
- `analysis/embedding_arithmetic_verified_anchors.js` (manually-verified vector-arithmetic
  anchors): **done**, fully local — compositionality gap widened from 0.123 to 0.189; also found
  no chip in the 725-chip grid is actually majority open water.
- Notebook 09 (multi-season composite, cropland/grassland confusion): **done** — overall ARI
  improved (0.275 to 0.347) but Cropland/Grassland separation specifically got *worse* (purity
  0.859→0.587 and 0.329→0.286), rejecting the phenology hypothesis; likely explanation is the
  second season reinforcing the existing terrain/snow axis rather than adding a phenology signal.

The project has three documentation surfaces that all need updating together when a new finding
lands — see "Writing up a new finding" below. Check `docs/log.html`'s last entry and recent git
log for what's actually been done most recently; this file won't stay current on its own.

## Environment constraints (important, easy to forget)
- **No Python on the local machine.** Only git + bash + node are available locally. All
  Sentinel-2 fetching, chipping, and Clay embedding happens in **Google Colab** (the user has
  free-tier T4 GPU access). This session's job is authoring notebooks/code and managing git;
  the user runs notebooks in Colab and reports back results/errors.
- Colab workflow: notebooks clone the repo fresh each session (`REPO_URL` in the first cell).
  **Editing a notebook here and pushing does NOT update a notebook tab the user already has
  open in their browser** — `git pull` inside Colab updates files on disk but not the
  in-memory cell content of an already-open tab. Always tell the user to close and reopen the
  notebook fresh from the `colab.research.google.com/github/...` link after any edit here.
- **Free-tier Colab GPU quota runs out.** When it does, check whether the notebook's AOI is
  small (a self-contained follow-up like notebooks 05/08, ~50 chips, not the 725-chip main
  pipeline) — those are small enough to fall back to a plain CPU runtime instead of waiting out
  the quota, just slower. Pattern: detect with `torch.cuda.is_available()` and set
  `device = "cpu"` rather than hard-asserting on CUDA (see notebook 08's setup cell).
- Committing results back from Colab uses a GitHub personal access token stored as a Colab
  Secret (`GITHUB_TOKEN`, repo write access). Standard cell to add at the end of a notebook
  (this snippet only exists here now — it was previously typed live in Colab each time and
  never actually landed in a tracked file, so don't bother searching git history for it):
  ```python
  from google.colab import userdata

  GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
  GITHUB_USER = "ZanderHirman08"
  REPO_NAME = "SATEMB"

  !git config user.email "zander_hirman08@outlook.com"
  !git config user.name "ZanderHirman08"

  remote_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git"

  !git add docs/figures/<whatever_this_notebook_wrote>*.png
  !git commit -m "..."
  !git push {remote_url} main
  ```
  Never `print(remote_url)` or otherwise let it hit notebook output — it embeds the raw token.
- Local dev preview: `.claude/launch.json` has a `docs-static-site` config (`npx serve docs -l
  8080`) for previewing the map/log/paper locally via the Browser pane tools.
- The Browser pane in this environment does not composite frames unless actively displayed to
  the user, which breaks anything relying on `requestAnimationFrame` (e.g. MapLibre's `load`
  event) and screenshot verification. Work around it by calling the same setup logic directly
  via `javascript_tool` instead of waiting on browser events, and verify via DOM/network state
  rather than pixels.

## Repo structure
- `notebooks/01-07` — the pipeline and follow-up experiments, run in Colab in order.
  01-03 build the main 725-chip dataset; 04-07 are independent follow-ups (04 and 06 read the
  committed `docs/data/` files directly with no Drive round-trip; 05 and 07 are fully
  self-contained with their own small AOIs).
- `src/` — shared helpers (`stac_utils.py`, `clay_embed.py`, `viz_utils.py`) imported by every
  notebook. Check here first before writing new STAC-search or embedding logic — most patterns
  already exist (e.g. `select_clearest_scene` for snow/cloud-aware date picking,
  `select_least_cloudy_per_tile` for multi-tile AOI coverage).
- `docs/` — the live site: `index.html` (interactive map), `log.html` (informal findings
  journal, chronological, includes failed attempts), `paper.html` (formal IMRaD+ academic
  write-up), `data/chips.geojson` + `embeddings.bin` (the committed dataset the map and
  `analysis/` scripts read directly, no Colab needed), `figures/` (everything referenced by
  the log and paper).
- `analysis/` — scripts that run locally against committed `docs/data/` files, no Colab needed
  (e.g. `embedding_arithmetic.js`).

## Writing up a new finding
Established three-tier pattern — update as much of this as the finding warrants:
1. **`docs/log.html`** — always. Full narrative including dead ends, exact numbers, at least
   one figure.
2. **`README.md`** — a concise paragraph closing whatever open question the finding answers.
3. **`docs/paper.html`** — only if formal/substantial. Keep within IMRaD+ length targets
   (Abstract 150-250 words, Conclusion 200-350, total body 3000-4500) — this has needed trimming
   after nearly every addition; check word counts with a quick node script before considering it
   done, and compress older supplementary-results prose rather than letting it grow unbounded.

## Collaboration notes
- **Never write a precise number (correlation, percentage, similarity score) into any doc from
  reading a chart.** Always ask the user to paste the notebook's actual printed output first.
  This was a hard rule established early and held throughout — a figure alone is not a citable
  number.
- When a Colab run fails, dig for root cause rather than suggesting "try again" — the real bugs
  so far were substantive (an xarray attribute/method name collision, a seasonal confound in
  date selection, `eo:cloud_cover` metadata not accounting for snow, T4 GPU OOM from repeated
  embedding calls in one session) and each got a real code fix pushed before asking for a re-run.
- Verify citations/facts for the paper via web search, not from training-data memory —
  citation accuracy was explicit in the user's requirements.
- The user is comfortable with reasonable autonomous calls (region/AOI choices, which
  correlations to check, notebook structure) — brief confirmation, then execute in full rather
  than over-asking.
