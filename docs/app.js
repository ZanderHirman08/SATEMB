// Front Range Satellite Embeddings -- interactive map.
//
// Reads two static, precomputed files (produced by notebooks/03_analyze_and_export.ipynb):
//   data/chips.geojson  - one polygon per image chip, with pca_color / cluster / ndvi properties
//   data/embeddings.bin - raw little-endian float32, shape (N, D) row-major, same feature order
//                         as chips.geojson, used to compute cosine-similarity nearest neighbors
//                         entirely client-side (no backend).

const FRONT_RANGE_CENTER = [-105.15, 39.93];
const CLUSTER_PALETTE = [
  "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
  "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
];
const NOISE_COLOR = "#555555"; // HDBSCAN cluster == -1

// Default similarity-mode reference: the chip nearest Boulder Reservoir, so
// switching into this mode shows something immediately, before any click.
const BOULDER_RESERVOIR = [40.0800, -105.2280];

// Sequential color ramp for continuous cosine-similarity coloring. Stops
// chosen from this dataset's observed range (baseline mean ~0.83, top
// matches ~0.95+) -- MapLibre clamps outside the defined range rather than
// extrapolating, so this is safe even if a query's true range is narrower.
const SIMILARITY_STOPS = [
  0.6, "#151a30",
  0.75, "#2f5f8a",
  0.85, "#35b6a8",
  0.93, "#ffcf5c",
  1.0, "#ffffff",
];

let chipsData = null;   // parsed GeoJSON FeatureCollection
let embeddings = null;  // Float32Array, length N * dim
let embeddingDim = 0;
let colorMode = "pca_color";
let similarityReferenceIdx = null; // index into chipsData.features, or null

const map = new maplibregl.Map({
  container: "map",
  center: FRONT_RANGE_CENTER,
  zoom: 9,
  style: "style.json",
});

map.addControl(new maplibregl.NavigationControl(), "bottom-right");

function clusterColorExpression() {
  const expr = ["match", ["to-number", ["get", "cluster"]]];
  for (let i = 0; i < CLUSTER_PALETTE.length; i++) {
    expr.push(i, CLUSTER_PALETTE[i]);
  }
  expr.push(NOISE_COLOR); // fallback, covers -1 (noise) and anything unmapped
  return expr;
}

function fillColorExpression(mode) {
  if (mode === "cluster") return clusterColorExpression();
  if (mode === "similarity") return ["interpolate", ["linear"], ["feature-state", "similarity"], ...SIMILARITY_STOPS];
  return ["get", "pca_color"];
}

function renderLegend(mode) {
  const legend = document.getElementById("legend");
  if (mode === "pca_color") {
    legend.innerHTML =
      '<span style="color: var(--text-dim)">Colors = PCA(embedding) &rarr; RGB. ' +
      "Similar color = similar embedding, not a fixed category.</span>";
    return;
  }
  if (mode === "similarity") {
    const refId = similarityReferenceIdx !== null ? chipsData.features[similarityReferenceIdx].properties.id : "?";
    const stopColors = SIMILARITY_STOPS.filter((_, i) => i % 2 === 1);
    legend.innerHTML = `
      <div style="width:100%">
        <div class="similarity-gradient" style="background:linear-gradient(90deg, ${stopColors.join(", ")})"></div>
        <div style="display:flex; justify-content:space-between; color:var(--text-dim); font-size:10.5px; margin-top:3px;">
          <span>less similar</span><span>more similar</span>
        </div>
        <div style="color:var(--text-dim); margin-top:4px;">Reference chip: <strong style="color:var(--text)">${refId}</strong> &mdash; click any chip to re-center</div>
      </div>`;
    return;
  }
  const clusterIds = [...new Set(chipsData.features.map((f) => f.properties.cluster))].sort(
    (a, b) => a - b
  );
  legend.innerHTML = clusterIds
    .map((c) => {
      const color = c === -1 ? NOISE_COLOR : CLUSTER_PALETTE[c % CLUSTER_PALETTE.length];
      const label = c === -1 ? "noise" : `cluster ${c}`;
      return `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${label}</span>`;
    })
    .join("");
}

function setColorMode(mode) {
  colorMode = mode;
  if (map.getLayer("chips-fill")) {
    map.setPaintProperty("chips-fill", "fill-color", fillColorExpression(mode));
  }
  document.querySelectorAll("#color-toggle button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  if (mode === "similarity" && similarityReferenceIdx === null) {
    const [lat, lon] = BOULDER_RESERVOIR;
    applySimilarityReference(closestChipTo(lat, lon));
    return; // applySimilarityReference() already re-renders the legend
  }
  if (chipsData) renderLegend(mode);
}

function chipCentroid(feature) {
  const coords = feature.geometry.coordinates[0];
  const lats = coords.map((c) => c[1]);
  const lons = coords.map((c) => c[0]);
  return [(Math.min(...lats) + Math.max(...lats)) / 2, (Math.min(...lons) + Math.max(...lons)) / 2];
}

function closestChipTo(lat, lon) {
  let best = -1, bestDist = Infinity;
  chipsData.features.forEach((f, i) => {
    const [flat, flon] = chipCentroid(f);
    const d = (flat - lat) ** 2 + (flon - lon) ** 2;
    if (d < bestDist) { bestDist = d; best = i; }
  });
  return best;
}

function applySimilarityReference(queryIdx) {
  similarityReferenceIdx = queryIdx;
  const query = embeddings.subarray(queryIdx * embeddingDim, (queryIdx + 1) * embeddingDim);
  chipsData.features.forEach((f, i) => {
    const vec = embeddings.subarray(i * embeddingDim, (i + 1) * embeddingDim);
    const sim = cosineSimilarity(query, vec, embeddingDim);
    map.setFeatureState({ source: "chips", id: f.properties.id }, { similarity: sim });
  });
  map.setFilter("chips-highlight", ["in", ["get", "id"], ["literal", [chipsData.features[queryIdx].properties.id]]]);
  renderLegend("similarity");
}

function cosineSimilarity(vecA, vecB, dim) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < dim; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-9);
}

function nearestNeighbors(queryIdx, k = 10) {
  const n = chipsData.features.length;
  const query = embeddings.subarray(queryIdx * embeddingDim, (queryIdx + 1) * embeddingDim);
  const sims = [];
  for (let i = 0; i < n; i++) {
    if (i === queryIdx) continue;
    const vec = embeddings.subarray(i * embeddingDim, (i + 1) * embeddingDim);
    sims.push([i, cosineSimilarity(query, vec, embeddingDim)]);
  }
  sims.sort((a, b) => b[1] - a[1]);
  return sims.slice(0, k);
}

function showInfo(feature, neighborIds) {
  const p = feature.properties;
  const info = document.getElementById("info");
  info.classList.remove("info-empty");
  info.innerHTML = `
    <dl style="margin:0">
      <div><dt>Chip</dt>${p.id}</div>
      <div><dt>Cluster</dt>${p.cluster === -1 ? "noise" : p.cluster}</div>
      <div><dt>NDVI</dt>${p.ndvi.toFixed(3)}</div>
      <div><dt>Neighbors</dt>${neighborIds.length} highlighted (outlined)</div>
    </dl>`;
}

function showSimilarityInfo(feature) {
  const p = feature.properties;
  const info = document.getElementById("info");
  info.classList.remove("info-empty");
  info.innerHTML = `
    <dl style="margin:0">
      <div><dt>Reference</dt>${p.id}</div>
      <div><dt>Cluster</dt>${p.cluster === -1 ? "noise" : p.cluster}</div>
      <div><dt>NDVI</dt>${p.ndvi.toFixed(3)}</div>
      <div><dt>Showing</dt>every chip's similarity to this one</div>
    </dl>`;
}

async function init() {
  const [geojsonResp, binResp] = await Promise.all([
    fetch("data/chips.geojson"),
    fetch("data/embeddings.bin"),
  ]);
  chipsData = await geojsonResp.json();
  const buffer = await binResp.arrayBuffer();
  embeddings = new Float32Array(buffer);
  embeddingDim = embeddings.length / chipsData.features.length;

  map.on("load", () => {
    map.addSource("chips", {
      type: "geojson",
      data: chipsData,
      promoteId: "id",
    });

    map.addLayer({
      id: "chips-fill",
      type: "fill",
      source: "chips",
      paint: {
        "fill-color": fillColorExpression(colorMode),
        "fill-opacity": 0.75,
      },
    });

    map.addLayer({
      id: "chips-outline",
      type: "line",
      source: "chips",
      paint: { "line-color": "rgba(255,255,255,0.15)", "line-width": 0.5 },
    });

    map.addLayer({
      id: "chips-highlight",
      type: "line",
      source: "chips",
      paint: { "line-color": "#ffffff", "line-width": 2.5 },
      filter: ["in", ["get", "id"], ["literal", []]],
    });

    renderLegend(colorMode);

    map.on("mouseenter", "chips-fill", () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", "chips-fill", () => (map.getCanvas().style.cursor = ""));

    map.on("click", "chips-fill", (e) => {
      const feature = e.features[0];
      const queryIdx = chipsData.features.findIndex((f) => f.properties.id === feature.properties.id);
      if (queryIdx === -1) return;

      if (colorMode === "similarity") {
        applySimilarityReference(queryIdx);
        showSimilarityInfo(feature);
        return;
      }

      const neighbors = nearestNeighbors(queryIdx, 10);
      const neighborIds = neighbors.map(([idx]) => chipsData.features[idx].properties.id);

      map.setFilter("chips-highlight", ["in", ["get", "id"], ["literal", [feature.properties.id, ...neighborIds]]]);
      showInfo(feature, neighborIds);
    });
  });
}

document.getElementById("color-toggle").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (btn) setColorMode(btn.dataset.mode);
});

init();
