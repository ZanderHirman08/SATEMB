// Embedding arithmetic, take two: does the compositionality gap (0.951 vs.
// 0.828 baseline in the original test) improve with manually verified pure
// exemplars instead of whole-cluster means and a single proximity-based pick?
//
// The original analysis/embedding_arithmetic.js used the mean of cluster 6
// ("Built-up") and cluster 1 ("Tree cover") wholesale, plus a single chip
// picked by nearest centroid distance to Boulder Reservoir for "water" --
// none of those three anchors were ever actually looked at. This script
// fetches real Esri World Imagery tiles for the most promising candidates in
// each concept (see scratchpad/fetch_candidate_tiles.js) and builds anchors
// only from chips visually confirmed to be dominated by that concept.
//
// Runs entirely against files already committed in docs/data/ -- no Colab, no
// GPU, no Python needed.
//
// Usage: node analysis/embedding_arithmetic_verified_anchors.js

const fs = require("fs");

const gj = JSON.parse(fs.readFileSync("docs/data/chips.geojson", "utf8"));
const features = gj.features;
const buf = fs.readFileSync("docs/data/embeddings.bin");
const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const DIM = flat.length / features.length;

const idOf = (id) => features.findIndex((f) => f.properties.id === id);
function vec(i) { return flat.subarray(i * DIM, (i + 1) * DIM); }
function meanVec(indices) {
  const out = new Float64Array(DIM);
  indices.forEach((i) => { const v = vec(i); for (let d = 0; d < DIM; d++) out[d] += v[d]; });
  for (let d = 0; d < DIM; d++) out[d] /= indices.length;
  return out;
}
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let d = 0; d < DIM; d++) { dot += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}
function centroid(feature) {
  const coords = feature.geometry.coordinates[0];
  const lons = coords.map((c) => c[0]);
  const lats = coords.map((c) => c[1]);
  return [(Math.min(...lats) + Math.max(...lats)) / 2, (Math.min(...lons) + Math.max(...lons)) / 2];
}

// --- Manually verified exemplars ---
// Each ID below was visually checked against a real Esri World Imagery tile
// (see scratchpad/fetch_candidate_tiles.js) before being included, not just
// picked by an index extremum or cluster membership.
//
// Urban: dense, near-total built-up in true color. Candidate r021c019 was
// dropped after inspection -- despite low NDVI and cluster-6 membership, its
// tile shows Sloan's Lake covering roughly a third of the chip.
const URBAN_IDS = ["r026c021", "r029c021", "r032c025", "r025c016"];
// Vegetation: dense, near-continuous forest canopy. Candidate r031c012 was
// dropped -- only ~60% forest, with substantial open grass/bare ground.
const VEG_IDS = ["r030c001", "r032c012", "r025c003", "r033c012"];
// Water: honest finding -- no chip in this 725-chip grid is majority open
// water. These four have the most visible open water of anything checked
// (including the chip originally used as the single water anchor,
// r009c012, which turned out on inspection to be mostly suburb/farmland
// with only a sliver of Boulder Reservoir visible). r021c019 is included
// here instead of in the urban set, for the same reason it was dropped
// from urban -- Sloan's Lake is the single largest contiguous water body
// found in any candidate chip.
const WATER_IDS = ["r007c013", "r009c013", "r010c012", "r021c019"];

const urbanIdxs = URBAN_IDS.map(idOf);
const vegIdxs = VEG_IDS.map(idOf);
const waterIdxs = WATER_IDS.map(idOf);
[...urbanIdxs, ...vegIdxs, ...waterIdxs].forEach((i, n) => {
  if (i < 0) throw new Error(`Chip ID not found in chips.geojson (index ${n})`);
});

const urban = meanVec(urbanIdxs);
const veg = meanVec(vegIdxs);
const water = meanVec(waterIdxs);

console.log("Verified-anchor construction:");
console.log(`  urban:      mean of ${urbanIdxs.length} visually-confirmed built-up chips (${URBAN_IDS.join(", ")})`);
console.log(`  vegetation: mean of ${vegIdxs.length} visually-confirmed forest chips (${VEG_IDS.join(", ")})`);
console.log(`  water:      mean of ${waterIdxs.length} best-available (NOT majority-water) chips (${WATER_IDS.join(", ")})`);
console.log("  -- no chip in this AOI is majority open water; this anchor is honestly imperfect.\n");

const result = new Float64Array(DIM);
for (let d = 0; d < DIM; d++) result[d] = urban[d] - veg[d] + water[d];

const excludeSet = new Set([...urbanIdxs, ...vegIdxs, ...waterIdxs]);
const sims = features.map((f, i) => (excludeSet.has(i) ? -Infinity : cosine(result, vec(i))));
const ranked = sims.map((s, i) => [i, s]).sort((a, b) => b[1] - a[1]).slice(0, 10);

console.log("Top 10 chips nearest to (verified urban - verified vegetation + verified water):");
ranked.forEach(([i, s]) => {
  const f = features[i];
  const [lat, lon] = centroid(f);
  console.log(`  ${f.properties.id}  sim=${s.toFixed(3)}  cluster=${f.properties.cluster}  ndvi=${f.properties.ndvi.toFixed(3)}  (${lat.toFixed(4)}, ${lon.toFixed(4)})`);
});

let randSum = 0, randN = 0;
for (let i = 0; i < features.length; i++) {
  if (excludeSet.has(i)) continue;
  randSum += cosine(result, vec(i));
  randN++;
}
const baseline = randSum / randN;
const top1 = ranked[0][1];

console.log(`\nMean similarity to result across all other chips (baseline): ${baseline.toFixed(3)}`);
console.log(`Top-1 similarity: ${top1.toFixed(3)}`);
console.log(`Gap (top-1 minus baseline): ${(top1 - baseline).toFixed(3)}`);
console.log(`\nOriginal test's gap (whole-cluster / single-chip anchors): 0.951 - 0.828 = 0.123`);
console.log(`(rerun analysis/embedding_arithmetic.js to reproduce that number directly)`);
