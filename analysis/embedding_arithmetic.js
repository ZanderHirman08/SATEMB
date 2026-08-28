// Embedding arithmetic: urban - vegetation + water =~ ?
//
// The classic word2vec trick (king - man + woman ~= queen) applied to satellite
// chip embeddings, to test something stronger than "these chips cluster
// together": is the embedding space linearly structured, such that concepts
// can be added and subtracted like vectors?
//
// Runs entirely against files already committed in docs/data/ -- no Colab, no
// GPU, no Python needed. Those files hold the full 1024-dim vectors for all
// 725 real Front Range chips (from the run documented in the README).
//
// Usage: node analysis/embedding_arithmetic.js   (run from the repo root)

const fs = require("fs");

const gj = JSON.parse(fs.readFileSync("docs/data/chips.geojson", "utf8"));
const features = gj.features;
const buf = fs.readFileSync("docs/data/embeddings.bin");
const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const DIM = flat.length / features.length;

function vec(i) {
  return flat.subarray(i * DIM, (i + 1) * DIM);
}

function centroid(feature) {
  const coords = feature.geometry.coordinates[0];
  const lons = coords.map((c) => c[0]);
  const lats = coords.map((c) => c[1]);
  return [
    (Math.min(...lats) + Math.max(...lats)) / 2,
    (Math.min(...lons) + Math.max(...lons)) / 2,
  ];
}

function closestTo(lat, lon) {
  let best = -1, bestDist = Infinity;
  features.forEach((f, i) => {
    const [flat_, flon] = centroid(f);
    const d = (flat_ - lat) ** 2 + (flon - lon) ** 2;
    if (d < bestDist) { bestDist = d; best = i; }
  });
  return best;
}

function meanVec(indices) {
  const out = new Float64Array(DIM);
  indices.forEach((i) => {
    const v = vec(i);
    for (let d = 0; d < DIM; d++) out[d] += v[d];
  });
  for (let d = 0; d < DIM; d++) out[d] /= indices.length;
  return out;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let d = 0; d < DIM; d++) { dot += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

// Concept anchors, chosen from clusters already validated against ESA
// WorldCover in notebook 03 (see docs/figures/cluster_vs_worldcover.png):
//   cluster 6 = ~pure "Built-up"   -> "urban" anchor (mean of the cluster)
//   cluster 1 = ~pure "Tree cover" -> "vegetation" anchor (mean of the cluster)
//   water chips are too sparse/scattered to form their own cluster, so
//   "water" is a single chip: the one nearest Boulder Reservoir.
const urbanIdxs = features.map((f, i) => (f.properties.cluster === 6 ? i : -1)).filter((i) => i >= 0);
const vegIdxs = features.map((f, i) => (f.properties.cluster === 1 ? i : -1)).filter((i) => i >= 0);
const waterIdx = closestTo(40.0800, -105.2280); // Boulder Reservoir

const urban = meanVec(urbanIdxs);
const veg = meanVec(vegIdxs);
const water = vec(waterIdx);

console.log(`urban anchor: mean of ${urbanIdxs.length} cluster-6 (Built-up) chips`);
console.log(`vegetation anchor: mean of ${vegIdxs.length} cluster-1 (Tree cover) chips`);
console.log(`water anchor: ${features[waterIdx].properties.id} (nearest chip to Boulder Reservoir)`);

const result = new Float64Array(DIM);
for (let d = 0; d < DIM; d++) result[d] = urban[d] - veg[d] + water[d];

const excludeSet = new Set([...urbanIdxs, ...vegIdxs, waterIdx]);
const sims = features.map((f, i) => (excludeSet.has(i) ? -Infinity : cosine(result, vec(i))));
const ranked = sims.map((s, i) => [i, s]).sort((a, b) => b[1] - a[1]).slice(0, 10);

console.log("\nTop 10 chips nearest to (urban - vegetation + water):");
ranked.forEach(([i, s]) => {
  const f = features[i];
  const [lat, lon] = centroid(f);
  console.log(`  ${f.properties.id}  sim=${s.toFixed(3)}  cluster=${f.properties.cluster}  ndvi=${f.properties.ndvi.toFixed(3)}  (${lat.toFixed(4)}, ${lon.toFixed(4)})`);
});

// Baseline: mean similarity of the result vector to every other chip. If the
// top hits aren't meaningfully above this, the arithmetic isn't doing
// anything interpretable -- it'd just be close to everything.
let randSum = 0, randN = 0;
for (let i = 0; i < features.length; i++) {
  if (excludeSet.has(i)) continue;
  randSum += cosine(result, vec(i));
  randN++;
}
console.log(`\nMean similarity to result across all other chips: ${(randSum / randN).toFixed(3)}`);
console.log(`Top-1 similarity: ${ranked[0][1].toFixed(3)}`);
