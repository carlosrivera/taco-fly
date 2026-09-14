// Filter the raw Overpass dump down to taco-relevant POIs and emit
// data/taquerias.json in the shape the simulation expects.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const raw = JSON.parse(readFileSync(join(root, "data/raw-overpass.json"), "utf8"));

const TACO_TOKENS = [
  "taco", "tacos", "taqueria", "taquería", "tacos y", "birria", "barbacoa",
  "pastor", "carnitas", "guisado", "guisados", "canasta", "sudados", "cochinita",
];
const MEXICAN_TOKENS = ["mexican", "mexicana", "regional", "antojitos", "pambazos", "quesadillas", "sopes", "tlacoyos"];

// Type label + intensity boost inferred from name/tags; pastor is king.
function classify(name, cuisine, tags) {
  const hay = `${name} ${cuisine}`.toLowerCase();
  if (/pastor/.test(hay)) return "pastor";
  if (/birria|barbacoa/.test(hay)) return "birria";
  if (/carnitas/.test(hay)) return "carnitas";
  if (/canasta|sudado/.test(hay)) return "canasta";
  if (/guisado/.test(hay)) return "guisados";
  if (/pescado|camarón|camaron|mariscos/.test(hay)) return "pescado";
  if (/suadero/.test(hay)) return "suadero";
  if (tags["amenity"] === "fast_food" || /fast_food/.test(cuisine)) return "fast_food";
  return "mexican";
}

function hash01(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 10000) / 10000;
}

const out = [];
const seen = new Set();

for (const el of raw.elements) {
  const tags = el.tags ?? {};
  const name = tags.name;
  if (!name) continue;

  const lat = el.lat ?? el.center?.lat;
  const lng = el.lon ?? el.center?.lon;
  if (lat == null || lng == null) continue;

  const cuisine = (tags.cuisine ?? "").toLowerCase().replace(/;/g, ",");
  const hay = `${name.toLowerCase()} ${cuisine} ${tags["cuisine:en"] ?? ""} ${tags.description ?? ""} ${tags.comment ?? ""}`;

  const hasTacoToken = TACO_TOKENS.some((t) => hay.includes(t));
  const hasMexicanCuisine = cuisine.split(",").some((c) => MEXICAN_TOKENS.includes(c.trim()));
  // mexican_cuisine_repo etc. also count as generic mexican
  const hasMexicanTag = /mexican/.test(cuisine);
  if (!hasTacoToken && !hasMexicanCuisine && !hasMexicanTag) continue;

  const key = `${name}|${lat.toFixed(4)}|${lng.toFixed(4)}`;
  if (seen.has(key)) continue;
  seen.add(key);

  const type = classify(name, cuisine, tags);
  // Deterministic intensity in [0.55, 1.0]; canonical taco types skew stronger.
  const typeBoost = { pastor: 0.18, birria: 0.15, carnitas: 0.12, canasta: 0.1, guisados: 0.08, suadero: 0.1 }[type] ?? 0.04;
  const intensity = Math.min(1, 0.55 + typeBoost + hash01(key) * 0.2);

  out.push({
    id: `taco_${String(out.length + 1).padStart(3, "0")}`,
    name,
    lat: +lat.toFixed(6),
    lng: +lng.toFixed(6),
    type,
    intensity: +intensity.toFixed(2),
    openingHours: tags["opening_hours"] ?? null,
    osm: el.type[0] + el.id,
  });
}

out.sort((a, b) => a.id.localeCompare(b.id));
mkdirSync(join(root, "data"), { recursive: true });
writeFileSync(join(root, "data/taquerias.json"), JSON.stringify(out, null, 2));
console.log(`taquerías: ${out.length}`);
const byType = {};
for (const t of out) byType[t.type] = (byType[t.type] ?? 0) + 1;
console.log(byType);
