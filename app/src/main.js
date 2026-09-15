import { Map as MaplibreMap, setWorkerUrl } from "maplibre-gl";
// serve the worker + its shared chunk unprocessed from /public (bundling it
// dangles its internal chunk import on static hosts); relative to the page URL
setWorkerUrl(new URL("maplibre-gl-worker.mjs", document.baseURI).href);
import taquerias from "../data/taquerias.json" with { type: "json" };
import {
  makeScent, createFly, step, formatClock, neighborhoodAt, CENTER, BOUNDS,
  createNetwork, neuralNeuronCount,
} from "./sim.js";
import { createMaleCNS, MALECNS_TOTAL, POP as POPN } from "./malecns.js";

// ---------- deterministic environment ----------
// V0: constant simulated wind (PRD §8). Seed pins taquería ordering & fly spawn.
const SEED = 87231;
const WIND = { dir: (SEED % 360) * (Math.PI / 180), speed: 2.4 }; // radians FROM which it blows→ direction it drifts toward
const smell = makeScent(taquerias, WIND);

// ---------- map ----------
const map = new MaplibreMap({
  container: "map",
  center: [CENTER.lng, CENTER.lat],
  zoom: 13.2,
  pitch: 0,
  attributionControl: true,
  preserveDrawingBuffer: true, // needed for PNG export
  style: {
    version: 8,
    sources: {
      carto: {
        type: "raster",
        tiles: [
          "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors · Basemap © Esri",
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#06090f" } },
      { id: "carto", type: "raster", source: "carto", paint: { "raster-opacity": 0.55, "raster-saturation": -0.4 } },
    ],
  },
});

// study-area outline — scientific instrumentation look
map.on("load", () => {
  map.addSource("bounds", {
    type: "geojson",
    data: {
      type: "Feature",
      properties: {},
      geometry: {
        type: "Polygon",
        coordinates: [[
          [BOUNDS.w, BOUNDS.s], [BOUNDS.e, BOUNDS.s], [BOUNDS.e, BOUNDS.n], [BOUNDS.w, BOUNDS.n], [BOUNDS.w, BOUNDS.s],
        ]],
      },
    },
  });
  map.addLayer({
    id: "bounds-line", type: "line", source: "bounds",
    paint: { "line-color": "rgba(125,255,90,0.35)", "line-width": 1, "line-dasharray": [3, 3] },
  });

  map.addSource("tacos", {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: taquerias.map((t) => ({
        type: "Feature",
        properties: { name: t.name, type: t.type, intensity: t.intensity },
        geometry: { type: "Point", coordinates: [t.lng, t.lat] },
      })),
    },
  });
  // glow halo = the olfactory field at a glance
  map.addLayer({
    id: "taco-glow", type: "circle", source: "tacos",
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, ["*", ["get", "intensity"], 10], 15, ["*", ["get", "intensity"], 26]],
      "circle-color": "#ff7a1a",
      "circle-blur": 1.1,
      "circle-opacity": 0.4,
    },
  });
  map.addLayer({
    id: "taco-dot", type: "circle", source: "tacos",
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 2.2, 15, 4.5],
      "circle-color": "#ffb35a",
      "circle-stroke-width": 1,
      "circle-stroke-color": "rgba(255,122,26,0.9)",
    },
  });

  map.addSource("trail", { type: "geojson", data: EMPTY_TRAIL });
  map.addLayer({
    id: "trail-line", type: "line", source: "trail",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#7dff5a", "line-width": 2.2, "line-opacity": 0.9 },
  });

  map.addSource("visited", { type: "geojson", data: EMPTY_COLL });
  map.addLayer({
    id: "visited-ring", type: "circle", source: "visited",
    paint: {
      "circle-radius": 9,
      "circle-color": "rgba(255,210,63,0)",
      "circle-stroke-width": 2.5,
      "circle-stroke-color": "#ffd23f",
    },
  });

  map.addSource("fly", { type: "geojson", data: EMPTY_COLL });
  map.addLayer({
    id: "fly-dot", type: "circle", source: "fly",
    paint: {
      "circle-radius": 7,
      "circle-color": ["match", ["get", "dead"], "yes", "#ff4d4d", "#7dff5a"],
      "circle-stroke-width": 2,
      "circle-stroke-color": "#0a1206",
    },
  });

  map.addSource("drop", { type: "geojson", data: EMPTY_COLL });
  map.addLayer({
    id: "drop-marker", type: "circle", source: "drop",
    paint: {
      "circle-radius": 8,
      "circle-color": "rgba(125,255,90,0.15)",
      "circle-stroke-width": 2,
      "circle-stroke-color": "#7dff5a",
    },
  });
});

const EMPTY_TRAIL = { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [] } };
const EMPTY_COLL = { type: "FeatureCollection", features: [] };

// ---------- simulation state ----------
let fly = null;
let running = false;
let speed = 1;
let trailCoords = [];
let visitedFeatures = [];
let cameraFollow = false;
let controller = "procedural";
let net = null; // neural network (created per release)
let dropPoint = null; // {lat,lng} | null → random edge spawn
let picking = false;

const $ = (id) => document.getElementById(id);
const releaseBtn = $("release");
const resetBtn = $("reset");
const speedBox = $("speed");

function releaseFly() {
  fly = createFly(SEED, taquerias, dropPoint);
  net = controller === "neural" ? createNetwork(SEED)
      : controller === "malecns" ? createMaleCNS(SEED)
      : null;
  if (controller === "malecns") {
    // lineage: each new fly inherits the ancestor's learned KC→MBON weights
    const lineage = loadLineage();
    if (lineage.weights) net.setLearned(lineage.weights);
    updateLineageBadge();
  }
  $("brain").classList.toggle("hidden", controller !== "malecns");
  trailCoords = [[fly.lng, fly.lat]];
  visitedFeatures = [];
  deathAnnounced = false;
  running = true;
  cameraFollow = true;
  setPicking(false);
  releaseBtn.classList.add("hidden");
  $("pick-drop").classList.add("hidden");
  $("controller").classList.add("hidden");
  speedBox.classList.remove("hidden");
  resetBtn.classList.remove("hidden");
  $("feed").classList.remove("hidden");
  $("feed").innerHTML = "";
  map.flyTo({ center: [fly.lng, fly.lat], zoom: 14.6, duration: 1600 });
}

function resetSim() {
  running = false;
  fly = null;
  cameraFollow = false;
  trailCoords = [];
  visitedFeatures = [];
  releaseBtn.classList.remove("hidden");
  $("pick-drop").classList.remove("hidden");
  $("controller").classList.remove("hidden");
  speedBox.classList.add("hidden");
  resetBtn.classList.add("hidden");
  $("feed").classList.add("hidden");
  $("brain").classList.add("hidden");
  $("t-status").textContent = "STANDBY";
  $("t-energy").textContent = "—";
  if (map.getSource("trail")) map.getSource("trail").setData(EMPTY_TRAIL);
  if (map.getSource("fly")) map.getSource("fly").setData(EMPTY_COLL);
  if (map.getSource("visited")) map.getSource("visited").setData(EMPTY_COLL);
  map.flyTo({ center: [CENTER.lng, CENTER.lat], zoom: 13.2, duration: 1200 });
}

releaseBtn.addEventListener("click", releaseFly);
resetBtn.addEventListener("click", resetSim);

$("controller").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-ctl]");
  if (!btn) return;
  controller = btn.dataset.ctl;
  $("controller").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
  const label = $("t-neurons");
  if (controller === "malecns") {
    label.innerHTML = `${MALECNS_TOTAL.toLocaleString()} <em>(MaleCNS v0.9)</em>`;
    $("brain-count").textContent = MALECNS_TOTAL.toLocaleString();
  } else if (controller === "neural") {
    label.innerHTML = `${neuralNeuronCount} <em>(random wiring)</em>`;
  } else {
    label.innerHTML = `1 <em>(procedural)</em>`;
  }
});

function setPicking(on) {
  picking = on;
  document.getElementById("pick-drop").classList.toggle("picking", on);
  document.getElementById("pick-drop").textContent = on ? "CLICK MAP…" : "PICK DROP POINT";
  map.getCanvas().style.cursor = on ? "crosshair" : "";
}
$("pick-drop").addEventListener("click", () => setPicking(!picking));

// drop marker layer (created after load)
function setDropMarker(lngLat) {
  dropPoint = lngLat;
  const src = map.getSource("drop");
  if (src) {
    src.setData({
      type: "FeatureCollection",
      features: [{ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [lngLat.lng, lngLat.lat] } }],
    });
  }
}
speedBox.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-speed]");
  if (!btn) return;
  speed = +btn.dataset.speed;
  speedBox.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
});

$("toggle-map").addEventListener("click", (e) => {
  const on = document.body.classList.toggle("trail-only");
  e.target.textContent = on ? "SHOW MAP" : "HIDE MAP";
  // hide the basemap raster so only the fly's shape remains (PRD §15)
  if (map.getLayer("carto")) map.setLayoutProperty("carto", "visibility", on ? "none" : "visible");
  if (map.getLayer("bounds-line")) map.setLayoutProperty("bounds-line", "visibility", on ? "none" : "visible");
  if (map.getLayer("taco-glow")) map.setLayoutProperty("taco-glow", "visibility", on ? "none" : "visible");
  if (map.getLayer("taco-dot")) map.setLayoutProperty("taco-dot", "visibility", on ? "none" : "visible");
});

$("export").addEventListener("click", exportPng);

function exportPng() {
  const src = map.getCanvas();
  const out = document.createElement("canvas");
  out.width = src.width;
  out.height = src.height;
  const ctx = out.getContext("2d");
  ctx.drawImage(src, 0, 0);

  // caption block
  const pad = Math.round(out.height * 0.035);
  const fs = Math.max(14, Math.round(out.height * 0.022));
  ctx.font = `700 ${fs}px ui-monospace, Menlo, monospace`;
  ctx.fillStyle = "#ffd23f";
  const disc = fly ? fly.visited.length : 0;
  const km = fly ? (fly.distance / 1000).toFixed(1) : "0.0";
  const lines = [
    `TACO FLY · CDMX`,
    `${disc} taquerías · ${km} km · seed ${SEED}`,
  ];
  lines.forEach((line, i) => {
    const y = out.height - pad - (lines.length - 1 - i) * fs * 1.5;
    ctx.shadowColor = "rgba(0,0,0,0.9)";
    ctx.shadowBlur = 8;
    ctx.fillText(line, pad, y);
  });

  const a = document.createElement("a");
  a.download = `taco-fly-${SEED}-${disc}tacos.png`;
  a.href = out.toDataURL("image/png");
  a.click();
}

// ---------- comparison mode (PRD §16) ----------
const COMPARE_SEEDS = [87231, 4108];
const COMPARE_MINS = 40;

$("compare").addEventListener("click", openCompare);
$("compare-close").addEventListener("click", () => $("compare-panel").classList.add("hidden"));

function openCompare() {
  $("compare-panel").classList.remove("hidden");
  runComparison();
}

async function runComparison() {
  const body = $("compare-body");
  const status = $("compare-status");
  body.innerHTML = "";
  const rows = [];
  const spawns = COMPARE_SEEDS.map((s) => dropPoint ?? null);
  const controllers = [
    ["procedural", "DUMB FLY"],
    ["neural", "NEURAL FLY (untrained)"],
    ["malecns", "MALECNS (untrained)"],
  ];
  let netFor = null;
  for (const [ctl, label] of controllers) {
    for (let i = 0; i < COMPARE_SEEDS.length; i++) {
      status.textContent = `running ${label} · seed ${COMPARE_SEEDS[i]}…`;
      await new Promise((r) => setTimeout(r, 30)); // let the UI breathe
      const fly = createFly(COMPARE_SEEDS[i], taquerias, spawns[i]);
      netFor = ctl === "neural" ? createNetwork(COMPARE_SEEDS[i])
             : ctl === "malecns" ? createMaleCNS(COMPARE_SEEDS[i])
             : null;
      const ticks = COMPARE_MINS * 600 / 0.1;
      for (let t = 0; t < ticks && fly.alive; t++) {
        step(fly, smell, 0.1, WIND, ctl, netFor);
      }
      rows.push({
        label: `${label} <em>#${i + 1}</em>`,
        tacos: fly.visited.length,
        km: (fly.distance / 1000).toFixed(1),
        fate: fly.alive ? "survived" : `☠ starved ${Math.round(fly.age / 60)}m`,
        alive: fly.alive,
      });
      body.innerHTML = rows.map((r) => `<tr class="${r.alive ? "" : "dead-row"}"><td>${r.label}</td><td>${r.tacos}</td><td>${r.km}</td><td>${r.fate}</td></tr>`).join("");
    }
  }
  const byCtl = {};
  rows.forEach((r, i) => {
    const k = Math.floor(i / COMPARE_SEEDS.length);
    byCtl[k] = byCtl[k] || { tacos: 0, n: 0 };
    byCtl[k].tacos += r.tacos; byCtl[k].n++;
  });
  status.textContent = controllers.map(([_, label], k) =>
    `${label.split(" ")[0]} avg ${(byCtl[k].tacos / byCtl[k].n).toFixed(1)}`).join(" · ");
}

// ---------- fly lineage: learning persists across runs (localStorage) ----------
const LINEAGE_KEY = "tacofly-lineage-v1";
function loadLineage() {
  try {
    const raw = localStorage.getItem(LINEAGE_KEY);
    if (!raw) return { generations: 0, totalTacos: 0, weights: null };
    const obj = JSON.parse(raw);
    return {
      generations: obj.generations || 0,
      totalTacos: obj.totalTacos || 0,
      weights: obj.weights ? Float64Array.from(obj.weights) : null,
    };
  } catch { return { generations: 0, totalTacos: 0, weights: null }; }
}
function saveLineage(net, tacos) {
  const lin = loadLineage();
  lin.generations += 1;
  lin.totalTacos += tacos;
  lin.weights = Array.from(net.getLearned());
  try { localStorage.setItem(LINEAGE_KEY, JSON.stringify(lin)); } catch {}
  updateLineageBadge();
}
function updateLineageBadge() {
  const lin = loadLineage();
  $("brain-lineage").textContent =
    `LINEAGE gen ${lin.generations} · ${lin.totalTacos} tacos inherited`;
}
$("forget-lineage")?.addEventListener("click", () => {
  localStorage.removeItem(LINEAGE_KEY);
  updateLineageBadge();
});

// ---------- taquería index ----------
const tacoListEl = $("taco-list");
const tacoRowsEl = $("taco-rows");
let tacoListBuilt = false;

$("toggle-tacos").addEventListener("click", () => {
  const show = tacoListEl.classList.toggle("hidden");
  document.body.classList.toggle("taco-list-open", !show);
  if (!show) buildTacoList();
});

$("taco-search").addEventListener("input", (e) => renderTacoRows(e.target.value));

function buildTacoList() {
  if (tacoListBuilt) { renderTacoRows($("taco-search").value); return; }
  tacoListBuilt = true;
  renderTacoRows($("taco-search").value);
}

function renderTacoRows(filter = "") {
  const f = filter.trim().toLowerCase();
  const visitedSet = new Set(fly ? fly.visited : []);
  const rows = taquerias
    .filter((t) => !f || t.name.toLowerCase().includes(f) || t.type.toLowerCase().includes(f))
    .map((t) => {
      const visited = visitedSet.has(t.id);
      return `<div class="taco-row${visited ? " visited" : ""}" data-lat="${t.lat}" data-lng="${t.lng}">
        <span class="taco-row-name">${visited ? "✓ " : ""}${t.name}</span>
        <span class="taco-row-meta">${t.type} · ${t.intensity.toFixed(2)}</span>
      </div>`;
    }).join("");
  tacoRowsEl.innerHTML = rows || `<div class="taco-row-empty">no matches</div>`;
  $("taco-count").textContent = `${taquerias.length} total · ${visitedSet.size} visited`;
}

tacoRowsEl.addEventListener("click", (e) => {
  const row = e.target.closest(".taco-row");
  if (!row) return;
  cameraFollow = false; // don't yank the view back
  map.flyTo({ center: [+row.dataset.lng, +row.dataset.lat], zoom: Math.max(map.getZoom(), 15), duration: 1200 });
});

// ---------- brain activity visualization ----------
const brainCanvas = $("brain-canvas");
const brainCtx = brainCanvas.getContext("2d");
const REGIONS = [
  { key: "orn", label: "ORN", color: "#7dff5a", n: POPN.olfactory_receptor },
  { key: "al", label: "AL", color: "#57d7ff", n: POPN.antennal_lobe_local + POPN.antennal_lobe_projection },
  { key: "kc", label: "MB·KC", color: "#ffd23f", n: POPN.kenyon_cell },
  { key: "dan", label: "DAN", color: "#ff7ad9", n: POPN.dopaminergic },
  { key: "mbon", label: "MBON", color: "#ff7a1a", n: POPN.mbon },
  { key: "dn", label: "DN", color: "#b28dff", n: POPN.descending },
  { key: "vnc", label: "VNC", color: "#5affd5", n: POPN.vnc_intrinsic },
  { key: "mn", label: "MOTOR", color: "#ff4d4d", n: POPN.motor },
  { key: "rest", label: "REST", color: "#3d5568", n: POPN.optic + POPN.central_brain_rest + POPN.sensory_other + POPN.other },
];

function drawBrain() {
  if (!net || !net.activity) return;
  const w = brainCanvas.width, h = brainCanvas.height;
  brainCtx.clearRect(0, 0, w, h);
  const act = net.activity;
  const barH = h / REGIONS.length;
  REGIONS.forEach((r, i) => {
    const v = Math.min(1, act[r.key] ?? 0);
    const y = i * barH;
    brainCtx.fillStyle = "rgba(215,227,238,0.55)";
    brainCtx.font = "8px ui-monospace, monospace";
    brainCtx.fillText(r.label, 4, y + barH * 0.7);
    brainCtx.fillStyle = "rgba(61,85,104,0.35)";
    brainCtx.fillRect(40, y + 3, w - 60, barH - 7);
    brainCtx.fillStyle = r.color;
    brainCtx.fillRect(40, y + 3, (w - 60) * v, barH - 7);
  });
}
let lastT = performance.now();
let simAccum = 0;
let deathAnnounced = false;
let lastUiUpdate = 0;
let lastBrainUpdate = 0;

function frame(now) {
  const realDt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;

  if (running && fly) {
    const dt = 0.08; // fixed sim timestep, seconds
    simAccum += realDt * speed;
    const simSteps = Math.min(200, Math.floor(simAccum / dt));
    simAccum -= simSteps * dt;
    for (let i = 0; i < simSteps; i++) {
      const s = step(fly, smell, dt, WIND, controller, net);
      if (fly.lastEvent) onDiscovery(fly.lastEvent);
      if (i === simSteps - 1) fly.lastSense = s;
      if (!fly.alive) break;
    }
    if (!fly.alive && !deathAnnounced) {
      deathAnnounced = true;
      cameraFollow = false;
      if (controller === "malecns" && net) saveLineage(net, fly.visited.length);
      addFeedEvent(
        "☠ THE FLY STARVED",
        `after ${fly.visited.length} taco${fly.visited.length === 1 ? "" : "s"}, it ran out of energy`,
        `${formatClock(fly.age)} in · ${(fly.distance / 1000).toFixed(1)} km flown`,
        "death",
      );
    }

    // trail: append when moved far enough from last point
    const last = trailCoords[trailCoords.length - 1];
    const moved = Math.hypot(
      (fly.lng - last[0]) * Math.cos(fly.lat * Math.PI / 180) * 111320,
      (fly.lat - last[1]) * 111320,
    );
    if (moved > 12) {
      trailCoords.push([fly.lng, fly.lat]);
      if (trailCoords.length > 20000) trailCoords.shift();
    }

    updateMapSources();
    if (now - lastUiUpdate > 250) updateTelemetry(), (lastUiUpdate = now);
    if (now - lastBrainUpdate > 90 && controller === "malecns") drawBrain(), (lastBrainUpdate = now); // ~11 fps: alive, not jittery
    if (cameraFollow && simSteps > 0) smoothFollow(realDt);
  }
  requestAnimationFrame(frame);
}

// eased dead-zone follow: the fly roams freely inside the middle of the
// viewport; only when it nears an edge does the camera drift after it,
// exponentially — no lockstep, no shake (PRD §4 "loosely follows")
function smoothFollow(dt) {
  const c = map.getCenter();
  const p = map.project([fly.lng, fly.lat]);
  const canvas = map.getCanvas();
  const cx = canvas.clientWidth / 2, cy = canvas.clientHeight / 2;
  const limit = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.2;
  const ox = p.x - cx, oy = p.y - cy;
  const d = Math.hypot(ox, oy);
  if (d <= limit) return; // fly is comfortably inside — camera holds still
  const excess = d - limit;
  const k = 1 - Math.exp(-dt * 3); // exponential ease toward the boundary
  const np = map.unproject([
    cx + ox - (ox / d) * excess * k,
    cy + oy - (oy / d) * excess * k,
  ]);
  map.jumpTo({ center: np });
}

function updateMapSources() {
  map.getSource("trail")?.setData({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: trailCoords } });
  map.getSource("fly")?.setData({
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: { dead: fly.alive ? "no" : "yes" }, geometry: { type: "Point", coordinates: [fly.lng, fly.lat] } }],
  });
}

function onDiscovery(ev) {
  const t = ev.taqueria;
  if (tacoListBuilt && !tacoListEl.classList.contains("hidden")) renderTacoRows($("taco-search").value);
  visitedFeatures.push({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [t.lng, t.lat] } });
  map.getSource("visited")?.setData({ type: "FeatureCollection", features: visitedFeatures });

  addFeedEvent(
    "🌮 TACO FOUND",
    t.name,
    `${t.type} · after ${formatClock(ev.at)} of flying`,
  );
}

let eventSeq = 0;
function addFeedEvent(head, name, meta, kind = "obj") {
  const el = document.createElement("div");
  el.className = "event" + (kind === "death" ? " death" : "");
  const caseId = `EVT-${String(++eventSeq).padStart(4, "0")}`;
  el.innerHTML = `<div class="head"><span>${head}</span><span class="evt-id">${caseId}</span></div>
    <div class="name">${name}</div>
    <div class="meta">${meta}</div>`;
  const feed = $("feed");
  feed.prepend(el);
  while (feed.children.length > 8) feed.lastChild.remove();
}

let smellEma = 0;
function updateTelemetry() {
  window.__fly = fly; // debug handle
  const s = fly.lastSense ?? { total_smell: 0 };
  smellEma += (s.total_smell - smellEma) * 0.25;
  $("t-status").textContent = fly.alive ? "FLYING" : "STARVED";
  $("t-status").classList.toggle("dead", !fly.alive);
  $("t-energy").textContent = `${Math.round(fly.energy * 100)}%`;
  $("t-time").textContent = formatClock(fly.age);
  $("t-dist").textContent = `${(fly.distance / 1000).toFixed(2)} km`;
  $("t-disc").textContent = String(fly.visited.length);
  $("t-smell").textContent = smellEma.toFixed(3);
  $("t-hood").textContent = neighborhoodAt(fly.lat, fly.lng);

  // strongest nearby odor: nearest taquerías by perceived smell, no targeting
  let best = null, bestS = 0;
  for (const t of taquerias) {
    if (fly.visited.includes(t.id)) continue;
    const dx = (fly.lng - t.lng) * Math.cos(fly.lat * Math.PI / 180) * 111320;
    const dy = (fly.lat - t.lat) * 111320;
    const sv = t.intensity * Math.exp(-Math.hypot(dx, dy) / 220);
    if (sv > bestS) { bestS = sv; best = t; }
  }
  $("t-odor").textContent = best && bestS > 0.01 ? `${best.type}, ${bestS.toFixed(2)}` : "faint / none";
}

// user pans → stop chasing the fly (PRD §4: loosely follow, user can pan)
map.on("dragstart", () => { cameraFollow = false; });
map.on("click", (e) => {
  if (!picking || running) return;
  setDropMarker(e.lngLat);
  setPicking(false);
});

// live coordinate readout under cursor
map.on("mousemove", (e) => {
  const { lat, lng } = e.lngLat;
  $("coord-readout").textContent =
    `${lat.toFixed(4)}° ${lat >= 0 ? "N" : "S"} · ${Math.abs(lng).toFixed(4)}° ${lng >= 0 ? "E" : "W"} · SEED ${SEED}`;
});

window.__map = map; // debug handle
window.__smell = smell;

requestAnimationFrame(frame);
