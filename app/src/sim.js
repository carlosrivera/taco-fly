// Simulation core: scent field, sensory encoder, Controller A (Dumb Fly).
// Pure state + step function — no rendering, no map, no LLM (PRD §11).

export const CENTER = { lat: 19.412, lng: -99.162 };
const M_PER_DEG_LAT = 111320;
const M_PER_DEG_LNG = 111320 * Math.cos((CENTER.lat * Math.PI) / 180);

export const BOUNDS = {
  // west, south, east, north — Centro/Condesa/Roma/Narvarte/Del Valle/Escandón/Juárez
  w: -99.205, s: 19.355, e: -99.105, n: 19.455,
};

// deterministic PRNG (PRD §12: reproducible runs)
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Rough neighborhood centroids — enough to label telemetry honestly.
const HOODS = [
  { name: "Centro Histórico", lat: 19.4335, lng: -99.1390, r: 1600 },
  { name: "Juárez", lat: 19.4230, lng: -99.1600, r: 1100 },
  { name: "Roma Norte", lat: 19.4170, lng: -99.1630, r: 900 },
  { name: "Roma Sur", lat: 19.4070, lng: -99.1610, r: 900 },
  { name: "Condesa", lat: 19.4110, lng: -99.1760, r: 950 },
  { name: "Escandón", lat: 19.4010, lng: -99.1810, r: 1000 },
  { name: "Doctores", lat: 19.4250, lng: -99.1480, r: 900 },
  { name: "Narvarte", lat: 19.3950, lng: -99.1510, r: 1100 },
  { name: "Del Valle", lat: 19.3870, lng: -99.1630, r: 1200 },
  { name: "Tabacalera", lat: 19.4350, lng: -99.1560, r: 800 },
  { name: "Santa María la Ribera", lat: 19.4420, lng: -99.1700, r: 800 },
];

export function neighborhoodAt(lat, lng) {
  const dLat = (lat - CENTER.lat) * M_PER_DEG_LAT;
  for (const h of HOODS) {
    const dLng = (lng - h.lng) * M_PER_DEG_LNG;
    const dLatH = (lat - h.lat) * M_PER_DEG_LAT;
    if (Math.hypot(dLng, dLatH) < h.r) return h.name;
  }
  return "somewhere in CDMX";
}

// ---------- scent ----------

// S(d) = I · exp(−k·d), anisotropic under wind:
// effective distance shrinks downwind of the source, grows upwind.
export function makeScent(taquerias, wind) {
  const k = 1 / 220; // decay coefficient: perceptible within a few hundred meters
  const cosW = Math.cos(wind.dir), sinW = Math.sin(wind.dir);

  return function smell(lat, lng) {
    const x = (lng - CENTER.lng) * M_PER_DEG_LNG;
    const y = (lat - CENTER.lat) * M_PER_DEG_LAT;
    let total = 0;
    for (let i = 0; i < taquerias.length; i++) {
      const t = taquerias[i];
      const tx = (t.lng - CENTER.lng) * M_PER_DEG_LNG;
      const ty = (t.lat - CENTER.lat) * M_PER_DEG_LAT;
      let dx = x - tx, dy = y - ty;
      const d = Math.hypot(dx, dy);
      if (d > 900) continue; // cull: exp(-900/220) ≈ 0.017, invisible anyway
      if (wind.speed > 0 && d > 1) {
        // downwind projection: positive when the point lies downwind of source
        const proj = (dx * cosW + dy * sinW) / d;
        const stretch = 1 + (wind.speed / 6) * proj * 0.75;
        total += t.intensity * Math.exp(-k * d * Math.max(0.3, stretch));
      } else {
        total += t.intensity * Math.exp(-k * d);
      }
    }
    // receptor transduction: monotonic, never fully saturates, so antenna
    // differentials stay informative even in taco-dense neighborhoods
    return 1 - Math.exp(-total * 2);
  };
}

// ---------- sensory encoder ----------
// The fly receives ONLY these local samples (PRD §7). No coordinates.

export function sense(smell, lat, lng, headingRad) {
  const ant = 30; // antenna sampling distance, meters
  const dLat = (ant / M_PER_DEG_LAT);
  const dLng = (ant / M_PER_DEG_LNG);
  const sample = (ang) => smell(
    lat + Math.sin(ang) * dLat,
    lng + Math.cos(ang) * dLng,
  );
  const left = sample(headingRad - Math.PI / 5);
  const forward = sample(headingRad);
  const right = sample(headingRad + Math.PI / 5);
  return {
    smell_left: left,
    smell_forward: forward,
    smell_right: right,
    total_smell: smell(lat, lng),
    smell_gradient: forward - (left + right) / 2,
  };
}

// ---------- Controller A: Dumb Fly ----------
// Probabilistic chemotaxis: biased random walk with run/tumble dynamics.

export function createFly(seed, taquerias, drop = null) {
  const rand = mulberry32(seed ^ 0x9e3779b9);
  const f = (lo, hi) => lo + rand() * (hi - lo);
  // default: a random edge of the map — it has to find tacos the hard way
  const edge = Math.floor(rand() * 4);
  const lat = drop?.lat ?? (edge === 0 ? BOUNDS.s + 0.002 : edge === 1 ? BOUNDS.n - 0.002 : f(BOUNDS.s, BOUNDS.n));
  const lng = drop?.lng ?? (edge === 2 ? BOUNDS.w + 0.003 : edge === 3 ? BOUNDS.e - 0.003 : f(BOUNDS.w, BOUNDS.e));
  return {
    lat, lng,
    heading: rand() * Math.PI * 2,
    velocity: 9,
    energy: 1,
    age: 0,
    alive: true,
    rand,
    turnMomentum: 0,
    runTimer: 0,
    visited: [],
    visitedSet: new Set(),
    distance: 0,
    refactoryUntil: 0,
    lastEvent: null,
    taquerias,
  };
}

// ---------- Controller B: Neural Fly ----------
// A recurrent neural dynamical system per PRD §10:
//   aᵢ(t+1) = f(Σ wⱼᵢ aⱼ(t) + Iᵢ(t) − λaᵢ(t))
// Wiring is random with a fixed seed and DYNAMICS ONLY — this is NOT the
// MaleCNS connectome. It exists so the two controllers can be compared.

export const NEURAL_SPEC = { inputs: 5, hidden: 48, lambda: 0.25 };

export function createNetwork(seed) {
  const rand = mulberry32(seed ^ 0x1234abcd);
  const gauss = () => gaussian(rand);
  const { inputs, hidden } = NEURAL_SPEC;
  const n = inputs + hidden;
  const density = 0.2;

  const W = [];
  for (let i = 0; i < n; i++) {
    const row = new Float32Array(n);
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      // sensory → all; hidden → hidden sparse recurrent
      const fromSensory = j < inputs;
      if (fromSensory || rand() < density) {
        // balanced signs, scale by sqrt(fan-in proxy) to keep activity bounded
        row[j] = gauss() * (fromSensory ? 0.9 : 0.7) / Math.sqrt(n * density);
      }
    }
    W.push(row);
  }
  // motor readout: [turn_left, turn_right, thrust] over ALL populations.
  // The sensory part carries a weak chemotaxis PRIOR (jittered gains: turn
  // toward the stronger antenna, thrust when smell rises ahead) so the fly is
  // viable; the hidden part is pure random wiring and adds state-dependent
  // "personality". Untrained — no learning, no connectome.
  const prior = 2.5 + Math.abs(gauss()) * 1.5;
  const jit = () => gauss() * 0.05;
  const motor = [
    Float64Array.from({ length: n }, (_, j) => (j === 0 ? prior : j === 2 ? -prior * 0.9 : j < inputs ? 0 : gauss() * 0.2) + jit()),
    Float64Array.from({ length: n }, (_, j) => (j === 0 ? -prior * 0.9 : j === 2 ? prior : j < inputs ? 0 : gauss() * 0.2) + jit()),
    Float64Array.from({ length: n }, (_, j) => (j === 1 ? 0.6 : j === 3 ? 0.25 : j === 4 ? 0.6 : j < inputs ? 0 : Math.abs(gauss()) * 0.4 + 0.1)),
  ];
  return { n, W, motor, a: new Float32Array(n) };
}

// per sensory channel: [left-pool, right-pool, thrust] sign conventions
const SENSORY_MOTOR = [
  [1, 0, 0],    // smell_left  → turns left (when stronger on the left)
  [0, 0, 1],    // smell_forward → thrust
  [0, 1, 0],    // smell_right → turns right
  [0, 0, 0.5],  // total smell → mild thrust
  [0, 0, 1],    // rising gradient → thrust
];

export const neuralNeuronCount = NEURAL_SPEC.inputs + NEURAL_SPEC.hidden;

// one dynamics tick: inputs stimulate the sensory population, activity
// propagates, motor populations are read off the hidden layer
function neuralMotor(net, s, dt) {
  const { inputs, hidden, lambda } = NEURAL_SPEC;
  const { n, W, motor, a } = net;

  // sensory drive
  const drive = [
    (s.smell_left - s.smell_right) * 2,
    s.smell_forward * 4,
    (s.smell_right - s.smell_left) * 2,
    s.total_smell * 3 - 1,
    s.smell_gradient * 20,
  ];
  for (let i = 0; i < inputs; i++) a[i] += (drive[i] - a[i]) * 0.5;

  // recurrent update for the hidden population
  for (let i = inputs; i < n; i++) {
    const row = W[i];
    let sum = 0;
    for (let j = 0; j < n; j++) sum += row[j] * a[j];
    // f = tanh; leak λ
    a[i] += (Math.tanh(sum) - a[i]) * (1 - Math.exp(-lambda * dt * 10));
  }

  // motor pools: read out over sensory + hidden populations
  let left = 0, right = 0, thrust = 0;
  for (let j = 0; j < n; j++) {
    const act = a[j];
    left += motor[0][j] * act;
    right += motor[1][j] * act;
    thrust += motor[2][j] * act;
  }
  return {
    angular: (right - left) * 0.9,
    speed: 11 + Math.tanh(thrust) * 6,
  };
}

// ---------- shared tick ----------

export function step(fly, smell, dt, wind, controller, net) {
  const s = sense(smell, fly.lat, fly.lng, fly.heading);
  fly.age += dt;

  if (!fly.alive) {
    fly.velocity = 0;
    fly.lastEvent = null;
    return s; // a dead fly senses nothing, does nothing
  }

  let turn, targetV;
  if (controller === "malecns") {
    // dopaminergic teaching signal: pulse set when the previous tick found food
    const reward = fly.rewardPulse || 0;
    fly.rewardPulse = 0;
    const m = net.step(dt, s, reward, fly.age);
    turn = m.angular + gaussian(fly.rand) * 0.2;
    targetV = m.thrust;
  } else if (controller === "neural") {
    const m = neuralMotor(net, s, dt);
    turn = m.angular + gaussian(fly.rand) * 0.3;
    targetV = m.speed;
  } else {
    ({ turn, targetV } = proceduralMotor(fly, s, dt));
  }

  // starving flies fly slower
  if (fly.energy < 0.25) targetV *= 0.6;

  // momentum smoothing keeps the trace organic rather than jittery
  fly.turnMomentum = fly.turnMomentum * 0.82 + turn * 0.18;
  fly.heading += fly.turnMomentum * dt * 2.4;
  fly.velocity += (targetV - fly.velocity) * Math.min(1, dt * 2);

  // soft containment inside the study area: steer back toward center
  const cx = (CENTER.lng - fly.lng) * M_PER_DEG_LNG;
  const cy = (CENTER.lat - fly.lat) * M_PER_DEG_LAT;
  const cdist = Math.hypot(cx, cy);
  if (cdist > 4400) {
    const desired = Math.atan2(cy, cx);
    let diff = desired - fly.heading;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    // strength ramps up the deeper it strays
    const urgency = Math.min(3, (cdist - 4400) / 600);
    fly.heading += Math.max(-3 * dt, Math.min(3 * dt, diff)) * Math.max(0.5, urgency);
  }

  // integrate motion
  const stepM = fly.velocity * dt;
  fly.lat += Math.sin(fly.heading) * (stepM / M_PER_DEG_LAT);
  fly.lng += Math.cos(fly.heading) * (stepM / M_PER_DEG_LNG);
  fly.distance += stepM;

  // metabolism: flying burns energy; a meal restores it
  fly.energy -= dt * 0.0012;
  if (fly.energy <= 0) {
    fly.energy = 0;
    fly.alive = false;
    fly.velocity = 0;
    fly.diedAt = fly.age;
    return s;
  }

  // discovery check — every 5th tick is plenty (24 m radius, ~15 m/tick)
  fly.lastEvent = null;
  fly.tickCount = (fly.tickCount || 0) + 1;
  if (fly.age >= fly.refactoryUntil && fly.tickCount % 5 === 0) {
    for (const t of fly.taquerias) {
      if (fly.visitedSet.has(t.id)) continue;
      const dx = (fly.lng - t.lng) * M_PER_DEG_LNG;
      const dy = (fly.lat - t.lat) * M_PER_DEG_LAT;
      if (dx * dx + dy * dy < 24 * 24) {
        fly.visited.push(t.id);
        fly.visitedSet.add(t.id);
        fly.refactoryUntil = fly.age + 6; // land, eat, ignore nearby clones
        fly.energy = Math.min(1, fly.energy + 0.4); // a taco is a meal
        fly.rewardPulse = 1; // dopamine: this smell pattern led to food
        fly.lastEvent = { taqueria: t, at: fly.age };
        break;
      }
    }
  }
  return s;
}

function proceduralMotor(fly, s, dt) {
  const rand = fly.rand;
  // turn noise — bigger when smell is weak (casting), smaller on-scent
  const onScent = s.total_smell > 0.02;
  const saturated = s.total_smell > 0.5; // standing on the source: nothing to gain by circling
  const baseNoise = saturated ? 0.25 : onScent ? 0.9 : 1.8;
  let turn = gaussian(rand) * baseNoise;

  // weathervane: bias toward the stronger antenna (damp when saturated)
  const lateral = s.smell_right - s.smell_left;
  turn += lateral * (saturated ? 4 : 14);

  // gradient along track: negative smell → shorten the run (tumble sooner)
  fly.runTimer -= dt;
  if (s.smell_gradient < -0.004 && fly.runTimer <= 0) {
    fly.runTimer = 0.4 + rand() * 0.8; // refractory: avoid oscillation (PRD §10)
    turn += (rand() < 0.5 ? -1 : 1) * (1.5 + rand());
  }

  // speed: hustle in scent-poor air and when saturated; slow to sniff the plume edge
  const targetV = saturated ? 17 : onScent ? 8 + (1 - Math.min(1, s.total_smell * 3)) * 6 : 15;
  return { turn, targetV };
}

export function formatClock(seconds) {
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = Math.floor(seconds % 60);
  return [h, m, s].map((n) => String(n).padStart(2, "0")).join(":");
}
