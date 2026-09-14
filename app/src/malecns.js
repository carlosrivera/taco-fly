// Controller C: "MaleCNS 001"
// A 165,114-neuron model sized from REAL MaleCNS v0.9 annotated counts
// (Lee lab male brain+VNC connectome, CC-BY — see data/malecns.json).
//
// What is real (documented per PRD §10C):
//   - population sizes (ORN 2635, AL LNs 420, PNs 697, KCs 4064, DANs 340,
//     MBONs 97, DNs 1321, ascending 2240, motor 879, VNC 13137, optic 99138…)
//   - circuit motifs: ORN→glomerulus→PN, LN gain control, sparse KC coding,
//     DAN-gated plasticity at KC→MBON synapses (appetitive conditioning),
//     descending steering pools, wing-steering MNs modulating turn gain
// What is abstracted:
//   - synapse-level connectivity is scaffolded with seeded random wiring
//     (the 3.2 GB connectome edgelist is not embedded — see README)
//   - ORN tuning vectors per glomerulus are synthetic (no odorant-response data)
//   - the optic lobe / non-olfactory CNS is a slow random recurrent pool that
//     contributes only diffuse tone

import malecnsData from "../data/malecns.json" with { type: "json" };
import connectome from "../data/malecns_connectome.json" with { type: "json" };
import { mulberry32, gaussian } from "./sim.js";

const HAS_REAL_WIRING = !!(connectome && connectome.kc_pn && connectome.kc_pn.length > 0);

export const MALECNS_TOTAL = malecnsData.total_neurons;

const P = malecnsData.populations;
export const POP = P;
const GLOM = malecnsData.glomeruli; // 57

// population offsets
const ORN = 0;
const AL_LN = ORN + P.olfactory_receptor;
const PN = AL_LN + P.antennal_lobe_local + P.antennal_lobe_other;
const KC = PN + P.antennal_lobe_projection;
const DAN = KC + P.kenyon_cell;
const MBON = DAN + P.dopaminergic;
const DN = MBON + P.mbon;
const ASC = DN + P.descending;
const MN = ASC + P.ascending;
const VNC = MN + P.motor;
const REST = VNC + P.vnc_intrinsic;
const N = REST + P.optic + P.central_brain_rest + P.sensory_other + P.other;

const N_PATHWAY = VNC; // stepped every tick
const K_SMALL = 4; // in-connections per "rest" neuron

export function createMaleCNS(seed) {
  const rand = mulberry32(seed ^ 0x5eed1234);
  const gauss = () => gaussian(rand);

  const A = new Float32Array(N); // activities

  // --- ORN: real glomerulus + side assignment from MaleCNS cell types
  const ornAnt = new Uint8Array(P.olfactory_receptor); // 0 left, 1 right
  const ornGlom = new Uint8Array(P.olfactory_receptor);
  const ornTune = new Float32Array(P.olfactory_receptor);
  for (let i = 0; i < P.olfactory_receptor; i++) {
    const r = HAS_REAL_WIRING ? connectome.orn[i] : null;
    ornAnt[i] = r ? r.s : i % 2;
    ornGlom[i] = r && r.g >= 0 ? r.g : Math.floor(i / 2) % GLOM;
    ornTune[i] = 0.6 + Math.abs(gauss()) * 0.8;
  }

  // --- PN: real hemisphere + glomerulus per neuron
  const pnAnt = new Uint8Array(P.antennal_lobe_projection);
  const pnGlom = new Uint8Array(P.antennal_lobe_projection);
  for (let i = 0; i < pnGlom.length; i++) {
    const r = HAS_REAL_WIRING ? connectome.pn[i] : null;
    pnAnt[i] = r ? r.s : i % 2;
    pnGlom[i] = r && r.g >= 0 ? r.g : Math.floor(i / 2) % GLOM;
  }

  // helper: build prefix-offset wiring from real partner lists w/ fallback
  function buildWiring(partnerLists, popOffset, popSize, fallbackK, fallbackFn, wScale) {
    const nTargets = partnerLists.length;
    const offs = new Int32Array(nTargets + 1);
    const from = [], W = [];
    for (let i = 0; i < nTargets; i++) {
      offs[i] = from.length;
      let real = partnerLists[i];
      let list = [];
      if (real && real.length > 0) {
        // normalize synapse counts to weights
        let sum = 0;
        for (const [, w] of real) sum += w;
        for (const [src, w] of real) list.push([src, (w / sum) * wScale]);
      } else {
        for (let k = 0; k < fallbackK; k++) list.push([Math.floor(rand() * popSize), (1 / fallbackK) * wScale]);
      }
      for (const [src, w] of list) { from.push(popOffset + src); W.push(w); }
    }
    offs[nTargets] = from.length;
    return { offs, from: Int32Array.from(from), W: Float32Array.from(W) };
  }

  // --- KC: real PN→KC connectivity (claws), fallback random ~7
  const kcWiring = buildWiring(
    HAS_REAL_WIRING ? connectome.kc_pn : [], PN, P.antennal_lobe_projection, 7,
    null, 1.0);
  const kcFrom = kcWiring.from, kcW = kcWiring.W, kcOffs = kcWiring.offs;
  const kcTheta = new Float32Array(P.kenyon_cell);
  for (let i = 0; i < kcTheta.length; i++) kcTheta[i] = 1.2 + rand() * 0.8;

  // --- DAN: baseline decay + reward drive
  const danGain = new Float32Array(P.dopaminergic);
  for (let i = 0; i < danGain.length; i++) danGain[i] = 0.5 + rand();

  // --- MBON: real KC→MBON convergence; plastic weights (lineage-carried)
  const mbonKc = buildWiring(
    HAS_REAL_WIRING ? connectome.mbon_kc : [], KC, P.kenyon_cell, 120,
    null, 1.0);
  const mbFrom = mbonKc.from, mbOffs = mbonKc.offs;
  const mbW = new Float32Array(mbFrom.length); // plastic
  for (let i = 0; i < mbW.length; i++) mbW[i] = gauss() * 0.1;

  // --- DN: real MBON→DN inputs (+ diffuse PN tone via bias below)
  const dnWiring = buildWiring(
    HAS_REAL_WIRING ? connectome.dn_mbon : [], MBON, P.mbon, 8,
    null, 1.0);
  const dnFrom = dnWiring.from, dnW = dnWiring.W, dnOffs = dnWiring.offs;
  // steering polarity: left-drive, right-drive, or thrust neuron
  const dnRole = new Uint8Array(P.descending);
  for (let i = 0; i < P.descending; i++) dnRole[i] = i % 3;

  // --- VNC intrinsic recurrent pool (relay/state)
  const VNC_K = 5;
  const vncFrom = new Int32Array(P.vnc_intrinsic * VNC_K);
  const vncW = new Float32Array(P.vnc_intrinsic * VNC_K);
  for (let i = 0; i < vncFrom.length; i++) {
    vncFrom[i] = VNC + Math.floor(rand() * P.vnc_intrinsic);
    vncW[i] = gauss() * 0.8;
  }

  // --- MN: real DN→MN inputs (+ VNC tone via bias below)
  const mnWiring = buildWiring(
    HAS_REAL_WIRING ? connectome.mn_dn : [], DN, P.descending, 6,
    null, 1.0);
  const mnFrom = mnWiring.from, mnW = mnWiring.W, mnOffs = mnWiring.offs;
  // MN roles: 0 left-turn, 1 right-turn, 2 thrust (wing steering MNs boost turn gain)
  const mnRole = new Uint8Array(P.motor);
  for (let i = 0; i < P.motor; i++) mnRole[i] = i % 3;

  // --- rest of the fly: sparse random recurrent (stepped slowly)
  const REST_N = N - REST;
  const restFrom = new Int32Array(REST_N * K_SMALL);
  const restW = new Float32Array(REST_N * K_SMALL);
  for (let i = 0; i < restFrom.length; i++) {
    restFrom[i] = REST + Math.floor(rand() * REST_N);
    restW[i] = gauss();
  }

  // wing steering MNs (real count) add turn authority
  const wingSteering = malecnsData.wing_steering_motor;

  let tickCount = 0;
  let rewardTrace = 0;
  let mLRun = 0, mRRun = 0, mFRun = 0, pnLRun = 0, pnRRun = 0, thrustRunning = 6, ambient = 0.3;
  const activity = {
    orn: 0, al: 0, kc: 0, dan: 0, mbon: 0, dn: 0, vnc: 0, mn: 0, rest: 0,
  };

  function step(dt, s, reward, flyAge = 0) {
    tickCount++;
    // ----- ORN: per-antenna drive (each antenna smells its own side plus
    // shared forward/total channels), rectified and tuned per neuron -----
    const gradSigned = s.smell_gradient * 10;
    const dL = 2.5 * s.smell_left + 1.2 * s.smell_forward + 1.0 * s.total_smell + Math.max(0, -gradSigned);
    const dR = 2.5 * s.smell_right + 1.2 * s.smell_forward + 1.0 * s.total_smell + Math.max(0, gradSigned);
    let ornSum = 0;
    for (let i = 0; i < P.olfactory_receptor; i++) {
      const target = Math.max(0, (ornAnt[i] === 0 ? dL : dR) * ornTune[i]);
      A[ORN + i] += (target - A[ORN + i]) * Math.min(1, dt * 12);
      ornSum += A[ORN + i];
    }
    const ornMean = ornSum / P.olfactory_receptor;

    // ----- AL local neurons: pooled gain control -----
    const ln = 1 + Math.tanh((ornMean - 0.25) * 4) * 1.6; // inhibition factor

    // ----- PN: hemisphere x glomerulus pool read, inhibited by LNs -----
    const pnPool = new Float32Array(GLOM * 2);
    for (let i = 0; i < P.olfactory_receptor; i++) pnPool[ornAnt[i] * GLOM + ornGlom[i]] += A[ORN + i];
    let pnL = 0, pnR = 0;
    for (let i = 0; i < P.antennal_lobe_projection; i++) {
      const h = pnAnt[i];
      const x = pnPool[h * GLOM + pnGlom[i]] / 45 - ln * 0.35; // ~45 ORNs/glob/hemisphere
      const target = Math.tanh(x);
      A[PN + i] += (target - A[PN + i]) * Math.min(1, dt * 15);
      if (h === 0) pnL += target; else pnR += target;
    }
    pnL /= P.antennal_lobe_projection / 2; pnR /= P.antennal_lobe_projection / 2;
    // very slow running means cancel wiring asymmetry while preserving the
    // tonic cross-antenna asymmetry that orients the fly toward a source
    pnLRun += (pnL - pnLRun) * 0.001;
    pnRRun += (pnR - pnRRun) * 0.001;

    // ----- KC: sparse thresholded code -----
    let kcSum = 0, kcActive = 0;
    for (let i = 0; i < P.kenyon_cell; i++) {
      let sum = 0;
      for (let k = kcOffs[i]; k < kcOffs[i + 1]; k++) sum += kcW[k] * A[kcFrom[k]];
      const target = Math.max(0, sum - kcTheta[i]);
      A[KC + i] += (target - A[KC + i]) * Math.min(1, dt * 20);
      if (A[KC + i] > 0.05) kcActive++;
      kcSum += A[KC + i];
    }

    // ----- DAN: phasic reward + small tonic from smell rise (PAM/PPL1) -----
    rewardTrace = Math.max(0, rewardTrace - dt * 0.8) + reward * 1.2;
    const danDrive = Math.min(2, rewardTrace);
    let danMean = 0;
    for (let i = 0; i < P.dopaminergic; i++) {
      const target = danDrive * danGain[i];
      A[DAN + i] += (target - A[DAN + i]) * Math.min(1, dt * 6);
      danMean += A[DAN + i];
    }
    danMean /= P.dopaminergic;

    // ----- MBON: plastic KC→MBON synapses (appetitive conditioning) -----
    const eta = 0.02 * dt * 10;
    let mbonPos = 0, mbonNeg = 0;
    for (let i = 0; i < P.mbon; i++) {
      let sum = 0;
      for (let k = mbOffs[i]; k < mbOffs[i + 1]; k++) {
        const kc = A[mbFrom[k]];
        // dopamine-gated plasticity: strengthen active KC→MBON when rewarded
        mbW[k] = Math.max(-1.5, Math.min(1.5, mbW[k] + eta * danMean * (kc - 0.1)));
        sum += mbW[k] * kc;
      }
      const target = Math.tanh(sum * 0.5);
      A[MBON + i] += (target - A[MBON + i]) * Math.min(1, dt * 12);
      if (i % 2 === 0) mbonPos += A[MBON + i]; else mbonNeg += A[MBON + i];
    }
    mbonPos /= P.mbon / 2; mbonNeg /= P.mbon / 2;

    // ----- DN: integrate learned valence + innate PN attraction + tone -----
    let dnL = 0, dnR = 0, dnF = 0;
    const valence = mbonPos - mbonNeg; // learned
    for (let i = 0; i < P.descending; i++) {
      let sum = valence * 0.6;
      for (let k = dnOffs[i]; k < dnOffs[i + 1]; k++) sum += dnW[k] * A[dnFrom[k]];
      const target = Math.tanh(sum * 0.7);
      A[DN + i] += (target - A[DN + i]) * Math.min(1, dt * 10);
      if (dnRole[i] === 0) dnL += A[DN + i];
      else if (dnRole[i] === 1) dnR += A[DN + i];
      else dnF += A[DN + i];
    }
    dnL /= P.descending / 3; dnR /= P.descending / 3; dnF /= P.descending / 3;

    // ----- VNC intrinsic recurrent relay -----
    for (let i = 0; i < P.vnc_intrinsic; i++) {
      const o = i * VNC_K;
      let sum = 0;
      for (let k = 0; k < VNC_K; k++) sum += vncW[o + k] * A[vncFrom[o + k]];
      const target = Math.tanh(sum * 0.9);
      A[VNC + i] += (target - A[VNC + i]) * Math.min(1, dt * 8);
    }

    // ----- Motor neurons -----
    let mL = 0, mR = 0, mF = 0;
    for (let i = 0; i < P.motor; i++) {
      let sum = 0;
      for (let k = mnOffs[i]; k < mnOffs[i + 1]; k++) sum += mnW[k] * A[mnFrom[k]];
      const role = mnRole[i];
      const bias = role === 0 ? dnL * 0.8 : role === 1 ? dnR * 0.8 : dnF * 2.5;
      const target = Math.max(0, Math.tanh(sum * 0.7 + bias));
      A[MN + i] += (target - A[MN + i]) * Math.min(1, dt * 12);
      if (role === 0) mL += A[MN + i];
      else if (role === 1) mR += A[MN + i];
      else mF += A[MN + i];
    }
    mL /= P.motor / 3; mR /= P.motor / 3; mF /= P.motor / 3;

    // ----- rest of the fly (optic lobe etc.): slow, diffuse tone -----
    if (tickCount % 8 === 0) {
      let restSum = 0;
      for (let i = 0; i < REST_N; i++) {
        const o = i * K_SMALL;
        let sum = 0;
        for (let k = 0; k < K_SMALL; k++) sum += restW[o + k] * A[restFrom[o + k]];
        const target = Math.tanh(sum * 0.5);
        A[REST + i] += (target - A[REST + i]) * Math.min(1, dt * 2);
        restSum += target;
      }
      activity.rest = restSum / REST_N;
    }

    // ----- decode: DC-remove pool imbalance (wiring asymmetry), then
    // wing steering MNs (real 23) modulate turn authority -----
    // slow EMA tracks each pool's tonic rate so constant wiring bias
    // cancels and only phasic left/right commands steer the fly
    mLRun += (mL - mLRun) * 0.002;
    mRRun += (mR - mRRun) * 0.002;
    mFRun += (mF - mFRun) * 0.002;
    const turnGain = 2.2 + (wingSteering / 23) * 1.2;
    // innate chemotaxis: turn toward the hemisphere with rising PN activity
    const innate = ((pnR - pnRRun) - (pnL - pnLRun)) * 2 + (pnR - pnL) * 2.2;
    // learned/valence steering from the descending pathway (random wiring → personality)
    const learned = ((mR - mRRun) - (mL - mLRun)) * 22;
    // olfactory adaptation: track ambient smell so behavior keys on the
    // plume CONTRAST, not the absolute background (real ORN adaptation)
    ambient += (s.total_smell - ambient) * 0.0006;
    const contrast = s.total_smell - ambient;
    // cast-and-thrust: with no plume contact, flies fly crosswind zigzags
    const onPlume = contrast > 0.06;
    const casting = onPlume ? 0 : Math.sin(flyAge * 1.4) * 1.3;
    const angular = (innate * turnGain + learned + casting) + (activity.rest - 0.3) * 0.4;
    // clean air → cruise fast; on-scent → slow and steer
    let thrust = 8 + (onPlume ? 0 : 5) + (mF - mFRun) * 20;
    // both antennae saturated and symmetric → we're on top of the source:
    // slow to a hover-drift and let the discovery radius do the rest
    if (contrast > 0.2 && Math.abs(pnR - pnL) < 0.03) thrust = 3.5;
    thrustRunning = Math.max(3, thrust);

    // ----- aggregates for the activity visualization -----
    activity.orn = ornMean;
    activity.al = ln / 2.6;
    activity.kc = kcActive / P.kenyon_cell;
    activity.dan = danMean / 2;
    activity.mbon = Math.abs(mbonPos) + Math.abs(mbonNeg);
    activity.dn = (Math.abs(dnL) + Math.abs(dnR) + Math.abs(dnF)) / 3;
    activity.vnc = meanRange(A, VNC, P.vnc_intrinsic);
    activity.mn = (mL + mR + mF) / 3;

    return { angular, thrust: thrustRunning };
  }

  // ---------- lineage: learned KC→MBON weights persist across flies (V1.1) ----------
  function getLearned() {
    return Float64Array.from(mbW);
  }
  function setLearned(w) {
    const m = Math.min(w.length, mbW.length);
    for (let i = 0; i < m; i++) mbW[i] = w[i];
  }

  return { step, activity, total: N, getLearned, setLearned };
}

function meanRange(A, off, len) {
  let s = 0;
  for (let i = 0; i < len; i += 16) s += Math.abs(A[off + i]); // sampled mean
  return s / Math.ceil(len / 16);
}
