#!/usr/bin/env python3
"""Extract real olfactory-pathway wiring from the MaleCNS v0.9 edgelist.

Reads the 3.4 GB simple edgelist feather, keeps only edges whose pre/post
neurons belong to the olfactory pathway (ORN, LN, PN, KC, DAN, MBON, DN, MN),
and emits app/data/malecns_connectome.json:

  - pn_to_kc:   real PN→KC partner lists (KC claws)
  - kc_to_mbon: real KC→MBON convergence
  - mbon_to_dn: real MBON→DN inputs
  - dn_to_mn:   real DN→MN inputs
  - orn_glom / pn_glom: glomerulus label per neuron (from cell types)

Runtime: a few minutes; the feather is read once, then filtered with Arrow.
"""
import json, sys, time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from collections import defaultdict

EDGELIST = sys.argv[1] if len(sys.argv) > 1 else "data/malecns_09_simple_edgelist.feather"

print("reading meta…")
meta = feather.read_table("data/malecns_meta.feather").to_pydict()
n = len(meta["malecns_09_id"])
ids = [int(x) for x in meta["malecns_09_id"]]
id_to_idx = {v: i for i, v in enumerate(ids)}
cls = [x if x else "" for x in meta["cell_class"]]
sub = [x if x else "" for x in meta["cell_sub_class"]]
ctype = [x if x else "" for x in meta["cell_type"]]

def idx_where(pred):
    return [i for i in range(n) if pred(i)]

ORN_I = idx_where(lambda i: cls[i] == "olfactory_receptor_neuron")
LN_I = idx_where(lambda i: cls[i] == "antennal_lobe_local_neuron")
PN_I = idx_where(lambda i: cls[i] == "antennal_lobe_projection_neuron")
KC_I = idx_where(lambda i: cls[i] == "kenyon_cell")
DAN_I = idx_where(lambda i: "dopaminergic" in cls[i])
MBON_I = idx_where(lambda i: cls[i] == "mushroom_body_output_neuron")
DN_I = idx_where(lambda i: meta["super_class"][i] == "descending")
MN_I = idx_where(lambda i: meta["super_class"][i] == "motor")

ORN_S = set(ids[i] for i in ORN_I)
PN_S = set(ids[i] for i in PN_I)
KC_S = set(ids[i] for i in KC_I)
MBON_S = set(ids[i] for i in MBON_I)
DN_S = set(ids[i] for i in DN_I)
MN_S = set(ids[i] for i in MN_I)
LN_S = set(ids[i] for i in LN_I)
DAN_S = set(ids[i] for i in DAN_I)
PATH = ORN_S | PN_S | KC_S | MBON_S | DN_S | MN_S | LN_S | DAN_S
print(f"pathway neurons: {len(PATH)}")

# glomerulus per ORN and per uPN (from cell type names)
def glom_of_orn(ct):
    return ct[4:] if ct.startswith("ORN_") else None

def glom_of_pn(sub_c, ct):
    if sub_c != "uniglomerular_projection_neuron":
        return None
    # DA1_lPN / VM5d_adPN / VP1l+_lvPN → glomerulus is the first token
    for suffix in ("_lPN", "_adPN", "_lvPN"):
        if ct.endswith(suffix):
            return ct[: -len(suffix)]
    return None

orn_glom = {}
for i in ORN_I:
    g = glom_of_orn(ctype[i])
    if g: orn_glom[ids[i]] = g
pn_glom = {}
for i in PN_I:
    g = glom_of_pn(sub[i], ctype[i])
    if g: pn_glom[ids[i]] = g
print(f"ORN with glomerulus: {len(orn_glom)}, uPN with glomerulus: {len(pn_glom)}")

print("streaming edgelist batch-wise (int64 cast + numpy isin)…")
t0 = time.time()
f = pa.memory_map(EDGELIST, "r")
reader = ipc.open_file(f)
nb = reader.num_record_batches
path_ids = np.array(sorted(PATH), dtype=np.int64)
sel_pre, sel_post, sel_w = [], [], []
total_rows = 0
for b in range(nb):
    batch = reader.get_batch(b)
    total_rows += batch.num_rows
    p_np = batch.column(0).cast(pa.int64()).to_numpy()
    q_np = batch.column(1).cast(pa.int64()).to_numpy()
    w_np = batch.column(2).to_numpy()
    mask = np.isin(p_np, path_ids) & np.isin(q_np, path_ids)
    if mask.any():
        sel_pre.append(p_np[mask])
        sel_post.append(q_np[mask])
        sel_w.append(w_np[mask])
    if b % 200 == 0:
        print(f"  batch {b}/{nb} rows {total_rows/1e6:.0f}M kept {sum(len(x) for x in sel_pre)/1e6:.1f}M ({time.time()-t0:.0f}s)", flush=True)
pre_s = np.concatenate(sel_pre)
post_s = np.concatenate(sel_post)
w_s = np.concatenate(sel_w)
del sel_pre, sel_post, sel_w
print(f"pathway-internal edges: {len(pre_s)} ({time.time()-t0:.0f}s)")
print("diagnostic PN→KC hits:", int((np.isin(pre_s, np.array(sorted(PN_S))) & np.isin(post_s, np.array(sorted(KC_S)))).sum()))

# build partner lists between populations
def edges_between(src_set, dst_set, min_w=1):
    out = defaultdict(list)
    for p, q, w in zip(pre_s.tolist(), post_s.tolist(), w_s.tolist()):
        if p in src_set and q in dst_set and w >= min_w:
            out[q].append((p, int(w)))
    return out

print("diagnostics: sample kept pre", pre_s[:5].tolist(), "post", post_s[:5].tolist())
print("diagnostics: PN_S sample", sorted(PN_S)[:3], "KC_S sample", sorted(KC_S)[:3])
print("diagnostics: pre in PN_S", sum(1 for x in pre_s[:100000].tolist() if x in PN_S), "/100k")
print("diagnostics: post in KC_S", sum(1 for x in post_s[:100000].tolist() if x in KC_S), "/100k")
print("building PN→KC…")
_hits = [0]
def _dbg(src_set, dst_set):
    n_hits = 0
    for p, q in zip(pre_s.tolist(), post_s.tolist()):
        if p in src_set and q in dst_set: n_hits += 1
    return n_hits
print("  PN→KC hits via direct scan:", _dbg(PN_S, KC_S))
print("  types:", type(next(iter(PN_S))), type(pre_s[0]))
pn_to_kc = edges_between(PN_S, KC_S)
print("building KC→MBON…")
kc_to_mbon = edges_between(KC_S, MBON_S)
print("building MBON→DN…")
mbon_to_dn = edges_between(MBON_S, DN_S)
print("building DN→MN…")
dn_to_mn = edges_between(DN_S, MN_S)
print("building LN→PN (gain control)…")
ln_to_pn = edges_between(LN_S, PN_S)
print("building DAN→MBON…")
dan_to_mbon = edges_between(DAN_S, MBON_S)

def summarize(partners, name):
    if not partners:
        print(f"  {name}: EMPTY")
        return
    ks = np.array([len(v) for v in partners.values()])
    print(f"  {name}: {len(partners)} targets, inputs/target min {ks.min()} med {int(np.median(ks))} max {ks.max()}")

summarize(pn_to_kc, "PN→KC")
summarize(kc_to_mbon, "KC→MBON")
summarize(mbon_to_dn, "MBON→DN")
summarize(dn_to_mn, "DN→MN")

# serialize: per-target partner id lists (as meta row indices for compactness)
def pack(partners):
    return {str(id_to_idx[q]): [[id_to_idx[p], w] for p, w in v] for q, v in partners.items()}

out = {
    "source": "MaleCNS v0.9 simple edgelist, filtered to olfactory pathway",
    "pn_to_kc": pack(pn_to_kc),
    "kc_to_mbon": pack(kc_to_mbon),
    "mbon_to_dn": pack(mbon_to_dn),
    "dn_to_mn": pack(dn_to_mn),
    "orn_glom": {str(id_to_idx[k]): v for k, v in orn_glom.items()},
    "pn_glom": {str(id_to_idx[k]): v for k, v in pn_glom.items()},
}
with open("app/data/malecns_connectome.json", "w") as f:
    json.dump(out, f)
import os
print(f"wrote app/data/malecns_connectome.json ({os.path.getsize('app/data/malecns_connectome.json')/1e6:.1f} MB)")
