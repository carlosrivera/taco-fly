#!/usr/bin/env python3
"""Convert meta-index connectome JSON into final local-index format for malecns.js."""
import json

meta = feather_idx = None
import pyarrow.feather as feather
meta = feather.read_table("data/malecns_meta.feather").to_pydict()
n = len(meta["malecns_09_id"])
ids = [int(x) for x in meta["malecns_09_id"]]
idx_of = {v: i for i, v in enumerate(ids)}
cls = [x if x else "" for x in meta["cell_class"]]
sub = [x if x else "" for x in meta["cell_sub_class"]]
side = [x if x else "" for x in meta["side"]]

def local_list(pred):
    """ordered meta indices of a population → local index lookup"""
    metas = [i for i in range(n) if pred(i)]
    return metas, {m: k for k, m in enumerate(metas)}

ORN_M, orn_local = local_list(lambda i: cls[i] == "olfactory_receptor_neuron")
PN_M, pn_local = local_list(lambda i: cls[i] == "antennal_lobe_projection_neuron")
KC_M, kc_local = local_list(lambda i: cls[i] == "kenyon_cell")
MBON_M, mbon_local = local_list(lambda i: cls[i] == "mushroom_body_output_neuron")
DN_M, dn_local = local_list(lambda i: meta["super_class"][i] == "descending")
MN_M, mn_local = local_list(lambda i: meta["super_class"][i] == "motor")

with open("app/data/malecns_connectome.json") as f:
    raw = json.load(f)

def glom_orn(ct):
    return ct[4:] if ct.startswith("ORN_") else None

def glom_pn(sub_c, ct):
    if sub_c != "uniglomerular_projection_neuron":
        return None
    for suffix in ("_lPN", "_adPN", "_lvPN"):
        if ct.endswith(suffix):
            return ct[: -len(suffix)]
    return None

GL = {}
def glom_idx(name):
    if name not in GL:
        GL[name] = len(GL)
    return GL[name]

orn = []
for m in ORN_M:
    g = glom_orn(ctype := meta["cell_type"][m])
    s = side[m]
    orn.append({
        "g": glom_idx(g) if g else -1,
        "s": 0 if s == "left" else 1 if s == "right" else m % 2,
    })
pn = []
for m in PN_M:
    g = glom_pn(sub[m], meta["cell_type"][m])
    s = side[m]
    pn.append({
        "g": glom_idx(g) if g else -1,
        "s": 0 if s == "left" else 1 if s == "right" else m % 2,
        "u": sub[m] == "uniglomerular_projection_neuron",
    })

def remap(partners, dst_local_map, src_local_map, cap):
    """{dstMeta: [[srcMeta, w],...]} → [[ [srcLocal, w], ...] per dstLocal]"""
    out = []
    for m in sorted(dst_local_map, key=dst_local_map.get):
        pairs = partners.get(str(m), [])
        conv = [(src_local_map[p], w) for p, w in ((int(a), b) for a, b in pairs) if p in src_local_map]
        conv.sort(key=lambda x: -x[1])
        conv = conv[:cap]
        out.append(conv)
    return out

kc_pn = remap(raw["pn_to_kc"], kc_local, pn_local, 24)       # real claws (capped)
mbon_kc = remap(raw["kc_to_mbon"], mbon_local, kc_local, 400)  # real convergence (capped)
dn_mbon = remap(raw["mbon_to_dn"], dn_local, mbon_local, 30)
mn_dn = remap(raw["dn_to_mn"], mn_local, dn_local, 40)

def stats(rows, name):
    ks = [len(r) for r in rows]
    nz = sum(1 for k in ks if k > 0)
    print(f"{name}: {nz}/{len(rows)} wired, max {max(ks)}, median {sorted(ks)[len(ks)//2]}")

stats(kc_pn, "KC←PN")
stats(mbon_kc, "MBON←KC")
stats(dn_mbon, "DN←MBON")
stats(mn_dn, "MN←DN")

out = {
    "source": "MaleCNS v0.9 edgelist, olfactory pathway, local indices",
    "glomeruli": [k for k, _ in sorted(GL.items(), key=lambda kv: kv[1])],
    "orn": orn,
    "pn": pn,
    "kc_pn": kc_pn,
    "mbon_kc": mbon_kc,
    "dn_mbon": dn_mbon,
    "mn_dn": mn_dn,
}
with open("app/data/malecns_connectome.json", "w") as f:
    json.dump(out, f)
import os
print(f"wrote {os.path.getsize('app/data/malecns_connectome.json')/1e6:.1f} MB")
