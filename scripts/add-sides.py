#!/usr/bin/env python3
"""Add real left/right hemisphere assignment to the connectome bundle."""
import json
import pyarrow.feather as feather

meta = feather.read_table("data/malecns_meta.feather").to_pydict()
n = len(meta["malecns_09_id"])
ids = [int(x) for x in meta["malecns_09_id"]]
idx_of = {v: i for i, v in enumerate(ids)}
cls = [x if x else "" for x in meta["cell_class"]]
side = [x if x else "" for x in meta["side"]]

def side_map(want_cls):
    out = {}
    for i in range(n):
        if cls[i] == want_cls and side[i] in ("left", "right"):
            out[str(idx_of[ids[i]])] = 0 if side[i] == "left" else 1
    return out

with open("app/data/malecns_connectome.json") as f:
    data = json.load(f)

data["orn_side"] = side_map("olfactory_receptor_neuron")
data["pn_side"] = side_map("antennal_lobe_projection_neuron")

with open("app/data/malecns_connectome.json", "w") as f:
    json.dump(data, f)
print("added sides:", len(data["orn_side"]), "ORN,", len(data["pn_side"]), "PN")
