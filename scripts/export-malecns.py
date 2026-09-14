#!/usr/bin/env python3
"""Export MaleCNS v0.9 population counts for the Taco Fly V1 controller.

Downloads/reads the malecns_09 metadata feather and emits data/malecns.json
with the exact neuron counts used to size each simulated population.
Source: Lee lab brain-and-nerve-cord connectome (male-cns v0.9), CC-BY.
"""
import json, sys
import pyarrow.feather as feather
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "data/malecns_meta.feather"
t = feather.read_table(path).to_pydict()
n = len(t["malecns_09_id"])

def col(name):
    return [x if x is not None else "" for x in t[name]]

sup, cls, sub = col("super_class"), col("cell_class"), col("cell_sub_class")

def count(pred):
    return sum(1 for i in range(n) if pred(i))

orn = count(lambda i: cls[i] == "olfactory_receptor_neuron")
al_ln = count(lambda i: cls[i] == "antennal_lobe_local_neuron")
al_pn = count(lambda i: cls[i] == "antennal_lobe_projection_neuron")
al_other = count(lambda i: cls[i].startswith("antennal_lobe") and "local" not in cls[i] and "projection" not in cls[i])
kc = count(lambda i: cls[i] == "kenyon_cell")
dan = count(lambda i: "dopaminergic" in cls[i])
mbon = count(lambda i: cls[i] == "mushroom_body_output_neuron")
dn = count(lambda i: sup[i] == "descending")
an = count(lambda i: sup[i] in ("ascending", "sensory_ascending"))
mn = count(lambda i: sup[i] == "motor")
vnc = count(lambda i: sup[i] == "ventral_nerve_cord_intrinsic")
optic = count(lambda i: sup[i] in ("optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"))
central = count(lambda i: sup[i] == "central_brain_intrinsic")
sensory_other = count(lambda i: sup[i] == "sensory" and cls[i] != "olfactory_receptor_neuron")

# central brain subtotal already includes AL/KC/DAN/MBON — split it out
al_total = al_ln + al_pn + al_other
mb_total = kc + dan + mbon
central_rest = central - al_total - mb_total
other = n - (orn + al_total + mb_total + dn + an + mn + vnc + optic + central_rest + sensory_other)

kc_by_lob = {}
for i in range(n):
    if cls[i] == "kenyon_cell":
        kc_by_lob[sub[i] or "other"] = kc_by_lob.get(sub[i] or "other", 0) + 1

wing_steering = count(lambda i: sub[i] == "wing_steering_motor_neuron")

out = {
    "source": "MaleCNS v0.9 (male-cns:v0.9), Lee lab brain-and-nerve-cord fly connectome, CC-BY",
    "total_neurons": n,
    "populations": {
        "olfactory_receptor": orn,
        "antennal_lobe_local": al_ln,
        "antennal_lobe_projection": al_pn,
        "antennal_lobe_other": al_other,
        "kenyon_cell": kc,
        "dopaminergic": dan,
        "mbon": mbon,
        "descending": dn,
        "ascending": an,
        "motor": mn,
        "vnc_intrinsic": vnc,
        "optic": optic,
        "central_brain_rest": central_rest,
        "sensory_other": sensory_other,
        "other": other,
    },
    "kenyon_by_lobule": kc_by_lob,
    "wing_steering_motor": wing_steering,
    "glomeruli": 57,  # published count of antennal lobe glomeruli (abstraction: ORNs assigned round-robin)
}
with open("data/malecns.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["populations"], indent=1))
print("total:", n)
