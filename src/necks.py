"""029 step 4: NARROW NECKS (<= 4 px wide) of the welded sample (matter + step-3 welds <= 4 px). Morphological opening with a
disk of radius 2 (a structure <= 4 px wide disappears); every matter pixel takes the label of the nearest opened (thick) part;
lattice bonds (fsi_kernel_002.build_bonds order on the welded matter) joining two DIFFERENT thick parts = neck bonds = the only
breakable bonds (the centrifuge breaks them at >= 2 % strain). CONTROL: the same number of breakable bonds drawn at random among
the other lattice bonds (seed 5). -> /out/necks_z{Z}.npz (cut = breakable mask), /out/necks_z{Z}_ctrl.npz, /out/welded_z{Z}.npz
usage: necks.py Z"""
import json, sys
import numpy as np
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import fsi_kernel_002 as k2
from slice_prep import prep
Z = int(sys.argv[1]) if len(sys.argv) > 1 else 4224
S = prep(Z); m0 = S["matter"]; W = np.load(f"/out/etape2_welds_z{Z}_g4.npz")["weld"]; m = m0 | W
np.savez_compressed(f"/out/welded_z{Z}.npz", weld=W)
disk = np.hypot(*np.mgrid[-2:3, -2:3]) <= 2.0
opened = ndi.binary_opening(m, structure=disk)
lab, n = ndi.label(opened, structure=np.ones((3, 3)))
_, (iy, ix) = ndi.distance_transform_edt(~opened, return_indices=True); near = lab[iy, ix] * m
ys, xs = np.nonzero(m)
_, _, bi, bj, _, _, _, _ = k2.build_bonds(m, S["tx"], S["ty"], 1.0, 0.1, 6.0, 0.1)
li, lj = near[ys[bi], xs[bi]], near[ys[bj], xs[bj]]
neck = (li != lj) & (li > 0) & (lj > 0)
rng = np.random.default_rng(5); ctrl = np.zeros(len(bi), bool); ctrl[rng.choice(np.nonzero(~neck)[0], int(neck.sum()), replace=False)] = True
np.savez_compressed(f"/out/necks_z{Z}.npz", cut=neck); np.savez_compressed(f"/out/necks_z{Z}_ctrl.npz", cut=ctrl)
npx = np.zeros(m.shape, bool); npx[ys[bi[neck]], xs[bi[neck]]] = True; npx[ys[bj[neck]], xs[bj[neck]]] = True
res = {"z": Z, "matter_px": int(m.sum()), "weld_px": int(W.sum()), "thick_parts": int(n), "neck_bonds": int(neck.sum()), "neck_bond_share": float(neck.mean()),
       "neck_px": int(npx.sum()), "lattice_bonds": int(len(bi))}
print(json.dumps(res, indent=1)); json.dump(res, open(f"/out/necks_z{Z}.json", "w"), indent=1)
g = S["gray"]; rgb = np.stack([np.where(m, g / 255.0, 0)] * 3, -1) * 0.85; rgb[W] = (0.1, 0.9, 0.1); rgb[npx] = (1, 0.1, 0.1)
import glob
fig, ax = plt.subplots(1, 4, figsize=(28, 7.6))
for j, f in enumerate(sorted(glob.glob("/vt/DEV/*/zone.json"))):
    w = json.load(open(f)); y0, x0, s = w["y0"], w["x0"], w["size_px"]
    ax[j].imshow(np.clip(rgb[y0:y0 + s, x0:x0 + s], 0, 1)); ax[j].axis("off"); ax[j].set_title(f"{w['zone']} — cols ≤ 4 px (rouge), soudures (vert)", fontsize=12)
fig.suptitle(f"029 étape 4 — Z{Z} : cols étroits (≤ 4 px) entre parties épaisses = seuls liens cassables ({int(neck.sum())} liens)", fontsize=14)
fig.tight_layout(); fig.savefig(f"/out/figures/etape4_cols_Z{Z}.png", dpi=70); plt.close(fig)
