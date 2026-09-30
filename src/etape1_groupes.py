"""029 step 1 (user): colour every SEPARATE group of matter differently. Matter = frozen slice_prep mask (grey >= 88, 12-px
cleaning); groups = connected components, 8-connectivity (the rule prep uses for its cleaning) and, for information, 4-connectivity.
Random distinct colours (largest groups get the most distinct hues), full slice + the 4 DEV zones. usage: etape1_groupes.py Z"""
import glob, json, sys
import numpy as np
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from slice_prep import prep

Z = int(sys.argv[1]) if len(sys.argv) > 1 else 4224
IMG = "/out/figures/etape1_groupes"; import os; os.makedirs(IMG, exist_ok=True)
S = prep(Z); m, g = S["matter"], S["gray"]
lab8, n8 = ndi.label(m, structure=np.ones((3, 3), bool)); lab4, n4 = ndi.label(m)
sz = np.bincount(lab8.ravel())[1:]; order = np.argsort(sz)[::-1]
rng = np.random.default_rng(1); cols = np.zeros((n8 + 1, 3))
# largest groups: well-separated hues (golden ratio), the rest random bright colours
import colorsys
for r, k in enumerate(order):
    h = (r * 0.618034) % 1.0 if r < 200 else rng.random()
    cols[k + 1] = colorsys.hsv_to_rgb(h, 0.55 + 0.45 * rng.random(), 0.75 + 0.25 * rng.random())
rgb = cols[lab8]; rgb[~m] = 0.0
# shade by the grey level so the texture stays visible
shade = np.clip(g.astype(float) / 255 * 1.4, 0.35, 1.0)[..., None]; rgbs = rgb * shade; rgbs[~m] = 0
yy, xx = np.nonzero(m); b = (yy.min() - 10, yy.max() + 10, xx.min() - 10, xx.max() + 10)
plt.imsave(f"{IMG}/Z{Z}_groupes_coupe_entiere.png", rgbs[b[0]:b[1], b[2]:b[3]])
fig, ax = plt.subplots(2, 4, figsize=(28, 14.5))
for j, f in enumerate(sorted(glob.glob("/vt/DEV/*/zone.json"))):
    w = json.load(open(f)); y0, x0, s = w["y0"], w["x0"], w["size_px"]
    ax[0, j].imshow(g[y0:y0 + s, x0:x0 + s], cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(f"{w['zone']} — original", fontsize=13)
    ax[1, j].imshow(rgbs[y0:y0 + s, x0:x0 + s]); nz = len(np.unique(lab8[y0:y0 + s, x0:x0 + s])) - 1
    ax[1, j].set_title(f"{nz} groupes de matière séparés dans la fenêtre", fontsize=13)
    for a in ax[:, j]: a.axis("off")
fig.suptitle(f"029 étape 1 — Z{Z} : chaque groupe de matière séparé (connexité 8) a sa couleur", fontsize=16)
fig.tight_layout(); fig.savefig(f"{IMG}/Z{Z}_groupes_zones_DEV.png", dpi=70); plt.close(fig)
tot = sz.sum(); cum = np.cumsum(np.sort(sz)[::-1]) / tot
res = {"z": Z, "matter_px": int(tot), "groups_8conn": int(n8), "groups_4conn": int(n4),
       "largest_group_share": float(sz.max() / tot), "groups_for_50pct_matter": int(np.searchsorted(cum, 0.5) + 1), "groups_for_90pct_matter": int(np.searchsorted(cum, 0.9) + 1),
       "size_px_percentiles_10_50_90_99": np.percentile(sz, [10, 50, 90, 99]).tolist(), "groups_ge_1000px": int((sz >= 1000).sum()), "groups_ge_10000px": int((sz >= 10000).sum()),
       "top10_sizes_px": np.sort(sz)[::-1][:10].tolist()}
print(json.dumps(res, indent=1)); json.dump(res, open(f"/out/etape1_groupes_z{Z}.json", "w"), indent=1)
np.savez_compressed(f"/out/etape1_labels_z{Z}.npz", lab8=lab8.astype(np.int32))
