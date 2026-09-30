"""029 step 2 (user): trace the material TANGENT of the sheets and WELD gaps of up to 2 px along it (welds in green).
Tangent = the frozen slice_prep structure-tensor tangent (grey, sigma 1 / rho 4) at every pixel. From every matter pixel p, in
both directions +/- t: if the pixels at p + k t (k = 1..g, rounded, 1-px steps) are VOID and p + (g + 1) t is MATTER, for
g = 1 or 2, the g void pixels are filled (weld). Guard (added, reported): the matter reached must have a tangent within
30 deg of t (weld along the same sheet); the unguarded count is also reported. Effect on the matter groups of step 1.
usage: etape2_soudure.py Z [GMAX]  (step 3: GMAX 4 + coloured groups after welding)"""
import glob, json, os, sys
import numpy as np
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from slice_prep import prep

Z = int(sys.argv[1]) if len(sys.argv) > 1 else 4224; GMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 2; ANG = 30.0
IMG = "/out/figures/" + ("etape2_soudure" if GMAX == 2 else f"etape3_soudure{GMAX}px_groupes"); os.makedirs(IMG, exist_ok=True)
S = prep(Z); m, g = S["matter"], S["gray"]; tx, ty = S["tx"], S["ty"]; N = m.shape[0]
ys, xs = np.nonzero(m); t_x, t_y = tx[ys, xs], ty[ys, xs]


def welds(guard):
    W = np.zeros_like(m)
    for sgn in (1, -1):
        for gap in range(1, GMAX + 1):
            qs = [(np.rint(ys + sgn * k * t_y).astype(int), np.rint(xs + sgn * k * t_x).astype(int)) for k in range(1, gap + 2)]
            ok = np.ones(len(ys), bool)
            for qy, qx in qs:
                ok &= (qy >= 0) & (qy < N) & (qx >= 0) & (qx < N)
            qs = [(np.clip(qy, 0, N - 1), np.clip(qx, 0, N - 1)) for qy, qx in qs]
            for qy, qx in qs[:-1]:
                ok &= ~m[qy, qx]
            ey, ex = qs[-1]; ok &= m[ey, ex]
            # the gap must really be g long (the pixel before the far matter must not be the start pixel itself)
            if guard:
                ok &= np.abs(t_x * tx[ey, ex] + t_y * ty[ey, ex]) >= np.cos(np.radians(ANG))
            for qy, qx in qs[:-1]:
                W[qy[ok], qx[ok]] = True
    return W & ~m


W = welds(True); W0 = welds(False); mw = m | W
lab0, n0 = ndi.label(m, structure=np.ones((3, 3))); lab1, n1 = ndi.label(mw, structure=np.ones((3, 3)))
sz1 = np.bincount(lab1.ravel())[1:]
_, nw = ndi.label(W, structure=np.ones((3, 3)))
res = {"z": Z, "gap_max_px": GMAX, "guard_parallel_deg": ANG, "weld_px": int(W.sum()), "weld_sites": int(nw), "weld_px_share_of_matter": float(W.sum() / m.sum()),
       "weld_px_without_guard": int(W0.sum()), "groups_before": int(n0), "groups_after": int(n1), "largest_group_share_after": float(sz1.max() / sz1.sum())}
print(json.dumps(res, indent=1)); json.dump(res, open(f"/out/etape2_soudure_z{Z}_g{GMAX}.json", "w"), indent=1)
np.savez_compressed(f"/out/etape2_welds_z{Z}_g{GMAX}.npz", weld=W, weld_noguard=W0, lab_after=lab1.astype(np.int32))
# images: grey matter, welds in green; tangent lines drawn on a zoom
rgb = np.stack([np.where(m, g / 255.0, 0)] * 3, -1) * 0.9; rgb[W] = (0.1, 1.0, 0.1)
yy, xx = np.nonzero(m); b = (yy.min() - 10, yy.max() + 10, xx.min() - 10, xx.max() + 10)
plt.imsave(f"{IMG}/Z{Z}_soudures_coupe_entiere.png", np.clip(rgb[b[0]:b[1], b[2]:b[3]], 0, 1))
fig, ax = plt.subplots(2, 4, figsize=(28, 14.5))
for j, f in enumerate(sorted(glob.glob("/vt/DEV/*/zone.json"))):
    w = json.load(open(f)); y0, x0, s = w["y0"], w["x0"], w["size_px"]; ext = (x0, x0 + s, y0 + s, y0)
    ax[0, j].imshow(np.clip(rgb[y0:y0 + s, x0:x0 + s], 0, 1), extent=ext); ax[0, j].set_title(f"{w['zone']} — soudures ≤ {GMAX} px en vert ({int(W[y0:y0 + s, x0:x0 + s].sum())} px)", fontsize=12)
    # tangent lines on the central 100 px of the window, every 6 px on matter
    c0y, c0x = y0 + s // 2 - 50, x0 + s // 2 - 50; ax[1, j].imshow(np.clip(rgb[c0y:c0y + 100, c0x:c0x + 100], 0, 1), extent=(c0x, c0x + 100, c0y + 100, c0y))
    gy, gx = np.mgrid[c0y:c0y + 100:6, c0x:c0x + 100:6]; sel = m[gy, gx]; gy, gx = gy[sel], gx[sel]
    for yv, xv in zip(gy, gx):
        ax[1, j].plot([xv - 2.5 * tx[yv, xv], xv + 2.5 * tx[yv, xv]], [yv - 2.5 * ty[yv, xv], yv + 2.5 * ty[yv, xv]], color="orange", lw=1.2)
    ax[1, j].set_xlim(c0x, c0x + 100); ax[1, j].set_ylim(c0y + 100, c0y); ax[1, j].set_title("zoom 100 px : tangente de la matière (orange), soudures (vert)", fontsize=11)
    for a in ax[:, j]: a.axis("off")
fig.suptitle(f"029 étape 2 — Z{Z} : soudure des sauts ≤ {GMAX} px le long de la tangente de la matière ; groupes {n0} → {n1}", fontsize=16)
fig.tight_layout(); fig.savefig(f"{IMG}/Z{Z}_soudures_zones_DEV.png", dpi=70); plt.close(fig)

# step 3: the matter GROUPS after welding, one colour per group (as step 1), welds in green
import colorsys
sz = np.bincount(lab1.ravel())[1:]; order = np.argsort(sz)[::-1]; rng = np.random.default_rng(1); cols = np.zeros((n1 + 1, 3))
for r, k in enumerate(order):
    cols[k + 1] = colorsys.hsv_to_rgb((r * 0.618034) % 1.0 if r < 200 else rng.random(), 0.55 + 0.45 * rng.random(), 0.75 + 0.25 * rng.random())
shade = np.clip(g.astype(float) / 255 * 1.4, 0.35, 1.0)[..., None]; G = cols[lab1] * shade; G[~mw] = 0; G[W] = (0.1, 1.0, 0.1)
plt.imsave(f"{IMG}/Z{Z}_groupes_apres_soudure_{GMAX}px_coupe_entiere.png", np.clip(G[b[0]:b[1], b[2]:b[3]], 0, 1))
fig, ax = plt.subplots(1, 4, figsize=(28, 7.6))
for j, f in enumerate(sorted(glob.glob("/vt/DEV/*/zone.json"))):
    w = json.load(open(f)); y0, x0, s = w["y0"], w["x0"], w["size_px"]
    ax[j].imshow(np.clip(G[y0:y0 + s, x0:x0 + s], 0, 1)); ax[j].axis("off")
    ax[j].set_title(f"{w['zone']} — {len(np.unique(lab1[y0:y0 + s, x0:x0 + s])) - 1} groupes après soudure", fontsize=12)
fig.suptitle(f"029 — Z{Z} : groupes de matière après soudure ≤ {GMAX} px (une couleur par groupe, soudures en vert) : {n0} → {n1} groupes, le plus grand = {100 * sz.max() / sz.sum():.1f} % de la matière", fontsize=14)
fig.tight_layout(); fig.savefig(f"{IMG}/Z{Z}_groupes_apres_soudure_{GMAX}px_zones_DEV.png", dpi=70); plt.close(fig)
