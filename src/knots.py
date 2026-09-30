"""025 STRESS KNOTS: automatic detection of the points where sheets stay pinned (could break but do not), from a
centrifuged state without rupture (default cf_g3000d: beta 3000, medium, grey mass, 2000 steps). usage: knots.py [STATE]
Detector A (mechanical): bonds oriented near the material NORMAL (|e.n| >= cos 30 deg) with tensile strain > 2 % (they would
  break under the 2 % rule but rupture is not allowed there); nodes carrying such bonds, grouped (8-connected after a 1 px
  dilation), components of >= 5 nodes = knots A.
Detector B (geometric, change of radius of curvature): every C1 line (original) is followed on the deformed lattice; local
  curvature (tangent turn over +/- 5 px of arc) before and after; points where |kappa_after - kappa_before| is in the top
  0.5 % AND the deformed radius < 20 px (190 um); grouped within 8 px = knots B.
Outputs: knot masks per node (for the unlocking experiment), random control masks (same component shapes moved to random
matter positions away from the knots), agreement A/B, maps and zooms. -> /out/knots_STATE.npz / .json, LBZ figures"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import fsi_kernel_002 as k2
from predetect import material_fields
from slice_prep import prep
from vt_eval import resample

STATE = sys.argv[1] if len(sys.argv) > 1 else "g3000d"
S = prep(4224); m, g = S["matter"], S["gray"]; N = m.shape[0]; ys, xs = np.nonzero(m); P = len(ys)
d = np.load(f"/out/cf_{STATE}/final.npz"); ux, uy = d["ux"].astype(float), d["uy"].astype(float)
px, py, bi, bj, L0, _, _, _ = k2.build_bonds(m, S["tx"], S["ty"], 1.0, 0.1, 6.0, 0.1)
B0 = np.load("/p19/var_sig05_W60_40/z4224.npz"); bands = [B0[k][:, :3].astype(float) for k in B0.files]
tx, ty, _ = material_fields(bands, m.shape, S["tx"], S["ty"])
ex, ey = (px[bj] - px[bi]), (py[bj] - py[bi]); ln = np.hypot(ex, ey); ex, ey = ex / ln, ey / ln
c2 = (tx[ys[bi], xs[bi]] ** 2 - ty[ys[bi], xs[bi]] ** 2) + (tx[ys[bj], xs[bj]] ** 2 - ty[ys[bj], xs[bj]] ** 2)
s2 = 2 * tx[ys[bi], xs[bi]] * ty[ys[bi], xs[bi]] + 2 * tx[ys[bj], xs[bj]] * ty[ys[bj], xs[bj]]
th = 0.5 * np.arctan2(s2, c2); en = np.abs(-ex * np.sin(th) + ey * np.cos(th))
dx = (px[bj] + ux[bj]) - (px[bi] + ux[bi]); dy = (py[bj] + uy[bj]) - (py[bi] + uy[bi]); strain = (np.hypot(dx, dy) - L0) / L0
# ---- detector A
# v2 (first run: a fixed 2 % threshold flagged 16 % of the matter at beta 3000): a knot = CONCENTRATION of normal tension --
# normal-bond strain in the top 0.5 % AND >= 3 x the local mean normal strain (Gaussian 8 px ~ 80 um around)
nb = en >= np.cos(np.radians(30))
smap = np.zeros(m.shape); cnt = np.zeros(m.shape); np.add.at(smap, (ys[bi[nb]], xs[bi[nb]]), np.maximum(strain[nb], 0)); np.add.at(cnt, (ys[bi[nb]], xs[bi[nb]]), 1)
loc = ndi.gaussian_filter(smap, 8) / np.maximum(ndi.gaussian_filter(cnt, 8), 1e-9)
hot = nb & (strain >= np.quantile(strain[nb], 0.995)) & (strain >= 3 * loc[ys[bi], xs[bi]])
nodeA = np.zeros(P, bool); nodeA[bi[hot]] = True; nodeA[bj[hot]] = True
img = np.zeros(m.shape, bool); img[ys[nodeA], xs[nodeA]] = True
lab, n = ndi.label(ndi.binary_dilation(img, np.ones((3, 3))) & m, structure=np.ones((3, 3)))
sizes = np.bincount(lab.ravel()); keep = np.nonzero(sizes >= 5)[0]; keep = keep[keep > 0]
knA_img = np.isin(lab, keep); knotA = knA_img[ys, xs]
compsA = [np.argwhere(lab == k) for k in keep] if len(keep) < 20000 else []
# ---- detector B
idx = -np.ones(m.shape, np.int64); idx[ys, xs] = np.arange(P); T = cKDTree(np.c_[ys, xs])
def curv(C, h=5):
    if len(C) < 2 * h + 3:
        return np.zeros(len(C))
    t = np.gradient(C, axis=0); a = np.unwrap(np.arctan2(t[:, 0], t[:, 1]))
    k = np.zeros(len(C)); k[h:-h] = (a[2 * h:] - a[:-2 * h]) / (2 * h); return k
ptsB, dk_all, rd_all = [], [], []
for b in bands:
    C = resample(b[:, :2])
    if len(C) < 20:
        continue
    dd, j = T.query(C); ok = dd <= 2
    if ok.sum() < 20:
        continue
    C = C[ok]; j = j[ok]; Cd = C + np.c_[uy[j], ux[j]]
    k0, k1 = curv(ndi.gaussian_filter1d(C, 2, axis=0)), curv(ndi.gaussian_filter1d(Cd, 2, axis=0))
    ptsB.append(C); dk_all.append(np.abs(k1 - k0)); rd_all.append(1 / np.maximum(np.abs(k1), 1e-9))
ptsB, dk_all, rd_all = np.concatenate(ptsB), np.concatenate(dk_all), np.concatenate(rd_all)
selB = (dk_all >= np.quantile(dk_all, 0.995)) & (rd_all < 20)
imgB = np.zeros(m.shape, bool); pb = np.rint(ptsB[selB]).astype(int); imgB[pb[:, 0], pb[:, 1]] = True
labB, nB = ndi.label(ndi.binary_dilation(imgB, iterations=4) & m, structure=np.ones((3, 3))); knotB = labB[ys, xs] > 0
keepB = np.arange(1, nB + 1)
# ---- agreement
cA = np.array([c.mean(0) for c in compsA]) if compsA else np.zeros((0, 2))
cB = np.array(ndi.center_of_mass(labB > 0, labB, keepB)) if nB else np.zeros((0, 2))
agree_BinA = float((cKDTree(cA).query(cB)[0] <= 10).mean()) if len(cA) and len(cB) else 0.0
agree_AinB = float((cKDTree(cB).query(cA)[0] <= 10).mean()) if len(cA) and len(cB) else 0.0
# ---- random control: every knot-A component moved to a random matter position >= 30 px from any knot
rng = np.random.default_rng(7); ctrl_img = np.zeros(m.shape, bool)
far = ndi.distance_transform_edt(~(knA_img | (labB > 0))) >= 30; cand = np.argwhere(far & m)
for c in compsA:                                            # same shape, random place; retried until >= 80 % lands on matter
    off = c - c.mean(0).astype(int)
    for _try in range(50):
        ctr = cand[rng.integers(len(cand))]; q = off + ctr
        ok = (q[:, 0] >= 0) & (q[:, 0] < N) & (q[:, 1] >= 0) & (q[:, 1] < N); q = q[ok]
        if len(q) and m[q[:, 0], q[:, 1]].mean() >= 0.8:
            break
    ctrl_img[q[:, 0], q[:, 1]] = True
ctrl = (ctrl_img & m)[ys, xs]
res = {"state": STATE, "hot_bonds": int(hot.sum()), "knotsA": int(len(keep)), "knotA_nodes": int(knotA.sum()), "knotsB": int(nB), "knotB_nodes": int(knotB.sum()),
       "ctrl_nodes": int(ctrl.sum()), "B_centres_within_10px_of_A": agree_BinA, "A_centres_within_10px_of_B": agree_AinB,
       "strain_normal_p99": float(np.percentile(strain[en >= np.cos(np.radians(30))], 99)), "curv_threshold_dk": float(np.quantile(dk_all, 0.995))}
print(json.dumps(res, indent=1)); json.dump(res, open(f"/out/knots_{STATE}.json", "w"), indent=1)
np.savez_compressed(f"/out/knots_{STATE}.npz", knotA=knotA, knotB=knotB, ctrl=ctrl)
# ---- figures
IMG = "/out/figures/noeuds_contrainte"; import os; os.makedirs(IMG, exist_ok=True)
yy, xx = np.nonzero(m); bb = (yy.min() - 20, yy.max() + 20, xx.min() - 20, xx.max() + 20)
fig, ax = plt.subplots(figsize=(18, 13)); ax.imshow(np.where(m, g, 0)[bb[0]:bb[1], bb[2]:bb[3]], cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
ay, axx = np.nonzero(knA_img); ax.plot(axx - bb[2], ay - bb[0], ".", color="red", ms=0.8, label=f"A mécanique ({len(keep)} nœuds)")
by_, bx_ = np.nonzero(labB > 0); ax.plot(bx_ - bb[2], by_ - bb[0], ".", color="cyan", ms=0.8, label=f"B courbure ({nB} nœuds)")
ax.legend(markerscale=15, fontsize=12, loc="lower left"); ax.axis("off")
ax.set_title(f"025 — nœuds de contrainte détectés automatiquement sur l'état {STATE} (rouge : concentration de tension normale (top 0,5 % et ≥ 3× l'entourage) ; cyan : forte variation de courbure)", fontsize=12)
fig.savefig(f"{IMG}/carte_noeuds_{STATE}.png", dpi=80, bbox_inches="tight"); plt.close(fig)
# zooms on the 6 largest knots A: original / deformed with hot bonds
from render3 import render, void_level
dm, dimg, _ = render(ys, xs, ux, uy, g, m, void_level(g, m, S["scroll"]), bi, bj, None)
order = np.argsort([-len(c) for c in compsA])[:6]
fig, ax = plt.subplots(3, 4, figsize=(24, 18))
for k, oi in enumerate(order):
    c = compsA[oi].mean(0).astype(int); y0, x0, s = c[0] - 50, c[1] - 50, 100
    sel = (ys >= y0) & (ys < y0 + s) & (xs >= x0) & (xs < x0 + s); dyy, dxx = int(np.rint(np.median(uy[sel]))), int(np.rint(np.median(ux[sel])))
    a0, a1 = ax[(2 * k) // 4, (2 * k) % 4], ax[(2 * k + 1) // 4, (2 * k + 1) % 4]
    a0.imshow(np.where(m, g, 0)[y0:y0 + s, x0:x0 + s], cmap="gray", vmin=0, vmax=255); a0.set_title(f"nœud A{k + 1} original ({c[0]}, {c[1]}), {len(compsA[oi])} px")
    a1.imshow(np.where(dm, dimg, 0)[y0 + dyy:y0 + dyy + s, x0 + dxx:x0 + dxx + s], cmap="gray", vmin=0, vmax=255)
    hb = hot & (ys[bi] >= y0) & (ys[bi] < y0 + s) & (xs[bi] >= x0) & (xs[bi] < x0 + s)
    for i, j in zip(bi[hb], bj[hb]):
        a1.plot([xs[i] + ux[i] - x0 - dxx, xs[j] + ux[j] - x0 - dxx], [ys[i] + uy[i] - y0 - dyy, ys[j] + uy[j] - y0 - dyy], "r-", lw=1.5)
    a1.set_title("centrifugé : liens en concentration de tension (rouge)")
    for a in (a0, a1): a.axis("off")
fig.savefig(f"{IMG}/zooms_noeuds_{STATE}.png", dpi=60, bbox_inches="tight"); plt.close(fig); print("ok")
