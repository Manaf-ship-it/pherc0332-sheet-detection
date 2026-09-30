"""029 step 4 evaluation: for each centrifuge state on the WELDED sample (w4_base, w4_necks, w4_ctrl) and the undeformed welded
sample ('w4_orig'): (1) matter GROUPS = connected components of the lattice through INTACT bonds (broken bonds removed);
(2) deformed slice coloured by group (as step 1), DEV zones; (3) line detection: frozen C1 on the deformed state mapped back
to the original coordinates, draws base / T1 / J0, errors / S1 / N50 and the draw consensus, frozen VT evaluator (DEV).
Reference: C1 on the ORIGINAL (unwelded) slice (025 risk lines). usage: etape4_eval.py"""
import glob, json
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import colorsys
import fsi_kernel_002 as k2
from metrics3 import detect_v3
from render3 import render, void_level
from slice_prep import prep
from vt_eval import resample
from lib026 import load_ann, bands_of, evaluate, TOL
Z = 4224; ann = load_ann("/vt/DEV"); IMG = "/out/figures/etape4_centrifugeuse_cols"
import os; os.makedirs(IMG, exist_ok=True)
S = prep(Z); g = S["gray"]; m = S["matter"] | np.load(f"/out/welded_z{Z}.npz")["weld"]; N = m.shape[0]; ys, xs = np.nonzero(m); P = len(ys)
_, _, bi, bj, _, _, _, _ = k2.build_bonds(m, S["tx"], S["ty"], 1.0, 0.1, 6.0, 0.1); VL = void_level(g, m, S["scroll"])
DR = {"base": (0, 0, 0), "T1": (0, 1, 0), "J0": (300, 0, 0)}; out = {}


def score(L):
    r = {}
    for d in L:
        s = evaluate(ann, Z, bands_of(L[d]["pts"], L[d]["lid"], L[d]["wid"], np.ones(len(L[d]["pts"]), bool)), True); r[d] = [s["distinct_n"], round(s["S1"], 4), round(s["paths_N50_mm"], 3)]
    p = L["base"]["pts"]; k = np.all([cKDTree(L[d]["pts"]).query(p)[0] <= TOL for d in ("T1", "J0")], axis=0)
    s = evaluate(ann, Z, bands_of(p, L["base"]["lid"], L["base"]["wid"], k), True)
    r["mean3"] = [round(float(np.mean([r[d][i] for d in DR])), 3) for i in range(3)]; r["CONS"] = [s["distinct_n"], round(s["S1"], 4), round(s["paths_N50_mm"], 3)]
    return r


# reference: original unwelded slice (025 risk lines)
out["original"] = score({d: {k: (np.load(f"/p25/risk_z{Z}__{d}.npz")[k2_].astype(float) if k2_ != "lid" else np.load(f"/p25/risk_z{Z}__{d}.npz")[k2_]) for k, k2_ in (("pts", "pts"), ("lid", "lid"), ("wid", "width"))} for d in DR})
print("original", out["original"], flush=True)
fig, ax = plt.subplots(4, 4, figsize=(26, 27))
for row, st in enumerate(("w4_orig", "w4_base", "w4_necks", "w4_ctrl")):
    if st == "w4_orig":
        ux, uy, brk = np.zeros(P), np.zeros(P), np.zeros(len(bi), bool)
    else:
        d = np.load(f"/out/cf_{st}/final.npz"); ux, uy, brk = d["ux"].astype(float), d["uy"].astype(float), d["broken"].astype(bool)[:len(bi)]
    ok = ~brk; A = coo_matrix((np.ones(ok.sum()), (bi[ok], bj[ok])), shape=(P, P))
    ncomp, comp = connected_components(A, directed=False); csz = np.bincount(comp)
    dm, dimg, R = render(ys, xs, ux, uy, g, m, VL, bi, bj, brk)
    own = R["owner"]; lab = np.where(own >= 0, comp[np.maximum(own, 0)] + 1, 0)
    order = np.argsort(csz)[::-1]; rng = np.random.default_rng(1); cols = np.zeros((ncomp + 1, 3))
    for r_, k in enumerate(order):
        cols[k + 1] = colorsys.hsv_to_rgb((r_ * 0.618034) % 1.0 if r_ < 200 else rng.random(), 0.55 + 0.45 * rng.random(), 0.75 + 0.25 * rng.random())
    shade = np.clip(dimg.astype(float) / 255 * 1.4, 0.35, 1.0)[..., None]; rgb = cols[lab] * shade; rgb[~dm] = 0
    for j, f in enumerate(sorted(glob.glob("/vt/DEV/*/zone.json"))):
        w = json.load(open(f)); y0, x0, s = w["y0"], w["x0"], w["size_px"]; sel = (ys >= y0) & (ys < y0 + s) & (xs >= x0) & (xs < x0 + s)
        dy, dx = int(np.rint(np.median(uy[sel]))), int(np.rint(np.median(ux[sel])))
        ax[row, j].imshow(np.clip(rgb[max(y0 + dy, 0):y0 + dy + s, max(x0 + dx, 0):x0 + dx + s], 0, 1)); ax[row, j].axis("off")
        ax[row, j].set_title(f"{st} — {w['zone']}", fontsize=11)
    L = {}
    for dn, (seed, ty_, tx_) in DR.items():
        a, b = ux + tx_, uy + ty_
        if dn[0] == "J":
            r_ = np.random.default_rng(seed); a = a + r_.uniform(-0.6, 0.6, P); b = b + r_.uniform(-0.6, 0.6, P)
        _, mapped, _, _, _ = detect_v3(ys, xs, a, b, g, m, VL, bi, bj, brk)
        Ls = [resample(q["pts"][:, :2]) if len(q["pts"]) > 1 else q["pts"][:, :2] for q in mapped]
        Ws = [np.interp(np.linspace(0, len(q["w"]) - 1, len(l)), np.arange(len(q["w"])), q["w"]) for q, l in zip(mapped, Ls)]
        L[dn] = {"pts": np.concatenate(Ls).astype(float), "lid": np.concatenate([np.full(len(l), k) for k, l in enumerate(Ls)]), "wid": np.concatenate(Ws).astype(float)}
    out[st] = {**score(L), "groups": int(ncomp), "largest_group_share": float(csz.max() / P), "groups_ge_1000px": int((csz >= 1000).sum()), "broken": int(brk.sum())}
    print(st, out[st], flush=True); json.dump(out, open("/out/etape4_eval.json", "w"), indent=1)
fig.suptitle("029 étape 4 — Z4224 : groupes de matière restés reliés (une couleur par groupe) : soudé sans centrifugeuse / centrifugé sans rupture / cols ≤ 4 px cassés à 2 % / témoin", fontsize=15)
fig.tight_layout(); fig.savefig(f"{IMG}/groupes_apres_centrifugeuse_zones_DEV.png", dpi=60); plt.close(fig)
