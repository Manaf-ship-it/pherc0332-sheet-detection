"""022 metrics v3.
M1 v3 (review of 021, reproduced: a 20 px slide far from the ends and a sheet crossing the other both scored 100 % clean
with v2's +/-12-point window): for a pair (i on sheet A, j = nearest point of sheet B), SIGNED point-to-curve distance
from i to the WHOLE trace of B, before (P) and after (Q) deformation, sign = side of B's local orientation.
  crossed  : the sign changed (A went through B) -> never an opening, reported as M1_crossed
  end      : the nearest point of B (before or after) is an END of B's trace -> not observable, excluded (M1_end_excluded)
  opening  : |s1| - |s0| for the other pairs; open if >= 2 px; clean = open, not crossed, no tear within +/- W on both traces
Tear and slide as before. Tight pairs (spacing < 1 band width) reported separately.
M2 v3: render3 (area-conserving, no fusion) -> C1 (frozen) -> map_back v3 (exact inverse of the raster) -> evaluator v2."""
import numpy as np
from scipy import ndimage as ndi
import bands019
from gt_metrics import evaluate_bands, W80
from predetect import predetect
from render3 import render, map_back


def _signed(X, C):
    """signed distance from X to polyline C, and whether the nearest point is an end of C."""
    A, B = C[:-1], C[1:]; AB = B - A; L2 = np.maximum((AB * AB).sum(1), 1e-12)
    t = np.clip(((X - A) * AB).sum(1) / L2, 0, 1); F = A + t[:, None] * AB; d = np.hypot(*(X - F).T); k = int(np.argmin(d))
    cr = AB[k, 1] * (X[0] - A[k, 0]) - AB[k, 0] * (X[1] - A[k, 1])          # (y, x) coords: cross of AB and AX
    end = (k == 0 and t[k] <= 0) or (k == len(A) - 1 and t[k] >= 1)
    return float(np.sign(cr) * d[k]) if cr != 0 else float(d[k]), bool(end)


def m1v3(prep, ux, uy, per_pair=False):
    res, agg = {}, {k: 0 for k in ("pairs", "obs", "open", "clean", "cross", "end", "sp", "torn", "tp", "tobs", "tclean")}; slides = []; pp = {}
    for zn, p in prep.items():
        P, T, lat = p["P"], p["T"], p["lat"]; ok = lat >= 0
        U = np.zeros_like(P); U[ok] = np.c_[uy[lat[ok]], ux[lat[ok]]]; Q = P + U
        a_, b_ = p["pairs"]; n = len(a_)
        op = np.zeros(n); crossed = np.zeros(n, bool); endx = np.zeros(n, bool)
        tr = {t: np.nonzero(T == t)[0] for t in np.unique(T)}
        for q, (i, j) in enumerate(zip(a_, b_)):
            idx = tr[T[j]]
            if len(idx) < 2:
                endx[q] = True; continue
            s0, e0 = _signed(P[i], P[idx]); s1, e1 = _signed(Q[i], Q[idx])
            endx[q] = e0 or e1; crossed[q] = (np.sign(s0) != np.sign(s1)) and not endx[q]; op[q] = abs(s1) - abs(s0)
        obs = ~endx
        same = np.r_[T[1:] == T[:-1], False] & np.r_[ok[1:] & ok[:-1], False]
        sp = np.r_[np.hypot(*np.diff(Q, axis=0).T), 0]; torn = same & (sp > 3.0)
        tornW = ndi.maximum_filter1d(torn.astype(np.uint8), size=int(2 * W80) + 1) > 0
        opn = obs & ~crossed & (op >= 2.0); clean = opn & ~tornW[a_] & ~tornW[b_]
        tight = np.hypot(*(P[a_] - P[b_]).T) < W80 if n else np.zeros(0, bool)
        sl = np.abs(((U[a_] - U[b_]) * p["tang"][a_]).sum(1)); slides += list(sl[obs])
        r = lambda v, m: float(v[m].mean()) if m.any() else 0.0
        res[zn] = {"pairs": n, "observable": int(obs.sum()), "M1_open": r(opn, obs), "M1_clean": r(clean, obs), "M1_crossed": r(crossed, obs),
                   "M1_end_excluded": int(endx.sum()), "M1_tear": float(torn[same].mean()) if same.any() else 0.0,
                   "M1_slide_px": float(np.median(sl[obs])) if obs.any() else 0.0, "tight_pairs": int((tight & obs).sum()),
                   "M1_clean_tight": r(clean, tight & obs), "M1_clean_other": r(clean, ~tight & obs)}
        if per_pair:
            pp[zn] = {"a": a_, "b": b_, "open_px": op, "crossed": crossed, "end": endx, "clean": clean, "tight": tight}
        for k_, v in (("pairs", n), ("obs", obs.sum()), ("open", opn.sum()), ("clean", clean.sum()), ("cross", crossed.sum()), ("end", endx.sum()),
                      ("sp", same.sum()), ("torn", torn.sum()), ("tp", (tight & obs).sum()), ("tclean", (clean & tight).sum())):
            agg[k_] += int(v)
    o = max(agg["obs"], 1)
    res["ALL"] = {"pairs": agg["pairs"], "observable": agg["obs"], "M1_open": agg["open"] / o, "M1_clean": agg["clean"] / o, "M1_crossed": agg["cross"] / o,
                  "M1_end_excluded": agg["end"], "M1_tear": agg["torn"] / max(agg["sp"], 1), "M1_slide_px": float(np.median(slides)) if slides else 0.0,
                  "tight_pairs": agg["tp"], "M1_clean_tight": agg["tclean"] / max(agg["tp"], 1), "M1_clean_other": (agg["clean"] - agg["tclean"]) / max(o - agg["tp"], 1)}
    return (res, pp) if per_pair else res


def detect_v3(ys, xs, ux, uy, gray, matter, vlevel, bi=None, bj=None, broken=None):
    dm, dimg, R = render(ys, xs, ux, uy, gray, matter, vlevel, bi, bj, broken)
    tx, ty, coh = bands019.structure_tangent(dimg)
    bd = predetect(dm, tx, ty, coh)
    return bd, map_back(bd, R, ys, xs), dm, dimg, R


def m2v3(gt, ys, xs, ux, uy, gray, matter, vlevel, bi=None, bj=None, broken=None, z=4224):
    bd, mapped, dm, dimg, R = detect_v3(ys, xs, ux, uy, gray, matter, vlevel, bi, bj, broken)
    return evaluate_bands(gt, mapped, z), mapped, dm, dimg, R
