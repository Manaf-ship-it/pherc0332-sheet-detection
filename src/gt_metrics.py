"""020: ground-truth metrics of a mechanical state (PROTOCOL metrics_ground_truth).
M1 separation of the ANNOTATED sheets (VT_v1 DEV, one slice) and M2 detection after re-running C1 on the separated state.
A mechanical state = (ys, xs) original lattice pixels + (ux, uy) displacements (+ broken bonds for the raster)."""
import glob, json
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from vt_eval import resample, evaluate_slice, summarise

W80 = 80.0 / 9.596
ZONES = ["zone1_pli", "zone2_dechirure_bord", "zone3_centre_ecrase", "zone4_temoin_paralleles"]


def load_gt(z=4224, root="/vt/DEV"):
    out = {}
    for zn in ZONES:
        vt = json.load(open(glob.glob(f"{root}/{zn}/annotation_DEV_{zn}_VT_v1.json")[0]))
        tr = [(q["sheet_id"], resample(q["points"])) for q in vt["polylines"] if q["z"] == z and q["confidence"] == "sur"]
        out[zn] = {"vt": vt, "traces": tr}
    return out


def m1_prepare(gt, matter, ys, xs):
    """attach every annotated sure point to a lattice point (nearest matter pixel <= 3 px); build neighbour pairs."""
    idx = -np.ones(matter.shape, np.int64); idx[ys, xs] = np.arange(len(ys))
    dist, (iy, ix) = ndi.distance_transform_edt(~matter, return_indices=True)
    prep = {}
    for zn, g in gt.items():
        P = np.concatenate([t for _, t in g["traces"]]); S = np.concatenate([np.full(len(t), s) for s, t in g["traces"]])
        T = np.concatenate([np.full(len(t), k) for k, (_, t) in enumerate(g["traces"])])            # trace index
        yi = np.clip(np.rint(P[:, 0]).astype(int), 0, matter.shape[0] - 1); xi = np.clip(np.rint(P[:, 1]).astype(int), 0, matter.shape[1] - 1)
        lat = np.where(dist[yi, xi] <= 3, idx[iy[yi, xi], ix[yi, xi]], -1)
        tree = cKDTree(P); a_, b_ = [], []
        for i in range(0, len(P), 3):
            cand = np.array(tree.query_ball_point(P[i], 20.0), int); cand = cand[S[cand] != S[i]]
            if not len(cand) or lat[i] < 0:
                continue
            j = int(cand[np.argmin(np.hypot(*(P[cand] - P[i]).T))])
            if lat[j] < 0:
                continue
            # no other annotated trace between i and j
            n = max(int(np.ceil(np.hypot(*(P[j] - P[i])))) * 2, 2); seg = P[i][None] + np.linspace(0, 1, n)[:, None] * (P[j] - P[i])[None]
            dd, kk = tree.query(seg, distance_upper_bound=1.5)
            other = np.isfinite(dd) & ~np.isin(S[np.minimum(kk, len(S) - 1)], (S[i], S[j]))
            if not other.any():
                a_.append(i); b_.append(j)
        tang = np.zeros_like(P)
        for k in np.unique(T):
            m = T == k; g_ = np.gradient(P[m], axis=0) if m.sum() > 1 else np.zeros((m.sum(), 2))
            tang[m] = g_ / np.maximum(np.hypot(g_[:, 0], g_[:, 1]), 1e-9)[:, None]
        prep[zn] = {"P": P, "T": T, "lat": lat, "pairs": (np.array(a_, int), np.array(b_, int)), "tang": tang}
    return prep


def m1(prep, ux, uy):
    res, agg = {}, {"pairs": 0, "open": 0, "clean": 0, "sp": 0, "torn": 0, "slide": []}
    for zn, p in prep.items():
        P, T, lat = p["P"], p["T"], p["lat"]; ok = lat >= 0
        U = np.zeros_like(P); U[ok] = np.c_[uy[lat[ok]], ux[lat[ok]]]; Q = P + U
        a_, b_ = p["pairs"]
        op = np.hypot(*(Q[a_] - Q[b_]).T) - np.hypot(*(P[a_] - P[b_]).T)
        same = np.r_[T[1:] == T[:-1], False] & np.r_[ok[1:] & ok[:-1], False]
        sp = np.r_[np.hypot(*np.diff(Q, axis=0).T), 0]; torn = same & (sp > 3.0)
        tornW = ndi.maximum_filter1d(torn.astype(np.uint8), size=int(2 * W80) + 1) > 0
        clean = (op >= 2.0) & ~tornW[a_] & ~tornW[b_]
        dU = U[a_] - U[b_]; slide = np.abs((dU * p["tang"][a_]).sum(1))
        res[zn] = {"pairs": int(len(a_)), "M1_open": float((op >= 2).mean()) if len(op) else 0.0, "M1_clean": float(clean.mean()) if len(op) else 0.0,
                   "M1_tear": float(torn[same].mean()) if same.any() else 0.0, "M1_slide_px": float(np.median(slide)) if len(slide) else 0.0}
        agg["pairs"] += len(a_); agg["open"] += int((op >= 2).sum()); agg["clean"] += int(clean.sum()); agg["sp"] += int(same.sum()); agg["torn"] += int(torn.sum()); agg["slide"] += list(slide)
    res["ALL"] = {"pairs": agg["pairs"], "M1_open": agg["open"] / max(agg["pairs"], 1), "M1_clean": agg["clean"] / max(agg["pairs"], 1),
                  "M1_tear": agg["torn"] / max(agg["sp"], 1), "M1_slide_px": float(np.median(agg["slide"])) if agg["slide"] else 0.0}
    return res


def m2(gt, bands_deformed, ys, xs, ux, uy, z=4224):
    """bands found on the deformed state -> each band point mapped to the ORIGINAL position of the nearest deformed lattice
    point -> frozen evaluator v2 on the DEV zones of slice z."""
    tree = cKDTree(np.c_[ys + uy, xs + ux])
    mapped = []
    for b in bands_deformed:
        _, j = tree.query(b[:, :2]); mapped.append({"pts": np.c_[ys[j], xs[j]].astype(float), "w": b[:, 2], "conn": np.zeros(len(b), bool)})
    return evaluate_bands(gt, mapped, z)


def evaluate_bands(gt, bands, z=4224):
    res, parts = {}, []
    for zn, g in gt.items():
        w = g["vt"]["window"]; win = (w["y0"], w["x0"], w["size_px"])
        bs = [b for b in bands if ((b["pts"][:, 0] >= win[0] - 5) & (b["pts"][:, 0] < win[0] + win[2] + 5) & (b["pts"][:, 1] >= win[1] - 5) & (b["pts"][:, 1] < win[1] + win[2] + 5)).any()]
        r = evaluate_slice(g["vt"], z, bs, win); parts.append(r); res[zn] = summarise([r])
    res["ALL"] = summarise(parts)
    return res
