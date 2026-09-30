"""026 shared helpers: C1 lines of the 6 draws (025 outputs, read-only on /p25), draw-consensus keep masks (025 rule, frozen:
T1+J0 / 5 voters, TOL 3.5 px, unanimity, pieces < 8 points dropped), bands with connector flags, and the frozen VT evaluator
restricted to the zone windows (same code path as 025 h2_eval.py)."""
import glob, json
import numpy as np
from scipy.spatial import cKDTree
from vt_eval import evaluate_slice, summarise

TOL, MINP = 3.5, 8
VOTERS = ("T1", "T2", "J0", "J1", "S0")
SETS = {"dev": (range(4222, 4227), "/p25/risk_z{z}__{d}.npz", "/vt/DEV"),
        "h2": (range(4274, 4279), "/p25/h2lines_z{z}__{d}.npz", "/ann/HOLDOUT2"),
        "h3": (range(4174, 4179), "/out/h2lines_z{z}__{d}.npz", "/ann/HOLDOUT3")}


def load_ann(root):
    """one annotation per zone window, the "_VT_" file preferred (025 h2_eval loader)"""
    byw = {}
    for f in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
        a = json.load(open(f))
        if "polylines" not in a or "window" not in a:
            continue
        w = a["window"]; key = (w["y0"], w["x0"], w["size_px"])
        if key not in byw or ("_VT_" in f and "_VT_" not in byw[key][0]):
            byw[key] = (f, a)
    return [a for f, a in byw.values()]


def load_lines(pat, z):
    D = {d: np.load(pat.format(z=z, d=d)) for d in ("base",) + VOTERS}
    b = D["base"]; pts, lid, wid = b["pts"].astype(float), b["lid"], b["width"].astype(float)
    dist = {v: cKDTree(D[v]["pts"]).query(pts)[0] for v in VOTERS}
    k1 = (dist["T1"] <= TOL) & (dist["J0"] <= TOL)
    k2 = np.all([dist[v] <= TOL for v in VOTERS], axis=0)
    return {"pts": pts, "lid": lid, "wid": wid, "k1": k1, "k2": k2, "dist": dist, "voters": {v: D[v] for v in VOTERS}}


def runs_of(mask):
    """[(start, end_inclusive)] of True runs"""
    i = np.nonzero(mask)[0]
    if not len(i):
        return []
    br = np.nonzero(np.diff(i) > 1)[0]
    return list(zip(i[np.r_[0, br + 1]], i[np.r_[br, len(i) - 1]]))


def bands_of(pts, lid, wid, keep, conn=None):
    """025 rule: kept runs of each base line, runs < MINP dropped; conn (per point) flags connector (bridge) points"""
    res = []
    for k in np.unique(lid):
        idx = np.nonzero(lid == k)[0]
        for a, e in runs_of(keep[idx]):
            run = idx[a:e + 1]
            if len(run) >= MINP:
                res.append({"pts": pts[run], "w": wid[run], "conn": np.zeros(len(run), bool) if conn is None else conn[run]})
    return res


def evaluate(ann, z, bands, full=False):
    parts = []
    for a in ann:
        if not any(q["z"] == z for q in a["polylines"]):
            continue
        w = a["window"]; win = (w["y0"], w["x0"], w["size_px"])
        bs = [b for b in bands if ((b["pts"][:, 0] >= win[0] - 5) & (b["pts"][:, 0] < win[0] + win[2] + 5) & (b["pts"][:, 1] >= win[1] - 5) & (b["pts"][:, 1] < win[1] + win[2] + 5)).any()]
        parts.append(evaluate_slice(a, z, bs, win))
    s = summarise(parts)
    return s if full else (s["distinct_n"], s["S1"])


def gt_sheet(ann, z, pts, tol=4.0):
    """annotated SURE sheet id of each point (nearest sure trace point within tol, sheet ids made unique per zone), -1 else;
    used ONLY for diagnostics / oracles on annotated data, never by a method"""
    P, S = [], []
    for zi, a in enumerate(ann):
        for q in a["polylines"]:
            if q["z"] == z and q["confidence"] == "sur":
                from vt_eval import resample
                R = resample(q["points"]); P.append(R); S.append(np.full(len(R), zi * 10000 + q["sheet_id"]))
    if not P:
        return np.full(len(pts), -1)
    P, S = np.concatenate(P), np.concatenate(S)
    d, j = cKDTree(P).query(pts, distance_upper_bound=tol)
    return np.where(np.isfinite(d), S[np.minimum(j, len(S) - 1)], -1)
