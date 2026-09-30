"""019 (evaluator v2): evaluation of detected bands against the frozen ground truth (CRITERES_VALIDES_v1).

Per zone and slice, restricted to the zone window:
  - annotated SURE traces (confidence 'sur') resampled at 1 px; DOUBTFUL traces are "don't care"; undecidable polygons
    are excluded (band points inside them are ignored, annotated points inside them are already doubtful by rule);
  - every band point (1 px) is matched to the nearest annotated point within TOL (38 um = 4 px); a match to a doubtful
    trace or no match = neutral;
  - along each band, the sequence of matched SURE sheet ids is cut into runs; a run is ACCEPTED if it lasts >= RUN_MIN
    (one band width, 8.3 px) -- shorter runs are matching noise where two annotated sheets are closer than TOL;
  - S1 (correctly followed length): annotated sure point p of sheet s is followed if a band point within TOL belongs to an
    accepted run of sheet s; S1 = followed length / sure length;
  - S2 (sheet changes): transitions between consecutive accepted runs of different sheets along a band;
  - S3 (false joins): weld connectors of the algorithm whose accepted runs just before and just after are different sheets;
  - a change or a connector whose gap crosses an undecidable zone is not counted (zones excluded);
  - distinct error events: S2 transitions and S3 connectors merged when closer than MERGE (2 band widths) along the band;
  - S4 (overlaps): share of sure annotated length within half-width of >= 2 different bands;
  - v2 fixes: D1 annotated length measured geometrically (v1 counted points); D3 a change/connector whose gap crosses
    an unobserved stretch (undecidable zone OR outside the window) is not counted (v1: only zones);
  - v2 addition: correctly-followed UNINTERRUPTED paths (same sheet, gaps <= RUN_MIN, no unobserved stretch) -> N50 etc.;
  - all counts normalised per 10 mm of SURE annotated length of the zone (same denominator for every method).
"""
import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree

PX_UM = 9.596
TOL = 4.0
W80 = 80.0 / PX_UM
RUN_MIN = W80
MERGE = 2 * W80


def resample(P, step=1.0):
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    n = max(int(np.ceil(L[-1] / step)), 1)
    s = np.linspace(0, L[-1], n + 1)
    return np.c_[np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])]


def resample_w(P, w, step=1.0):
    P = np.asarray(P, float); w = np.asarray(w, float)
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    R = resample(P, step)
    s = np.linspace(0, L[-1], len(R))
    return R, np.interp(s, L, w)


def in_window(R, win):
    y0, x0, S = win
    return (R[:, 0] >= y0) & (R[:, 0] < y0 + S) & (R[:, 1] >= x0) & (R[:, 1] < x0 + S)


def evaluate_slice(ann, z, bands, win):
    """ann: VT json dict; z: slice; bands: list of dicts {pts (n,2), w (n,), conn (n, bool)}; win: (y0, x0, size)."""
    polys = [Path(np.array(a["points"])) for a in ann["ambiguous"] if a["z"] == z]
    sure_pts, sure_sid, dbt_pts, sure_w = [], [], [], []
    for q in ann["polylines"]:
        if q["z"] != z:
            continue
        R = resample(q["points"])
        if q["confidence"] == "sur":
            # v2 fix D1: each point weighs its share of the GEOMETRIC length (v1 counted points: +1 px per trace)
            seg = np.hypot(*np.diff(R, axis=0).T) if len(R) > 1 else np.zeros(0)
            wgt = np.r_[seg, 0] * 0.5 + np.r_[0, seg] * 0.5
            sure_pts.append(R); sure_sid.append(np.full(len(R), q["sheet_id"])); sure_w.append(wgt)
        else:
            dbt_pts.append(R)
    SP = np.concatenate(sure_pts) if sure_pts else np.zeros((0, 2)); SS = np.concatenate(sure_sid) if sure_sid else np.zeros(0, int)
    DP = np.concatenate(dbt_pts) if dbt_pts else np.zeros((0, 2))
    SW = np.concatenate(sure_w) if sure_w else np.zeros(0)
    paths = []                                                  # v2: correctly-followed uninterrupted paths (px)
    allP = np.concatenate([SP, DP]); allS = np.r_[SS, np.full(len(DP), -1)]          # -1 = doubtful (neutral)
    tree = cKDTree(allP) if len(allP) else None
    followed = np.zeros(len(SP), bool)
    cover_count = np.zeros(len(SP), int)
    s2_events, s3_events, distinct = 0, 0, 0
    event_log = []
    for bi, b in enumerate(bands):
        R, Wd = resample_w(b["pts"], b["w"])
        if "conn" in b:                                   # connector flags carried by ARC LENGTH (fix: not by index)
            _, cf = resample_w(b["pts"], np.asarray(b["conn"], float))
            C = cf > 0.5
        else:
            C = np.zeros(len(R), bool)
        inw = in_window(R, win)
        amb = np.zeros(len(R), bool)
        for pth in polys:
            amb |= pth.contains_points(R)
        keep = inw & ~amb
        blind = amb | ~inw                               # v2 fix D3: unobserved stretches (undecidable zone OR outside window)
        step = float(np.hypot(*np.diff(R, axis=0).T).mean()) if len(R) > 1 else 1.0
        if not keep.any() or tree is None:
            continue
        d, j = tree.query(R, distance_upper_bound=TOL)
        sid = np.where(np.isfinite(d) & keep, allS[np.minimum(j, len(allS) - 1)], -2)   # -2 = no match / excluded
        # coverage for S4 (band corridor = own half-width) on sure points
        if len(SP):
            T = cKDTree(R[keep])
            dd, jj = T.query(SP, distance_upper_bound=float(np.max(Wd)) / 2 + 1e-6)
            ok = np.isfinite(dd)
            ok[ok] = dd[ok] <= Wd[keep][jj[ok]] / 2
            cover_count += ok
        # runs of sure sheet ids along the band
        runs = []                                                   # (sheet, start, end)
        i = 0
        while i < len(sid):
            if sid[i] < 0:
                i += 1; continue
            k = i
            while k + 1 < len(sid) and sid[k + 1] == sid[i]:
                k += 1
            runs.append((int(sid[i]), i, k)); i = k + 1
        acc = [r for r in runs if r[2] - r[1] + 1 >= RUN_MIN]
        # merge consecutive accepted runs of the same sheet (separated by neutral / noise)
        merged = []
        for r in acc:
            if merged and merged[-1][0] == r[0]:
                merged[-1] = (r[0], merged[-1][1], r[2])
            else:
                merged.append(r)
        # v2: uninterrupted correctly-followed paths: consecutive accepted runs of the SAME sheet joined only if the gap is
        # <= RUN_MIN and crosses no unobserved stretch; a path ends at a change of sheet, a longer gap, a zone or the window
        cur = None
        for r in acc:
            if cur and r[0] == cur[0] and r[1] - cur[2] - 1 <= RUN_MIN and not blind[cur[2]:r[1] + 1].any():
                cur = (cur[0], cur[1], r[2])
            else:
                if cur: paths.append((cur[2] - cur[1]) * step)
                cur = r
        if cur: paths.append((cur[2] - cur[1]) * step)
        # S1: sure points of sheet s within TOL of accepted-run band points of sheet s
        for s, a, e in merged:
            seg = R[a:e + 1]
            m = SS == s
            if m.any():
                Ts = cKDTree(seg); dd, _ = Ts.query(SP[m], distance_upper_bound=TOL)
                idx = np.nonzero(m)[0][np.isfinite(dd)]
                followed[idx] = True
        # S2 / S3 / distinct events
        # a change whose gap crosses an undecidable zone is not counted (zones are excluded)
        trans = [(merged[t][2], merged[t + 1][1]) for t in range(len(merged) - 1)
                 if merged[t][0] != merged[t + 1][0] and not blind[merged[t][2]:merged[t + 1][1] + 1].any()]
        conn_idx = np.nonzero(C & keep)[0]
        cuts = np.split(conn_idx, np.nonzero(np.diff(conn_idx) > 1)[0] + 1) if len(conn_idx) else []
        s3_here = []
        for cc in cuts:
            if not len(cc):
                continue
            a, e = cc[0], cc[-1]
            # runs may extend a few px into the connector: the run that STARTS before it and the one that ENDS after it
            before = [r for r in merged if r[1] < a]; after = [r for r in merged if r[2] > e]
            if before and after and before[-1][0] != after[0][0] and not blind[min(before[-1][2], a):max(after[0][1], e) + 1].any():
                s3_here.append((a, e))
        s2_events += len(trans); s3_events += len(s3_here)
        ev = sorted([(a, e, "S2") for a, e in trans] + [(a, e, "S3") for a, e in s3_here])
        groups = []
        for a, e, k in ev:
            if groups and a - groups[-1][1] <= MERGE:
                groups[-1] = (groups[-1][0], max(groups[-1][1], e), groups[-1][2] | {k})
            else:
                groups.append((a, e, {k}))
        distinct += len(groups)
        for a, e, ks in groups:
            event_log.append({"band": bi, "y": float(R[a, 0]), "x": float(R[a, 1]), "kinds": sorted(ks)})
    return {"sure_len_px": float(SW.sum()), "followed_px": float(SW[followed].sum()), "overlap_px": float(SW[cover_count >= 2].sum()),
            "S2": s2_events, "S3": s3_events, "distinct": distinct, "events": event_log, "paths_px": [float(p) for p in paths]}


def summarise(parts):
    L = sum(p["sure_len_px"] for p in parts)
    per10 = lambda n: n / max(L * PX_UM / 1e4, 1e-12)
    return {"sure_len_mm": L * PX_UM / 1000, "S1": sum(p["followed_px"] for p in parts) / max(L, 1e-9),
            "S2_per10mm": per10(sum(p["S2"] for p in parts)), "S3_per10mm": per10(sum(p["S3"] for p in parts)),
            "distinct_per10mm": per10(sum(p["distinct"] for p in parts)), "S4": sum(p["overlap_px"] for p in parts) / max(L, 1e-9),
            "S2_n": sum(p["S2"] for p in parts), "S3_n": sum(p["S3"] for p in parts), "distinct_n": sum(p["distinct"] for p in parts),
            **paths_summary([x for p in parts for x in p.get("paths_px", [])])}


def paths_summary(P):
    """v2: correctly-followed uninterrupted paths (mm): count, total, median, N50 (half of the followed length lies in
    paths at least this long), max."""
    P = np.sort(np.array(P, float))[::-1] * PX_UM / 1000
    if not len(P):
        return {"paths_n": 0, "paths_total_mm": 0.0, "paths_median_mm": 0.0, "paths_N50_mm": 0.0, "paths_max_mm": 0.0}
    return {"paths_n": int(len(P)), "paths_total_mm": float(P.sum()), "paths_median_mm": float(np.median(P)),
            "paths_N50_mm": float(P[np.searchsorted(np.cumsum(P), P.sum() / 2)]), "paths_max_mm": float(P[0])}
