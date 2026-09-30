"""019: copy of the archived 013 bands013.py (sha ed667457...) with ONE added parameter JUMP_MIN_BITS (default 1 =
identical behaviour). Variants override module parameters; the reference V itself is always run from /ref.

013: copy of 011 bands011.py with the two review defects fixed:
 (1) resample() kept the start but dropped the end (arange excluded the last sample): every call shortened a curve by
     up to 1 px (10 -> 9 -> 8 px on a test segment). Now both ends are kept, uniform spacing <= step, and lengths are
     measured geometrically (curve_length);
 (2) the connector clearance (weld_ends and link_graph) looked only at the 4 nearest centreline points and then
     discarded those of the two joined bands, so a third band could be missed. Now EVERY centreline point within
     w_join/2 + w_other/2 (+ 0 px) of any connector point is checked, with each band's own (per-point) width.
 (3) link_graph keeps the width of every portion (per-point width arrays) instead of the chain minimum.

011: copy of 010 bands.py (v6) + set_width (idea 2), density/raw overrides for 3D evidence (idea 1), initial
forbidden zone for a fill pass, and a graph linking of band pieces into sheets (idea 3). The 010 metrics are imported
unchanged from bands.py for comparability.

010: 80 um sheet bands on one 2D raster (original slice or deformed raster), and the shared metrics.

A band = centreline (1 px steps) + corridor of half-width W/2. Tracing = pure pursuit toward the sheet mid-line
(ridge of the in-matter distance transform, inside the matter run containing the centreline), heading change clamped
to STEP / R_MIN per step (curvature radius >= R_MIN by construction), no corridor overlap, straight void crossing
<= BRIDGE with tangent re-entry check, then a welding pass of facing band ends. See PROTOCOL.json.
"""
import math

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

PIXEL_UM = 9.596
W = 80.0 / PIXEL_UM
R_MIN = 150.0 / PIXEL_UM
CONTACT_R = W + 1.0
STEP = 1.0
DTH = STEP / (1.05 * R_MIN)       # v2: numerical margin
JUMP_MIN_BITS = 1                 # 019: minimum overlapping corridor samples between consecutive steps (1 = 013 rule)
SIG_B, MEM_T = 1.5, 0.3           # v2: bundle-scale sheet membership (closes 1-2 px internal gaps)
LOOK = 4.0
PROF = W / 2 + 1.0
BRIDGE = 2 * W                    # max tear bridged by the post-pass weld
TRACE_BRIDGE = W                  # v6: short void crossing while tracing (protected by the JUMP rule + tangent re-entry)
COAST = 2.0
COS20 = math.cos(math.radians(20))
MIN_LEN = 3 * W
SEED_STRIDE = 4
MAX_STEPS = 40000
SELF_LAG = int(math.ceil(3 * CONTACT_R / STEP))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Tracer:
    def __init__(self, matter, tx, ty, dens=None, raw=None, forbid0=None):
        self.m = matter.astype(bool)
        self.raw = self.m.copy() if raw is None else raw.astype(bool)          # 011: raw matter for the JUMP rule
        self.H, self.Wd = matter.shape
        self.dt = ndi.gaussian_filter(self.m.astype(np.float64), SIG_B) if dens is None else dens.astype(np.float64)
        self.m = self.dt > MEM_T                                                 # v2: bundle membership
        self.c2 = (tx.astype(np.float64) ** 2 - ty.astype(np.float64) ** 2)
        self.s2 = 2.0 * tx.astype(np.float64) * ty.astype(np.float64)
        r = int(math.ceil(CONTACT_R))
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        self.r = r
        self.disk = (dy ** 2 + dx ** 2) <= CONTACT_R ** 2
        self.forbid = np.zeros((self.H, self.Wd), np.int32) if forbid0 is None else forbid0.astype(np.int32).copy()
        self.own = np.zeros((self.H, self.Wd), bool)
        self.soff = np.arange(-PROF, PROF + 1e-9, 0.5)
        self.coff = [float(v) for v in np.arange(-W / 2, W / 2 + 1e-9, 0.5)]

    # ---------------------------------------------------------------- field access
    def bil(self, a, y, x):
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
                + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)

    def inside(self, y, x, pad=2):
        return pad <= y < self.H - pad - 1 and pad <= x < self.Wd - pad - 1

    def tangent(self, y, x):
        th = 0.5 * math.atan2(self.bil(self.s2, y, x), self.bil(self.c2, y, x))
        return math.cos(th), math.sin(th)

    def on(self, y, x):
        return bool(self.m[int(round(y)), int(round(x))])

    def ridge(self, y, x, nx, ny):
        """Offset along n of the distance-transform maximum inside the matter run containing the point (None if the
        point is in void)."""
        if not self.inside(y, x) or not self.m[int(round(y)), int(round(x))]:
            return None
        best, bs = -1.0, None
        # walk outward from 0 in both directions while in matter
        for sgn in (1.0, -1.0):
            for s in self.soff[self.soff >= 0] if sgn > 0 else -self.soff[self.soff > 0]:
                yy, xx = y + s * ny, x + s * nx
                if not self.inside(yy, xx) or not self.m[int(round(yy)), int(round(xx))]:
                    break
                v = self.bil(self.dt, yy, xx)
                if v > best:
                    best, bs = v, s
        return bs

    def corridor_profile(self, y, x, th):
        """Raw-matter samples across the corridor (|s| <= W/2, 0.5 px) as a bit mask; None if no matter."""
        nx, ny = -math.sin(th), math.cos(th)
        bits = 0
        for k, s in enumerate(self.coff):
            yy, xx = int(round(y + s * ny)), int(round(x + s * nx))
            if self.raw[yy, xx]:
                bits |= 1 << k
        return bits or None

    # ---------------------------------------------------------------- painting
    def paint(self, arr, y, x, val):
        r, H, Wd = self.r, self.H, self.Wd
        yi, xi = int(round(y)), int(round(x))
        y0, y1, x0, x1 = yi - r, yi + r + 1, xi - r, xi + r + 1
        cy0, cx0, cy1, cx1 = max(y0, 0), max(x0, 0), min(y1, H), min(x1, Wd)
        d = self.disk[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        sub = arr[cy0:cy1, cx0:cx1]
        if arr.dtype == bool:
            sub[d] = val
        else:
            sub[d & (sub == 0)] = val

    # ---------------------------------------------------------------- one direction
    def trace(self, y, x, th, prev=()):
        pts, painted = [], []
        void_run, welds, reason = 0.0, 0, "MAX_STEPS"
        last = None                                          # v4: last corridor matter profile (no-jump rule)
        hist = list(prev)
        for _ in range(MAX_STEPS):
            hx, hy = math.cos(th), math.sin(th)
            if not self.inside(y + LOOK * hy, x + LOOK * hx, pad=int(PROF) + 2):
                reason = "FRAME_EDGE"; break
            if void_run == 0.0:
                tx, ty = self.tangent(y, x)
                if tx * hx + ty * hy < 0:
                    tx, ty = -tx, -ty
                want = math.atan2(ty, tx)
                # pure pursuit toward the ridge a look-ahead distance ahead
                ly, lx = y + LOOK * math.sin(want), x + LOOK * math.cos(want)
                s = self.ridge(ly, lx, -math.sin(want), math.cos(want))
                if s is not None:
                    want = want + math.atan2(s, LOOK)
                th = th + max(-DTH, min(DTH, wrap(want - th)))
            ny_, nx_ = y + STEP * math.sin(th), x + STEP * math.cos(th)
            yi, xi = int(round(ny_)), int(round(nx_))
            if self.forbid[yi, xi]:
                reason = "CONTACT"; break
            if self.own[yi, xi]:
                reason = "SELF_CONTACT"; break
            prof = self.corridor_profile(ny_, nx_, th)
            if prof is not None:
                if last is not None and bin(prof & (last | (last << 1) | (last >> 1))).count("1") < JUMP_MIN_BITS:
                    reason = "JUMP"; break
                last = prof
            if self.m[yi, xi]:
                if void_run > COAST:          # re-entry after a tear: the sheet must continue in the same direction
                    tx, ty = self.tangent(ny_, nx_)
                    if abs(tx * math.cos(th) + ty * math.sin(th)) < COS20:
                        reason = "WELD_REJECTED"; break
                    welds += 1
                void_run = 0.0
            else:
                void_run += STEP
                if void_run > TRACE_BRIDGE:
                    reason = "LEFT_MATTER"; break
            y, x = ny_, nx_
            pts.append((y, x)); hist.append(th)
            if len(pts) > SELF_LAG:
                py, px = pts[-SELF_LAG]
                self.paint(self.own, py, px, True); painted.append((py, px))
        while pts and not self.on(*pts[-1]):
            pts.pop()
        return pts, reason, painted, welds

    # ---------------------------------------------------------------- all bands
    def run(self, seed_score, log=None):
        sk = skeletonize(self.m)                     # v2: skeleton of the bundle membership
        sy, sx = np.nonzero(sk)
        order = np.argsort(-seed_score[sy, sx], kind="stable")[::SEED_STRIDE]
        bands, reasons, nwelds = [], {}, 0
        for i in order:
            y, x = float(sy[i]), float(sx[i])
            if self.forbid[int(y), int(x)] or not self.inside(y, x, pad=int(PROF) + 6):
                continue
            tx, ty = self.tangent(y, x)
            th = math.atan2(ty, tx)
            pa, ra, pda, wa = self.trace(y, x, th)
            for p in pda:
                self.paint(self.own, *p, False)
            for p in pa[SELF_LAG:]:
                self.paint(self.own, *p, True)
            pb, rb, pdb, wb = self.trace(y, x, th + math.pi)
            for p in pa + pda + pdb:
                self.paint(self.own, *p, False)
            line = pb[::-1] + [(y, x)] + pa
            if len(line) * STEP < MIN_LEN:
                continue
            bid = len(bands) + 1
            for p in line:
                self.paint(self.forbid, *p, bid)
            bands.append(np.array(line, np.float64))
            nwelds += wa + wb
            for rr in (ra, rb):
                reasons[rr] = reasons.get(rr, 0) + 1
            if log and len(bands) % 500 == 0:
                log(f"{len(bands)} bands")
        return bands, {"stop_reasons": reasons, "welds_during_tracing": nwelds}


def end_dir(b, head):
    """Outward unit direction at an end, over an arc of W."""
    n = min(int(round(W)), len(b) - 1)
    p, q = (b[0], b[n]) if head else (b[-1], b[-1 - n])
    d = p - q
    return d / max(np.hypot(*d), 1e-9)


TRIM = int(round(W))
COS30 = math.cos(math.radians(30))


def hermite(p, dp, q, dq):
    """C1 connector from p (outward direction dp) to q (outward direction dq), 1 px spacing, end points excluded."""
    L = float(np.hypot(*(q - p)))
    n = max(int(math.ceil(L * 1.3)), 2)
    t = np.linspace(0, 1, n + 1)[:, None]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    c = h00 * p + h10 * L * dp + h01 * q + h11 * L * (-dq)
    return resample(c)[1:-1]


def min_radius(c):
    c = resample(c)
    if len(c) < int(round(W)) + 3:
        return np.inf
    g = np.gradient(c, axis=0)
    th = np.unwrap(np.arctan2(g[:, 0], g[:, 1]))
    win = int(round(W))
    return float(win / max(np.abs(th[win:] - th[:-win]).max(), 1e-9))


def weld_ends(bands, member, raw=None):
    out, nm, _ = weld_ends_w(bands, member, raw)
    return out, nm


def weld_ends_w(bands, member, raw=None, widths=None):
    """v3 post-pass: join facing band ends of the same sheet. Both ends trimmed by W, C1 Hermite connector; accepted if
    the ends face each other (30 deg), the lateral offset is <= W/2, the joined curve keeps R >= R_MIN over +/- 3 W
    around the connector, and the connector has no void run longer than BRIDGE (bundle membership). Greedy by distance."""
    from scipy.spatial import cKDTree
    maxd = CONTACT_R + BRIDGE
    ends = []
    for i, b in enumerate(bands):
        ends.append((i, True, b[0], end_dir(b, True)))
        ends.append((i, False, b[-1], end_dir(b, False)))
    P = np.array([e[2] for e in ends]); D = np.array([e[3] for e in ends])
    H, Wd = member.shape
    PTS_ALL = np.concatenate(bands); TREE = cKDTree(PTS_ALL)
    if widths is None:
        widths = [np.full(len(b), W) for b in bands]
    OWN = np.concatenate([np.full(len(b), i) for i, b in enumerate(bands)]); WPT_ALL = np.concatenate(widths)
    cand = []
    for a in range(len(ends)):
        d = P - P[a]; dist = np.hypot(d[:, 0], d[:, 1])
        for b in np.nonzero((dist <= maxd) & (dist > 0))[0]:
            if b <= a or ends[a][0] == ends[b][0]:
                continue
            if -(D[a] @ D[b]) < COS30 or (d[b] @ D[a]) <= 0 or (-d[b] @ D[b]) <= 0:
                continue
            lat = abs(d[b][0] * D[a][1] - d[b][1] * D[a][0])
            if lat > W:                                              # v5: up to one band width
                continue
            ba, bb = bands[ends[a][0]], bands[ends[b][0]]
            if len(ba) <= 3 * TRIM or len(bb) <= 3 * TRIM:
                continue
            sa = ba[TRIM:] if ends[a][1] else ba[:-TRIM][::-1]        # oriented so that sa[0] is the trimmed end
            sb = bb[TRIM:] if ends[b][1] else bb[:-TRIM][::-1]
            pa_, pb_ = sa[0], sb[0]
            da = sa[0] - sa[min(TRIM, len(sa) - 1)]; da = da / np.hypot(*da)
            db = sb[0] - sb[min(TRIM, len(sb) - 1)]; db = db / np.hypot(*db)
            con = hermite(pa_, da, pb_, db)
            k = 3 * int(round(W))
            joined = np.concatenate([sa[:k][::-1], con, sb[:k]])
            if min_radius(joined) < R_MIN:
                continue
            yi = np.clip(np.rint(con[:, 0]).astype(int), 0, H - 1); xi = np.clip(np.rint(con[:, 1]).astype(int), 0, Wd - 1)
            vo = ~member[yi, xi]
            run, worst = 0, 0
            for v in vo:
                run = run + 1 if v else 0; worst = max(worst, run)
            if worst > (BRIDGE if lat <= W / 2 else COAST):         # v5: offset welds may not cross void
                continue
            # 013 fix (2): full clearance test (all points within w/2 + w_other/2, own widths)
            wj = min(widths[ends[a][0]][0 if ends[a][1] else -1], widths[ends[b][0]][0 if ends[b][1] else -1])
            if not clear_of_others(con, wj, TREE, PTS_ALL, OWN, WPT_ALL, (ends[a][0], ends[b][0])):
                continue
            cand.append((float(dist[b]) + lat, a, b))
    cand.sort()
    used, merges = set(), []
    for dist, a, b in cand:
        if a in used or b in used:
            continue
        used.add(a); used.add(b); merges.append((a, b, dist))
    link = {}
    for a, b, _ in merges:
        link[a] = b; link[b] = a
    out, outw, seen = [], [], set()

    def walk(s, enter_head):
        chain, wch, cur, fwd = [], [], s, enter_head
        while cur not in seen:
            seen.add(cur)
            b = bands[cur] if fwd else bands[cur][::-1]
            wb = widths[cur] if fwd else widths[cur][::-1]
            if chain:                                        # trim both welded ends by TRIM, C1 connector
                prev = chain[-1][:-TRIM]; wprev = wch[-1][:-TRIM]
                b = b[TRIM:]; wb = wb[TRIM:]
                dp = prev[-1] - prev[-1 - TRIM]; dp = dp / np.hypot(*dp)
                dq = b[0] - b[TRIM]; dq = dq / np.hypot(*dq)
                chain[-1] = prev; wch[-1] = wprev
                con = hermite(prev[-1], dp, b[0], dq)
                chain.append(con); wch.append(np.full(len(con), min(wprev[-1], wb[0])))
            chain.append(b); wch.append(wb)
            e = 2 * cur + (1 if fwd else 0)                  # exit end
            if e not in link:
                break
            o = link[e]; cur = o // 2; fwd = (o % 2 == 0)     # enter next band through its linked end
        c_, w_ = resample_w(np.concatenate(chain), np.concatenate(wch))
        outw.append(w_)
        return c_
    for i in range(len(bands)):                               # chains with a free end first
        if i in seen:
            continue
        if 2 * i not in link:
            out.append(walk(i, True))
        elif 2 * i + 1 not in link:
            out.append(walk(i, False))
    for i in range(len(bands)):                               # remaining = closed cycles, cut anywhere
        if i not in seen:
            out.append(walk(i, True))
    return out, len(merges), outw


def _connector(p, q):
    n = int(math.ceil(np.hypot(*(q - p)) / STEP))
    t = np.linspace(0, 1, n + 1)[1:-1]
    return p[None, :] + t[:, None] * (q - p)[None, :] if len(t) else np.zeros((0, 2))


def resample(c, step=1.0):
    """013 fix: both end points kept, n = ceil(L / step) segments of equal length (<= step)."""
    c = np.asarray(c, np.float64)
    if len(c) < 2:
        return c
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    if d[-1] <= 0:
        return c[:1]
    n = max(int(math.ceil(d[-1] / step - 1e-9)), 1)
    s = np.linspace(0.0, d[-1], n + 1)
    return np.c_[np.interp(s, d, c[:, 0]), np.interp(s, d, c[:, 1])]


def resample_w(c, w, step=1.0):
    """resample a curve and its per-point width array together."""
    c = np.asarray(c, np.float64); w = np.asarray(w, np.float64)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    r = resample(c, step)
    if len(r) < 2 or d[-1] <= 0:
        return r, w[:len(r)]
    s = np.linspace(0.0, d[-1], len(r))
    return r, np.interp(s, d, w)


def curve_length(c):
    c = np.asarray(c, np.float64)
    return float(np.hypot(*np.diff(c, axis=0).T).sum()) if len(c) > 1 else 0.0


def clear_of_others(con, w_join, TREE, PTS, OWN, WPT, exclude):
    """013 fix (2): True if no centreline point of another band lies within w_join/2 + w_point/2 of any connector
    point (all points in the radius are examined, each with its own width)."""
    if not len(con):
        return True
    rmax = w_join / 2 + float(WPT.max()) / 2
    for q, pts in zip(con, TREE.query_ball_point(con, rmax)):
        if not pts:
            continue
        pts = np.asarray(pts)
        pts = pts[~np.isin(OWN[pts], exclude)]
        if not len(pts):
            continue
        dist = np.hypot(PTS[pts, 0] - q[0], PTS[pts, 1] - q[1])
        if (dist < w_join / 2 + WPT[pts] / 2).any():
            return False
    return True


def corridors(bands, shape):
    cl = np.zeros(shape, np.int32)
    for i, b in enumerate(bands, 1):
        yi = np.clip(np.rint(b[:, 0]).astype(int), 0, shape[0] - 1); xi = np.clip(np.rint(b[:, 1]).astype(int), 0, shape[1] - 1)
        cl[yi, xi] = i
    dist, (iy, ix) = ndi.distance_transform_edt(cl == 0, return_indices=True)
    return np.where(dist <= W / 2, cl[iy, ix], 0).astype(np.int32), cl


def metrics(curves, matter, with_corridor=True, sk=None):
    """with_corridor=False: 009 lines, a VIRTUAL 80 um corridor is still used for skeleton_in_corridor."""
    """Same metric code for 009 lines and 010 bands (PROTOCOL metrics_same_code...)."""
    H, Wd = matter.shape
    if sk is None:
        sk = skeletonize(matter)
    cs = [resample(np.asarray(c, np.float64)) for c in curves if len(c) >= 3]
    cs = [c for c in cs if len(c) >= 5]
    raster = np.zeros(matter.shape, bool)
    on_all, jumps, bridges, blen, rad_len, total = [], 0, 0, 0.0, [], 0.0
    has_all, bjumps = [], 0
    win = int(round(W))
    soff = np.arange(-PROF - 2, PROF + 2 + 1e-9, 0.5)
    for c in cs:
        yi = np.clip(np.rint(c[:, 0]).astype(int), 0, H - 1); xi = np.clip(np.rint(c[:, 1]).astype(int), 0, Wd - 1)
        raster[yi, xi] = True
        on = matter[yi, xi]; on_all.append(on)
        total += curve_length(c)                       # 013: geometric length
        # bridges: void runs > COAST along the centreline
        lab, n = ndi.label(~on)
        if n:
            sz = np.bincount(lab.ravel())[1:]
            bridges += int((sz > COAST).sum()); blen += float(sz[sz > COAST].sum())
        # local direction and normal
        g = np.gradient(c, axis=0); g /= np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-9)[:, None]
        ny, nx = g[:, 1], -g[:, 0]
        Y = c[:, 0][:, None] + soff[None, :] * ny[:, None]; X = c[:, 1][:, None] + soff[None, :] * nx[:, None]
        prof = matter[np.clip(np.rint(Y).astype(int), 0, H - 1), np.clip(np.rint(X).astype(int), 0, Wd - 1)]
        a = np.full(len(c), np.nan); b = np.full(len(c), np.nan)
        z = len(soff) // 2
        for i in range(len(c)):
            pr = prof[i]
            cand = np.nonzero(pr & (np.abs(soff) <= W / 2))[0]
            if not len(cand):
                continue
            j = cand[np.argmin(np.abs(soff[cand]))]
            lo = j
            while lo > 0 and pr[lo - 1]:
                lo -= 1
            hi = j
            while hi < len(soff) - 1 and pr[hi + 1]:
                hi += 1
            a[i], b[i] = soff[lo], soff[hi]
        # v2 band scale: matter runs inside the corridor; jump = consecutive points (with matter) whose run sets do not
        # overlap at all (tolerance 1 sample), also across void runs (bridge re-entry on another sheet counts)
        inc = np.abs(soff) <= W / 2
        pc = prof & inc[None, :]
        has = pc.any(1); has_all.append(has)
        idx = np.nonzero(has)[0]
        if len(idx) > 1:
            p1 = pc[idx[:-1]]; p2 = pc[idx[1:]]
            p1d = p1 | np.roll(p1, 1, axis=1) | np.roll(p1, -1, axis=1)
            bjumps += int((~(p1d & p2).any(1)).sum())
        ok = np.isfinite(a[1:]) & np.isfinite(a[:-1])
        jumps += int((ok & ((np.maximum(a[1:], a[:-1]) > np.minimum(b[1:], b[:-1]) + 0.5))).sum())
        # curvature over an arc of W
        if len(c) > win + 1:
            th = np.unwrap(np.arctan2(g[:, 0], g[:, 1]))
            dth = np.abs(th[win:] - th[:-win])
            rad_len.append(win / np.maximum(dth, 1e-9))
    on_all = np.concatenate(on_all) if on_all else np.zeros(0, bool)
    rr = np.concatenate(rad_len) if rad_len else np.zeros(0)
    lens = np.array([curve_length(c) for c in cs]) * PIXEL_UM / 1000
    out = {"n": len(cs), "total_mm": float(total * PIXEL_UM / 1000),
           "len_median_mm": float(np.median(lens)) if len(lens) else 0.0,
           "len_p90_mm": float(np.percentile(lens, 90)) if len(lens) else 0.0,
           "centreline_on_matter": float(on_all.mean()) if len(on_all) else 0.0,
           "skeleton_coverage": float(ndi.binary_dilation(raster, iterations=2)[sk].mean()),
           "band_jumps": bjumps, "band_jumps_per_10mm": float(bjumps / max(total * PIXEL_UM / 1e4, 1e-9)),
           "corridor_has_matter": float(np.concatenate(has_all).mean()) if has_all else 0.0,
           "lateral_jumps": jumps, "lateral_jumps_per_10mm": float(jumps / max(total * PIXEL_UM / 1e4, 1e-9)),
           "bridges": bridges, "bridge_len_mm": float(blen * PIXEL_UM / 1000),
           "share_R_lt_150um": float((rr < R_MIN).mean()) if len(rr) else 0.0,
           "share_R_lt_300um": float((rr < 2 * R_MIN).mean()) if len(rr) else 0.0,
           "R_min_um": float(rr.min() * PIXEL_UM) if len(rr) else 0.0}
    cor, _ = corridors(cs, matter.shape)
    out["skeleton_in_corridor"] = float((cor[sk] > 0).mean())
    if with_corridor:
        out["matter_coverage"] = float((cor[matter] > 0).mean())
        out["corridor_on_matter"] = float(matter[cor > 0].mean()) if (cor > 0).any() else 0.0
    return out


def structure_tangent(img, sigma=1.0, rho=4.0):
    g = img.astype(np.float32)
    gx = ndi.gaussian_filter(g, sigma, order=(0, 1)); gy = ndi.gaussian_filter(g, sigma, order=(1, 0))
    jxx, jxy, jyy = (ndi.gaussian_filter(a, rho) for a in (gx * gx, gx * gy, gy * gy))
    th = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    tr = jxx + jyy
    coh = np.where(tr > 1e-6, (np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-6)) ** 2, 0)
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32), coh.astype(np.float32)


def run_global(T, seed_score, log=None):
    """v5: every seed traced freely (no contact rule; self-contact, void and JUMP rules kept), seeds already on a
    candidate (<= 2 px) skipped; then non-overlapping selection, longest first: a candidate whose points fall within
    CONTACT_R of accepted bands is split into its free runs, which go back to the queue."""
    import heapq
    sk = skeletonize(T.m)
    sy, sx = np.nonzero(sk)
    order = np.argsort(-seed_score[sy, sx], kind="stable")[::SEED_STRIDE]
    covered = np.zeros((T.H, T.Wd), bool)
    cands, reasons = [], {}
    for i in order:
        y, x = float(sy[i]), float(sx[i])
        if covered[int(y), int(x)] or not T.inside(y, x, pad=int(PROF) + 6):
            continue
        tx, ty = T.tangent(y, x)
        th = math.atan2(ty, tx)
        pa, ra, pda, _ = T.trace(y, x, th)
        for p in pda:
            T.paint(T.own, *p, False)
        for p in pa[SELF_LAG:]:
            T.paint(T.own, *p, True)
        pb, rb, pdb, _ = T.trace(y, x, th + math.pi)
        for p in pa + pda + pdb:
            T.paint(T.own, *p, False)
        line = np.array(pb[::-1] + [(y, x)] + pa, np.float64)
        yi = np.clip(np.rint(line[:, 0]).astype(int), 0, T.H - 1); xi = np.clip(np.rint(line[:, 1]).astype(int), 0, T.Wd - 1)
        covered[yi, xi] = True
        for dy in (-2, -1, 1, 2):
            covered[np.clip(yi + dy, 0, T.H - 1), xi] = True; covered[yi, np.clip(xi + dy, 0, T.Wd - 1)] = True
        if len(line) * STEP >= MIN_LEN:
            cands.append(line)
            for rr in (ra, rb):
                reasons[rr] = reasons.get(rr, 0) + 1
        if log and len(cands) % 1000 == 0:
            log(f"{len(cands)} candidates")
    heap = [(-len(c), k) for k, c in enumerate(cands)]
    heapq.heapify(heap)
    bands = []
    while heap:
        _, k = heapq.heappop(heap)
        c = cands[k]
        yi = np.clip(np.rint(c[:, 0]).astype(int), 0, T.H - 1); xi = np.clip(np.rint(c[:, 1]).astype(int), 0, T.Wd - 1)
        free = T.forbid[yi, xi] == 0
        if free.all():
            bid = len(bands) + 1
            for p in c:
                T.paint(T.forbid, *p, bid)
            bands.append(c)
            continue
        lab, n = ndi.label(free)
        for r in range(1, n + 1):
            seg = c[lab == r]
            if len(seg) * STEP >= MIN_LEN:
                cands.append(seg); heapq.heappush(heap, (-len(seg), len(cands) - 1))
    return bands, {"stop_reasons_candidates": reasons, "n_candidates_traced": len(reasons) and sum(reasons.values()) // 2}


def detect(matter, tx, ty, seed_score, log=None, mode="global", dens=None, raw=None, forbid0=None):
    T = Tracer(matter, tx, ty, dens=dens, raw=raw, forbid0=forbid0)
    if mode == "global":
        raw, info = run_global(T, seed_score, log=log)
    else:
        raw, info = T.run(seed_score, log=log)
    welded, nm = weld_ends(raw, T.m, T.raw)
    info["weld_merges"] = nm
    info["n_raw"] = len(raw)
    return [resample(b) for b in welded], info


# ================================================================== 011 additions
def set_width(um):
    """Idea 2: all width-dependent constants for a band width of `um` micrometres."""
    global W, CONTACT_R, PROF, BRIDGE, TRACE_BRIDGE, MIN_LEN, SELF_LAG, TRIM
    W = um / PIXEL_UM
    CONTACT_R = W + 1.0
    PROF = W / 2 + 1.0
    BRIDGE = 2 * W
    TRACE_BRIDGE = W
    MIN_LEN = 3 * W
    SELF_LAG = int(math.ceil(3 * CONTACT_R / STEP))
    TRIM = int(round(W))


def forbid_from(bands, shape, w_old_px, w_new_px):
    """Fill pass: centrelines of new bands must stay >= w_old/2 + w_new/2 + 1 px from the existing ones."""
    cl = np.zeros(shape, bool)
    for b in bands:
        cl[np.clip(np.rint(b[:, 0]).astype(int), 0, shape[0] - 1), np.clip(np.rint(b[:, 1]).astype(int), 0, shape[1] - 1)] = True
    d = ndi.distance_transform_edt(~cl)
    return np.where(d < w_old_px / 2 + w_new_px / 2 + 1.0, -1, 0).astype(np.int32)


def link_graph(bands, member3d, raw3d, maxd_w=4.0, log=None, widths=None):
    """Idea 3: link band pieces into sheets. Candidate link = two ends facing within 45 deg, lateral offset <= W,
    distance <= maxd_w x W; connector = C1 Hermite between the ends trimmed by W, R >= R_MIN over +/- 3 W, clear
    (> W/2) of every other band, and every connector point inside the 3D membership (the gap must be filled in a
    neighbouring slice: idea 1 evidence). Score = distance + 2 lateral + 20 (1 - cos). Greedy on the score, each end
    used once, no cycle. Returns chains (lists of band indices with orientation) and the linked curves."""
    from scipy.spatial import cKDTree
    COS45 = math.cos(math.radians(45))
    maxd = maxd_w * W
    ends = []
    for i, b in enumerate(bands):
        ends.append((i, True, b[0], end_dir(b, True))); ends.append((i, False, b[-1], end_dir(b, False)))
    P = np.array([e[2] for e in ends]); D = np.array([e[3] for e in ends])
    H, Wd = member3d.shape
    if widths is None:
        widths = [np.full(len(b), W) for b in bands]
    widths = [np.broadcast_to(np.asarray(w, np.float64), (len(b),)).copy() for b, w in zip(bands, widths)]
    PTS_ALL = np.concatenate(bands); TREE = cKDTree(PTS_ALL)
    OWN = np.concatenate([np.full(len(b), i) for i, b in enumerate(bands)]); WPT_ALL = np.concatenate(widths)
    ET = cKDTree(P)
    cand = []
    for a, b in ET.query_pairs(maxd):
        if ends[a][0] == ends[b][0]:
            continue
        d = P[b] - P[a]; dist = float(np.hypot(*d))
        if dist == 0:
            continue
        u = d / dist
        if -(D[a] @ D[b]) < COS45 or (u @ D[a]) < COS45 * 0.5 or (-u @ D[b]) < COS45 * 0.5:
            continue
        lat = abs(d[0] * D[a][1] - d[1] * D[a][0])
        if lat > W:
            continue
        ba, bb = bands[ends[a][0]], bands[ends[b][0]]
        if len(ba) <= 3 * TRIM or len(bb) <= 3 * TRIM:
            continue
        sa = ba[TRIM:] if ends[a][1] else ba[:-TRIM][::-1]
        sb = bb[TRIM:] if ends[b][1] else bb[:-TRIM][::-1]
        da = sa[0] - sa[TRIM]; da = da / np.hypot(*da); db = sb[0] - sb[TRIM]; db = db / np.hypot(*db)
        con = hermite(sa[0], da, sb[0], db)
        k = 3 * int(round(W))
        if min_radius(np.concatenate([sa[:k][::-1], con, sb[:k]])) < R_MIN:
            continue
        if len(con):
            yi = np.clip(np.rint(con[:, 0]).astype(int), 0, H - 1); xi = np.clip(np.rint(con[:, 1]).astype(int), 0, Wd - 1)
            if not member3d[yi, xi].all():
                continue
            wj = min(widths[ends[a][0]][0 if ends[a][1] else -1], widths[ends[b][0]][0 if ends[b][1] else -1])
            if not clear_of_others(con, wj, TREE, PTS_ALL, OWN, WPT_ALL, (ends[a][0], ends[b][0])):
                continue
        cand.append((dist + 2 * lat + 20 * (1 + float(D[a] @ D[b])), a, b))
    cand.sort()
    parent = list(range(len(bands)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    used, link = set(), {}
    for sc, a, b in cand:
        if a in used or b in used:
            continue
        ra, rb = root(ends[a][0]), root(ends[b][0])
        if ra == rb:
            continue                                            # no cycle
        parent[ra] = rb; used.add(a); used.add(b); link[a] = b; link[b] = a
    seen, chains, curves, cwidths = set(), [], [], []
    for i in range(len(bands)):
        if i in seen:
            continue
        # find a chain end
        cur, fwd, guard = i, True, 0
        while True:
            e = 2 * cur + (0 if fwd else 1)                      # entry end when walking forward
            if e not in link or guard > len(bands):
                break
            o = link[e]; cur = o // 2; fwd = (o % 2 == 1); guard += 1
        chain, parts, wparts = [], [], []
        while cur not in seen:
            seen.add(cur)
            b = bands[cur] if fwd else bands[cur][::-1]
            wb = widths[cur] if fwd else widths[cur][::-1]
            if parts:
                prev = parts[-1][:-TRIM]; wprev = wparts[-1][:-TRIM]; b = b[TRIM:]; wb = wb[TRIM:]
                dp = prev[-1] - prev[-1 - TRIM]; dp = dp / np.hypot(*dp); dq = b[0] - b[TRIM]; dq = dq / np.hypot(*dq)
                parts[-1] = prev; wparts[-1] = wprev
                con = hermite(prev[-1], dp, b[0], dq)
                parts.append(con); wparts.append(np.full(len(con), min(wprev[-1], wb[0])))   # 013 (3): per-portion width
            parts.append(b); wparts.append(wb); chain.append((cur, fwd))
            e = 2 * cur + (1 if fwd else 0)
            if e not in link:
                break
            o = link[e]; cur = o // 2; fwd = (o % 2 == 0)
        c_, w_ = resample_w(np.concatenate(parts), np.concatenate(wparts))
        chains.append(chain); curves.append(c_); cwidths.append(w_)
    return chains, curves, {"link_candidates": len(cand), "links": len(link) // 2, "widths": cwidths}


# ================================================================== 013 variable width (variant V)
W_HI_UM, W_LO_UM, RAMP_UM = 80.0, 50.0, 30.0
_disks = {}


def _disk(r):
    k = round(r, 2)
    if k not in _disks:
        R = int(math.ceil(r)); dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
        _disks[k] = (R, (dy ** 2 + dx ** 2) <= r * r)
    return _disks[k]


def _paint_r(arr, y, x, r):
    R, dk = _disk(r)
    H, Wd = arr.shape
    yi, xi = int(round(y)), int(round(x))
    y0, y1, x0, x1 = yi - R, yi + R + 1, xi - R, xi + R + 1
    cy0, cx0, cy1, cx1 = max(y0, 0), max(x0, 0), min(y1, H), min(x1, Wd)
    arr[cy0:cy1, cx0:cx1] |= dk[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]


def width_profile(raw_hi):
    """raw_hi: bool per point (True = an 80 um corridor fits). Width = 50 + min(30, slope * distance to the nearest
    50-only point), slope = 30 um per band width: the width never changes by more than 30 um over one 80 um."""
    hi, lo = W_HI_UM / PIXEL_UM, W_LO_UM / PIXEL_UM
    if raw_hi.all():
        return np.full(len(raw_hi), hi)
    idx = np.arange(len(raw_hi)); lo_idx = idx[~raw_hi]
    dist = np.min(np.abs(idx[:, None] - lo_idx[None, :]), axis=1) if len(lo_idx) < 4000 else ndi.distance_transform_edt(raw_hi)
    return np.minimum(hi, lo + (hi - lo) * dist / (W_HI_UM / PIXEL_UM))


def run_global_var(T, seed_score, log=None):
    """Variant V: candidates traced freely (tracer at 80 um, as v6); longest-first selection with per-point width:
    80 um where the 80 um corridor is free, 50 um where only 50 um fits, split where neither fits; ramped profile;
    accepted points paint the forbidden zones with their own width."""
    import heapq
    set_width(W_HI_UM)
    hi, lo = W_HI_UM / PIXEL_UM, W_LO_UM / PIXEL_UM
    # candidates exactly as run_global (reuse its tracing part by running it on an empty selection)
    sk = skeletonize(T.m)
    sy, sx = np.nonzero(sk)
    order = np.argsort(-seed_score[sy, sx], kind="stable")[::SEED_STRIDE]
    covered = np.zeros((T.H, T.Wd), bool)
    cands = []
    for i in order:
        y, x = float(sy[i]), float(sx[i])
        if covered[int(y), int(x)] or not T.inside(y, x, pad=int(PROF) + 6):
            continue
        tx, ty = T.tangent(y, x)
        th = math.atan2(ty, tx)
        pa, ra, pda, _ = T.trace(y, x, th)
        for p in pda:
            T.paint(T.own, *p, False)
        for p in pa[SELF_LAG:]:
            T.paint(T.own, *p, True)
        pb, rb, pdb, _ = T.trace(y, x, th + math.pi)
        for p in pa + pda + pdb:
            T.paint(T.own, *p, False)
        line = np.array(pb[::-1] + [(y, x)] + pa, np.float64)
        yi = np.clip(np.rint(line[:, 0]).astype(int), 0, T.H - 1); xi = np.clip(np.rint(line[:, 1]).astype(int), 0, T.Wd - 1)
        covered[yi, xi] = True
        for dy in (-2, -1, 1, 2):
            covered[np.clip(yi + dy, 0, T.H - 1), xi] = True; covered[yi, np.clip(xi + dy, 0, T.Wd - 1)] = True
        if len(line) * STEP >= 3 * lo:
            cands.append(line)
    # forbidden zones for a NEW point of width 80 (f_hi) and of width 50 (f_lo)
    f_hi = np.zeros((T.H, T.Wd), bool); f_lo = np.zeros((T.H, T.Wd), bool)
    heap = [(-len(c), k) for k, c in enumerate(cands)]
    heapq.heapify(heap)
    bands, widths, n_split = [], [], 0
    while heap:
        _, k = heapq.heappop(heap)
        c = cands[k]
        yi = np.clip(np.rint(c[:, 0]).astype(int), 0, T.H - 1); xi = np.clip(np.rint(c[:, 1]).astype(int), 0, T.Wd - 1)
        ok_lo = ~f_lo[yi, xi]
        if not ok_lo.all():
            n_split += 1
            lab, n = ndi.label(ok_lo)
            for r in range(1, n + 1):
                seg = c[lab == r]
                if len(seg) * STEP >= 3 * lo:
                    cands.append(seg); heapq.heappush(heap, (-len(seg), len(cands) - 1))
            continue
        w = width_profile(~f_hi[yi, xi])
        for p, wp in zip(c, w):
            _paint_r(f_hi, p[0], p[1], wp / 2 + hi / 2 + 1.0)
            _paint_r(f_lo, p[0], p[1], wp / 2 + lo / 2 + 1.0)
        bands.append(c); widths.append(w)
    return bands, widths, {"n_candidates": len(cands), "splits": n_split,
                           "share_points_80um": float(np.mean(np.concatenate(widths) >= hi - 1e-6))}


def detect_var(matter, tx, ty, seed_score, dens=None, raw=None):
    set_width(W_HI_UM)
    T = Tracer(matter, tx, ty, dens=dens, raw=raw)
    raw_b, raw_w, info = run_global_var(T, seed_score)
    welded, nm, ww = weld_ends_w(raw_b, T.m, T.raw, widths=raw_w)
    info["weld_merges"] = nm; info["n_raw"] = len(raw_b)
    return welded, ww, info
