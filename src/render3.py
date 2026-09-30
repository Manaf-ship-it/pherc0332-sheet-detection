"""022 deformed raster v3 + inverse map v3 (replace render2 / gt_metrics2.map_bands).
Review of 021 (reproduced, test_reproduce_021.py): the bilinear splat of render2 with a 0.25 mask threshold doubles a
1-px line under a 0.26 px shift and closes a 1-px gap -> it can fuse separate sheets.
v3 raster: each matter pixel is a unit square moved rigidly by its displacement -> it covers exactly ONE pixel centre,
q = rint(p + u) (area-conserving, no blur, no fusion of separate sheets). Collisions: grey = mean of the colliding points.
Holes: only where an INTACT bond has its two rendered pixels at Chebyshev distance >= 2 (the material really stretched
open), the bond midpoint pixel is filled if uncovered (grey = mean of the two ends). The void is Eulerian (original grey;
median void grey in newly uncovered matter pixels). At rest, and under any displacement with rint(u) == 0, the image and
mask are exactly the original; an integer translation gives the exactly translated image.
Inverse map v3: every rendered pixel has an OWNER lattice point (collisions: the one whose p + u is closest to the pixel
centre; midpoint pixels: the bond end i); a band point x is mapped back with the RENDERED offset of the owner of rint(x)
(or of the nearest owned pixel): X = x - (q_o - p_o). It inverts the raster exactly (null -> identity, sub-pixel band
coordinates kept, integer translation -> exact)."""
import numpy as np
from scipy import ndimage as ndi


def void_level(gray, matter, scroll):
    return float(np.median(gray[scroll & ~matter]))


def render(ys, xs, ux, uy, gray, matter, vlevel, bi=None, bj=None, broken=None):
    N0, N1 = gray.shape; P = len(ys)
    qy = np.rint(ys + uy).astype(np.int64); qx = np.rint(xs + ux).astype(np.int64)
    ok = (qy >= 0) & (qy < N0) & (qx >= 0) & (qx < N1); k = qy * N1 + qx
    gv = gray[ys, xs].astype(np.float64)
    cnt = np.bincount(k[ok], minlength=N0 * N1); sg = np.bincount(k[ok], weights=gv[ok], minlength=N0 * N1)
    # owner: closest rendered point to the pixel centre
    d2 = (ys + uy - qy) ** 2 + (xs + ux - qx) ** 2
    order = np.lexsort((d2, k)); ks = k[order]; first = np.r_[True, ks[1:] != ks[:-1]] & ok[order]
    owner = -np.ones(N0 * N1, np.int64); owner[ks[first]] = order[first]
    mask = cnt > 0; collided = int(ok.sum() - mask.sum())
    img = np.where(matter, np.float64(vlevel), gray.astype(np.float64)).ravel()
    img[mask] = sg[mask] / cnt[mask]
    nfill = 0
    if bi is not None:
        intact = ~broken if broken is not None else np.ones(len(bi), bool)
        cheb = np.maximum(np.abs(qy[bi] - qy[bj]), np.abs(qx[bi] - qx[bj]))
        sel = intact & (cheb >= 2) & ok[bi] & ok[bj]
        my = np.rint(0.5 * (ys[bi[sel]] + uy[bi[sel]] + ys[bj[sel]] + uy[bj[sel]])).astype(np.int64)
        mx = np.rint(0.5 * (xs[bi[sel]] + ux[bi[sel]] + xs[bj[sel]] + ux[bj[sel]])).astype(np.int64)
        km = my * N1 + mx; free = ~mask[km]
        km, gi, gj, oi = km[free], gv[bi[sel]][free], gv[bj[sel]][free], bi[sel][free]
        km_u, first_m = np.unique(km, return_index=True)
        img[km_u] = 0.5 * (gi + gj)[first_m]; mask[km_u] = True; owner[km_u] = oi[first_m]; nfill = len(km_u)
    return mask.reshape(N0, N1), img.reshape(N0, N1).astype(np.float32), {"owner": owner.reshape(N0, N1), "qy": qy, "qx": qx,
                                                                         "collided_points": collided, "midpoint_fills": nfill}


def map_back(bands, R, ys, xs):
    """bands: list of (n, >=2) arrays in rendered coordinates -> list of dicts for the evaluator (original coordinates)."""
    own = R["owner"]; H, W = own.shape
    _, (iy, ix) = ndi.distance_transform_edt(own < 0, return_indices=True)
    dy = (R["qy"] - ys).astype(np.float64); dx = (R["qx"] - xs).astype(np.float64); out = []
    for b in bands:
        py = np.clip(np.rint(b[:, 0]).astype(int), 0, H - 1); px = np.clip(np.rint(b[:, 1]).astype(int), 0, W - 1)
        o = own[iy[py, px], ix[py, px]]
        out.append({"pts": np.c_[b[:, 0] - dy[o], b[:, 1] - dx[o]], "w": b[:, 2], "conn": np.zeros(len(b), bool)})
    return out
