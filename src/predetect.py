"""020: pre-detection of the material lines with the FROZEN C1 (bands019 + C1 parameters) and the orthotropic material
fields that follow them.
  predetect(matter, tx, ty, coh) -> list of (n, 3) arrays (y, x, width_px) = C1 bands
  material_fields(bands, shape, tx0, ty0) -> tangent fields following the bands (nearest band centreline inside its
      territory, structure tensor elsewhere) and the territory label image (nearest band within 60 um, 0 = none)
"""
import hashlib, json, os
import numpy as np
from scipy import ndimage as ndi
import bands019 as B

C1 = {"SIG_B": 0.5, "W_HI_UM": 60, "W_LO_UM": 40, "RAMP_UM": 20}
_here = os.path.dirname(os.path.abspath(__file__))
_expected = {"bands019.py": "8cd79f932d0d1f07"}                         # prefix of the frozen C1 archive hash
assert hashlib.sha256(open(os.path.join(_here, "bands019.py"), "rb").read()).hexdigest().startswith(_expected["bands019.py"]), "bands019 differs from frozen C1"
for k, v in C1.items():
    setattr(B, k, v)


def predetect(matter, tx, ty, coh):
    B.set_width(B.W_HI_UM)
    T = B.Tracer(matter, tx, ty, dens=None, raw=None)
    raw_b, raw_w, info = B.run_global_var(T, coh)
    welded, nm, ww = B.weld_ends_w(raw_b, T.m, T.raw, widths=raw_w)
    B.set_width(80.0)
    return [np.c_[c, w] for c, w in zip(welded, ww)]


def material_fields(bands, shape, tx0, ty0, reach_um=60.0):
    H, W = shape
    cl = np.zeros(shape, np.int64); tcx = np.zeros(shape, np.float32); tcy = np.zeros(shape, np.float32)
    for i, b in enumerate(bands, 1):
        c = b[:, :2]
        g = np.gradient(c, axis=0) if len(c) > 1 else np.array([[0.0, 1.0]])
        g = g / np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-9)[:, None]
        yi = np.clip(np.rint(c[:, 0]).astype(int), 0, H - 1); xi = np.clip(np.rint(c[:, 1]).astype(int), 0, W - 1)
        cl[yi, xi] = i; tcx[yi, xi] = g[:, 1]; tcy[yi, xi] = g[:, 0]
    dist, (iy, ix) = ndi.distance_transform_edt(cl == 0, return_indices=True)
    terr = np.where(dist <= reach_um / 9.596, cl[iy, ix], 0)
    tx = np.where(terr > 0, tcx[iy, ix], tx0).astype(np.float32); ty = np.where(terr > 0, tcy[iy, ix], ty0).astype(np.float32)
    return tx, ty, terr
