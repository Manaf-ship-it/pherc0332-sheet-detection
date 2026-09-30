"""Per-slice preparation with the SAME rules as 001 (Otsu threshold fixed at 88 from Z 4224, 12 px cleaning,
grey structure tensor sigma 1 / rho 4). Z 4224 is checked against the 001 outputs bit for bit."""
import numpy as np
from scipy import ndimage as ndi

SAMPLE = "/sample/PHerc0332_L2_Z4124-4324_uint8.npy"
Z0 = 4124
OTSU = 88.0
NOISE = 12


def prep(z):
    vol = np.load(SAMPLE, mmap_mode="r")
    gray = np.ascontiguousarray(vol[z - Z0])
    m_raw = gray >= OTSU
    lab, _ = ndi.label(m_raw, structure=np.ones((3, 3), bool))
    small_m = (np.bincount(lab.ravel()) < NOISE)[lab] & m_raw
    m1 = m_raw & ~small_m
    labv, _ = ndi.label(~m1)
    small_v = (np.bincount(labv.ravel()) < NOISE)[labv] & ~m1
    matter = m1 | small_v
    g = gray.astype(np.float32)
    gx = ndi.gaussian_filter(g, 1.0, order=(0, 1)); gy = ndi.gaussian_filter(g, 1.0, order=(1, 0))
    jxx, jxy, jyy = (ndi.gaussian_filter(a, 4.0) for a in (gx * gx, gx * gy, gy * gy))
    th = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    tx, ty = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
    tr = jxx + jyy
    coh = np.where(tr > 1e-6, (np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-6)) ** 2, 0).astype(np.float32)
    scroll = ndi.binary_fill_holes(ndi.binary_closing(matter, np.ones((3, 3)), iterations=15))
    yc, xc = ndi.center_of_mass(scroll)
    return {"gray": gray, "matter": matter, "tx": tx, "ty": ty, "coh": coh, "centre_yx": (float(yc), float(xc)), "scroll": scroll}


if __name__ == "__main__":
    import json
    d = prep(4224)
    ref_m = np.load("/prev/02_clean/matter.npy"); ref_g = np.load("/prev/00_sample/gray.npy")
    f = np.load("/prev/03_flow/flow.npz")
    res = {"gray_identical": bool((d["gray"] == ref_g).all()), "matter_identical": bool((d["matter"] == ref_m).all()),
           "tx_max_abs_diff": float(np.abs(np.abs(d["tx"]) - np.abs(f["tx"])).max()), "centre_yx": d["centre_yx"]}
    print(json.dumps(res))
