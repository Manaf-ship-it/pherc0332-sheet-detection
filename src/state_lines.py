"""027: C1 lines traced on a CENTRIFUGED state (cf_RUN final displacement + broken bonds) for one perturbation draw (025
h2_lines definitions: base, T1 (1,0), T2 (0,1), J0 / J1 jitter +/-0.6 px seeds 300 / 301, S0 smooth field), mapped back to the
ORIGINAL coordinates, resampled at 1 px -> /out/slines_RUN__DRAW.npz (pts, lid, width) = same format as 025 h2lines.
RUN = 'orig' -> no state (the original slice). usage: state_lines.py RUN Z DRAW"""
import os, sys
import numpy as np
from scipy import ndimage as ndi
if os.environ.get("BLOCK1CM"):
    import block1cm  # noqa: F401
import fsi_kernel_002 as k2
from metrics3 import detect_v3
from render3 import void_level
from slice_prep import prep
from vt_eval import resample
RUN, Z, DRAW = sys.argv[1], int(sys.argv[2]), sys.argv[3]; f = f"/out/slines_{RUN}__{DRAW}.npz"
if not os.path.exists(f):
    S = prep(Z); m, g = S["matter"], S["gray"]; N = m.shape[0]; ys, xs = np.nonzero(m); P = len(ys)
    _, _, bi, bj, _, _, _, _ = k2.build_bonds(m, S["tx"], S["ty"], 1.0, 0.1, 6.0, 0.1)
    if RUN == "orig" or RUN.startswith("orig_"):
        sx, sy, brk = np.zeros(P), np.zeros(P), np.zeros(len(bi), bool)
    else:
        d = np.load(f"/out/cf_{RUN}/final.npz"); sx, sy = d["ux"].astype(float), d["uy"].astype(float); brk = d["broken"].astype(bool)[:len(bi)]
    DR = {"base": (0, 0, 0), "T1": (0, 1, 0), "T2": (0, 0, 1), "J0": (300, 0, 0), "J1": (301, 0, 0), "S0": (400, 0, 0)}
    seed, ty_, tx_ = DR[DRAW]; ux, uy = sx + float(tx_), sy + float(ty_)
    if DRAW[0] == "J":
        r = np.random.default_rng(seed); ux += r.uniform(-0.6, 0.6, P); uy += r.uniform(-0.6, 0.6, P)
    elif DRAW[0] == "S":
        r = np.random.default_rng(seed); fl = [ndi.gaussian_filter(r.standard_normal((N, N)), 30.0) for _ in range(2)]; fl = [v * 2.0 / v[m].std() for v in fl]
        ux += fl[0][ys, xs]; uy += fl[1][ys, xs]
    _, mapped, _, _, _ = detect_v3(ys, xs, ux, uy, g, m, void_level(g, m, S["scroll"]), bi, bj, brk)
    L = [resample(b["pts"][:, :2]) if len(b["pts"]) > 1 else b["pts"][:, :2] for b in mapped]
    W = [np.interp(np.linspace(0, len(b["w"]) - 1, len(l)), np.arange(len(b["w"])), b["w"]) for b, l in zip(mapped, L)]
    np.savez_compressed(f, pts=np.concatenate(L).astype(np.float32), lid=np.concatenate([np.full(len(l), k) for k, l in enumerate(L)]), width=np.concatenate(W).astype(np.float32))
print("DONE", RUN, Z, DRAW, flush=True)
