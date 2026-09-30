"""025: frozen C1 on one ORIGINAL slice -> /out/c1orig/z{Z}.npz (keys b{i}: (n, 3) y, x, width). usage: c1_slice.py Z"""
import os, sys
import numpy as np
from predetect import predetect
from slice_prep import prep
Z = int(sys.argv[1]); os.makedirs("/out/c1orig", exist_ok=True)
if not os.path.exists(f"/out/c1orig/z{Z}.npz"):
    S = prep(Z); bd = predetect(S["matter"], S["tx"], S["ty"], S["coh"])
    np.savez_compressed(f"/out/c1orig/z{Z}.npz", **{f"b{i}": b.astype(np.float32) for i, b in enumerate(bd)})
print("DONE", Z, flush=True)
