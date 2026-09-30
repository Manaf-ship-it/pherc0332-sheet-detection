"""027 KNOTS of material blocking = 025 lens-tip detector v3 (validated by the user, section 14), made generic (any slice /
state). On the DEFORMED centrifuged state: voids inside the wide envelope (closing 40 it.), lenses = void components with area
>= 3000 px and elongation >= 3; tips = the 2 ends of each lens along its major axis; knot = deformed matter within 14 px of a
tip, mapped back to the ORIGINAL matter (render owner). Also each tip's original position (the owner node of the deformed
matter pixel nearest the tip) and the lens axis. usage: knots027.py RUN Z  -> /out/knots_RUN.npz / .json"""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops, label as sklabel
if os.environ.get("BLOCK1CM"):
    import block1cm  # noqa: F401
import fsi_kernel_002 as k2
from render3 import render, void_level
from slice_prep import prep

AMIN, ELONG, TIP_R, CLOSE_IT = 3000, 3.0, 14, 40


def knots(run, z):
    S = prep(z); m, g = S["matter"], S["gray"]; N = m.shape[0]; ys, xs = np.nonzero(m)
    _, _, bi, bj, _, _, _, _ = k2.build_bonds(m, S["tx"], S["ty"], 1.0, 0.1, 6.0, 0.1)
    d = np.load(f"/out/cf_{run}/final.npz"); ux, uy = d["ux"].astype(float), d["uy"].astype(float)
    brk = d["broken"].astype(bool) if len(d["broken"]) == len(bi) else None
    dm, dimg, R = render(ys, xs, ux, uy, g, m, void_level(g, m, S["scroll"]), bi, bj, brk)
    env = ndi.binary_fill_holes(ndi.binary_closing(dm, np.ones((3, 3)), iterations=CLOSE_IT))
    voids = env & ~ndi.binary_closing(dm, np.ones((3, 3)))
    lab = sklabel(voids, connectivity=1); tips = []
    for rp in regionprops(lab):
        if rp.area < AMIN or rp.minor_axis_length < 1 or rp.major_axis_length / rp.minor_axis_length < ELONG:
            continue
        c = np.array(rp.centroid); o = rp.orientation; ax_ = np.array([np.cos(o), np.sin(o)])
        co = rp.coords; proj = (co - c) @ ax_
        tips += [(co[np.argmin(proj)], -ax_, rp.major_axis_length), (co[np.argmax(proj)], ax_, rp.major_axis_length)]
    knot_img = np.zeros(m.shape, bool); yy, xx = np.ogrid[-TIP_R:TIP_R + 1, -TIP_R:TIP_R + 1]; disc = yy ** 2 + xx ** 2 <= TIP_R ** 2
    own = R["owner"]; dmy, dmx = np.nonzero(dm); tdm = cKDTree_(np.c_[dmy, dmx])
    tip_orig, tip_axis, tip_len = [], [], []
    for (ty, tx), a, L in tips:
        y0, x0 = ty - TIP_R, tx - TIP_R
        if y0 < 0 or x0 < 0 or y0 + 2 * TIP_R + 1 > N or x0 + 2 * TIP_R + 1 > N:
            continue
        knot_img[y0:y0 + 2 * TIP_R + 1, x0:x0 + 2 * TIP_R + 1] |= disc
        _, q = tdm.query([ty, tx]); o_ = own[dmy[q], dmx[q]]
        if o_ >= 0:
            tip_orig.append((ys[o_], xs[o_])); tip_axis.append(a); tip_len.append(L)
    knot_img &= dm
    oo = own[knot_img]; oo = oo[oo >= 0]; node_mask = np.zeros(len(ys), bool); node_mask[oo] = True
    orig = np.zeros(m.shape, bool); orig[ys[node_mask], xs[node_mask]] = True
    return {"node_mask": node_mask, "orig_mask": orig, "tip_orig": np.array(tip_orig, float).reshape(-1, 2), "tip_axis_def": np.array(tip_axis).reshape(-1, 2),
            "tip_len": np.array(tip_len), "n_lenses": len(tips) // 2, "dm": dm, "dimg": dimg, "owner": own}


def cKDTree_(P):
    from scipy.spatial import cKDTree
    return cKDTree(P)


if __name__ == "__main__":
    RUN, Z = sys.argv[1], int(sys.argv[2])
    K = knots(RUN, Z)
    res = {"run": RUN, "z": Z, "lenses": K["n_lenses"], "tips_mapped": int(len(K["tip_orig"])), "knot_nodes": int(K["node_mask"].sum()),
           "share_of_matter": float(K["node_mask"].mean()), "params": {"AMIN": AMIN, "ELONG": ELONG, "TIP_R": TIP_R, "CLOSE_IT": CLOSE_IT}}
    print(json.dumps(res)); json.dump(res, open(f"/out/knots_{RUN}.json", "w"), indent=1)
    np.savez_compressed(f"/out/knots_{RUN}.npz", node_mask=K["node_mask"], orig_mask=K["orig_mask"], tip_orig=K["tip_orig"], tip_len=K["tip_len"])
