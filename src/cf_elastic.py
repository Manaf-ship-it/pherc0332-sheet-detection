"""025: CENTRIFUGE, PURELY ELASTIC material WITHOUT RUPTURE, slice Z (020 lattice: one point per matter pixel, bonds to the
8 neighbours, contact, solid007 kernels, quasi-static relaxation by kinetic damping, centre anchored r < 52 px).
Material (orthotropic, frame = the C1 bands): k = k_t cos^2 + k_n sin^2 w.r.t. the local band tangent (k_t = 1);
bonds joining two DIFFERENT C1 band territories (sheet interfaces) scaled by k_if. No rupture (threshold infinite).
Centrifugal body force per point 2 beta 1e-3 r / R^2 (020 law). usage: cf_elastic.py NAME '{"kn":..,"kif":..,"beta":..,"steps":..}'
Outputs: history (KE, mean displacement, displacement change, max tangential / normal bond strain), convergence, M1 v3 on
the DEV annotations (clean opening, tight pairs, crossing, tear), render3 collisions, full-slice + DEV-zone images
(progress image every 4000 steps in LBZ)."""
import json, os, sys, time
import cupy as cp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import fsi_kernel_002 as k2
import solid007
from gt_metrics import load_gt, m1_prepare, ZONES
from metrics3 import m1v3
from predetect import material_fields
from render3 import render, void_level
from slice_prep import prep
if os.environ.get("BLOCK1CM"):                      # 027: 1-cm block slices (block1cm redirects the frozen slice_prep)
    import block1cm  # noqa: F401

NAME = sys.argv[1]; PAR = {"z": 4224, "kn": 0.1, "kif": 1.0, "beta": 3.0, "steps": 20000, "crop_margin": None, "lists_every": 1, "mass_gray": False, "cdrag": 0.0, "rupt_radial": None, "rupt_cone_deg": 30.0, "cut_1px": False, "between_sheets_only": False, "rupt_dir": "radial", "rupt_bands": None, "rupt_zone": None, "seam_only": False, "seam_c": 20.0, "precut": False, "wedge_g0": 0.0, "wedge_p": 1e-2, "cut_file": None, "vcrack": None, "vc_every": 50, "vc_zone_r": 40.0, "vc_apex_r": 3.0, "vc_trig": 3.0, "vc_persist": 2, "vc_prop": 2.0, "vc_maxdist": 80.0, "vc_ang": 30.0, "vc_emin": 0.02, "vc_rank": None, "save_steps": None, "kt": 1.0, "dt": 1.0, "fluid": False, "fl_rho": 0.2, "fl_K": 1.0, "fl_D": 2.0, "fl_phim": 0.1, "fl_kapm": 0.05, "fl_sig": 1.5, "images": True, "matter_file": None, "break_file": None}
PAR.update(json.loads(sys.argv[2])); Z = int(PAR["z"])
OUT = f"/out/cf_{NAME}"; IMG = f"/out/figures/cf_{NAME}"; os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
KT, ZETA, KC, M_, MNUM, DT, R_ANCHOR = 1.0, 0.1, 1.0, 1.5, 6.0, 1.0, 52.0
# 025 piste 1 (stiff sheets): in-sheet stiffness kt (default 1) and time step dt (explicit stability: dt ~ 1 / sqrt(kt); the
# same simulated time needs steps = T / dt). The centrifugal load does not depend on k or dt.
KT, DT = float(PAR["kt"]), float(PAR["dt"])
t00 = time.time()
S = prep(Z); m, g = S["matter"], S["gray"]; N = m.shape[0]
if PAR["matter_file"]:                                   # 029: extra matter (step-3 welds) OR-ed into the matter
    m = m | np.load(f"/out/{PAR['matter_file']}")["weld"]; S["matter"] = m
YC, XC = S["centre_yx"]; VL = void_level(g, m, S["scroll"])
# 027: C1 bands of the slice: 019 variant file for the DEV slices, else the frozen C1 of 025 (c1orig / c1orig1cm, same C1)
_bf = [f for f in (f"/p19/var_sig05_W60_40/z{Z}.npz", f"/p25/c1orig/z{Z}.npz", f"/p25/c1orig1cm/z{Z}.npz", f"/out/c1orig/z{Z}.npz") if os.path.exists(f)]
B0 = np.load(_bf[0]); bands = [B0[k][:, :3].astype(float) for k in B0.files if k.startswith("b")]
px, py, bi, bj, L0, _, _, _ = k2.build_bonds(m, S["tx"], S["ty"], 1.0, 0.1, MNUM, ZETA)
P, Bn0 = len(px), len(bi); ys, xs = py.astype(int), px.astype(int)
# fragments: matter pieces not bonded to the anchored centre have no elastic equilibrium under a centrifugal load (025
# probe: 6 % of the matter in 696 pieces, the mean displacement kept growing). Model choice: each piece is linked to the
# nearest matter of the already-connected set within 3 px by ADHESION bonds (stiffness = weak interface: k_n x k_if),
# iterated until no piece can be reached; unreachable pieces are held fixed and unloaded (reported).
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
labc, ncomp = ndi.label(m, structure=np.ones((3, 3))); lab_pt = labc[ys, xs]
r0_ = np.hypot(px - XC, py - YC); conn = np.zeros(ncomp + 1, bool); conn[np.unique(lab_pt[r0_ < 52.0])] = True; conn[0] = False
add_i, add_j = [], []
while True:
    inC = conn[lab_pt]
    if inC.all():
        break
    d, q = cKDTree(np.c_[ys[inC], xs[inC]]).query(np.c_[ys[~inC], xs[~inC]], distance_upper_bound=3.0)
    ok = np.isfinite(d)
    if not ok.any():
        break
    cand = np.nonzero(~inC)[0][ok]; tgt_ = np.nonzero(inC)[0][q[ok]]
    add_i += list(cand); add_j += list(tgt_); conn[np.unique(lab_pt[cand])] = True
add_i, add_j = np.array(add_i, np.int64), np.array(add_j, np.int64)
bridge = np.r_[np.zeros(Bn0, bool), np.ones(len(add_i), bool)]
bi = np.r_[bi, add_i].astype(np.int32); bj = np.r_[bj, add_j].astype(np.int32)
L0 = np.r_[L0, np.hypot(px[add_j] - px[add_i], py[add_j] - py[add_i])].astype(np.float32); Bn = len(bi)
floating = ~conn[lab_pt]
frag_info = {"components": int(ncomp), "adhesion_bonds": int(len(add_i)), "pieces_bridged": int(len(np.unique(lab_pt[add_i]))) if len(add_i) else 0,
             "floating_points_fixed": int(floating.sum()), "floating_share": float(floating.mean())}
print(json.dumps(frag_info), flush=True)
r0 = np.hypot(px - XC, py - YC); R = float(r0.max()); OM2 = 2 * PAR["beta"] * 1e-3 / (M_ * R ** 2)
ex, ey = (xs[bj] - xs[bi]).astype(float), (ys[bj] - ys[bi]).astype(float); ln = np.maximum(np.hypot(ex, ey), 1e-9); ex /= ln; ey /= ln
tx, ty, terr = material_fields(bands, m.shape, S["tx"], S["ty"])
c2 = (tx[ys[bi], xs[bi]] ** 2 - ty[ys[bi], xs[bi]] ** 2) + (tx[ys[bj], xs[bj]] ** 2 - ty[ys[bj], xs[bj]] ** 2)
s2 = 2 * tx[ys[bi], xs[bi]] * ty[ys[bi], xs[bi]] + 2 * tx[ys[bj], xs[bj]] * ty[ys[bj], xs[bj]]
th = 0.5 * np.arctan2(s2, c2); cos2 = (ex * np.cos(th) + ey * np.sin(th)) ** 2
lab = terr[ys, xs]; inter = ((lab[bi] > 0) & (lab[bj] > 0) & (lab[bi] != lab[bj])) | bridge
k_ = np.where(bridge, PAR["kn"] * PAR["kif"], (KT * cos2 + PAR["kn"] * (1 - cos2)) * np.where(inter, PAR["kif"], 1.0))
cd = ZETA * 2 * np.sqrt(k_ * MNUM / 2)                                   # damping consistent with the actual stiffness
tang = (cos2 >= 0.5) & ~bridge                                                        # bond closer to the band tangent than to the normal

SRC = r"""
extern "C" __global__ void integrate_w(const float* px0, const float* py0, float* ux, float* uy, float* vx, float* vy,
    const float* ax, const float* ay, const unsigned char* fixed, const float* wcf, const float* Mp, float om2, float xc, float yc,
    int P, float dt, float xmax, float ymax, int* nclamp, float cdrag) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
    if (fixed[p]) { vx[p] = 0.f; vy[p] = 0.f; return; }
    float M = Mp[p];                                      // per-node mass (inertia AND centrifugal force)
    float cx = (px0[p] - xc) + ux[p], cy = (py0[p] - yc) + uy[p];
    // cdrag: drag of the surrounding medium (-c v, same c for every node): with it a heavier node reaches a higher speed
    vx[p] += (ax[p] + M * om2 * wcf[p] * cx - cdrag * vx[p]) / M * dt; vy[p] += (ay[p] + M * om2 * wcf[p] * cy - cdrag * vy[p]) / M * dt;
    ux[p] += vx[p] * dt; uy[p] += vy[p] * dt;
    float X = px0[p] + ux[p], Y = py0[p] + uy[p];
    if (X < 0.f || X > xmax || Y < 0.f || Y > ymax) { ux[p] = fminf(fmaxf(X, 0.f), xmax) - px0[p]; uy[p] = fminf(fmaxf(Y, 0.f), ymax) - py0[p];
        vx[p] = 0.f; vy[p] = 0.f; atomicAdd(nclamp, 1); }
}"""
SRC += r"""
extern "C" __global__ void wedge(const int* wi, const int* wj, const float* nx, const float* ny, const float* s0, int W, float g0,
    float p, const float* px0, const float* py0, const float* ux, const float* uy, float* ax, float* ay) {
    int w = blockIdx.x * blockDim.x + threadIdx.x; if (w >= W) return;
    int i = wi[w], j = wj[w];
    float s = ((px0[i] + ux[i]) - (px0[j] + ux[j])) * nx[w] + ((py0[i] + uy[i]) - (py0[j] + uy[j])) * ny[w];
    if (s < s0[w] + g0) { atomicAdd(&ax[i], p * nx[w]); atomicAdd(&ay[i], p * ny[w]); atomicAdd(&ax[j], -p * nx[w]); atomicAdd(&ay[j], -p * ny[w]); }
}"""
_mod = cp.RawModule(code=SRC); f_intw = _mod.get_function("integrate_w"); f_wedge = _mod.get_function("wedge")
m7 = solid007.module(); f_node, f_lists, f_contact, f_bonds = (m7.get_function(n) for n in ("node_of_u", "build_lists", "contact_u", "bond_forces_u"))
gpu = cp.asarray
PX0, PY0 = gpu(px), gpu(py); UX, UY, VX, VY = (cp.zeros(P, cp.float32) for _ in range(4)); AX, AY = cp.zeros(P, cp.float32), cp.zeros(P, cp.float32)
# 025 radial rupture (option): a LATTICE bond (not adhesion) within rupt_cone_deg of the local radial direction (from the scroll
# centre through the bond midpoint) breaks when its tensile strain exceeds rupt_radial; kernel test fel * en > Fcrit with
# en = 1 / k and Fcrit = strain threshold. Other bonds never break.
mx_, my_ = 0.5 * (px[bi] + px[bj]), 0.5 * (py[bi] + py[bj]); erx, ery = mx_ - XC, my_ - YC; ern = np.maximum(np.hypot(erx, ery), 1e-9)
cos_rad = np.abs(ex * erx / ern + ey * ery / ern)
if PAR["rupt_dir"] == "normal":                       # cone around the local MATERIAL normal (C1 band normal), not the radius
    cos_dir = np.sqrt(np.clip(1 - cos2, 0, 1))       # |e . n| with cos2 = (e . t)^2 w.r.t. the band tangent
else:
    cos_dir = cos_rad
radial = (cos_dir >= np.cos(np.radians(PAR["rupt_cone_deg"]))) & ~bridge
# rupt_bands: rupture only inside the first K bands of 50 um measured inward from the OUTER envelope of the scroll
# rupt_zone: "FILE:KEY" -> per-node mask (knots.py): rupture only for bonds with both ends in the zone
if PAR["rupt_zone"]:
    zf, zk = PAR["rupt_zone"].split(":"); zone = np.load(f"/out/{zf}")[zk].astype(bool)
    zone_b = np.r_[zone, np.zeros(max(0, P - len(zone)), bool)][:P]; radial &= zone_b[bi] & zone_b[bj]
# seam_only: only bonds touching a grey VALLEY (dark seam between two bright sheets along the structure-tensor normal: the
# darker end >= seam_c grey levels below the matter pixels 2 px away on both sides) = sheet INTERFACES, not inside a sheet
if PAR["seam_only"]:
    gf_ = g.astype(np.float32); nyp, nxp = S["tx"][ys, xs], -S["ty"][ys, xs]                   # structure-tensor normal (y, x)
    smp = lambda a, yy, xx, o: ndi.map_coordinates(a, [yy, xx], order=o, mode="nearest")
    gp_, gm2 = smp(gf_, ys + 2 * nyp, xs + 2 * nxp, 1), smp(gf_, ys - 2 * nyp, xs - 2 * nxp, 1)
    mp_, mm_ = smp(m.astype(np.uint8), ys + 2 * nyp, xs + 2 * nxp, 0) > 0, smp(m.astype(np.uint8), ys - 2 * nyp, xs - 2 * nxp, 0) > 0
    valley = mp_ & mm_ & (gp_ - gf_[ys, xs] >= PAR["seam_c"]) & (gm2 - gf_[ys, xs] >= PAR["seam_c"])
    valley_b = np.r_[valley, np.zeros(max(0, P - len(valley)), bool)][:P]; radial &= valley_b[bi] | valley_b[bj]
if PAR["rupt_bands"]:
    depth_ = ndi.distance_transform_edt(S["scroll"])[ys, xs]; inb = depth_ <= PAR["rupt_bands"] * 50.0 / 9.596
    radial &= inb[bi] & inb[bj]
# between_sheets_only: restrict cuts and radial rupture to bonds joining two DIFFERENT C1 band territories (between sheets);
# first 025 run: 88 % of the 1-px cuts were inside a single band territory (thin parts of one sheet) -> tears doubled
betw = (lab[bi] > 0) & (lab[bj] > 0) & (lab[bi] != lab[bj])
if PAR["between_sheets_only"]:
    radial &= betw
if PAR["break_file"]:                                    # 029: the ONLY breakable bonds = this per-bond mask (build_bonds order), any direction
    _bf = np.load(f"/out/{PAR['break_file']}")["cut"]; radial = np.r_[_bf, np.zeros(len(bi) - len(_bf), bool)]
EN_h = np.where(radial & (PAR["rupt_radial"] is not None), 1.0 / np.maximum(k_, 1e-9), 0.0).astype(np.float32)
FCRIT = float(PAR["rupt_radial"]) if PAR["rupt_radial"] is not None else 1e30
# 025 1-pixel links (option): contacts through a single pixel between two masses of matter. The matter opened by a 3x3 cross
# loses every 1-px-thick structure; each pixel gets the label of the nearest opened component; lattice bonds joining two
# different labels (the bonds of the 1-px bridges) are removed (broken from the start). Adhesion bonds are kept.
cut1 = np.zeros(len(bi), bool)
if PAR["cut_1px"]:
    opened = ndi.binary_opening(m, structure=ndi.generate_binary_structure(2, 1))
    lab_o, _ = ndi.label(opened, structure=np.ones((3, 3)))
    _, (iy_o, ix_o) = ndi.distance_transform_edt(~opened, return_indices=True)
    lp_o = lab_o[iy_o[ys, xs], ix_o[ys, xs]]
    cut1 = (lp_o[bi] != lp_o[bj]) & ~bridge
    if PAR["between_sheets_only"]:
        cut1 &= betw
BI, BJ, L0g = gpu(bi.astype(np.int32)), gpu(bj.astype(np.int32)), gpu(L0.astype(np.float32)); CD = gpu(cd.astype(np.float32)); KB = gpu(k_.astype(np.float32)); EN = gpu(EN_h)
# resource optimisation (025-opt): contact grid cropped to the scroll bounding box + margin (positions shifted for the
# cell lists only; forces are shift-invariant) and cell lists rebuilt every `lists_every` steps
if PAR["crop_margin"] is not None:
    Mg = int(PAR["crop_margin"]); GY0_, GX0_ = max(int(ys.min()) - Mg, 0), max(int(xs.min()) - Mg, 0)
    GH, GW = min(int(ys.max()) + Mg + 1, N) - GY0_, min(int(xs.max()) + Mg + 1, N) - GX0_
else:
    GY0_, GX0_, GH, GW = 0, 0, N, N
GPX0, GPY0 = gpu((px - GX0_).astype(np.float32)), gpu((py - GY0_).astype(np.float32))
PRECUT = radial.copy() if PAR["precut"] else np.zeros(len(bi), bool)     # precut: the breakable set is cut from the start
CUTF = None
if PAR["cut_file"]:                                    # lens_crack.py: crack extension at the knots (cut + wedge normal per bond)
    CUTF = np.load(f"/out/{PAR['cut_file']}"); PRECUT = np.r_[CUTF["cut"], np.zeros(len(bi) - len(CUTF["cut"]), bool)]
# wedge (022 oracle mechanism): every precut interface bond becomes a wedge pair pushing its two ends apart along the
# material normal (force p) while their normal separation < initial + wedge_g0 -> opening of the interface, no cut of sheets
W_on = PAR["wedge_g0"] > 0 and PRECUT.any()
if W_on:
    thb = 0.5 * np.arctan2(s2, c2); nbx, nby = -np.sin(thb), np.cos(thb)
    sgn = np.sign((px[bi] - px[bj]) * nbx + (py[bi] - py[bj]) * nby); sgn[sgn == 0] = 1
    wsel = np.nonzero(PRECUT)[0]; WNX, WNY = (nbx * sgn)[wsel].astype(np.float32), (nby * sgn)[wsel].astype(np.float32)
    if CUTF is not None:                               # wedge normal = the lens normal (crack opening direction)
        WNX, WNY = CUTF["nx"][wsel].astype(np.float32), CUTF["ny"][wsel].astype(np.float32)
    WS0 = ((px[bi[wsel]] - px[bj[wsel]]) * WNX + (py[bi[wsel]] - py[bj[wsel]]) * WNY).astype(np.float32)
    GWI, GWJ, GWNX, GWNY, GWS0 = gpu(bi[wsel].astype(np.int32)), gpu(bj[wsel].astype(np.int32)), gpu(WNX), gpu(WNY), gpu(WS0); NW = len(wsel)
BR = gpu((cut1 | PRECUT).astype(np.uint8)); node_of = cp.empty(P, cp.int32); head = cp.empty((GH, GW), cp.int32); nxt = cp.empty(P, cp.int32)
c1, c2_, c3 = cp.zeros(1, cp.int32), cp.zeros(1, cp.int32), cp.zeros(1, cp.int32)
# 025 VCRACK (user 2026-09-27: "dans le nœud du V, une fois la certitude de passage de concentration de contrainte, force la
# rupture et suis le développement de la contrainte pendant le calcul"). V apexes from vtrace.py (original coordinates).
# Candidates = lattice bonds CROSSING the sheet (>= vc_ang deg from the band tangent: no cut along a sheet) with one end within
# vc_zone_r px of an apex. Every vc_every steps, on the GPU, per V:
#   SCF = max strain of the crossing bonds at the apex (<= vc_apex_r px) / mean positive strain of the zone's crossing bonds;
#   TRIGGER when SCF >= vc_trig on vc_persist consecutive checks (certainty that the concentration passes there) -> forced
#   rupture of the apex crossing bonds whose strain >= the zone mean;
#   PROPAGATION (following the stress): an intact candidate bond touching a node of an already broken vcrack bond breaks when
#   its strain >= vc_prop x the zone mean and it lies within vc_maxdist px of the apex -> the crack advances where the stress
#   concentrates at its own tip. History per check (triggered V, broken bonds, SCF) + break step of every bond.
VC = None
if PAR["vcrack"]:
    import cupyx
    VF = np.load(f"/out/{PAR['vcrack']}"); APX = VF["apex"]; TNODE = cKDTree(np.c_[py, px]); nV = len(APX)
    crossing = np.zeros(Bn, bool); crossing[:Bn0] = cos2[:Bn0] <= np.cos(np.radians(PAR["vc_ang"])) ** 2
    cb, cv, ca, cdist = [], [], [], []
    order = np.argsort(bi[:Bn0], kind="stable"); starts = np.searchsorted(bi[:Bn0][order], np.arange(P + 1))
    order2 = np.argsort(bj[:Bn0], kind="stable"); starts2 = np.searchsorted(bj[:Bn0][order2], np.arange(P + 1))
    for v, a in enumerate(APX):
        nodes = np.array(TNODE.query_ball_point(a, PAR["vc_zone_r"]), int)
        if not len(nodes):
            continue
        bl = np.unique(np.concatenate([order[starts[n]:starts[n + 1]] for n in nodes] + [order2[starts2[n]:starts2[n + 1]] for n in nodes]))
        bl = bl[crossing[bl]]
        mid = np.c_[(py[bi[bl]] + py[bj[bl]]) / 2, (px[bi[bl]] + px[bj[bl]]) / 2]; dd = np.hypot(*(mid - a).T)
        cb.append(bl); cv.append(np.full(len(bl), v)); ca.append(dd <= PAR["vc_apex_r"]); cdist.append(dd)
    cb, cv, ca, cdist = (np.concatenate(z) for z in (cb, cv, ca, cdist))
    VC = {"b": gpu(cb.astype(np.int64)), "v": gpu(cv.astype(np.int64)), "apex": gpu(ca), "near": gpu(cdist <= PAR["vc_maxdist"]),
          "bi": gpu(bi[cb].astype(np.int64)), "bj": gpu(bj[cb].astype(np.int64)), "L0": gpu(L0[cb].astype(np.float32)),
          "persist": cp.zeros(nV, cp.int32), "trig": cp.zeros(nV, bool), "trig_step": -np.ones(nV, int), "dmg": cp.zeros(P, bool),
          "bstep": -np.ones(Bn0, np.int32), "hist": [], "nV": nV, "cand": int(len(cb)), "apex_bonds": int(ca.sum())}
    print(json.dumps({"vcrack": PAR["vcrack"], "V": nV, "candidate_entries": int(len(cb)), "apex_bonds": int(ca.sum())}), flush=True)


def vcrack_check(step):
    b, v = VC["b"], VC["v"]; nV = VC["nV"]
    dx = (PX0[VC["bj"]] + UX[VC["bj"]]) - (PX0[VC["bi"]] + UX[VC["bi"]]); dy = (PY0[VC["bj"]] + UY[VC["bj"]]) - (PY0[VC["bi"]] + UY[VC["bi"]])
    st = (cp.sqrt(dx * dx + dy * dy) - VC["L0"]) / VC["L0"]; intact = BR[b] == 0; sp = cp.where(intact, cp.maximum(st, 0), 0)
    zmean = cp.bincount(v, weights=sp, minlength=nV) / cp.maximum(cp.bincount(v, weights=intact.astype(cp.float64), minlength=nV), 1)
    amax = cp.zeros(nV, cp.float64); sel = VC["apex"] & intact; cupyx.scatter_max(amax, v[sel], sp[sel].astype(cp.float64))
    scf = amax / cp.maximum(zmean, 1e-9)
    # vc_emin (first test: at step 50 the zone strains are ~1e-4 and any small ratio triggered, the crack then ran away):
    # the apex must ALSO carry the user's rupture strain (2 %), and a propagating bond must reach it
    above = cp.bincount(v, weights=(intact & (sp > amax[v])).astype(cp.float64), minlength=nV) / cp.maximum(cp.bincount(v, weights=intact.astype(cp.float64), minlength=nV), 1)
    # vc_rank (calibrated on the no-rupture diagnostic runs vd_V / vd_Vctrl, Z4224): the ratio SCF is as high at random points
    # as at V apexes; "certainty" = the apex max strain is in the top vc_rank share of its zone AND >= vc_emin, persistently
    cond = (above <= PAR["vc_rank"]) if PAR["vc_rank"] is not None else (scf >= PAR["vc_trig"])
    VC["persist"] = cp.where(cond & (amax >= PAR["vc_emin"]), VC["persist"] + 1, 0)
    new = (VC["persist"] >= PAR["vc_persist"]) & ~VC["trig"]; VC["trig"] |= new
    brk = VC["apex"] & intact & new[v] & (sp >= cp.maximum(zmean[v], PAR["vc_emin"]))                                   # forced rupture at the apex
    brk |= intact & VC["trig"][v] & VC["near"] & (VC["dmg"][VC["bi"]] | VC["dmg"][VC["bj"]]) & (sp >= cp.maximum(PAR["vc_prop"] * zmean[v], PAR["vc_emin"]))   # follow the stress
    idx = cp.unique(b[brk])
    if len(idx):
        BR[idx] = 1; VC["dmg"][VC["bi"][brk]] = True; VC["dmg"][VC["bj"][brk]] = True
        ih = idx.get(); ih = ih[ih < Bn0]; VC["bstep"][ih[VC["bstep"][ih] < 0]] = step
    # per-V diagnostics (calibration of the trigger): SCF, apex max strain, rank of the apex max in its zone (share of zone
    # bonds stretched more), mean of the 3 largest apex strains / zone p90
    VC.setdefault("diag", []).append(np.stack([np.full(nV, step, float), scf.get(), amax.get(), above.get()]))
    nh = new.get(); VC["trig_step"][nh] = step; tr = VC["trig"].get(); sc = scf.get()
    VC["hist"].append({"step": step, "triggered": int(tr.sum()), "new_broken": int(len(idx)), "broken_total": int((VC["bstep"] >= 0).sum()),
                       "scf_p50": float(np.median(sc)), "scf_p90": float(np.percentile(sc, 90)), "scf_ge_trig": int((sc >= PAR["vc_trig"]).sum()),
                       "zone_mean_strain_p50": float(np.median(zmean.get()))})
# node mass (025 mass_gray): m_p = MNUM x max(grey_p, 88) / median matter grey (multiplying factor = grey / median grey; mean mass
# ~ unchanged); it drives the inertia and the centrifugal force (denser = brighter pixels pulled harder)
GMED = float(np.median(g[ys, xs]))
# grey floored at the matter threshold (88): 5 % of the matter pixels are darker (small voids filled by the cleaning, down
# to grey 4) and a near-zero mass made the explicit integration unstable (first 025 mass run: KE 1.2e7 at step 20)
MASSF = (np.maximum(g[ys, xs].astype(np.float64), 88.0) / GMED) if PAR["mass_gray"] else np.ones(P)
MP = gpu((MNUM * MASSF).astype(np.float32))
FIX = gpu(((r0 < R_ANCHOR) | floating).astype(np.uint8)); WC = gpu((~floating).astype(np.float32)); gp, gb = ((P + 255) // 256,), ((Bn + 255) // 256,)


def snapshot(step, ux, uy):
    brk_ = BR.get().astype(bool)[:Bn0]
    dm, dimg, _ = render(ys, xs, ux, uy, g, m, VL, bi[:Bn0], bj[:Bn0], inter[:Bn0] | brk_)                 # weak interface bonds not drawn (display only)
    yy, xx = np.nonzero(dm); b = (max(yy.min() - 20, 0), min(yy.max() + 20, N), max(xx.min() - 20, 0), min(xx.max() + 20, N))
    if not PAR["images"]:
        return dm, dimg
    fig, ax = plt.subplots(figsize=(14, 12)); ax.imshow(dimg[b[0]:b[1], b[2]:b[3]], cmap="gray", vmin=0, vmax=255); ax.axis("off")
    ax.set_title(f"025 {NAME} pas {step}: k_n {PAR['kn']}, interfaces x{PAR['kif']}, beta {PAR['beta']} — élastique sans rupture")
    fig.savefig(f"{IMG}/EN_COURS.png", dpi=45, bbox_inches="tight"); plt.close(fig); return dm, dimg


if PAR["save_steps"]:                                  # bond data for stress_field.py (force law F = k (L - L0) / L0 along the bond)
    np.savez_compressed(f"{OUT}/bonds.npz", bi=bi, bj=bj, L0=L0, k=k_.astype(np.float32), px=px, py=py, n_lattice=Bn0)
# 025 FLUID (user 2026-09-27: "rajoute un fluide dans tout le vide, qui est aussi centrifugé, il a une pression qui peut
# augmenter et il peut se diffuser"). Compressible Darcy fluid on the contact grid (every cell, void AND matter pores):
#   content c per cell; porosity phi = 1 in the void, fl_phim where matter sits (3x3-smoothed occupancy of the nodes, updated
#   with the cell lists); pressure p = K (c / phi - 1): a void squeezed by the matter (phi drops) -> the pressure rises;
#   flux between neighbour cells a -> b: J = D kappa [(p_a - p_b) + rho_f a_c . (x_b - x_a)], kappa = 1 in the void, fl_kapm in
#   matter (min of the two cells), a_c = the SAME centrifugal acceleration as the matter (om2 r): the fluid is centrifuged too;
#   start: hydrostatic centrifugal equilibrium p0 = rho_f om2 r^2 / 2 (spinning fluid at rest in the rotating frame);
#   closed container (no flux through the grid edges); explicit update once per mechanics step (stability D dt K 4 < 1);
#   force on every matter node: -grad p (1 px^2): buoyancy + push of a pressurised void on the sheets around it.
FL = None
if PAR["fluid"]:
    OMF = float(OM2 * M_ / MNUM); RHO = float(PAR["fl_rho"] * MNUM); K_ = float(PAR["fl_K"]); Ddt = float(PAR["fl_D"] * DT)
    assert Ddt * K_ * 4 < 1.0 and PAR["fl_kapm"] <= PAR["fl_phim"], "fluid explicit update unstable"
    gy_, gx_ = cp.mgrid[0:GH, 0:GW].astype(cp.float32); RX = gx_ + GX0_ - XC; RY = gy_ + GY0_ - YC
    P0 = 0.5 * RHO * OMF * (RX ** 2 + RY ** 2)
    FL = {"p0": P0, "c": None, "phi": None, "kap": None, "hist": []}


def fluid_occupancy():
    cnt = cp.bincount(node_of[node_of >= 0], minlength=GH * GW)[:GH * GW].reshape(GH, GW).astype(cp.float32)
    import cupyx.scipy.ndimage as cndi
    # first test (3x3 box occupancy, K 1): the rounded node positions flip cells between matter and void -> pressure noise of
    # order K / phi_m, far above the centrifugal force per node (~0.004): Gaussian occupancy (sigma fl_sig) + softer fluid
    occ = cp.minimum(cndi.gaussian_filter(cnt, PAR["fl_sig"]), 1.0)
    FL["phi"] = 1.0 - (1.0 - PAR["fl_phim"]) * occ; FL["kap"] = 1.0 - (1.0 - PAR["fl_kapm"]) * occ
    if FL["c"] is None:
        FL["c"] = FL["phi"] * (1.0 + FL["p0"] / K_)


def fluid_step():
    c, phi, kap = FL["c"], FL["phi"], FL["kap"]; p = K_ * (c / phi - 1.0)
    Jx = Ddt * cp.minimum(kap[:, :-1], kap[:, 1:]) * ((p[:, :-1] - p[:, 1:]) + RHO * OMF * 0.5 * (RX[:, :-1] + RX[:, 1:]))
    Jy = Ddt * cp.minimum(kap[:-1, :], kap[1:, :]) * ((p[:-1, :] - p[1:, :]) + RHO * OMF * 0.5 * (RY[:-1, :] + RY[1:, :]))
    c[:, :-1] -= Jx; c[:, 1:] += Jx; c[:-1, :] -= Jy; c[1:, :] += Jy
    gpx = cp.zeros_like(p); gpy = cp.zeros_like(p); gpx[:, 1:-1] = 0.5 * (p[:, 2:] - p[:, :-2]); gpy[1:-1, :] = 0.5 * (p[2:, :] - p[:-2, :])
    ok = node_of >= 0; idx = cp.where(ok, node_of, 0)
    AX[:] -= cp.where(ok, gpx.ravel()[idx], 0); AY[:] -= cp.where(ok, gpy.ravel()[idx], 0)
    return p


hist, ke_prev, u_prev = [], 0.0, np.zeros((2, P)); cp.cuda.Device().synchronize(); T_LOOP0 = time.time(); T_DIAG = 0.0
for step in range(1, PAR["steps"] + 1):
    if (step - 1) % PAR["lists_every"] == 0:
        f_node(gp, (256,), (GPX0, GPY0, UX, UY, np.int32(P), np.int32(GW), np.int32(GH), node_of))
        head.fill(-1); f_lists(gp, (256,), (node_of, np.int32(P), head, nxt))
    AX.fill(0); AY.fill(0)
    f_bonds(gb, (256,), (BI, BJ, L0g, KB, EN, CD, BR, PX0, PY0, UX, UY, VX, VY, np.int32(Bn), np.float32(FCRIT), AX, AY, c1))
    f_contact(gp, (256,), (GPX0, GPY0, UX, UY, node_of, head, nxt, np.int32(P), np.int32(GW), np.int32(GH), np.float32(KC), AX, AY, c2_))
    if FL is not None:
        if (step - 1) % PAR["lists_every"] == 0:
            fluid_occupancy()
        PF = fluid_step()
        if step % 2000 == 0 or step == PAR["steps"]:
            pe = PF - FL["p0"]; FL["hist"].append({"step": step, "p_max": float(PF.max()), "excess_p_p99": float(cp.percentile(pe, 99)), "excess_p_p1": float(cp.percentile(pe, 1)),
                                                   "fluid_total": float(FL["c"].sum())})
    if W_on:
        f_wedge(((NW + 255) // 256,), (256,), (GWI, GWJ, GWNX, GWNY, GWS0, np.int32(NW), np.float32(PAR["wedge_g0"]), np.float32(PAR["wedge_p"]), PX0, PY0, UX, UY, AX, AY))
    f_intw(gp, (256,), (PX0, PY0, UX, UY, VX, VY, AX, AY, FIX, WC, MP, np.float32(OM2 * M_ / MNUM), np.float32(XC), np.float32(YC),
                        np.int32(P), np.float32(DT), np.float32(N - 1), np.float32(N - 1), c3, np.float32(PAR["cdrag"])))
    if PAR["save_steps"] and step in PAR["save_steps"]:          # 025 stress field: displacement snapshots during loading
        np.savez_compressed(f"{OUT}/snap_{step}.npz", ux=UX.get(), uy=UY.get(), broken=BR.get().astype(bool))
    if VC is not None and step % PAR["vc_every"] == 0:
        vcrack_check(step)
    if step % 10 == 0:
        ke = float((VX * VX + VY * VY).sum())
        if ke < ke_prev: VX.fill(0); VY.fill(0)
        ke_prev = ke if ke >= ke_prev else 0.0
    if step % 2000 == 0 or step == PAR["steps"]:
        cp.cuda.Device().synchronize(); _td = time.time()
        ux, uy = UX.get().astype(float), UY.get().astype(float); d = np.hypot(ux, uy)
        dchg = float(np.hypot(ux - u_prev[0], uy - u_prev[1]).mean()); u_prev = np.stack([ux, uy])
        dx = (px[bj] + ux[bj]) - (px[bi] + ux[bi]); dy = (py[bj] + uy[bj]) - (py[bi] + uy[bi]); strain = (np.hypot(dx, dy) - L0) / np.maximum(L0, 1e-6)
        h = {"step": step, "ke": float((VX * VX + VY * VY).sum()), "disp_mean_um": float(d.mean() * 9.596), "disp_change_last_px": dchg,
             "strain_tang_p99": float(np.percentile(strain[tang], 99)), "strain_norm_p99": float(np.percentile(strain[~tang], 99)),
             "strain_if_p99": float(np.percentile(strain[inter], 99)) if inter.any() else 0.0, "clamped": int(c3.get()[0])}
        hist.append(h); print(json.dumps(h), flush=True)
        if step % 4000 == 0:
            snapshot(step, ux, uy)
        T_DIAG += time.time() - _td
cp.cuda.Device().synchronize(); T_LOOP = time.time() - T_LOOP0 - T_DIAG
ux, uy = UX.get().astype(float), UY.get().astype(float)
DEVZ = 4222 <= Z <= 4226 and os.path.isdir("/vt/DEV")      # DEV metrics only when the annotation is mounted on /vt
if DEVZ:
    G = load_gt(Z); PR = m1_prepare(G, m, ys, xs); M1 = m1v3(PR, ux, uy)
else:
    G, M1 = {}, {"ALL": None}
dm, dimg = snapshot(PAR["steps"], ux, uy); _, _, RR = render(ys, xs, ux, uy, g, m, VL, bi[:Bn0], bj[:Bn0], inter[:Bn0] | BR.get().astype(bool)[:Bn0])
conv = hist[-1]["disp_change_last_px"] / max(np.hypot(ux, uy).mean(), 1e-9)
BRF = BR.get().astype(bool)
res = {"params": PAR, "rupture": {"cut_1px_bonds": int(cut1.sum()), "cut_1px_share": float(cut1.mean()), "radial_breakable": int(radial.sum()),
       "radial_broken": int((BRF & ~cut1).sum()), "precut": int(PRECUT.sum()), "tangential_bonds_broken": int((BRF & (np.sqrt(np.clip(1 - cos2, 0, 1)) < np.cos(np.radians(45)))).sum()), "radial_broken_share_of_bonds": float((BRF & ~cut1).mean()), "broken_total_share": float(BRF.mean())}, "mass": {"grey_median": GMED, "factor_min_p50_max": [float(MASSF.min()), float(np.median(MASSF)), float(MASSF.max())], "mass_mean": float(MNUM * MASSF.mean())}, "timing": {"mechanics_loop_s": T_LOOP, "ms_per_step": 1000 * T_LOOP / PAR["steps"], "grid": [GH, GW], "grid_share": GH * GW / N / N,
       "points_outside_grid_end": int(((np.rint(py + UY.get()) - GY0_ < 0) | (np.rint(py + UY.get()) - GY0_ >= GH) | (np.rint(px + UX.get()) - GX0_ < 0) | (np.rint(px + UX.get()) - GX0_ >= GW)).sum())},
       "fragments": frag_info, "history": hist, "converged_rel_change_last2000": conv, "M1": M1["ALL"], "M1_zones": {zn: M1[zn] for zn in ZONES} if DEVZ else None,
       "render_collided_share": RR["collided_points"] / P, "interface_bonds": int(inter.sum()), "time_s": time.time() - t00}
if FL is not None:
    PF = K_ * (FL["c"] / FL["phi"] - 1.0); res["fluid"] = {"history": FL["hist"]}
    np.savez_compressed(f"{OUT}/fluid.npz", p=PF.get().astype(np.float32), p0=FL["p0"].get().astype(np.float32), phi=FL["phi"].get().astype(np.float32), origin=[GY0_, GX0_])
if VC is not None:
    res["vcrack"] = {"V": VC["nV"], "candidate_entries": VC["cand"], "apex_bonds": VC["apex_bonds"], "triggered": int(VC["trig"].get().sum()),
                     "trigger_step_p50": float(np.median(VC["trig_step"][VC["trig_step"] >= 0])) if (VC["trig_step"] >= 0).any() else None,
                     "broken": int((VC["bstep"] >= 0).sum()), "broken_along_sheet_lt30": int(((VC["bstep"] >= 0) & (cos2[:Bn0] > np.cos(np.radians(30)) ** 2)).sum()),
                     "history": VC["hist"]}
    np.savez_compressed(f"{OUT}/vcrack.npz", bstep=VC["bstep"], trig_step=VC["trig_step"], diag=np.array(VC.get("diag", [])))
json.dump(res, open(f"{OUT}/result.json", "w"), indent=1); np.savez_compressed(f"{OUT}/final.npz", ux=ux, uy=uy, broken=BR.get().astype(bool)[:Bn0], cut1=cut1[:Bn0])
if not (PAR["images"] and DEVZ):
    print(json.dumps({"M1": M1["ALL"], "conv": conv, "collided": res["render_collided_share"]}), flush=True); print("DONE", NAME, time.time() - t00, flush=True); sys.exit(0)
fig, ax = plt.subplots(2, 4, figsize=(24, 12))
for k, (zn, gz) in enumerate(G.items()):
    w = gz["vt"]["window"]; y0, x0, s = w["y0"], w["x0"], w["size_px"]; sel = (ys >= y0) & (ys < y0 + s) & (xs >= x0) & (xs < x0 + s)
    dy, dx = np.rint(np.median(uy[sel])).astype(int), np.rint(np.median(ux[sel])).astype(int)
    ax[0, k].imshow(g[y0:y0 + s, x0:x0 + s], cmap="gray", vmin=0, vmax=255); ax[0, k].set_title(f"{zn} — original")
    ax[1, k].imshow(dimg[y0 + dy:y0 + dy + s, x0 + dx:x0 + dx + s], cmap="gray", vmin=0, vmax=255)
    ax[1, k].set_title(f"centrifugé — ouverture propre {100 * M1[zn]['M1_clean']:.0f} %")
    for a_ in ax[:, k]: a_.axis("off")
# full-resolution images (no display aliasing): deformed slice and original, same crop
yy, xx = np.nonzero(dm | m); b = (max(yy.min() - 20, 0), min(yy.max() + 20, N), max(xx.min() - 20, 0), min(xx.max() + 20, N))
plt.imsave(f"{IMG}/coupe_centrifugee_pleine_resolution.png", dimg[b[0]:b[1], b[2]:b[3]], cmap="gray", vmin=0, vmax=255)
plt.imsave(f"{IMG}/coupe_originale_pleine_resolution.png", np.where(m, g, 0)[b[0]:b[1], b[2]:b[3]].astype(np.float32), cmap="gray", vmin=0, vmax=255)
fig2, ax2 = plt.subplots(1, 2, figsize=(30, 14))
ax2[0].imshow(np.where(m, g, 0)[b[0]:b[1], b[2]:b[3]], cmap="gray", vmin=0, vmax=255, interpolation="antialiased"); ax2[0].set_title(f"Z{Z} original")
ax2[1].imshow(dimg[b[0]:b[1], b[2]:b[3]], cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
ax2[1].set_title(f"Z{Z} centrifugé, élastique sans rupture (k_n {PAR['kn']}, interfaces x{PAR['kif']}, beta {PAR['beta']}), déplacement moyen {np.hypot(ux, uy).mean() * 9.596:.0f} µm")
[a.axis("off") for a in ax2]; fig2.savefig(f"{IMG}/coupe_complete_avant_apres.png", dpi=110, bbox_inches="tight"); plt.close(fig2)
fig.suptitle(f"025 {NAME}: élastique sans rupture, k_n {PAR['kn']}, interfaces x{PAR['kif']}, beta {PAR['beta']} — M1 clean {M1['ALL']['M1_clean']:.3f} (serrées {M1['ALL']['M1_clean_tight']:.3f}), croisées {M1['ALL']['M1_crossed']:.3f}")
fig.savefig(f"{IMG}/zones.png", dpi=55, bbox_inches="tight"); plt.close(fig)
print(json.dumps({"M1": M1["ALL"], "conv": conv, "collided": res["render_collided_share"]}), flush=True)
print("DONE", NAME, time.time() - t00, flush=True)
