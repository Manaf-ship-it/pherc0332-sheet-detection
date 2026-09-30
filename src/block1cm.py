"""025 1-cm block (Z3703-4744, 1042 slices, L2 9.596 um, downloaded 2026-09-27 from the same public volume): points the FROZEN
slice_prep at the new block without modifying it (its SAMPLE / Z0 globals are read at call time). Import this module
BEFORE using slice_prep.prep when the container mounts the 1-cm block at /sample1cm.
Also: matter_only(z) = the matter mask of prep() only (same rules: grey >= 88, 12-px cleaning), without the structure tensor."""
import numpy as np
from scipy import ndimage as ndi
import slice_prep

Z0, Z1 = 3703, 4744
slice_prep.SAMPLE = "/sample1cm/PHerc0332_L2_Z3703-4744_uint8.npy"; slice_prep.Z0 = Z0


def matter_only(z, vol=None):
    vol = np.load(slice_prep.SAMPLE, mmap_mode="r") if vol is None else vol
    gray = np.ascontiguousarray(vol[z - Z0]); m_raw = gray >= slice_prep.OTSU
    lab, _ = ndi.label(m_raw, structure=np.ones((3, 3), bool)); small_m = (np.bincount(lab.ravel()) < slice_prep.NOISE)[lab] & m_raw
    m1 = m_raw & ~small_m; labv, _ = ndi.label(~m1); small_v = (np.bincount(labv.ravel()) < slice_prep.NOISE)[labv] & ~m1
    return gray, m1 | small_v
