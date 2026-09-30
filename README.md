# Mechanical separation of papyrus sheets for sheet detection — PHerc0332

In dense regions of a carbonized scroll, neighbouring papyrus sheets touch each other: at 9.6 µm the matter of a whole
cross-section forms essentially **one connected block** (94 % of the matter in a single component on PHerc0332 slice Z4224).
A line tracer then has to guess where one sheet ends and the next begins, and its errors are sheet jumps.

This repository explores a **mechanical aid**: the CT slice is turned into a lattice of material points (one per matter voxel,
bonds to the 8 neighbours, contact), sheets are made stiff along their tangent, and the scroll is loaded by a **centrifugal force**
around its axis. The goal is to make the fused matter **separate where sheets are only glued** (stress knots, thin necks), so that
sheet detection has distinct pieces to follow. Every effect below is measured against a **same-cost random control**.

Data: public PHerc0332 scan `20251211183505-2.399um-0.2m-78keV-masked.zarr`, level 2 (9.596 µm), slices Z4222–4226, with
human annotations of sure sheet traces in four 2.9 mm windows (evaluation only). Hardware: one laptop GPU (RTX 4070 Laptop).

## Results

### 1. Stiff sheets make the centrifuge separate instead of stretch

Orthotropic lattice: stiffness k_t along the local sheet tangent (structure tensor), k_n = 0.3 across; centrifugal load β = 3000,
medium drag 0.06, grey-level mass, no rupture. The frozen line tracer (C1) is run on the deformed slice and its lines mapped back
to the original coordinates (Z4224, 11 sub-pixel perturbation draws, frozen evaluator):

| sheet stiffness | stretch along sheets p50 / p99 | sheet-jump errors (mean ± sd, 11 draws) | S1 (followed sheet length) |
|---|---|---|---|
| k_t = 1 (isotropic-like) | 6.0 % / 59 % | 42.7 ± 5.2 | 0.748 |
| k_t = 30 | 0.8 % / 14 % | 29.5 ± 3.7 | 0.797 |
| **k_t = 100** | **0.3 % / 8 %** | **24.8 ± 3.2** (11/11 draws better than k_t = 1) | **0.801** |

Stiff sheets divide the stretching by 20, move the load to the interfaces between sheets (inter-sheet peeling stress ×7),
halve collisions and crossings, and cut the tracing errors on the deformed state by 42 %.
(`cf_elastic.py`, `results/stiff_sheets_c1_errors_z4224.json`, `docs/figures/stiff_sheets.png`)

### 2. The centrifuge finds the stress knots where sheets are pinned

On the loaded state without rupture, a bond oriented across the sheets is a **knot bond** if its strain is in the top 0.5 % and at
least 3 times the mean across-sheet strain of its surroundings (8 px); knots = groups of ≥ 5 such nodes (2.5 % of the matter).
Test: allow rupture (across-sheet bonds, 2 % strain) **only** in the knots, versus the same shapes placed at random in the matter.

| rupture allowed in | breakable bonds | bonds that actually break | crossed sheets (DEV) |
|---|---|---|---|
| nowhere (reference) | — | — | 1.42 % |
| **mechanical knots** | 25,568 | **15,485 (61 %)** | **0.93 %** |
| random zones, same shapes (control) | 22,230 | 255 (1 %) | 1.46 % |

The knots are exactly the places that are held under tension: unlocking them removes a third of the crossed sheets.
(`knots.py`, `results/knots_*.json`, `docs/figures/stress_knots_*.png`)

### 3. Tangent welding + neck breaking splits the fused block into sheet packets

1. Matter groups of the raw slice: 698 connected groups, one holding 94.1 % of the matter (`etape1_groupes.py`).
2. **Tangent welding**: gaps of ≤ 4 px along the local sheet tangent are welded (211,683 px): 255 groups, the largest 97.8 %
   (`etape2_soudure.py`).
3. **Necks**: structures ≤ 4 px wide joining two thick parts (morphological opening, disk of radius 2); their 66,376 lattice bonds
   are the only breakable ones (`necks.py`).
4. **Centrifuge** (stiff sheets as in §1): a neck bond breaks at 2 % strain.

| Z4224, welded sample | broken bonds | connected groups | largest group | groups ≥ 1000 px |
|---|---|---|---|---|
| centrifuge, no rupture | 0 | 255 | 97.8 % | 8 |
| **centrifuge, necks breakable** | 12,152 | **466** | **30 %** | **77** |
| control: same number of random breakable bonds | 6,558 | 255 | 97.8 % | 8 |

Only the necks open the block: it falls apart into distinct **packets of sheets** (`docs/figures/neck_breaking_matter_groups.png`),
while the random control does not separate anything. (`results/neck_breaking_groups_and_detection_z4224.json`,
`results/centrifuge_w4_*.json`)

## Current limits (next steps)

- The pieces obtained in §3 are **packets of several sheets**, not yet individual sheets; half of the broken neck bonds lie along
  a sheet (thin parts of the sheet itself), so part of the separation cuts sheets.
- **Line tracing on the deformed states is not yet better than on the undeformed slice**: on Z4224 the original slice gives
  13.9 errors (11 draws) versus 24.8 for the best deformed state (§1), and 27 on the neck-broken state versus 13.7 for the
  original (3 draws). The tracer was tuned on undeformed data; mapping the separated packets back to the original geometry
  and tracing inside each packet is the next step.
- Evaluation covers five consecutive slices and four 2.9 mm windows of one scroll region, with one annotator.

## Files

- `src/cf_elastic.py`: 2D centrifuge solver (GPU, cupy): orthotropic stiff-sheet lattice, contact, fragments, damping, options for
  rupture zones, breakable-bond masks, pre-cut bonds, welded matter. `fsi_kernel_002.py`, `solid007.py`: lattice / contact kernels.
- `src/knots.py` (stress knots), `knots027.py` (lens-tip knots on the deformed state), `etape1_groupes.py`, `etape2_soudure.py`,
  `necks.py`, `etape4_eval.py` (groups through intact bonds + tracing on the deformed state), `state_lines.py`.
- Frozen tracer and evaluator: `slice_prep.py`, `predetect.py`, `bands019.py`, `metrics3.py`, `render3.py`, `vt_eval.py`,
  `gt_metrics.py` (sha256 in `FROZEN_CORE.sha`).
- `src/c1_slice.py`: frozen tracer on one slice (its lines give the sheet frame of the lattice).
- `tools/download_pherc0332_block.py`: extraction of an L2 block from the public volume (HTTP byte ranges).
- `Dockerfile`: CUDA 12.2 + the exact Python package versions used.

Scripts use the mount points of the study (`/sample` = L2 block, `/vt` = annotations, `/out` = outputs); each script documents its
inputs and outputs in its header. Example:
```
docker build -t pherc0332-mech .
docker run --gpus all -v <L2 block dir>:/sample:ro -v <out>:/out pherc0332-mech python3 c1_slice.py 4224   # frozen tracer lines (material frame)
docker run --gpus all -v <L2 block dir>:/sample:ro -v <out>:/out pherc0332-mech \
  python3 cf_elastic.py kt100 '{"z": 4224, "kn": 0.3, "beta": 3000, "steps": 40000, "crop_margin": 200, "lists_every": 10, "mass_gray": true, "cdrag": 0.06, "kt": 100, "dt": 0.1, "images": false}'
```

## License

MIT (see `LICENSE`).
