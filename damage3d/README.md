# damage3d: multi-view 3D damage mapping (geometric pipeline)

`damage3d` is the geometric part of the thesis pipeline:

```
2D probability maps (6 model channels; 5 mapped, delamination closed) -> Metashape cameras -> visible 3D points -> multi-view fusion -> 3D outputs
```

**The deep-learning model is not part of this package.** It is represented by a
`ProbabilityProvider` (`damage3d/providers.py`). This README covers two providers:
`synthetic` (test patterns, not an AI prediction) and `files` (maps in `.npz`
format, which the future model will write). When the model is ready, only the
maps change. Projection, visibility, fusion and export stay as they are.

The primary output is where each damage type is located across the 3D structure.
This package does not estimate metric extent.

---

## 1. Folder layout (inside the MS-Thesis repository)

```
MS-Thesis/
├── damage3d/                       # NEW
│   ├── __init__.py, __main__.py    # python -m damage3d
│   ├── cli.py                      # argparse CLI (all options below)
│   ├── paths.py                    # project root + default file names
│   ├── classes.py                  # class order imported from src/data/class_mapping.py, colors, states
│   ├── metashape_xml.py            # cameras/sensors/adjusted calibration/chunk transform, camera selection
│   ├── camera_model.py             # verified Metashape projection + radial fold-over guard + frustum test
│   ├── frames.py                   # PLY/OBJ frame vs chunk frame (auto check with the marker)
│   ├── ply_io.py                   # PLY header parsing, read-only memmap, reproducible sampling, PLY writer
│   ├── mesh_visibility.py          # OBJ occlusion test with Open3D ray casting
│   ├── providers.py                # ProbabilityProvider, SyntheticProvider, FileProvider, save_probability_npz
│   ├── sampling.py                 # nearest / bilinear sampling of the six channels
│   ├── fusion.py                   # on-disk per-chunk accumulators, fused scores, labels, states, colors
│   ├── run_state.py                # manifest/fingerprint, atomic writes, resume bookkeeping
│   ├── export.py                   # fused PLY, NPZ shards, CSV, per-camera stats, summary.json
│   ├── pipeline.py                 # orchestration
│   ├── requirements.txt
│   ├── README.md
│   └── tests/                      # pytest suite (synthetic Metashape-like project + real-data regression)
└── scripts/20_map_damage_to_3d.py  # NEW thin wrapper (same CLI), consistent with the numbered scripts
```

Large inputs and outputs stay in `bridge_model` and are never committed.

## 2. Installation

Use the environment you already use to run the `bridge_model/scripts` Open3D
scripts. If you use the thesis environment instead, check that Open3D has a
wheel for its Python version.

```powershell
cd C:\path\to\MS-Thesis
python -m pip install -r damage3d\requirements.txt
```

## 3. Inputs and what has been verified

Default file names come from the existing, working scripts. Relative paths,
whether default or passed on the CLI, are resolved against the project root,
not the working directory. The project root comes from `--project-root`, then
`$env:DAMAGE3D_PROJECT_ROOT`, then `C:\Users\Samuele_Caruso\Desktop\bridge_model`.

| Input | Default | Origin |
|---|---|---|
| cameras XML | `cameras.xml` | `inspect_camera_geometry.py`, `verify_marker_projection.py` |
| point cloud | `bridge_3dpc_point_cloud.ply` | `check_obj_ply_alignment.py` (binary LE, `x y z nx ny nz` float + `class` uchar, no RGB) |
| mesh | `model_medium_quality.obj` | `check_obj_ply_alignment.py`, `check_surface_alignment.py` |
| marker reference | `marker_reference.txt` | `verify_marker_projection.py` (`chunk_x/y/z`, `pixel_u/v`) |
| synthetic mask | `{stem}_mask_test.png` | `project_test_mask.py` (grayscale, white >= 128) |
| photos | `--images-dir`, default project root | **not verified**; only used for an optional size/EXIF check |

What has been verified already:
- Projection formula (`project_chunk_point`) for a marker on DSC02150: 0.000000 px difference against Metashape. The marker was given in chunk coordinates.
- The OBJ and PLY share the same frame.

What has **not** been verified, and how the pipeline handles it:
- **PLY frame vs chunk frame.** `project_test_mask.py` assumed PLY = chunk frame without checking. `inspect_camera_geometry.py` looked for `<transform><matrix>`, but Metashape usually writes `<rotation>/<translation>/<scale>`. So a chunk transform could have been reported as "not present". With `--point-frame auto` (the default):
  - if there is no chunk transform, or it is the identity, the two frames coincide;
  - otherwise the marker is placed in both candidate frames and its distance to the OBJ is measured. A frame is accepted only if one distance is at least 10x smaller than the other. If the test is inconclusive, or `marker_reference.txt` is missing, the run stops.
- **Pixel convention.** Metashape uses a corner convention (manual, Appendix C): the centre of pixel (i, j) is at (i+0.5, j+0.5). The prototype indexed the mask with `rint(u)`. The pipeline uses `floor(u)` (nearest) or bilinear interpolation between pixel centres. This can shift the pixel read by up to half a pixel compared with the prototype. The projection itself is unchanged.

## 4. Geometry

1. **Frames.** PLY/OBJ coordinates are in the "source frame". The pipeline converts them to the chunk frame (identity, or the inverse of the chunk transform) before projecting. Occlusion is computed in the source frame, in PLY/OBJ units.
2. **Projection.** The verified equations (`camera_model.py`): chunk to camera with `inv(<camera><transform>)`, then `x = X/Z, y = Y/Z`, then radial k1..k3 plus tangential p1, p2 (Metashape ordering), then `u = W/2 + cx + f x'`, `v = H/2 + cy + f y'`. Non-zero `k4, b1, b2, p3, p4` are rejected, as in the prototype, because they were not part of the verified model.
3. **Rejections**, applied in this order:
   - **behind the camera:** `Z <= --min-depth` (default 0);
   - **radial guard:** undistorted radius larger than `r_limit`, which prevents polynomial fold-over;
   - **outside the image:** outside `[m, W-m) x [m, H-m)`, where `m = --border-margin-px`;
   - **occluded by the mesh**;
   - **invalid provider pixel.**
4. **Occlusion** (`--occlusion mesh`). A ray is cast from the camera centre towards the point. The point is visible if `t_hit >= d - (abs_tol + rel_tol * d)`. A miss counts as visible.
   - **Units:** PLY/OBJ units. They are metric only if the export is scaled, and the XML alone does not establish that.
   - **Default `abs_tol`:** `max(P95 point-to-mesh distance on <= 20k selected points, 1e-4 x mesh diagonal)`. It is written in `manifest.json`. Override it with `--occlusion-tolerance`.
   - **Precision:** Open3D is float32. Mesh and rays are shifted by the mesh bounding-box centre.
   - **Cost:** one BVH ray per (point inside the image, camera), processed in batches of 2M rays.
   - **Not reliable for:**
     - mesh holes (hidden points counted as visible);
     - points more than `tol` behind the surface (visible points dropped, which is the conservative case);
     - elements thinner than `tol` (back-face points counted as visible);
     - grazing rays;
     - any area where the medium-quality mesh departs from the cloud.
   - `--occlusion none` disables the test and adds a warning to the summary.
5. **Sampling** (`sampling.py`). The map may be a downscaled version of the image with the same aspect ratio (1% tolerance). Coordinates are scaled by `Wm/W` and `Hm/H`.
   - `bilinear` (default): interpolates between pixel centres; indices are clamped (edge replication) in the outer half-pixel band.
   - `nearest`: reads pixel `floor()`.
   - NaN in a channel means that class is invalid for this view. A `False` in the `valid` mask means the whole view is invalid.

## 5. Fusion, labels and states

For each point i and class c, counting valid views only (`fusion.py`):
- `score = sum(p) / count`: the mean probability, NaN if the class has no valid view;
- `views_c = count`;
- `votes_c` = number of views with `p >= --vote-threshold`;
- `n_views` = number of geometrically valid views.

A class label is positive when `views_c >= --min-views` and `score_c >= threshold_c`. Each class is decided independently, so several labels can be true on the same point. There is no softmax and no argmax.

| state | value | meaning | RGB |
|---|---|---|---|
| not_observed | 0 | no valid view. **Never shown as healthy.** | dark blue (40, 70, 160) |
| insufficient_views | 1 | seen, but some class has fewer than `min_views` valid views | dark grey (90, 90, 90) |
| observed_no_damage | 2 | every class has at least `min_views` views and no label | light grey (200, 200, 200) |
| damage | 3 | at least one positive label | colour of the dominant class |

Class colours:
- crack: red (228, 26, 28);
- spalling: orange (255, 127, 0);
- corrosion: brown (166, 86, 40);
- moisture: cyan (0, 190, 255);
- delamination: purple (152, 78, 163);
- surface: green (77, 175, 74).

**Points with more than one class.** The RGB colour is the positive class with the largest `score/threshold` ratio. It is only a display choice. All the positive classes are kept in:
- `label_mask`, a bitmask: crack = 1, spalling = 2, corrosion = 4, moisture = 8, surface = 16 (delamination closed since 2026-10-26: never mapped; outputs produced before that date used delamination = 16, surface = 32);
- `n_classes`;
- the six `score_*` fields.

In CloudCompare, switch the active scalar field to `score_<class>` or `label_mask`, or filter `n_classes >= 2`, to inspect overlaps.

Thresholds (`--thresholds 0.5` or `crack=0.6,spalling=0.7,...`) and `--min-views` only affect export. You can change them with `--resume`, and the run is re-exported without being re-accumulated.
- **Synthetic runs:** 0.5 is a declared **test** threshold.
- **Real runs:** use thresholds selected on DACL10K validation for the chosen model. Never calibrate them on the external bridge.

## Direction A: resolution-conditioned observation likelihood

This experimental extension is based on commit `186819950939369ad0d83321ee1a389d46781bb8`.
It implements the primary Direction A model, its estimation workflow, controlled
baselines and evaluation, not a verified bridge-performance or publication claim.
Read `DIRECTION_A_METHOD.md` for the derivation, prior-art audit and assumptions.
The mean path remains the default; the extension does not retrain the detector.

### Observation model and changes to the earlier proposal

For each of the five active classes:
\[
\lambda_{ivc}=\operatorname{logit}p_{ivc}-\operatorname{logit}\rho_c,\quad
\pi_c(g)=\operatorname{sigmoid}((\log g_c^{50}-\log g)/\beta_c),
\]
\[
\widetilde\lambda_{ivc}=\log(1-\pi_c+\pi_c e^{\lambda_{ivc}}),\quad
s_{ic}=\operatorname{sigmoid}\{\operatorname{logit}b_c^{prior}
+\alpha_c\sum_v\widetilde\lambda_{ivc}\}.
\]
The implementation uses `logaddexp` for numerical stability. `rho` is the measured
prevalence in the detector's Platt-calibration set; optional `base_rate` is the target
prior and defaults to `rho`. Do not infer either from T. `--fusion llr` omits the
mixture; `--fusion censored_llr` uses it. Missing/invalid per-class curves are errors,
not implicit defaults. Legacy `kappa` is rejected: the report uses log-GSD width
`beta`, not its reciprocal.

- **Evidence interpretation:** a negative contribution means `p < rho`, not
  necessarily `p < detection_threshold`. Both evidence signs approach zero as
  `pi -> 0`; the method does not unconditionally preserve every positive.
- **Identification:** measured recall is `q(g) = background_fpr +
  (resolved_tpr-background_fpr)*pi(g)`, not `pi(g)`. Two externally fixed anchors
  are required; four unanchored curve parameters are never fitted together.
- **Calibration:** a calibrated ensemble average is not guaranteed calibrated.
  Real likelihood runs require a single `files+platt` provider, matching calibration
  hash and prevalence. Mean/ensemble legacy inference is unaffected.
- **Geometry:** `g = (Z*s/f) * map_scale / sqrt(max(abs_cos,min_cos))`.
  The default floor is `min_cos=0.0872`, approximately cos(85 degrees). `map_scale`
  is measured `W/Wm`, multiplied by `--inference-scale` only for an additional
  inference resize not already counted. Native sliding-window inference needs 1.
  This scalar pinhole approximation is not an exact distortion Jacobian or crack width.
- **Tempering:** class-specific alpha in `(0,1]` regularizes evidence on C_cal;
  it is not a pairwise camera-correlation estimator. The uncensored and
  pi-weighted-LLR comparators receive their own fitted alpha, on the same labels.
- **Uncertainty:** the likelihood is a working model. Pooled DACL10K calibration
  may not represent fully resolved positives, and transfer/background-shift
  assumptions require experimental checks. Do not call its output a certified POD.
- **Optional terms:** primary `omega=1`; no invented quality weight or anisotropic
  crack-direction correction. `min_views` remains an export/coverage policy, not
  the new observation model.

### Added and modified implementation

The following file inventory is the change log. Existing geometry, occlusion and
sampling algorithms are reused, not replaced.

- **`detectability.py`, `fusion_params.py`:** validated class parameters, calibrated
  prior extraction, stable mixture/posterior, explicit units and immutable JSON output.
- **`observations.py`:** atomic schema-3 point-view logs, original PLY row IDs,
  camera IDs, projected pixels, camera depth, Euclidean source-frame range,
  frontal GSD, unsigned incidence, effective map scale and five sampled probabilities.
  Each record is 54 bytes. NaN channel means invalid, not no damage.
- **`pipeline.py`, `cli.py`:** `--fusion`, `--fusion-params`, `--inference-scale`,
  `--observation-log`, `--observation-subset`; source-frame incidence and range;
  strict calibration/normal checks; versioned resume fingerprints. Logging happens
  before accumulator commit, so replay overwrites a shard rather than duplicating it.
- **`fusion.py`, `export.py`:** optional evidence accumulator and posterior export;
  likelihood shards also retain `score_mean` and `llr`. Main PLY schema is preserved.
  Default likelihood decision threshold is 0.5, not a transferred detector threshold.
- **`experiment_data.py`:** sorted unique original IDs, class order, unknown labels,
  positive surface areas, region-level splits and spatial buffer validation;
  completed-batch checks, a sealed shard inventory and SHA-256 binding of every observation shard,
  regions, config, parameters and downstream scores. Files are not overwritten.
- **`resolvability_fit.py`:** leave-view/group-out pseudo-positive fitting on C_fit,
  equal region/point mass, anchored Bernoulli MLE, multistart bounded optimization,
  identifiability checks, empirical bins and optional region-bootstrap intervals.
- **`tempering_fit.py`, `freeze_direction_a.py`:** C_cal-only NLL minimization;
  independent comparator tempering; per-method/class F1 thresholds on a declared
  0.01:0.01:0.99 grid, highest-threshold tie break, no tuning on T.
- **`fusion_benchmark.py`:** same-observation offline comparisons, nested global
  camera subsets, optional label-assisted per-point/class view oracle. This is
  explicitly not the best single physical camera for an entire region.
- **`experiment_metrics.py`, `evaluate_direction_a.py`:** area-weighted AP/F1/IoU,
  NLL/Brier, coverage, common-support comparisons, paired region-bootstrap delta
  intervals, frozen GSD/incidence bins and descriptive area-weighted bin slopes.
- **`crack_axis_metrics.py`:** tolerance IoU for supplied, uniformly sampled 3D
  medial axes. No automatic skeletonization or crack-width estimate.
- **`scripts/29_direction_a_degradation.py`, `prepare_direction_a.py`:** frozen-model
  RGB downsampling/re-inference on DACL10K, native-grid evaluation, per-image counts
  and optional native-scale anchor configuration. Merely resizing predictions
  is not the degradation experiment.
- **`synthetic_direction_a.py`:** known-generative and deliberately misspecified
  simulations. These are not semi-synthetic image experiments.
- **`roi_views.py`:** includes the previously proposed ROI coverage/GSD/Labelme
  helper absent from this main branch. It prepares views and annotation inputs,
  not independent GT. It is optional for Direction A fitting.
- **`requirements.txt`, tests:** SciPy >=1.10 for optimization/KD-trees; analytical,
  geometry/logging/resume, fitting/metric, leakage and end-to-end experiment tests.
  `tests/scene.py` adds known surface normals to the synthetic Metashape fixture.

### Required data contract and frozen design

Prepare `regions.npz` with `allow_pickle=False` compatible arrays. The external
annotations must be independent of model output, cover negative surfaces and
confounders, and retain unassessed areas as unknown. Metashape is the intended
3D review/annotation environment.

```text
source_index : (N,) unique nonnegative integer ORIGINAL PLY row IDs
region       : (N,) integer or fixed-width string independent region/block IDs
split        : (N,) strings "C_fit", "C_cal", "T"; each region belongs to one split
xyz          : (N,3) finite source-PLY coordinates, same units as --buffer
area         : (N,) positive physical surface-support weights
labels       : (N,5) -1 unknown, 0 assessed absent, 1 assessed present
classes      : (5,) ["crack","spalling","corrosion","moisture","surface"]
```

Do not use row numbers from a filtered/reordered export as `source_index`. Verify
coordinates against those original rows. Area weights must come from surface
support (for example fixed-area sampling or mesh-associated areas); setting all
weights to 1 is only defensible for independently verified uniform-area sampling.
The buffer checks supplied point coordinates, not continuous polygon boundaries;
ensure full ROI boundaries and neighbourhood/context sizes are separated too.
Region IDs must encode the independent experimental unit, not arbitrary
subdivisions of one correlated patch.

C_fit can have all labels -1. Its fit needs three supported regions per class by
default, hits and misses, at least 50 eligible observations and a >=2x GSD span.
C_cal needs both labels and three supported regions per class by default. These
are software guardrails, not sample-size/power guarantees. If one class fails,
the all-class fit stops: collect more information or explicitly redesign the
class-specific experiment, rather than inventing a `g50`.

From a reviewed CSV with columns `source_index,region,split,x,y,z,area` and the
five class labels, this one-time conversion preserves the contract:

```python
import numpy as np
from damage3d.classes import DAMAGE_CLASSES
d = np.genfromtxt("regions_reviewed.csv", delimiter=",", names=True, dtype=None, encoding="utf8")
d = np.atleast_1d(d)
np.savez_compressed("regions.npz", source_index=d["source_index"], region=d["region"],
    split=d["split"], xyz=np.column_stack([d[k] for k in ("x","y","z")]), area=d["area"],
    labels=np.column_stack([d[c] for c in DAMAGE_CLASSES]), classes=np.array(DAMAGE_CLASSES))
np.save("roi_source_ids.npy", d["source_index"])
```

The code cannot certify annotation independence or surface-area correctness.
Freeze region membership, buffers, camera groups, pseudo-reference choices,
threshold policy, bins and camera-subset seeds before examining T outcomes.

### Executable workflow

Run from the repository root. Set these absolute paths to your actual files;
the example buffer of 0.5 is valid only if you deliberately choose half a source
unit and justify it against geometry and spatial correlation. Helpers resolve
paths from the current directory; the main mapper resolves relative paths from
`--project-root`, so absolute paths avoid ambiguity.

```powershell
$Repo = (Get-Location).Path
$P = "C:\Users\Samuele_Caruso\Desktop\bridge_model"
$Run2D = Join-Path $Repo "outputs\runs\p1ml_t12_upernet_convnext_base_512_ade20k"
$Cal = Join-Path $Run2D "eval_curves\probability_calibration.json"
$Maps = Join-Path $P "damage3d_maps\p1ml_t12_upernet_convnext_base_512_ade20k"
$E = Join-Path $P "direction_a_experiment_01"
$Obs = Join-Path $E "observations_run"
$Regions = Join-Path $E "regions.npz"
$Subset = Join-Path $E "roi_source_ids.npy"
$B = 0.5
New-Item -ItemType Directory -Force $E | Out-Null
python -m pip install -r damage3d\requirements.txt
python -m pytest damage3d\tests -q
```

Place the reviewed ROI files in `$E`. Ensure `$Maps` holds raw probabilities
from `scripts/23_infer_probability_maps.py`, from the checkpoint referenced by
`$Cal`: the mapper applies Platt once. Do not supply already calibrated maps.
Supply the same explicit `--camera-list` to inference and projection if restricting
views; omitting selectors below means all aligned/enabled cameras need maps.

```powershell
# Project once and retain all valid observations for the ROI subset.
# No fitted curves or ROI labels are needed at this stage.
python -m damage3d --project-root $P --probability-source files `
  --probabilities-dir $Maps --calibration $Cal --fusion mean `
  --observation-log --observation-subset $Subset --inference-scale 1 `
  --output-dir $Obs --point-chunk-size 200000 --camera-batch-size 4

# Base priors are measured from the stored Platt-calibration diagnostics.
python -m damage3d.fusion_params --calibration $Cal --out "$E\base_params.json"

# Requires the existing training/inference environment, weights and dataset.
# First use --limit 2 with a separate output as a local GPU smoke test.
python scripts\29_direction_a_degradation.py --run-dir $Run2D `
  --data-root "$Repo\datasets\dacl10k" --split validation `
  --scales 1 2 3 4 5 6 7 8 --out "$E\degradation.json"
python -m damage3d.prepare_direction_a --degradation "$E\degradation.json" `
  --out "$E\fit_config.json"

# Review/freeze fit_config.json BEFORE fitting.
python -m damage3d.resolvability_fit --run $Obs --regions $Regions --buffer $B `
  --params "$E\base_params.json" --config "$E\fit_config.json" `
  --bootstrap 200 --seed 0 --out "$E\resolvability_params.json"

# All methods use the same visibility log; no re-projection.
python -m damage3d.fusion_benchmark --run $Obs --regions $Regions --buffer $B `
  --params "$E\resolvability_params.json" --out "$E\scores_untempered.npz"
python -m damage3d.tempering_fit --scores "$E\scores_untempered.npz" `
  --regions $Regions --buffer $B --params "$E\resolvability_params.json" `
  --out "$E\final_params.json"
python -m damage3d.fusion_benchmark --run $Obs --regions $Regions --buffer $B `
  --params "$E\final_params.json" --out "$E\scores_full.npz"
python -m damage3d.freeze_direction_a --scores "$E\scores_full.npz" `
  --regions $Regions --buffer $B --out "$E\thresholds.json"
```

`fit_config.json` contains measured `resolved_tpr`, `background_fpr`, threshold
and anchor-source provenance for every class. Native-scale DACL10K recall is only
a reference anchor, not proof of a fully resolved observation. The preparer uses
declared defaults `top_k=2`, `min_resolution_ratio=1.25`,
`reference_threshold=max(0.9,detector_threshold)`. They trade reference precision
against support; they are design choices, not measured optima. Perform sensitivity
analysis on C only. Optional `common.camera_groups` maps string camera IDs
from `observations/meta.json` to group names; unlisted cameras are individual groups.
Use groups for overlapping/nearly duplicate frames. The candidate's group is
excluded, then the k finest eligible distinct groups must unanimously be positive.

Prepare `bins.json` with finite strictly increasing `g_edges` in source-units/map-pixel
and `incidence_edges` in degrees, chosen from C/geometry before viewing T metrics.
For example, an explicitly justified metric-scale experiment might use
`{"g_edges":[0,0.001,0.002,0.004,0.008,0.02,1],"incidence_edges":[0,30,60,75,90]}`;
these numbers are illustrative, not measured Lincoln Street bounds.

```powershell
python -m damage3d.evaluate_direction_a --scores "$E\scores_full.npz" `
  --regions $Regions --buffer $B --thresholds "$E\thresholds.json" `
  --bins "$E\bins.json" --bootstrap 1000 --seed 0 --out "$E\metrics_full.json"

# Nested global camera subsets, same seeds across fractions. Never refit on T.
foreach ($Seed in 0,1,2,3,4) {
  foreach ($Fraction in 0.25,0.5,1.0) {
    $Tag = "f${Fraction}_s${Seed}"
    python -m damage3d.fusion_benchmark --run $Obs --regions $Regions --buffer $B `
      --params "$E\final_params.json" --fraction $Fraction --seed $Seed --out "$E\scores_$Tag.npz"
    python -m damage3d.evaluate_direction_a --scores "$E\scores_$Tag.npz" `
      --regions $Regions --buffer $B --thresholds "$E\thresholds.json" --bins "$E\bins.json" `
      --bootstrap 1000 --seed 0 --out "$E\metrics_$Tag.json"
  }
}
```

The full-view score and JSON sidecar must travel together. Each output includes
hashes, selected cameras and parameters. Output names must be new; deliberately
delete only disposable smoke-test outputs, not frozen experiment results.
Resolvability bootstrap intervals condition on fixed anchors and pseudo-positive
selection; they do not include uncertainty in transferred detector rates.

### Baselines, diagnostics and interpretation

The offline `METHODS` are `mean`, `max`, `angle_range`, `distance_range`,
`distance_weighted`, `naive_bayes`, `tempered_bayes`, `censored_untempered`,
`censored`, `pi_weighted_mean`, `pi_weighted_llr`. `angle_range` keeps unsigned
incidence <=50 degrees. Distance range uses Euclidean range below the per-point
median; strict ties can leave no support. Distance weighting uses
`1-(d-min)/(max-min)`; equal distances explicitly receive equal weights. These
are documented adaptations, not a claim of exact reproduction of another codebase.

Evaluation reports each method's area coverage and computes comparisons on the
intersection of valid support for the requested methods. A restrictive geometric
baseline may shrink that intersection: report it and predeclare a second primary
method group with `--methods mean max naive_bayes tempered_bayes censored_untempered censored pi_weighted_mean pi_weighted_llr`.
Never select the group after inspecting its accuracy.

AP is step-integrated precision-recall with tied scores grouped. NLL clips scores
to `[1e-7,1-1e-7]`; Brier does not. Empty sets produce JSON null, not zero. An
all-negative set without predicted positives has undefined F1/IoU. Bootstrap draws
whole regions, with identical draws for candidate/reference. Delta is candidate
minus reference (`max` by default); lower NLL/Brier is better. With too few
independent regions, intervals are not reliable even if many points exist.
Strata use minimum GSD and median incidence per point, not each view's metric;
their slopes are descriptive bin-midpoint regressions, not causal estimates.

For a frontal-only geometry ablation, create `base_params.json` with
`fusion_params --gsd frontal` in a separate experiment, then refit on C_fit and
C_cal. Do not simply change `gsd` after fitting effective-GSD curves. The no-pi
and no-tempering ablations are already included. Optional `fusion_benchmark
--oracle` adds `oracle_best_view`, which uses labels and must only be reported
as a diagnostic; fitting commands reject oracle score files. Evaluate it by
including `oracle_best_view` in `--methods`; its diagnostic threshold is fixed 0.5.

```powershell
# Known-model sanity and deliberate misspecification; no images involved.
python -m damage3d.synthetic_direction_a --seed 0 --n 10000 --views 16 --out "$E\synthetic.json"

# Only after extracting/resampling GT and predicted medial axes uniformly.
# --metres-per-unit must be independently established; 1 below assumes metres.
python -m damage3d.crack_axis_metrics --truth "$E\gt_axes.npy" --prediction "$E\pred_axes.npy" `
  --metres-per-unit 1 --sampling-mm 1 --tolerances-cm 0.5 1 2 4 8 --out "$E\axis_metrics.json"
```

To export the fitted method as a PLY, run the mapper into a NEW directory with
`--fusion censored_llr --fusion-params "$E\final_params.json"` and explicit
`--thresholds` copied from `thresholds.json["censored"]`, keeping the same geometry,
calibration, maps and scale. The mapper does not accept a multi-method threshold
JSON directly. Omitting thresholds gives 0.5 for a diagnostic export only.
In Metashape, overlay source and fused PLY to review localisation; retain the
NPZ class probabilities for quantitative multilabel evaluation.

### Reproducibility, storage and remaining limitations

The main accumulation is chunked. The offline fitter/benchmark filter to ROI IDs
before concatenating observations per point chunk, but use per-point loops;
fitting retains eligible ROI samples and benchmarking retains ROI scores.
They are research/ROI tools, not a benchmarked 550-million-point offline engine.
Logs cost approximately `54 * number_of_valid_point_views` bytes; prefer a
representative, uniformly sampled ROI subset. Hashing every shard adds disk I/O.
`--observation-subset` restricts logging, not projection or the exported cloud.

The inherited mean path and log/no-log equivalence are regression-tested.
Online float32 accumulation and offline float64 recomputation agree to a tested
numerical tolerance, not universally bit-for-bit. Determinism assumes fixed camera
order and software environment. Log schema 1/2 and previous development
likelihood fingerprints are incompatible: regenerate logs; do not resume them.
Ordinary pre-extension mean runs without logs preserve their fingerprint.

Software tests include stable limits/signs, scale/normal geometry, parameter
validation, known-curve recovery with imperfect detector anchors, held-out
reference exclusion, tied AP, paired bootstrap, spatial buffers, provenance,
resumed log replay, and a complete synthetic fitting-to-evaluation chain.
Changing all synthetic T labels leaves fitted C_fit and C_cal parameters unchanged.
The release validation record reports exact test counts and skips.

Real DACL10K degradation inference, real bridge fit/performance, annotation
reliability, cross-structure generalization, GPU runtime and full-cloud
scalability have NOT been established by these tests. The five-class fit must
fail if data are insufficient; software cannot create the missing scientific
evidence. Cross-run/structure transfer is currently rejected by strict provenance:
it requires a separately specified transfer protocol, not bypassing hash checks.
External GT creation, area estimation and medial-axis extraction remain human/
domain-data preparation steps, not silently approximated by model predictions.

## 6. Outputs (`<run>/outputs/`)

- **`fused_points.ply`**, binary little endian. Fields:
  - `x y z`: float32, identical to the source PLY, so it overlays the original cloud;
  - `nx ny nz`: float32, copied unchanged from the source PLY (`--normals auto|source|none`, default `auto` = when the source has them). Without them Metashape and CloudCompare estimate normals with arbitrary sign, which gives inverted shading on some surfaces;
  - `red green blue`;
  - `state`, `label_mask`, `n_classes`, `n_views`;
  - `score_<class>` x6 (NaN when unobserved) and `views_<class>` x6;
  - `source_index`: the record index in the original PLY.
- **`fused/chunk_XXXXXX.npz`**: machine-readable arrays `source_index, xyz, [normal], score(n,5), views(n,5), votes(n,5), n_views, label_mask, state, rgb`. `classes.json` gives the channel order.
- **`fused_points.csv`**: only written when the exported points are at most `--csv-max-points`.
- **`per_camera_stats.csv`**, per camera:
  - point counts: candidates, in_front, in_image, occluded, valid_views;
  - positive views per class.
- **`summary.json`**: synthetic flag, frame decision, occlusion tolerance, state and label counts, legend, warnings.

`--export-scope all|observed|damage` controls which points are written.
`--normals` is an export option: changing it with `--resume` on a finished run only rewrites the outputs (no re-accumulation).
For runs exported before `--normals` existed: `python -m damage3d.add_normals --fused <run>/outputs/fused_points.ply --source <project>/bridge_3dpc_point_cloud.ply` writes `fused_points_normals.ply` next to it (inputs untouched, coordinates checked against the source).

Run-level files:
- `manifest.json` (fingerprint);
- `progress.json`;
- `selected_cameras.csv` (effective selection);
- `selection_indices.npy` (sampled source indices; its SHA-256 is in the manifest);
- `chunks.npz` (chunk bounding spheres);
- `acc/` (accumulators);
- `run.log`.

## 7. Selection rules

- **Camera order.** Only aligned and enabled cameras are selectable. They are sorted with a natural sort on the label stem. `--start-image` and `--end-image` are **0-based positions in this sorted list**.
- **`--end-image` is EXCLUSIVE** (Python slice): `--start-image 10 --end-image 20` selects 10 cameras.
- **Incompatible selectors.** The run is refused (ambiguous) if you combine:
  - `--camera` with `--camera-list`;
  - `--camera` or `--camera-list` with `--start-image` or `--end-image`.
- **`--limit-images N`** is applied last, to the result of whichever selector was used.
- **Errors instead of silent fixes:**
  - an out-of-range position;
  - an unknown, unaligned, disabled or duplicate camera.
- **Output order.** The result always follows the global order and is written to `selected_cameras.csv`.
- **`--max-points N`** takes a reproducible subset without modifying the source file.
  - `random` (default): `default_rng(--seed).choice(N_total, N, replace=False)`, sorted.
  - `stride`: `floor(k * N_total / N)`.
  - Only the pages holding the selected records are read, through a read-only memmap.

## 8. Scalability, RAM and disk

- **PLY access.** The PLY is never loaded in full. Points are processed in chunks (`--point-chunk-size`, default 2,000,000).
- **Memory per chunk.** Every chunk has its own accumulator file (about 50 B per point). Nothing of size N_points x N_photos x 6 is ever built.
- **Camera batches.** Cameras are processed in batches of `--camera-batch-size` (default 4). The probability maps of one batch are kept in RAM: a full-resolution float32 6-channel map is `24 x W x H` bytes, which is why the provider accepts downscaled maps.
- **Chunk culling.** A first pass computes one bounding sphere per chunk. A chunk is skipped for a camera when its sphere lies outside the camera's field-of-view cone (a conservative test). The benefit depends on how spatially coherent the file order of the PLY is. The log reports "N chunks updated" for each batch.
- **Disk estimate for the full cloud** (about 13.76 GB, 25 B per point, so about 550M points):
  - accumulators: about 27.5 GB;
  - `fused_points.ply` with `--export-scope all`: about 30 GB without normals, about 36 GB with them (54 / 66 B per point);
  - shards: similar size;
  - prefer `--export-scope observed` or `damage`. `--dry-run` prints the estimates.
- **Measured in the development sandbox** (2 CPU cores, Linux, synthetic 30M-point PLY, 0.75 GB, 5 cameras, chunk 2M, batch 2): peak RSS 1.84 GB and 131 s wall time, export included.
- **The full 13.76 GB run has NOT been executed or tested.** Its cost scales with (points inside each camera's cone) x (number of cameras).

## 9. Resume

`--resume` continues a run in `--output-dir`.
1. **Fingerprint check.** The stored fingerprint must match, otherwise the run stops and lists the keys that differ. The fingerprint covers:
   - inputs: XML SHA-256, PLY and OBJ size + mtime + head/tail SHA-256, PLY header, marker file;
   - the list of selected cameras;
   - the point-selection SHA-256;
   - frame and occlusion parameters;
   - interpolation;
   - chunk and batch sizes;
   - the provider identity, including every `.npz` map.
2. **Atomic accumulator writes.** Each accumulator file is written to a temporary file, flushed with fsync, then renamed (`os.replace`). It records the last camera batch applied to it.
3. **Resume logic.** When the run restarts, chunks that already contain the current batch are skipped. An interrupted batch is completed without adding any camera twice.
4. **Existing runs are never overwritten.** Without `--resume`, an existing run directory is refused.

On Windows, close `fused_points.ply` in CloudCompare before re-exporting: `os.replace` cannot replace an open file.

## 10. Tests

```powershell
python -m pytest damage3d\tests -q
# includes the real DSC02150 regression when the data are reachable:
$env:DAMAGE3D_PROJECT_ROOT = "C:\Users\Samuele_Caruso\Desktop\bridge_model"
python -m pytest damage3d\tests\test_real_dsc02150.py -v
```

Coverage:
- **XML parsing:** groups, unaligned and disabled cameras, adjusted calibration, chunk transform, rejection of unsupported coefficients.
- **Selection rules**, including every ambiguous combination.
- **Frames:** round trip and the automatic frame decision.
- **Projection:**
  - the vectorised projection against a verbatim copy of the verified formula (< 1e-9 px);
  - the synthetic marker regression;
  - points behind the camera;
  - image bounds with the corner convention;
  - the radial fold-over guard;
  - the conservative chunk culling.
- **Occlusion:** hidden points, tolerance, misses.
- **Providers:** six overlapping channels, the mask provider, the file provider round trip and class-order check.
- **Sampling:** nearest, bilinear, borders, downscaled maps, NaN, valid mask.
- **Fusion:** means, counts, votes, states, `min_views`, dominant colour.
- **End to end (synthetic Metashape-like project):**
  - one camera with few points, producing a PLY readable by Open3D;
  - a camera range;
  - identical results with two chunk sizes and batch sizes (bitwise);
  - interruption then resume without double counting;
  - resume refused when parameters change;
  - dry run writes nothing;
  - missing inputs are reported;
  - the file provider.
- **Real data:** `test_real_dsc02150.py` reproduces `verify_marker_projection.py` on the real XML with the vectorised projector. It is skipped when the data are not reachable.

## 11. PowerShell commands

```powershell
cd C:\path\to\MS-Thesis
$env:DAMAGE3D_PROJECT_ROOT = "C:\Users\Samuele_Caruso\Desktop\bridge_model"

# 0) Dry run: validates inputs, prints chunk transform, selection, PLY header, estimates. Writes nothing.
python -m damage3d --camera DSC02150 --max-points 200000 --dry-run

# 1) One photo, few points, synthetic six-class pattern (NOT an AI prediction)
python -m damage3d --camera DSC02150 --max-points 200000 --output-dir damage3d_runs\t1_dsc02150_regions

# 1b) Same photo with the prototype mask (white -> crack), nearest sampling
python -m damage3d --camera DSC02150 --max-points 200000 --synthetic-pattern mask `
    --synthetic-mask-classes crack --interpolation nearest --output-dir damage3d_runs\t1b_dsc02150_mask

# 2) Small range of photos (positions 0..9; end is exclusive)
python -m damage3d --start-image 0 --end-image 10 --max-points 2000000 --camera-batch-size 5 `
    --output-dir damage3d_runs\t2_range_0_10

# 2b) Resume after an interruption (same command + --resume)
python -m damage3d --start-image 0 --end-image 10 --max-points 2000000 --camera-batch-size 5 `
    --output-dir damage3d_runs\t2_range_0_10 --resume

# 3) Future full run with model maps (<stem>.npz in predictions\<model>); NOT tested
python -m damage3d --probability-source files --probabilities-dir predictions\p1ml_model `
    --point-chunk-size 2000000 --camera-batch-size 4 --min-views 2 `
    --thresholds "crack=<val>,spalling=<val>,corrosion=<val>,moisture=<val>,delamination=<val>,surface=<val>" `
    --export-scope observed --output-dir damage3d_runs\full_p1ml_model
```

Pass `--point-frame chunk` or `--point-frame transformed` explicitly if the automatic frame check is inconclusive.

## 12. Plugging in the model later

The inference script only has to write, for every photo:

```python
from damage3d.providers import save_probability_npz
save_probability_npz(out_dir / f"{stem}.npz", probs)   # probs: (6, H, W) sigmoid outputs, class order of UNIFIED_DAMAGE_CLASSES
```

Requirements on the maps:
- **Pixel frame:** the raw Metashape pixel frame. Do not rotate by EXIF: `cv2.imread` applies the EXIF orientation by default, so use `cv2.IMREAD_IGNORE_ORIENTATION` or Pillow without `exif_transpose`.
- **Downscaling:** allowed if it keeps the aspect ratio.
- **Unreliable pixels:** mark them with NaN or with a `valid` mask.

Then run with `--probability-source files --probabilities-dir <dir>`.

`scripts/23_infer_probability_maps.py` implements this for a P1ML run. It reuses the sliding-window protocol of step 12, the damage3d camera selection (same flags, same order) and reads the photos with `IMREAD_IGNORE_ORIENTATION`:

```powershell
python scripts\23_infer_probability_maps.py --run-dir outputs\runs\<p1ml_run> --start-image 76 --end-image 86
```

Maps go to `<project-root>\damage3d_maps\<run name>\` with a `maps_manifest.json` (checkpoint SHA-256, inference settings, per-map size and timing). Existing maps are skipped; a different setting in the same folder is refused. The script prints the frozen validation thresholds of the run (`eval_multilabel_sliding_calibrated`) as a ready-made `--thresholds` string. Never re-select thresholds on the bridge photos.

Add `--save-overlays` to also write `<maps dir>\overlays\<stem>_overlay.jpg` for visual inspection: the photo, all classes together (mean color where labels overlap, one outline per class) and one panel per class with its threshold and positive-pixel fraction. Overlays are rendered from the stored map, i.e. the exact input of damage3d, using the frozen thresholds (or `--overlay-thresholds`). For maps that already exist only the overlay is written; the model is not re-run on them. Panels are in the raw sensor frame, like the maps.

## 13. Known limitations and missing inputs

- The frame of the real PLY/OBJ is still unknown. It is decided at runtime from `marker_reference.txt` and the OBJ, or with `--point-frame`.
- Only frame cameras are supported (no fisheye or spherical), with f, cx, cy, k1..k3, p1, p2. Rolling-shutter data are ignored, with a warning.
- CRS conversions and export shifts chosen in the Metashape export dialog are not stored in the XML. The marker test is the only check against them.
- The photos folder location is not verified. Photos are not needed by the synthetic or file providers.
- The full 13.76 GB run has not been executed.
