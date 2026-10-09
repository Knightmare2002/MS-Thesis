# damage3d: multi-view 3D damage mapping (geometric pipeline)

`damage3d` is the geometric part of the thesis pipeline:

```
2D probability maps (6 classes) -> Metashape cameras -> visible 3D points -> multi-view fusion -> 3D outputs
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
- `label_mask`, a bitmask: crack = 1, spalling = 2, corrosion = 4, moisture = 8, delamination = 16, surface = 32;
- `n_classes`;
- the six `score_*` fields.

In CloudCompare, switch the active scalar field to `score_<class>` or `label_mask`, or filter `n_classes >= 2`, to inspect overlaps.

Thresholds (`--thresholds 0.5` or `crack=0.6,spalling=0.7,...`) and `--min-views` only affect export. You can change them with `--resume`, and the run is re-exported without being re-accumulated.
- **Synthetic runs:** 0.5 is a declared **test** threshold.
- **Real runs:** use thresholds selected on DACL10K validation for the chosen model. Never calibrate them on the external bridge.

## 6. Outputs (`<run>/outputs/`)

- **`fused_points.ply`**, binary little endian. Fields:
  - `x y z`: float32, identical to the source PLY, so it overlays the original cloud;
  - `red green blue`;
  - `state`, `label_mask`, `n_classes`, `n_views`;
  - `score_<class>` x6 (NaN when unobserved) and `views_<class>` x6;
  - `source_index`: the record index in the original PLY.
- **`fused/chunk_XXXXXX.npz`**: machine-readable arrays `source_index, xyz, score(n,6), views(n,6), votes(n,6), n_views, label_mask, state, rgb`. `classes.json` gives the channel order.
- **`fused_points.csv`**: only written when the exported points are at most `--csv-max-points`.
- **`per_camera_stats.csv`**, per camera:
  - point counts: candidates, in_front, in_image, occluded, valid_views;
  - positive views per class.
- **`summary.json`**: synthetic flag, frame decision, occlusion tolerance, state and label counts, legend, warnings.

`--export-scope all|observed|damage` controls which points are written.

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
  - `fused_points.ply` with `--export-scope all`: about 33 GB;
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
