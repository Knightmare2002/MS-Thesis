# P1ML + damage3d — bridge damage segmentation and 3D mapping

Release of the MS-thesis pipeline (Politecnico di Torino / SENSORS group, UMass Lowell).
It contains:

1. **P1ML**: a multilabel semantic-segmentation model trained on DACL10K. It has 6 independent
   sigmoid channels (`crack, spalling, corrosion, moisture, delamination, surface`). The
   `delamination` channel is closed (no reliable DACL10K source class), so all reported
   metrics are computed on the other **5 classes**.
2. **Calibration**: per-class thresholds (scripts/17), threshold + minimum-area operating
   points (scripts/20), Platt scaling of the probabilities (scripts/27) and an optional
   weighted ensemble of models (scripts/28).
3. **damage3d**: it projects the calibrated per-image probabilities onto the photogrammetric
   point cloud of a bridge (Agisoft Metashape `cameras.xml`) and votes over the views.

All paths are relative to the folder that contains this README (the *release root*).
Nothing has to be edited after unzipping.

## Direction A experimental extension

The Direction A patch adds class-dependent resolvability inside a marginalized
observation likelihood, with spatially separated fitting/calibration/testing,
same-observation baselines and reproducible evaluation. The original mean fusion
remains the default; no detector retraining or changes to unrelated experiments
are required.

See `damage3d/README.md`, section “Direction A”, for the complete change log,
input schemas, executable PowerShell commands and limitations. The derivation,
identification correction and publication-novelty audit are in
`damage3d/DIRECTION_A_METHOD.md`; software validation does not establish real
bridge accuracy or guarantee novelty.

---

## Layout

```
<release root>/
├── README.md                 this file
├── requirements.txt          deep-learning stack (PyTorch CUDA 11.8 wheels)
├── run_paths.ps1             optional PowerShell helper with the standard paths
├── configs/                  config_p1ml.yaml (T9), config_p1ml_t12.yaml (T12), config_p1ml_t13.yaml (T13)
├── src/                      library: data, models, losses, engine, metrics, calibration
├── scripts/                  numbered entry points (see "Scripts")
├── damage3d/                 2D -> 3D mapping package (python -m damage3d), tests/, requirements.txt
├── outputs/
│   ├── stats/                cached DACL10K class weights (training only)
│   ├── runs/<run>/           trained models: config.yaml, best.pt, calibration files, metrics
│   └── ensembles/<name>/     ensemble.json (if an ensemble is shipped)
├── bridge_model/             3D project: cameras.xml, point cloud, mesh, marker file, images/
└── datasets/dacl10k/         empty: put DACL10K here only to evaluate or retrain
```

Shipped runs (only the ones present in `outputs/runs/` are available):

| Run | Architecture | Init |
|---|---|---|
| `p1ml_t9_unetplusplus_effb4_512_imagenet` | U-Net++ / EfficientNet-B4 | ImageNet |
| `p1ml_t12_upernet_convnext_base_512_ade20k` | UPerNet / ConvNeXt-B | ADE20K |
| `p1ml_t13_upernet_convnext_base_512_ade20k_reg` | UPerNet / ConvNeXt-B, regularised | ADE20K |

`best.pt` contains only the inference weights (EMA). Training cannot be resumed from it;
retraining from scratch is possible (see "Training").

---

## Installation

Requirements: Python **3.11 or 3.12** (64 bit), an NVIDIA GPU with ≥ 8 GB VRAM and a driver
supporting CUDA 11.8 (tested on an RTX A2000 12 GB, Windows 11). CPU-only inference works but is slow.

Windows (PowerShell), from the release root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt            # PyTorch 2.7.1 + cu118 from the PyTorch index
pip install -r damage3d/requirements.txt   # open3d, pytest
```

Linux / macOS:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r damage3d/requirements.txt
```

CPU only / different CUDA: install `torch==2.7.1` and `torchvision==0.22.1` from the matching
index of <https://pytorch.org/get-started/locally/> first, then remove the two torch lines
from `requirements.txt` and install the rest.

Windows long paths: run directories are long. If you get `FileNotFoundError` on an existing
file, unzip the release close to the drive root (e.g. `C:\p1ml`) or enable long paths
(`LongPathsEnabled = 1` in `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`).

Internet: T12 and T13 build the UPerNet-ConvNeXt architecture from the Hugging Face hub
(`smp-hub/upernet-convnext-base`), downloaded once into the Hugging Face cache, also for inference.
On an offline machine, run any T12/T13 command once while online, or copy `~/.cache/huggingface`.

### Check the installation

```powershell
python scripts/13_smoke_test_multilabel.py   # synthetic data, ~1-2 min, must end with PASS
python -m pytest -q damage3d/tests           # 2D->3D geometry and I/O tests
```

---

## Inference on a bridge (main use case)

Put the photogrammetric project in `bridge_model/`:

| File | Content |
|---|---|
| `cameras.xml` | Metashape camera export (intrinsics + poses) |
| `bridge_3dpc_point_cloud.ply` | dense point cloud |
| `model_medium_quality.obj` | mesh (optional, occlusion/visualisation) |
| `marker_reference.txt` | marker coordinates (optional) |
| `images/` | the original photos referenced by `cameras.xml` |

Different names or locations can be passed with `--project-root`, `--images-dir`,
`--cameras-xml`, `--point-cloud`, or the environment variable `DAMAGE3D_PROJECT_ROOT`.
Relative paths given to `damage3d` and `scripts/23` are resolved **from the project root
(`bridge_model/`)**, hence the `../outputs/...` below.

### 1. Probability maps (scripts/23)

```powershell
$RUN = "outputs/runs/p1ml_t13_upernet_convnext_base_512_ade20k_reg"
python scripts/23_infer_probability_maps.py --run-dir $RUN --project-root bridge_model `
    --calibration auto --dry-run                 # lists the cameras, writes nothing
python scripts/23_infer_probability_maps.py --run-dir $RUN --project-root bridge_model `
    --calibration auto --save-overlays
```

Output: `bridge_model/damage3d_maps/<run>/<image>.npz` (+ overlays). Useful options:
`--limit-images N` or `--camera-list file.txt` (subset), `--downsample 2` (default; map
resolution), `--batch-size` (lower it on out-of-memory), `--dtype uint8` (smaller maps).
`--calibration auto` reads `<run>/eval_curves/probability_calibration.json` and checks that
it was fitted on the same `best.pt`.

### 2. Projection on the point cloud (damage3d)

Single model:

```powershell
python -m damage3d --project-root bridge_model --probability-source files `
    --probabilities-dir damage3d_maps/p1ml_t13_upernet_convnext_base_512_ade20k_reg `
    --calibration ../outputs/runs/p1ml_t13_upernet_convnext_base_512_ade20k_reg/eval_curves/probability_calibration.json `
    --min-views 2 --export-scope observed --output-dir damage3d_runs/t13 --dry-run
```

Remove `--dry-run` to run; add `--resume` to continue an interrupted run.

Ensemble (run step 1 for every member first, with the same `--downsample`):

```powershell
python -m damage3d --project-root bridge_model --probability-source ensemble `
    --ensemble ../outputs/ensembles/T12_T13/ensemble.json `
    --member-maps T12=damage3d_maps/p1ml_t12_upernet_convnext_base_512_ade20k `
    --member-maps T13=damage3d_maps/p1ml_t13_upernet_convnext_base_512_ade20k_reg `
    --min-views 2 --export-scope observed --output-dir damage3d_runs/ens_t12_t13
```

Output in `bridge_model/damage3d_runs/<name>/`: per-point scores and labels (PLY/CSV) and a
run manifest. See `damage3d/README.md` for every option and output file.

---

## Reproducing the DACL10K evaluation

Download DACL10K v2 (`dacl10k_v2_devphase.zip`) from the
[dacl10k-toolkit](https://github.com/phiyodr/dacl10k-toolkit) and unzip it so that
`datasets/dacl10k/images/{train,validation}/*.jpg` and
`datasets/dacl10k/annotations/{train,validation}/*.json` exist. The validation split is the
evaluation set (the official test labels are not public).

Per run (`$RUN` as above; same order as in the thesis):

```powershell
$CAL = "$RUN/eval_multilabel_sliding_calibrated"
$PPD = "$RUN/eval_multilabel_sliding_postproc_objrec05"
$PP  = @("--min-object-recall","0.5","--thresholds","0.3","0.4","0.5","0.6","0.7","0.8","0.9",
         "--min-area-fractions","0","0.0003","0.001","0.003","0.005","0.01","0.02")

python scripts/12_evaluate_dacl10k_multilabel_sliding.py --run-dir $RUN --output-dir $CAL  # sliding window, global sweep
python scripts/17_calibrate_multilabel_thresholds.py   --run-dir $RUN --eval-dir $CAL      # per-class thresholds
python scripts/20_calibrate_threshold_min_area.py      --run-dir $RUN @PP --output-dir $PPD # 2-fold tau x min-area
python scripts/25_pr_curves_confusion.py --run-dir $RUN --operating-points $PPD/operating_points_frozen.json  # PR, AP, histograms
python scripts/27_fit_probability_calibration.py       --run-dir $RUN                      # Platt scaling
```

Ensemble of calibrated members (after 25 + 27 on every member):

```powershell
python scripts/28_ensemble_calibrate_thresholds.py `
    --member name=T12 run_dir=outputs/runs/p1ml_t12_upernet_convnext_base_512_ade20k weight=1 `
    --member name=T13 run_dir=outputs/runs/p1ml_t13_upernet_convnext_base_512_ade20k_reg weight=1 `
    --output-dir outputs/ensembles/T12_T13
```

Test-time augmentation (optional, ~6× slower): `python scripts/29_make_tta_run.py --run-dir $RUN --hflip`
creates `<run>_tta_<tag>`, which is evaluated with the same commands.

Shipped metric files: `<run>/eval_multilabel_sliding_calibrated/*.csv`,
`<run>/eval_multilabel_sliding_postproc_objrec05/metrics_postproc_*.csv`,
`<run>/eval_curves/average_precision.csv`. Re-running a script overwrites them.

---

## Training

```powershell
python scripts/11_train_dacl10k_multilabel.py --config configs/config_p1ml.yaml     --run-name p1ml_t9_unetplusplus_effb4_512_imagenet
python scripts/11_train_dacl10k_multilabel.py --config configs/config_p1ml_t12.yaml --run-name p1ml_t12_upernet_convnext_base_512_ade20k
python scripts/11_train_dacl10k_multilabel.py --config configs/config_p1ml_t13.yaml --run-name p1ml_t13_upernet_convnext_base_512_ade20k_reg
```

Any config key can be overridden with `--set key=value` (e.g. `--set train.batch_size=2`).
Use a new `--run-name` to avoid overwriting a shipped run. The class weights are read from
`outputs/stats/dacl10k_multilabel_class_weights.json` (recomputed on the train split if missing).
Then run the evaluation chain above.

---

## Scripts

| Script | Purpose |
|---|---|
| `11_train_dacl10k_multilabel.py` | training (patch sampling, BCE + Dice, EMA, best checkpoint on 5-class macro Dice) |
| `12_evaluate_dacl10k_multilabel_sliding.py` | sliding-window evaluation, global threshold sweep |
| `13_smoke_test_multilabel.py` | installation test on synthetic data |
| `17_calibrate_multilabel_thresholds.py` | per-class thresholds on validation |
| `20_calibrate_threshold_min_area.py` | 2-fold threshold + min-area operating points |
| `22_map_damage_to_3d.py` | thin wrapper around `python -m damage3d` |
| `23_infer_probability_maps.py` | probability maps for the bridge photos |
| `25_pr_curves_confusion.py` | PR curves, AP, inter-class confusion, histograms for 27 |
| `27_fit_probability_calibration.py` | Platt scaling per class |
| `28_ensemble_calibrate_thresholds.py` | weighted ensemble of Platt-calibrated members |
| `29_make_tta_run.py` | test-time-augmentation variant of a run |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | `--batch-size 1` (scripts/23), or `--set train.batch_size=2` (training) |
| `... was fitted on checkpoint ...` | `best.pt` and `probability_calibration.json` come from different trainings: re-run scripts/25 + 27 |
| `Maps of member 'X' not found` | pass `--member-maps X=<folder>` |
| T12/T13 hang or fail at model creation | no internet for the Hugging Face download (see Installation) |
| `FileNotFoundError` on an existing file (Windows) | long paths (see Installation) |
| `Expected datasets/dacl10k/images/...` | DACL10K missing: only needed for evaluation/training |

## References

| Topic | Link | Use |
|---|---|---|
| DACL10K dataset and toolkit | <https://github.com/phiyodr/dacl10k-toolkit> | dataset download and label definitions |
| segmentation_models_pytorch | <https://github.com/qubvel-org/segmentation_models.pytorch> | U-Net++ and UPerNet implementations |
| PyTorch installation | <https://pytorch.org/get-started/locally/> | wheels for other CUDA versions or CPU |
| Open3D | <https://www.open3d.org/> | point-cloud I/O used by damage3d |
