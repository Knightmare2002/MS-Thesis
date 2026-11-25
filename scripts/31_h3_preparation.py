"""Orchestrates H3 preparation: preflight checks, DACL degradation (step 29), and Direction A anchor preparation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from damage3d.classes import DAMAGE_CLASSES
from damage3d.experiment_data import regions
from damage3d.resolvability_fit import unsupported_classes
from damage3d.run_state import sha256_file
from src.data.dacl10k import list_samples


def validate_inputs(regions_path: Path, dacl_root: Path, run_dir: Path, out_dir: Path) -> None:
    required_files = [
        regions_path,
        run_dir / "config.yaml",
        run_dir / "best.pt",
        run_dir / "eval_curves" / "probability_calibration.json",
    ]
    for f in required_files:
        if not f.is_file():
            raise FileNotFoundError(f"File richiesto non trovato: {f}")

    required_dirs = [
        dacl_root / "images" / "validation",
        dacl_root / "annotations" / "validation",
    ]
    for d in required_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Directory DACL richiesta non trovata: {d}")

    for out_file in (out_dir / "degradation.json", out_dir / "fit_config.json"):
        if out_file.exists():
            raise FileExistsError(f"Output gia esistente: {out_file}. Lo script non lo sovrascrive.")


def run_preflight(regions_path: Path, buffer: float, dacl_root: Path, run_dir: Path) -> None:
    print("[1/3] Preflight: classi Lincoln Street, coppie DACL e checkpoint T13", flush=True)
    d = regions(regions_path, buffer)
    excluded = unsupported_classes(d)
    active = [name for name in DAMAGE_CLASSES if name not in excluded]

    print(json.dumps({"active_classes": active, "excluded_classes": excluded}, indent=2), flush=True)
    if not active:
        raise RuntimeError("Nessuna classe supera il controllo C_cal.")

    samples = list_samples(str(dacl_root), "validation")
    print(f"Coppie JPG/JSON DACL validation: {len(samples)}", flush=True)

    cal = json.loads((run_dir / "eval_curves" / "probability_calibration.json").read_text(encoding="utf-8"))
    if cal.get("checkpoint", {}).get("sha256") != sha256_file(run_dir / "best.pt"):
        raise RuntimeError("best.pt non corrisponde alla calibrazione Platt T13.")
    print("Checkpoint e calibrazione T13 corrispondono.", flush=True)


def run_step(cmd: list[str], step_name: str) -> None:
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"{step_name} non completato (exit code {res.returncode}).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--dacl-root", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--buffer", type=float, default=1.478)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    degradation = args.out / "degradation.json"
    fit_config = args.out / "fit_config.json"

    validate_inputs(args.regions, args.dacl_root, args.run_dir, args.out)
    run_preflight(args.regions, args.buffer, args.dacl_root, args.run_dir)

    args.out.mkdir(parents=True, exist_ok=True)

    print("[2/3] Degradazione DACL validation, scale 1-8", flush=True)
    script_29 = Path(__file__).resolve().parent / "29_direction_a_degradation.py"
    run_step([
        sys.executable, str(script_29),
        "--run-dir", str(args.run_dir),
        "--data-root", str(args.dacl_root),
        "--split", "validation",
        "--scales", "1", "2", "3", "4", "5", "6", "7", "8",
        "--out", str(degradation),
    ], "Script 29")

    print("[3/3] Preparazione degli anchor a scala 1", flush=True)
    run_step([
        sys.executable, "-m", "damage3d.prepare_direction_a",
        "--degradation", str(degradation),
        "--out", str(fit_config),
    ], "prepare_direction_a")

    print(f"\nConfigurazione pronta per la revisione: {fit_config}", flush=True)
    print(fit_config.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())