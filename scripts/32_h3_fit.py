"""Orchestrates H3 resolvability fit: provenance validation, fusion_params, and resolvability_fit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from damage3d.run_state import sha256_file


def validate_inputs(cal_path: Path, obs_dir: Path, regions_path: Path, fit_path: Path) -> None:
    meta_path = obs_dir / "observations" / "meta.json"
    for f in (cal_path, meta_path, regions_path, fit_path):
        if not f.is_file():
            raise FileNotFoundError(f"File richiesto non trovato: {f}")

    fit_cfg = json.loads(fit_path.read_text(encoding="utf-8"))
    t7_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cal_hash = sha256_file(cal_path).lower()

    provider = t7_meta.get("provider", {})
    if provider.get("type") != "files+platt":
        raise RuntimeError(f"T7 non usa il provider files+platt atteso (trovato: {provider.get('type')}).")

    fit_hash = str(fit_cfg.get("calibration_sha256", "")).lower()
    t7_hash = str(provider.get("calibration", {}).get("sha256", "")).lower()
    if cal_hash != fit_hash or cal_hash != t7_hash:
        raise RuntimeError("La calibrazione non coincide tra file locale, fit_config e T7. Non avviare il fit.")

    print("[1/3] Validazione file e hash SHA-256 completata.", flush=True)


def run_step(cmd: list[str], step_name: str) -> None:
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"{step_name} non completato (exit code {res.returncode}).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--fit-config", type=Path, required=True)
    ap.add_argument("--buffer", type=float, default=1.478)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    validate_inputs(args.calibration, args.obs, args.regions, args.fit_config)
    args.out.mkdir(parents=True, exist_ok=True)

    base_params = args.out / "base_params.json"
    res_params = args.out / "resolvability_params.json"

    print("[2/3] Generazione base_params.json (damage3d.fusion_params)...", flush=True)
    run_step([
        sys.executable, "-m", "damage3d.fusion_params",
        "--calibration", str(args.calibration),
        "--out", str(base_params),
    ], "fusion_params")

    print("[3/3] Esecuzione fit di risolvibilita (damage3d.resolvability_fit)...", flush=True)
    run_step([
        sys.executable, "-m", "damage3d.resolvability_fit",
        "--run", str(args.obs),
        "--regions", str(args.regions),
        "--buffer", str(args.buffer),
        "--params", str(base_params),
        "--config", str(args.fit_config),
        "--skip-unsupported-classes",
        "--bootstrap", str(args.bootstrap),
        "--seed", str(args.seed),
        "--out", str(res_params),
    ], "Fit di risolvibilita")

    result = json.loads(res_params.read_text(encoding="utf-8"))
    print("\n--- Excluded Classes ---", flush=True)
    print(json.dumps(result.get("excluded_classes"), indent=2), flush=True)
    print("\n--- Fit Diagnostics ---", flush=True)
    print(json.dumps(result.get("fit_diagnostics"), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())