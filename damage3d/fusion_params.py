"""Write the ``--fusion-params`` JSON of the llr fusions from a scripts/27 calibration.

The Platt-calibrated probabilities refer to the prevalence of the calibration
set: rho_c = n_positive_pixels / (n_positive_pixels + n_negative_pixels), read
from ``diagnostics`` of ``probability_calibration.json``. Individually calibrated
ensembles are deliberately refused: their mean is not necessarily calibrated.

    python -m damage3d.fusion_params --calibration outputs/runs/<run>/eval_curves/probability_calibration.json \\
        --out fusion_params_llr.json
    python -m damage3d.fusion_params --calibration calibration.json \\
        --pod pod_fit.json --out fusion_params_censored.json

``--pod``: JSON {class: {"g50": g, "beta": b}} with pi_c(g) = sigmoid(-(log g - log g50) / beta)
(detectability fit, g in source units per MAP pixel); without it the file serves
``--fusion llr`` only. ``--tempering``: JSON {class: alpha_c}, alpha_c in (0, 1]
(default 1 = untempered naive Bayes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classes import DAMAGE_CLASSES


def calibration_priors(path: Path) -> dict[str, float]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    diag = d.get("diagnostics", {})
    out = {}
    for c in DAMAGE_CLASSES:
        if c not in diag or "n_positive_pixels" not in diag[c]:
            raise RuntimeError(f"{path}: no pixel counts for '{c}' in diagnostics (scripts/27 JSON expected).")
        pos, neg = int(diag[c]["n_positive_pixels"]), int(diag[c]["n_negative_pixels"])
        if pos <= 0 or neg <= 0:
            raise ValueError(f"{c}: calibration must contain positive and negative pixels.")
        out[c] = pos / (pos + neg)
    return out


def build(rho: dict[str, float], pod: dict | None, tempering: dict | None, gsd: str, source: str) -> dict:
    classes = {}
    for c in DAMAGE_CLASSES:
        entry = {"rho": rho[c], "alpha": float((tempering or {}).get(c, 1.0))}
        if pod is not None and c in pod:
            entry.update(g50=float(pod[c]["g50"]), beta=float(pod[c]["beta"]))
        classes[c] = entry
    return {"source": source, "gsd": gsd, "gsd_units": "source_units_per_map_px",
            "p_clip": 1e-4, "min_cos": 0.0872, "classes": classes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--calibration", type=Path, help="scripts/27 probability_calibration.json")
    src.add_argument("--ensemble", type=Path, help="legacy argument; explicitly rejected for likelihood inference")
    ap.add_argument("--pod", type=Path, help="JSON {class: {g50, beta}} for censored_llr")
    ap.add_argument("--tempering", type=Path, help="JSON {class: alpha_c}, alpha_c in (0, 1]")
    ap.add_argument("--gsd", choices=("effective", "frontal"), default="effective")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.calibration is not None:
        priors, source = calibration_priors(args.calibration), str(args.calibration)
    else:
        raise ValueError("An ensemble of individually calibrated scores is not necessarily calibrated. "
                         "Use a single model for the likelihood experiment.")
    pod = json.loads(args.pod.read_text(encoding="utf-8")) if args.pod else None
    tempering = json.loads(args.tempering.read_text(encoding="utf-8")) if args.tempering else None
    data = build(priors, pod, tempering, args.gsd, source)
    from .run_state import sha256_file
    data["calibration_sha256"] = sha256_file(args.calibration)
    from .detectability import FusionModel  # validate before writing
    FusionModel.from_dict("censored_llr" if pod else "llr", data)
    from .experiment_data import write_json
    write_json(args.out, data)
    print(json.dumps({c: round(v, 6) for c, v in priors.items()}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
