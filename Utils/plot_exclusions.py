import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Modifica questo percorso
FOLDER = Path(r"C:\\Users\\Samuele_Caruso\\OneDrive - UMass Lowell\Desktop\bridge_model\\labeling\\roi_views_13112026_V2\\results")

OUT = FOLDER / "plots_exclusions"
OUT.mkdir(parents=True, exist_ok=True)

cases = {
    "Tutti i blocchi": "h1_2d_transfer_global_median.json",
    "Senza B15": "h1_2d_transfer_global_median_width_omit_15.json",
    "Senza B16 e B20": "h1_2d_transfer_global_median_width_omit_16_20.json",
}

classes = ["crack", "spalling", "corrosion", "moisture", "surface"]

metrics = {
    "precision": "Precision",
    "recall": "Recall",
    "dice": "Dice / F1",
    "iou": "IoU",
    "ap": "AP",
}

reports = {}
for name, filename in cases.items():
    with (FOLDER / filename).open(encoding="utf-8-sig") as f:
        reports[name] = json.load(f)["results"]["all"]

colors = ["#4477AA", "#EEAA33", "#228833"]
labels = [
    f"{name}\n(n={report['n_images']} immagini)"
    for name, report in reports.items()
]

# Un'immagine per classe
x = np.arange(len(metrics))
bar_width = 0.25

for cls in classes:
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, report) in enumerate(reports.items()):
        scores = report["classes"][cls]
        values = [
            scores[m] if scores[m] is not None else np.nan
            for m in metrics
        ]

        bars = ax.bar(
            x + (i - 1) * bar_width,
            values,
            bar_width,
            label=labels[i],
            color=colors[i],
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)

    ax.set_xticks(x, list(metrics.values()))
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Valore della metrica")
    ax.set_title(f"{cls}: confronto esclusioni — tutte le viste")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(OUT / f"{cls}_comparison.png", dpi=200)
    plt.close(fig)

# Media aritmetica dei Dice delle cinque classi
macro_dice = []

for name, report in reports.items():
    values = [report["classes"][c]["dice"] for c in classes]

    if any(v is None for v in values):
        raise ValueError(f"{name}: manca il Dice di almeno una classe")

    macro_dice.append(float(np.mean(values)))
    print(f"{name}: macro-Dice = {macro_dice[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(labels, macro_dice, color=colors, width=0.6)
ax.bar_label(bars, fmt="%.3f", padding=4)

ax.set_ylim(0, 1)
ax.set_ylabel("Macro-Dice (media delle 5 classi)")
ax.set_title("Dice generale — tutte le viste")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT / "macro_dice_comparison.png", dpi=200)
plt.close(fig)

print(f"Immagini salvate in: {OUT}")