import json
from pathlib import Path

import matplotlib.pyplot as plt

# Modifica con la cartella che contiene i quattro report
FOLDER = Path(r"C:\\Users\\Samuele_Caruso\\OneDrive - UMass Lowell\Desktop\bridge_model\\labeling\\roi_views_13112026_V2\\results")

reports = {
    3: "h1_2d_transfer.json",
    11: "h1_2d_transfer_global_median.json",
    13: "h1_2d_transfer_all_views_median.json",
    17: "h1_2d_transfer_glb_central_median.json",
}

metrics = {
    "precision": "Precision",
    "recall": "Recall",
    "dice": "Dice / F1",
    "ap": "AP",
    "tol_f1": "F1 with tolerance",
}

values = {key: [] for key in metrics}

for width, filename in reports.items():
    with (FOLDER / filename).open(encoding="utf-8-sig") as f:
        report = json.load(f)

    crack = report["results"]["all"]["classes"]["crack"]

    for key in metrics:
        values[key].append(crack[key])

fig, ax = plt.subplots(figsize=(9, 5))

for key, label in metrics.items():
    ax.plot(list(reports), values[key], marker="o", label=label)

ax.set(
    xlabel="Thickness of the annotation --line-px [pixel]",
    ylabel="Value of the metric",
    title="Crack: sensitivity to thickness — all views",
    xticks=list(reports),
    ylim=(0, 1),
)

ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(FOLDER / "crack_vs_line_px.png", dpi=200)
plt.show()