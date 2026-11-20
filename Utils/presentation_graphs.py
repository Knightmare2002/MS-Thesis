import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Path to your JSON file
JSON_PATH = Path(r"C:\\Users\\Samuele_Caruso\\OneDrive - UMass Lowell\\Desktop\\bridge_model\\labeling\\roi_views_13112026_V2\\results\\h1_2d_transfer_global_median.json")
OUT_DIR = JSON_PATH.parent

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Global styling for clean PowerPoint figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Order classes by overall Dice score (descending)
all_res = data["results"]["all"]
total_groups_all = len(all_res["groups"])
class_order = sorted(
    all_res["classes"].keys(),
    key=lambda c: all_res["classes"][c]["dice"] or 0.0,
    reverse=True
)
class_labels = [c.capitalize() for c in class_order]

# ==============================================================================
# FIGURE 1: Overall Performance (Split 'all') - Forest Plot + Damage-Free FAR
# ==============================================================================
fig, axes = plt.subplots(
    1, 4, figsize=(13.5, 4.4),
    gridspec_kw={"width_ratios": [1, 1, 1, 0.85]},
    sharey=True
)

metrics_cfg = [
    ("dice", "Dice Score", "#1f77b4"),
    ("precision", "Precision", "#2ca02c"),
    ("recall", "Recall", "#ff7f0e"),
]

y_pos = np.arange(len(class_order))
y_tick_labels = [
    f"{c.capitalize()}\n({all_res['classes'][c]['n_positive_groups']}/{total_groups_all} pos. groups)"
    for c in class_order
]

for ax, (m_key, title, color) in zip(axes[:3], metrics_cfg):
    vals, err_low, err_high = [], [], []
    for c in class_order:
        m_dict = all_res["classes"][c]
        v = m_dict[m_key]
        ci = m_dict["ci95"].get(m_key, [v, v])
        vals.append(v)
        err_low.append(max(0.0, v - ci[0]))
        err_high.append(max(0.0, ci[1] - v))

    ax.errorbar(
        vals, y_pos,
        xerr=[err_low, err_high],
        fmt="o", color=color, ecolor=color,
        elinewidth=2.2, capsize=4, capthick=1.8, markersize=7, zorder=3
    )

    # Value annotations above each marker
    for y, v in zip(y_pos, vals):
        ax.text(v, y - 0.22, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#222222")

    ax.set_title(title, pad=10)
    ax.set_xlim(-0.05, 1.08)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", linestyle="--", alpha=0.45, zorder=0)
    ax.invert_yaxis()

# 4th subplot: False Alarm Rate on Damage-Free Images (%)
ax_far = axes[3]
far_vals = [
    (all_res["classes"][c]["false_alarm_rate_damage_free_images"] or 0.0) * 100.0
    for c in class_order
]
bars = ax_far.barh(y_pos, far_vals, height=0.45, color="#d62728", alpha=0.85, zorder=3)

for bar, v in zip(bars, far_vals):
    ax_far.text(
        v + 0.35, bar.get_y() + bar.get_height() / 2,
        f"{v:.2f}%", ha="left", va="center", fontsize=9.5, fontweight="bold", color="#222222"
    )

ax_far.set_title("Damage-Free FAR (%)", pad=10)
ax_far.set_xlim(0, max(far_vals) * 1.35)
ax_far.grid(axis="x", linestyle="--", alpha=0.45, zorder=0)

axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(y_tick_labels)

fig.suptitle(
    f"Overall 2D Transfer Performance (All Views: n = {all_res['n_images']} images, {total_groups_all} groups) — Point Estimate & 95% Group Bootstrap CI",
    fontsize=12.5, fontweight="bold", y=1.02
)
plt.tight_layout()

fig1_path = OUT_DIR / "fig1_overall_metrics_ci.png"
fig.savefig(fig1_path, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)

# ==============================================================================
# FIGURE 2: Split Comparison (C_fit vs C_cal vs T) - Grouped Bar Chart (Dice)
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(10.5, 4.6))

splits_cfg = [
    ("C_fit", f"C_fit (n={data['results']['C_fit']['n_images']}, {len(data['results']['C_fit']['groups'])} grp)", "#4e79a7"),
    ("C_cal", f"C_cal (n={data['results']['C_cal']['n_images']}, {len(data['results']['C_cal']['groups'])} grp)", "#f28e2b"),
    ("T",     f"Test T (n={data['results']['T']['n_images']}, {len(data['results']['T']['groups'])} grp)", "#59a14f"),
]

x = np.arange(len(class_order))
width = 0.25
offsets = [-width, 0.0, width]

for (split_key, split_label, color), offset in zip(splits_cfg, offsets):
    split_res = data["results"][split_key]
    vals, err_low, err_high, notes = [], [], [], []

    for c in class_order:
        m_dict = split_res["classes"][c]
        v = m_dict["dice"] if m_dict["dice"] is not None else 0.0
        ci = m_dict["ci95"].get("dice")
        pos_g = m_dict["n_positive_groups"]

        vals.append(v)
        if ci is not None:
            err_low.append(max(0.0, v - ci[0]))
            err_high.append(max(0.0, ci[1] - v))
            notes.append(f"{v:.2f}")
        else:
            err_low.append(0.0)
            err_high.append(0.0)
            notes.append(f"{v:.2f}*" if pos_g > 0 else "0 pos.")

    bars = ax2.bar(
        x + offset, vals, width=width * 0.9, label=split_label,
        color=color, alpha=0.9, zorder=3
    )
    ax2.errorbar(
        x + offset, vals, yerr=[err_low, err_high],
        fmt="none", ecolor="#222222", elinewidth=1.4, capsize=3.5, capthick=1.2, zorder=4
    )

    for bar, v, hi, note in zip(bars, vals, err_high, notes):
        y_text = v + hi + 0.025
        ax2.text(
            bar.get_x() + bar.get_width() / 2, y_text,
            note, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222"
        )

ax2.set_ylabel("Dice Score (95% Group Bootstrap CI)")
ax2.set_title("Dice Score Across Dataset Splits (C_fit vs. C_cal vs. Test T)", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(class_labels, fontweight="bold")
ax2.set_ylim(0, 1.08)
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
ax2.legend(frameon=True, facecolor="white", framealpha=0.8, loc="upper right")

# Footnote explaining missing CIs when positive groups <= 1
fig2.text(
    0.01, -0.02,
    "*Note: 95% CI omitted when a split contains ≤ 1 positive group for that class (insufficient non-zero bootstrap resamples).",
    fontsize=9, style="italic", color="#444444"
)

plt.tight_layout()
fig2_path = OUT_DIR / "fig2_split_comparison_dice.png"
fig2.savefig(fig2_path, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig2)

print(f"Saved:\n  1) {fig1_path}\n  2) {fig2_path}")