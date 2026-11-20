import json
from pathlib import Path

json_path = Path(r"C:\\Users\\Samuele_Caruso\\OneDrive - UMass Lowell\Desktop\bridge_model\\labeling\\roi_views_13112026_V2\\output_h1_2d_v2\\h1_2d_transfer.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

metrics = ["dice", "iou", "precision", "recall", "fpr"]

def fmt(val):
    return f"{val:.4f}" if val is not None else "N/A"

for split_name, split_data in data["results"].items():
    print(f"\n=== SPLIT: {split_name} (n_images={split_data['n_images']}) ===")
    print(f"{'Classe':<12} | {'Dice':<8} | {'IoU':<8} | {'Prec':<8} | {'Recall':<8} | {'FPR':<8}")
    print("-" * 64)
    
    for cls_name, cls_metrics in split_data["classes"].items():
        vals = [fmt(cls_metrics.get(m)) for m in metrics]
        print(f"{cls_name:<12} | {vals[0]:<8} | {vals[1]:<8} | {vals[2]:<8} | {vals[3]:<8} | {vals[4]:<8}")