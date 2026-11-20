import csv
import shutil
from pathlib import Path

# Percorsi di base
base_dir = Path(r"C:\\Users\Samuele_Caruso\\OneDrive - UMass Lowell\Desktop\\bridge_model\\labeling\\roi_views_19112026")
csv_path = base_dir / "summary_files" / "selected_views.csv"
source_dir = base_dir / "labelme_json"
dest_dir = base_dir / "labelme_json_selected"

# Crea la cartella di destinazione se non esiste
dest_dir.mkdir(parents=True, exist_ok=True)

# Legge tutti i valori univoci dalla colonna 'stem'
stems = set()
with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        stem_val = row.get("stem", "").strip()
        if stem_val:
            stems.add(stem_val)

print(f"Trovati {len(stems)} stem nel CSV.")

moved_files = 0
missing_stems = []

for stem in sorted(stems):
    # Cerca tutti i file nella cartella source_dir che hanno esattamente quello stem (es. DSC02069.json, DSC02069.jpg, ecc.)
    matching_files = [p for p in source_dir.iterdir() if p.is_file() and p.stem == stem]
    
    if not matching_files:
        missing_stems.append(stem)
        continue

    for file_path in matching_files:
        dest_path = dest_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))
        moved_files += 1

print(f"Spostati {moved_files} file in: {dest_dir}")

if missing_stems:
    print(f"Attenzione: {len(missing_stems)} stem non trovati in {source_dir.name}:")
    for s in missing_stems:
        print(f"  - {s}")