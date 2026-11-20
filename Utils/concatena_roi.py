from pathlib import Path
import csv
import json
import math
import re

ROOT = Path(r"C:\\Users\\Samuele_Caruso\\OneDrive - UMass Lowell\\Desktop\\bridge_model\\labeling\\roi_views_19112026_v2")


def main():
    blocks = sorted(
        (p for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"block\d+", p.name)),
        key=lambda p: int(p.name[5:]),
    )
    if not blocks:
        raise RuntimeError(f"Nessuna cartella blockN trovata in {ROOT}")

    rows, summaries, columns = [], {}, ["block_id", "block_folder"]
    for block in blocks:
        csv_path = block / "roi_views.csv"
        json_path = block / "summary.json"
        if not csv_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"Mancano roi_views.csv o summary.json in {block}")

        with json_path.open(encoding="utf-8-sig") as f:
            summaries[block.name] = json.load(f)

        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV senza intestazione: {csv_path}")
            if {"block_id", "block_folder"} & set(reader.fieldnames):
                raise ValueError(f"Colonne di provenienza gia presenti: {csv_path}")
            for name in reader.fieldnames:
                if name not in columns:
                    columns.append(name)
            for row in reader:
                if None in row:
                    raise ValueError(f"Riga CSV non valida: {csv_path}")
                rows.append({"block_id": int(block.name[5:]), "block_folder": block.name, **row})

    out = ROOT / "summary_files"
    out.mkdir(exist_ok=True)
    with (out / "roi_views_all.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    with (out / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(clean(summaries), f, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"Uniti {len(blocks)} blocchi e {len(rows)} righe CSV.")
    print(f"Risultati: {out}")


if __name__ == "__main__":
    main()
