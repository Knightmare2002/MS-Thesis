from pathlib import Path
from collections import Counter
import argparse
import csv
import math

ROOT = Path(r"C:\\Users\\Samuele_Caruso\\OneDrive - UMass Lowell\\Desktop\\bridge_model\\labeling\\roi_views_19112926")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estrae tutte le righe eligible=True dal CSV aggregato.")
    parser.add_argument("--input", type=Path, default=ROOT / "summary_files" / "roi_views_all.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.input.with_name("eligible_views_all.csv")
    if args.input.resolve() == output.resolve():
        raise ValueError("Input e output devono essere file diversi.")

    rows, counts = [], Counter()
    with args.input.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or not {"block_id", "eligible", "stem"}.issubset(fields):
            raise ValueError("Il CSV deve contenere block_id, eligible e stem.")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"Riga CSV non valida: {line}")
            block = int(row["block_id"])
            counts.setdefault(block, 0)
            value = (row["eligible"] or "").strip().lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(f"Valore eligible non valido alla riga {line}: {value!r}")
            if value in {"true", "1"}:
                rows.append(row)
                counts[block] += 1

    def gsd(row):
        try:
            value = float(row.get("gsd_eff_median", ""))
            return value if math.isfinite(value) else float("inf")
        except (ValueError, TypeError):
            return float("inf")

    rows.sort(key=lambda row: (int(row["block_id"]), gsd(row), row["stem"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for block, count in sorted(counts.items()):
        print(f"Blocco {block:02d}: {count} viste eleggibili")
    print(f"Totale: {len(rows)} righe. File: {output}")


if __name__ == "__main__":
    main()
