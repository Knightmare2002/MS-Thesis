import os


def analyze_group_views(file_paths):
    group_sets = {}

    for path in file_paths:
        group_name = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(path):
            print(f"File non trovato: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            # Rimuove estensioni ed eventuali spazi/maiuscole per un matching esatto
            cameras = {
                os.path.splitext(line.strip())[0].upper()
                for line in f
                if line.strip()
            }
            group_sets[group_name] = cameras
            print(f"{group_name}: {len(cameras)} camere lette.")

    if not group_sets:
        return

    all_unique = set.union(*group_sets.values())
    common_all = set.intersection(*group_sets.values())

    print("\n--- RISULTATI ---")
    print(f"Totale viste UNICHE complessive: {len(all_unique)}")
    print(f"Viste comuni a TUTTI i gruppi: {len(common_all)}")

    # Se vuoi salvare la lista complessiva unificata in un file
    out_all = r"C:\\Users\\Samuele_Caruso\\Desktop\\test_views\\all_unique_views.txt"
    with open(out_all, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(all_unique)))
    print(f"Lista unificata salvata in: {out_all}")


# Inserisci qui i percorsi dei file dei gruppi da confrontare
base_dir = r"C:\\Users\\Samuele_Caruso\\Desktop\\test_views"
files_to_check = [
    os.path.join(base_dir, "bridge_roof.txt"),
    os.path.join(base_dir, "bridge_face.txt"),
]

analyze_group_views(files_to_check)