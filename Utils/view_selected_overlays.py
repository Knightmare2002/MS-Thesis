import glob
import os
import re
import shutil

# 1. Definizione percorsi base
base_dir = r"C:\\Users\\Samuele_Caruso\\Desktop\\test_views"
overlays_src_dir = r"C:\\Users\\Samuele_Caruso\\Desktop\\damage3d_work\\damage3d_maps\\p1ml_t9_unetplusplus_effb4_512_imagenet\\overlays"

# 2. Cerca tutti i file che corrispondono al pattern "Group * views.txt"
selected_cameras = os.path.join(base_dir, "bridge_roof")  # Modifica qui per cambiare il gruppo selezionato
txt_file_path = os.path.join(base_dir, f"{selected_cameras}.txt")
matching_txt_files = glob.glob(txt_file_path)

if not matching_txt_files:
    print(f"Nessun file trovato corrispondente al pattern: {selected_cameras}")
else:
    # 3. Mappa tutti i file presenti nella cartella overlays per nome base (senza estensione)
    overlay_files_map = {}
    if os.path.exists(overlays_src_dir):
        for fname in os.listdir(overlays_src_dir):
            full_path = os.path.join(overlays_src_dir, fname)
            if os.path.isfile(full_path):
                base_name, _ = os.path.splitext(fname)
                overlay_files_map[base_name] = full_path
    else:
        print(f"Errore: la cartella sorgente overlays non esiste:\n{overlays_src_dir}")
        matching_txt_files = []

    # 4. Elabora ciascun file trovato
    for txt_file in matching_txt_files:
        filename = os.path.basename(txt_file)
        
        # Crea la cartella di destinazione "overlays_group_*"
        dest_dir = os.path.join(base_dir, f"{selected_cameras} overlays")
        os.makedirs(dest_dir, exist_ok=True)

        # Legge le etichette delle camere dal file txt
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        copied_count = 0
        missing_count = 0

        for line in lines:
            cam_label, _ = os.path.splitext(line)

            # Cerca il file overlay corrispondente
            matched_file = None
            if cam_label in overlay_files_map:
                matched_file = overlay_files_map[cam_label]
            else:
                # Gestisce eventuali prefissi/suffissi (es. "DSC02150_overlay.png")
                for base_key, path in overlay_files_map.items():
                    if base_key == cam_label or base_key.startswith(f"{cam_label}_") or base_key.endswith(f"_{cam_label}"):
                        matched_file = path
                        break

            if matched_file:
                shutil.copy2(matched_file, os.path.join(dest_dir, os.path.basename(matched_file)))
                copied_count += 1
            else:
                missing_count += 1

        print(f"[{filename}] -> Salvati {copied_count} overlay in: {dest_dir} (Non trovati: {missing_count})")