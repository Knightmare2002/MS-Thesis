# Baseline crack segmentation — Settimana 3

Codice per il primo risultato sperimentale della tesi: **EDA su CrackSeg9k + dacl10k** e
**baseline U-Net binaria** (crack / non-crack) su CrackSeg9k, con valutazione
zero-shot su dacl10k come prima misura di domain shift.

Il dataset RGB proprietario di ponti resta **fuori** da questo codice: è il blind
test set esterno e va usato solo alla fine della pipeline.

Ambiente di riferimento: **workstation locale, VS Code, NVIDIA RTX A2000 12 GB,
Intel i5-10500 (6 core / 12 thread)**.

## Struttura

```
crack-segmentation-baseline/
├── configs/config.yaml         # unica fonte di verità: path, iperparametri
├── src/
│   ├── utils.py                # config con override CLI, seeding, device+TF32, loader kwargs
│   ├── losses.py               # BCE + soft Dice (pos_weight per lo sbilanciamento)
│   ├── engine.py               # train/eval loop, AMP, SGDR, checkpoint resume
│   ├── data/
│   │   ├── class_mapping.py    # tassonomia dacl10k/CrackSeg9k/CODEBRIM → classi unificate
│   │   ├── crackseg9k.py       # pairing immagine/maschera, split, Dataset
│   │   ├── dacl10k.py          # poligoni JSON → maschere (binaria e multi-label)
│   │   ├── transforms.py       # pipeline albumentations
│   │   └── stats.py            # EDA: risoluzioni, % pixel crack, frequenza classi
│   ├── models/unet.py          # factory smp (unet / unet++ / deeplabv3+)
│   └── eval/metrics.py         # IoU, Dice, precision, recall (micro e macro)
└── scripts/
    ├── 00_smoke_test.py        # test end-to-end su dati sintetici (secondi)
    ├── 01_eda.py               # CSV + figure per il PPT
    ├── 02_train_unet.py        # training baseline
    └── 03_evaluate.py          # test set + threshold sweep + zero-shot dacl10k
```

## Setup locale (Windows + VS Code)

```powershell
# 1. driver e CUDA visibili
nvidia-smi

# 2. ambiente isolato nella cartella del progetto
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3. PyTorch con CUDA (NON la wheel CPU di default di PyPI)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. resto delle dipendenze
pip install -r requirements.txt

# 5. verifica che la GPU sia vista da torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Atteso al punto 5: `True` e `NVIDIA RTX A2000 12GB`. Se stampa `False`, la wheel
installata è quella CPU: disinstalla `torch`/`torchvision` e ripeti il punto 3.

In VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`. Le estensioni
utili sono Python + Pylance; il terminale integrato va aperto con il venv attivo.

Per il debug degli script, `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "train (dry run)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/02_train_unet.py",
      "args": ["--limit-train", "64", "--limit-val", "32",
               "--set", "train.epochs=1", "data.num_workers=0"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

`data.num_workers=0` in debug è voluto: con i worker attivi i breakpoint dentro il
`Dataset` non vengono raggiunti (su Windows i processi figli usano `spawn`).

## Configurazione dei path

In `configs/config.yaml` imposta le tue cartelle usando **slash normali**, anche su
Windows (`pathlib` li gestisce correttamente):

```yaml
data:
  crackseg9k:
    images_dir: "D:/datasets/crackseg9k/images"
    masks_dir: "D:/datasets/crackseg9k/masks"
  dacl10k:
    root: "D:/datasets/dacl10k"      # contiene images/<split> e annotations/<split>
```

## Uso

```powershell
# 0. verifica che tutto giri (nessun dataset richiesto, ~20 s)
python scripts/00_smoke_test.py

# 1. EDA — prima un giro rapido, poi completo
python scripts/01_eda.py --sample 500
python scripts/01_eda.py

# 2. training: dry run, poi run vero
python scripts/02_train_unet.py --limit-train 64 --limit-val 32 --set train.epochs=1
python scripts/02_train_unet.py --run-name unet_r34_512

# 3. valutazione + domain shift
python scripts/03_evaluate.py --run-dir outputs/runs/unet_r34_512 --cross-dataset
```

Qualsiasi campo del config è sovrascrivibile da CLI:

```powershell
python scripts/02_train_unet.py --set model.encoder=resnet50 data.image_size=384 train.batch_size=4
```

## Note per RTX A2000 12 GB

| Situazione | Cosa cambiare |
| --- | --- |
| Configurazione di partenza | `image_size 512`, `batch_size 8`, `amp: true` → circa 7–8 GB di VRAM |
| `CUDA out of memory` | `train.batch_size=4` **e** `train.accumulation_steps=2`: batch efficace invariato, VRAM dimezzata |
| VRAM ancora insufficiente | `data.image_size=384` (rivedi le metriche: le crepe sottili perdono pixel) |
| GPU sotto il 90% di utilizzo | alza `data.num_workers` (max 8–10) o `data.prefetch_factor` |
| Encoder più grande (resnet50, effnet-b3) | parti da `train.batch_size=4` |

`get_device()` attiva automaticamente TF32 e `cudnn.benchmark`: sull'Ampere sono
guadagni gratuiti a parità di risultato, corretti perché la risoluzione di input è
fissa. Con AMP + TF32 su A2000 un'epoca su ~9k immagini a 512 px sta nell'ordine
dei 5–8 minuti, quindi il run da 20 epoche è realistico in una sessione di lavoro.

Monitoraggio durante il training, in un secondo terminale:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 5
```

`train.resume: true` resta utile anche in locale: se interrompi con `Ctrl+C` il run
riprende da `last.pt` invece di ricominciare. Per ripartire da zero, cancella la
cartella del run in `outputs/runs/<run_name>`.

## Scelte metodologiche da difendere in presentazione

| Scelta | Motivazione |
| --- | --- |
| `Crack` + `ACrack` → classe unificata `crack` | Rende dacl10k confrontabile con CrackSeg9k senza addestrare nulla su dacl10k. |
| Loss BCE + Dice con `pos_weight` | Con 1–5% di pixel positivi la sola BCE converge a predizione vuota. |
| Dice/IoU riportati micro **e** macro | Il micro è dominato dalle crepe grandi; il macro mostra i fallimenti sulle immagini con poche crepe. |
| `false_alarm_rate_empty_gt` | Sulle immagini senza crepe Dice non è definito: i falsi allarmi vanno misurati a parte. |
| Threshold sweep in valutazione | La soglia 0.5 è arbitraria; la curva precision/recall è parte del risultato. |
| SGDR (cosine con warm restarts) | Aiuta a uscire dalle regioni piatte tipiche dei task densi sbilanciati. |
| Split congelato in `split.json` | Il test set resta identico tra run diversi: nessun leakage tra esperimenti. |
| Encoder ImageNet condiviso, nome da config | È lo stesso componente che sarà condiviso dalle due teste della dual-branch. |
| TF32 + `cudnn.benchmark` | Accelerazione hardware senza effetti misurabili sulla metrica, con input a risoluzione fissa. |

## Riferimenti

| Argomento trattato | Link (URL) | A cosa serve/Perché è utile |
| --- | --- | --- |
| dacl10k: formato annotazioni, layout cartelle, 19 classi | [dacl10k-toolkit](https://github.com/phiyodr/dacl10k-toolkit) | Fonte usata per implementare la rasterizzazione dei poligoni e l'ordine ufficiale delle classi. |
| dacl10k: benchmark e metriche di riferimento | [Paper WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Flotzinger_dacl10k_Benchmark_for_Semantic_Bridge_Damage_Segmentation_WACV_2024_paper.pdf) | Valori di mIoU da citare come baseline nel confronto. |
| segmentation_models_pytorch | [smp docs](https://smp.readthedocs.io/) | API di U-Net/UNet++/DeepLabV3+ con encoder pre-addestrati. |
| Albumentations | [docs](https://albumentations.ai/docs/) | Augmentation sincronizzate immagine/maschera. |
| Wheel PyTorch per CUDA | [pytorch.org](https://pytorch.org/get-started/locally/) | Comando di installazione corretto per la versione di CUDA locale. |
