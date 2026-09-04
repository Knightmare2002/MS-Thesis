# Baseline crack segmentation — Settimana 3

Codice per il primo risultato sperimentale della tesi: **EDA su CrackSeg9k + dacl10k** e
**baseline U-Net binaria** (crack / non-crack) su CrackSeg9k, con valutazione
zero-shot su dacl10k come prima misura di domain shift.

Il dataset RGB proprietario di ponti resta **fuori** da questo codice: è il blind
test set esterno e va usato solo alla fine della pipeline.

## Struttura

```
thesis/
├── configs/config.yaml         # unica fonte di verità: path, iperparametri
├── src/
│   ├── utils.py                # config con override CLI, seeding, device
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
    ├── 00_smoke_test.py        # test end-to-end su dati sintetici (CPU, secondi)
    ├── 01_eda.py               # CSV + figure per il PPT
    ├── 02_train_unet.py        # training baseline
    └── 03_evaluate.py          # test set + threshold sweep + zero-shot dacl10k
```

## Uso

```bash
pip install -r requirements.txt

# 0. verifica che tutto giri (nessun dataset richiesto)
python scripts/00_smoke_test.py

# 1. EDA — usa --sample per un giro rapido
python scripts/01_eda.py --config configs/config.yaml
python scripts/01_eda.py --config configs/config.yaml --sample 500

# 2. training (prima un dry run corto, poi il run vero)
python scripts/02_train_unet.py --limit-train 64 --limit-val 32 --set train.epochs=1
python scripts/02_train_unet.py --run-name unet_r34_512

# 3. valutazione + domain shift
python scripts/03_evaluate.py --run-dir outputs/runs/unet_r34_512 --cross-dataset
```

Qualsiasi campo del config è sovrascrivibile da CLI:

```bash
python scripts/02_train_unet.py --set model.encoder=resnet50 data.image_size=384 train.batch_size=4
```

## Note su Colab

1. Monta Drive e imposta `project.output_dir` su una cartella di Drive: i
   checkpoint sopravvivono al riavvio della sessione.
2. `train.resume: true` fa ripartire il run da `last.pt` se la sessione muore.
3. Se la VRAM non basta: `data.image_size=384`, `train.batch_size=4`,
   `train.accumulation_steps=2` (batch efficace invariato).
4. `data.num_workers=2` è il valore sano su Colab; valori più alti spesso peggiorano.

```python
from google.colab import drive; drive.mount('/content/drive')
!pip -q install segmentation-models-pytorch albumentations
!cd /content/thesis && python scripts/02_train_unet.py \
    --set project.output_dir=/content/drive/MyDrive/thesis_outputs
```

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

## Riferimenti

| Argomento trattato | Link (URL) | A cosa serve/Perché è utile |
| --- | --- | --- |
| dacl10k: formato annotazioni, layout cartelle, 19 classi | [dacl10k-toolkit](https://github.com/phiyodr/dacl10k-toolkit) | Fonte usata per implementare la rasterizzazione dei poligoni e l'ordine ufficiale delle classi. |
| dacl10k: benchmark e metriche di riferimento | [Paper WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Flotzinger_dacl10k_Benchmark_for_Semantic_Bridge_Damage_Segmentation_WACV_2024_paper.pdf) | Valori di mIoU da citare come baseline nel confronto. |
| segmentation_models_pytorch | [smp docs](https://smp.readthedocs.io/) | API di U-Net/UNet++/DeepLabV3+ con encoder pre-addestrati. |
| Albumentations | [docs](https://albumentations.ai/docs/) | Augmentation sincronizzate immagine/maschera. |
