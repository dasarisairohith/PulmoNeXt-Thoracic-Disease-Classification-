# PulmoNeXt

Long-tailed multi-label thoracic disease classification from chest X-rays using a pretrained ConvNeXt-Tiny, Class-Balanced Focal Loss, and Grad-CAM explainability.

## Dataset

Use the NIH ChestX-ray14 dataset. Download the official images and metadata separately and place them under a local data directory.

Expected files:

```text
data/
  Data_Entry_2017.csv
  train_val_list.txt
  test_list.txt
  00000001_000.png
  ...
```

The image files can also be stored elsewhere by setting:

```bash
set CHESTXRAY_DATA_DIR=C:\path\to\chestxray
```

Linux/macOS:

```bash
export CHESTXRAY_DATA_DIR=/path/to/chestxray
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Evaluate

```bash
python evaluate.py
```

Metrics are written to `metrics.csv`.

## Inference

```bash
python inference.py --image sample.jpg
```

## Grad-CAM

```bash
python gradcam.py --image sample.jpg --class_name Hernia
```

This creates `gradcam_hernia.png`.

## Important

Do not claim a specific PR-AUC such as 0.9791 unless the experiment actually produces that value on a clearly documented, leakage-free evaluation split. This project is for research/portfolio use and is not a clinical diagnostic system.
