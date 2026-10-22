# PulmoNeXt: Long-Tailed Thoracic Disease Classification & Explainability

PulmoNeXt is a deep-learning system for **multi-label thoracic disease classification from chest X-ray images**. The project uses a pretrained **ConvNeXt-Tiny** backbone and addresses severe class imbalance using **Class-Balanced Focal Loss**.

The system also integrates **Grad-CAM** to generate visual heatmaps showing the image regions that contributed most strongly to a model prediction.

> **Note:** This is a research/portfolio project and is not intended for clinical diagnosis or medical decision-making.

---

## 1. Project Overview

Chest X-ray datasets contain multiple thoracic diseases with highly uneven class distributions. Common diseases may have thousands of positive examples while rare diseases such as **Hernia** have substantially fewer samples.

A conventional binary cross-entropy objective can therefore become dominated by common classes.

PulmoNeXt addresses this problem using:

* **ConvNeXt-Tiny** for image representation
* **Transfer learning** from ImageNet-pretrained weights
* **Multi-label classification** for simultaneous disease predictions
* **Class-Balanced Focal Loss** for long-tailed class distributions
* **PR-AUC and ROC-AUC** for class-wise evaluation
* **Grad-CAM** for visual model explainability

### Pipeline

```text
                 Chest X-ray
                     │
                     ▼
              Image Preprocessing
                     │
                     ▼
          ConvNeXt-Tiny Backbone
                     │
                     ▼
             Classification Head
                     │
                     ▼
          14 Disease Probabilities
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     Evaluation              Grad-CAM
          │                     │
          ▼                     ▼
 PR-AUC / ROC-AUC          Visual Heatmap
 Precision / Recall
       F1-score
```

---

# 2. Objectives

The main objectives of PulmoNeXt are:

1. Build a multi-label thoracic disease classification model.
2. Use transfer learning to improve feature extraction.
3. Address extreme class imbalance.
4. Evaluate performance using class-wise metrics.
5. Investigate performance on rare diseases.
6. Generate visual explanations using Grad-CAM.
7. Provide a reproducible research-oriented pipeline.

---

# 3. Dataset

The project is designed for the **NIH ChestX-ray14** dataset.

ChestX-ray14 contains frontal chest radiographs with 14 disease categories.

The 14 classes used by this project are:

|  # | Disease            |
| -: | ------------------ |
|  1 | Atelectasis        |
|  2 | Cardiomegaly       |
|  3 | Consolidation      |
|  4 | Edema              |
|  5 | Effusion           |
|  6 | Emphysema          |
|  7 | Fibrosis           |
|  8 | Hernia             |
|  9 | Infiltration       |
| 10 | Mass               |
| 11 | Nodule             |
| 12 | Pleural Thickening |
| 13 | Pneumonia          |
| 14 | Pneumothorax       |

### Multi-label nature

A single X-ray can contain multiple abnormalities.

For example:

```text
Image:
00012345_000.png

Finding Labels:
Effusion|Cardiomegaly
```

The corresponding target vector becomes:

```text
Atelectasis        0
Cardiomegaly       1
Consolidation      0
Edema              0
Effusion           1
Emphysema          0
Fibrosis           0
Hernia             0
Infiltration       0
Mass               0
Nodule             0
Pleural Thickening 0
Pneumonia          0
Pneumothorax       0
```

Therefore, the model performs **14 simultaneous binary predictions** rather than choosing only one disease.

---

# 4. Dataset Setup

Download the ChestX-ray14 dataset and place the metadata and images in a local directory.

The expected structure is:

```text
chestxray/
│
├── Data_Entry_2017.csv
├── train_val_list.txt
├── test_list.txt
│
├── 00000001_000.png
├── 00000002_000.png
├── 00000003_000.png
├── ...
└── ...
```

The project itself can remain in a separate folder.

For example:

```text
C:\
├── PulmoNeXt\
│   ├── train.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── inference.py
│   ├── config.py
│   ├── utils.py
│   ├── requirements.txt
│   └── README.md
│
└── chestxray\
    ├── Data_Entry_2017.csv
    ├── train_val_list.txt
    ├── test_list.txt
    ├── 00000001_000.png
    └── ...
```

---

# 5. Configure Dataset Location

By default, the project expects:

```text
./data
```

You can change this using the `CHESTXRAY_DATA_DIR` environment variable.

### Windows PowerShell

```powershell
$env:CHESTXRAY_DATA_DIR="C:\path\to\chestxray"
```

### Windows CMD

```cmd
set CHESTXRAY_DATA_DIR=C:\path\to\chestxray
```

### Linux/macOS

```bash
export CHESTXRAY_DATA_DIR=/path/to/chestxray
```

After setting this variable, the Python scripts automatically locate:

```text
Data_Entry_2017.csv
train_val_list.txt
test_list.txt
```

and the corresponding image files.

---

# 6. Project Structure

All source files are intentionally kept in one folder.

```text
PulmoNeXt/
│
├── config.py
├── dataset.py
├── model.py
├── losses.py
├── utils.py
├── train.py
├── evaluate.py
├── gradcam.py
├── inference.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

After training, additional generated files can appear:

```text
PulmoNeXt/
│
├── pulmoNeXt_best.pth
├── metrics.csv
└── gradcam_hernia.png
```

---

# 7. Technology Stack

### Programming Language

* Python

### Deep Learning

* PyTorch
* TorchVision

### Model

* ConvNeXt-Tiny
* ImageNet pretrained weights

### Computer Vision

* Pillow
* TorchVision transforms
* Grad-CAM

### Machine Learning

* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

---

# 8. Model Architecture

PulmoNeXt uses **ConvNeXt-Tiny** as its feature extractor.

```text
Input X-ray
    │
    ▼
Resize 224 × 224
    │
    ▼
Grayscale → 3 channels
    │
    ▼
ConvNeXt-Tiny
    │
    ▼
Deep visual features
    │
    ▼
Linear classification layer
    │
    ▼
14 logits
    │
    ▼
Sigmoid
    │
    ▼
14 disease probabilities
```

The original ConvNeXt classification layer is replaced with a new linear layer containing 14 outputs.

```python
self.backbone.classifier[2] = nn.Linear(
    in_features,
    NUM_CLASSES
)
```

---

# 9. Why Multi-Label Classification?

Thoracic diseases are not mutually exclusive.

For example, an X-ray can contain:

```text
Cardiomegaly + Effusion
```

or:

```text
Mass + Atelectasis
```

Therefore, using a standard softmax classifier would be inappropriate because softmax assumes one mutually exclusive class.

PulmoNeXt instead uses:

```text
14 independent sigmoid outputs
```

Each output represents the estimated probability of one disease.

---

# 10. Image Preprocessing

The input images are transformed using:

```text
Original X-ray
      ↓
Resize to 224 × 224
      ↓
Convert grayscale → 3 channels
      ↓
Random horizontal flip
      ↓
Small random rotation
      ↓
Convert to tensor
      ↓
ImageNet normalization
```

Training augmentation:

* Random horizontal flip
* Random rotation

Validation/test images use deterministic preprocessing.

---

# 11. Handling Class Imbalance

One of the main challenges in ChestX-ray14 is **long-tailed class distribution**.

Some diseases occur much more frequently than others.

A model trained without imbalance handling may learn to focus heavily on common diseases while performing poorly on rare diseases.

PulmoNeXt uses **Class-Balanced Focal Loss**.

---

# 12. Class-Balanced Focal Loss

The method combines two ideas:

### Focal Loss

Focal Loss reduces the contribution of easy examples and emphasizes difficult examples.

The general form is:

```text
FL(pt) = -(1 - pt)^γ log(pt)
```

where:

```text
pt = probability assigned to the correct target
γ  = focusing parameter
```

The project uses:

```text
γ = 2.0
```

---

# 13. Effective Number of Samples

Class-balanced weighting uses the effective number of samples:

```text
Effective(N) = (1 - β^N) / (1 - β)
```

where:

```text
N = number of samples in a class
β = balancing parameter
```

The implementation uses:

```text
β = 0.9999
```

Rare classes consequently receive higher effective weighting than highly represented classes.

---

# 14. Why Hernia Is Important

Hernia is one of the rare classes in the dataset.

This makes it useful for investigating whether the imbalance-handling strategy improves performance on minority classes.

The project therefore reports:

```text
Hernia PR-AUC
```

separately rather than relying only on the overall average.

---

# 15. Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python train.py
```

The training script:

1. Loads the dataset.
2. Reads the class labels.
3. Calculates class frequencies.
4. Creates the Class-Balanced Focal Loss.
5. Loads pretrained ConvNeXt-Tiny.
6. Replaces the classification head.
7. Trains the network.
8. Evaluates validation performance.
9. Saves the best checkpoint.

---

# 16. Training Configuration

Default configuration:

```text
Image size       : 224 × 224
Batch size       : 32
Epochs           : 10
Learning rate    : 2e-4
Weight decay     : 1e-4
Focal gamma      : 2.0
Class beta       : 0.9999
Seed             : 42
```

These values can be modified in:

```text
config.py
```

For example:

```python
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4
```

---

# 17. GPU Training

CUDA is automatically detected.

If a compatible NVIDIA GPU is available:

```text
Device: cuda
```

Otherwise:

```text
Device: cpu
```

For large-scale ChestX-ray14 training, a GPU is strongly recommended.

---

# 18. Model Checkpoint

During training, the best model is saved as:

```text
pulmoNeXt_best.pth
```

The checkpoint contains:

```text
model weights
class information
```

The checkpoint is used by both:

```text
evaluate.py
inference.py
gradcam.py
```

---

# 19. Evaluation

After training:

```bash
python evaluate.py
```

The evaluation script generates class-level metrics.

The output file is:

```text
metrics.csv
```

It contains:

```text
Class
PR-AUC
ROC-AUC
Precision
Recall
F1
```

Example:

```text
Class,PR-AUC,ROC-AUC,Precision,Recall,F1
Atelectasis,...
Cardiomegaly,...
...
Hernia,...
...
```

---

# 20. Evaluation Metrics

## PR-AUC

Precision-Recall AUC measures the area under the precision-recall curve.

It is particularly informative for highly imbalanced classification problems because it focuses on the relationship between:

```text
Precision
Recall
```

For rare classes such as Hernia, PR-AUC can provide a more informative view than accuracy.

---

## ROC-AUC

ROC-AUC measures discrimination across different classification thresholds.

It evaluates:

```text
True Positive Rate
vs.
False Positive Rate
```

---

## Precision

```text
Precision = TP / (TP + FP)
```

It measures how many predicted positive cases were actually positive.

---

## Recall

```text
Recall = TP / (TP + FN)
```

It measures how many actual positive cases were detected.

---

## F1 Score

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

F1 balances precision and recall.

---

# 21. Inference

To run prediction on an individual image:

```bash
python inference.py --image sample.jpg
```

Example output:

```text
Hernia                 0.8124
Effusion               0.2731
Atelectasis            0.1842
Mass                   0.1034
...
```

These values are model probabilities and should not be interpreted as clinical diagnoses.

---

# 22. Grad-CAM Explainability

PulmoNeXt uses Grad-CAM to visualize regions associated with a particular prediction.

Run:

```bash
python gradcam.py --image sample.jpg --class_name Hernia
```

The result is saved as:

```text
gradcam_hernia.png
```

---

# 22a. Image Upload App

Start the interactive app:

```bash
streamlit run app.py
```

Upload a PNG or JPEG chest X-ray to see a graph of all 14 predicted disease
scores, a Grad-CAM overlay, and a short explanation of the selected model
output. The explanation includes the model's rank and score, general features
clinicians assess for that finding, and a clear note that the heatmap does not
confirm those features or establish a diagnosis. If a trained
`pulmoNeXt_best.pth` checkpoint is present, the app uses it. Otherwise, the app
uses the public TorchXRayVision `densenet121-res224-all` pretrained model and
downloads its weights on first use. This fallback does not require downloading
ChestX-ray14 or training locally. The displayed probabilities and
visualizations are for research only and are not clinical diagnoses.

## Vercel static preview

The `vercel-static/` folder contains a static image-preview page for Vercel.
It does not perform inference or show disease scores. The Python/PyTorch model
needs a separate inference host; do not present the static preview as an AI
diagnostic app.

To deploy the static preview from GitHub:

1. Import this repository in the Vercel dashboard.
2. Set **Root Directory** to `vercel-static`.
3. Select **Other** as the framework preset and leave the build command empty.
4. Deploy. The static page is served from `vercel-static/index.html`.

---

# 23. Grad-CAM Process

The Grad-CAM implementation performs:

```text
Input X-ray
      ↓
ConvNeXt feature extraction
      ↓
Target disease score
      ↓
Backward gradient calculation
      ↓
Feature-map weighting
      ↓
ReLU
      ↓
Normalization
      ↓
Heatmap
      ↓
Overlay on original X-ray
```

The highlighted regions represent image areas that had stronger influence on the selected model output.

---

# 24. Example Explainability Workflow

For Hernia:

```bash
python gradcam.py \
    --image sample.jpg \
    --class_name Hernia
```

Output:

```text
gradcam_hernia.png
```

The image contains:

```text
Original X-ray
+
Grad-CAM heatmap
```

This makes the model's prediction easier to inspect visually.

---

# 25. Research Reproducibility

The project uses:

```python
SEED = 42
```

The seed is applied to:

* Python random
* NumPy
* PyTorch
* CUDA

This improves reproducibility, although exact results can still vary depending on hardware and software versions.

---

# 26. Important Metric Claim

If reporting a result such as:

```text
Hernia PR-AUC = 0.9791
```

the value must come from an actual experiment.

The project does **not** artificially set or hard-code this number.

Before using it on a resume or publication, document:

```text
Dataset version
Train/test split
Patient-level split methodology
Preprocessing
Model version
Loss parameters
Training epochs
Random seed
Evaluation code
PR-AUC calculation method
```

A high score should also be checked for possible data leakage and evaluation artifacts.

---

# 27. Example Final Results Table

After running the experiment, the README can be updated with the actual results.

```text
| Disease              | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|----------------------|--------|---------|-----------|--------|----|
| Atelectasis          | --     | --      | --        | --     | -- |
| Cardiomegaly         | --     | --      | --        | --     | -- |
| Consolidation        | --     | --      | --        | --     | -- |
| Edema                | --     | --      | --        | --     | -- |
| Effusion             | --     | --      | --        | --     | -- |
| Emphysema            | --     | --      | --        | --     | -- |
| Fibrosis             | --     | --      | --        | --     | -- |
| Hernia               | --     | --      | --        | --     | -- |
| Infiltration         | --     | --      | --        | --     | -- |
| Mass                 | --     | --      | --        | --     | -- |
| Nodule               | --     | --      | --        | --     | -- |
| Pleural Thickening   | --     | --      | --        | --     | -- |
| Pneumonia            | --     | --      | --        | --     | -- |
| Pneumothorax         | --     | --      | --        | --     | -- |
```

Replace `--` with values from the actual `metrics.csv`.

---

# 28. Limitations

This project has several important limitations.

### Dataset labels

Chest X-ray disease labels are not equivalent to manually verified clinical ground truth for every image.

### Class imbalance

Although Class-Balanced Focal Loss addresses imbalance during optimization, it cannot create additional clinical information for rare classes.

### Model explanations

Grad-CAM highlights regions associated with model predictions. It does not prove that the highlighted region corresponds to the true pathological cause.

### Dataset shift

Performance on ChestX-ray14 does not guarantee equivalent performance on X-rays from another hospital, scanner, population, or acquisition protocol.

### Clinical use

The model has not been validated for clinical deployment and must not be used to diagnose or treat patients.

---

# 29. Future Improvements

Possible improvements include:

* ConvNeXtV2 backbone
* Higher-resolution input images
* Multi-scale feature extraction
* Spatial attention
* Multi-head self-attention
* Asymmetric Loss
* Label-distribution-aware margin loss
* Advanced long-tail sampling
* Patient-level stratification
* Threshold optimization per disease
* Calibration analysis
* External validation
* Confidence estimation
* Ensemble models
* More robust Grad-CAM variants
* Clinical expert evaluation

---

# 30. Recommended Experimentation

For a stronger research project, compare:

```text
Experiment 1
Binary Cross Entropy
        ↓
ConvNeXt-Tiny

Experiment 2
Focal Loss
        ↓
ConvNeXt-Tiny

Experiment 3
Class-Balanced Focal Loss
        ↓
ConvNeXt-Tiny
```

Then compare:

```text
Hernia PR-AUC
Hernia ROC-AUC
Macro PR-AUC
Macro ROC-AUC
Macro F1
```

This gives a much stronger experimental argument than reporting only one model.

---

# 31. Reproducibility Checklist

Before publishing the result, verify:

* [ ] Dataset downloaded correctly
* [ ] No corrupted images
* [ ] Patient-level separation considered
* [ ] Training and test lists verified
* [ ] Preprocessing documented
* [ ] Random seed recorded
* [ ] Model architecture recorded
* [ ] Loss parameters recorded
* [ ] Optimizer recorded
* [ ] Learning rate recorded
* [ ] Number of epochs recorded
* [ ] Hardware recorded
* [ ] Test metrics generated independently
* [ ] Rare-class performance verified
* [ ] Grad-CAM outputs inspected
* [ ] No test-set tuning
* [ ] Actual metrics used in resume

---

# 32. Running the Complete Pipeline

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Configure dataset

```powershell
$env:CHESTXRAY_DATA_DIR="C:\path\to\chestxray"
```

### Step 3 — Train

```bash
python train.py
```

### Step 4 — Evaluate

```bash
python evaluate.py
```

### Step 5 — Run inference

```bash
python inference.py --image sample.jpg
```

### Step 6 — Generate Grad-CAM

```bash
python gradcam.py --image sample.jpg --class_name Hernia
```

---

# 33. Project Highlights

### Deep Learning

```text
ConvNeXt-Tiny
Transfer Learning
Multi-label Classification
```

### Long-Tailed Learning

```text
Class-Balanced Focal Loss
Effective Number of Samples
Rare-Class Optimization
```

### Evaluation

```text
PR-AUC
ROC-AUC
Precision
Recall
F1
```

### Explainable AI

```text
Grad-CAM
Visual Heatmaps
Disease-specific Explanations
```

---

# 34. Resume Description

A concise resume version:

> Developed a multi-label thoracic disease classification system using ConvNeXt-Tiny and Class-Balanced Focal Loss to address severe class imbalance in chest X-rays, with class-wise PR-AUC/ROC-AUC evaluation and Grad-CAM visual explanations for model interpretability.

If the experiment actually produces the previously specified metric, the result can be incorporated:

> Mitigated extreme long-tailed dataset imbalance using Class-Balanced Focal Loss, achieving a **0.9791 PR-AUC on the Hernia class**, and integrated Grad-CAM visual heatmaps for model explainability.

---

# 35. Disclaimer

PulmoNeXt is an educational and research-oriented machine-learning project.

It is **not a medical device**, has not been clinically validated, and should not be used as a substitute for professional radiological assessment.

Model predictions and Grad-CAM visualizations should be treated as research outputs rather than medical diagnoses.
