# 🧠 ACDNet: Polyp Detection and Colonoscopy Analysis

An anatomy-conditioned deep learning framework for polyp detection, segmentation, and severity analysis in colonoscopy videos using the HyperKvasir dataset.

---

## 🚀 Overview

This repository implements ACDNet (Anatomy-Conditioned Dual Attention Network), a unified multi-task deep learning pipeline designed for clinical-grade colonoscopy analysis.

The system integrates:
- Detection (Polyp vs Normal)
- Segmentation (Pixel-wise mask)
- Severity classification (3-class UC grading)
- Temporal consistency learning (video-based)

---

## 🎯 Key Features

- Multi-task learning (Detection + Segmentation + Severity)
- Anatomy-aware conditioning using FiLM
- Attention mechanism (CBAM: Channel + Spatial)
- Temporal consistency across video frames
- MC Dropout for uncertainty estimation
- Grad-CAM support for explainability
- Gradio interface for real-time inference

---

## 🧠 Architecture Highlights

- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Attention: CBAM (Channel + Spatial)
- Conditioning: FiLM (Anatomy-aware feature modulation)
- Auxiliary model: Anatomy CNN (location embedding)

Outputs:
- Detection head
- Segmentation head
- Severity classification head

---

## 📂 Repository Structure

```text
.
├── src/
│   ├── dataset.py                  # Data loading, preprocessing, and splits
│   ├── models.py                   # AnatomyCNN + ACDNet architecture
│   └── engine.py                   # Training, evaluation, MC Dropout, Grad-CAM
├── notebooks/
│   └── ACDNet_Pipeline.ipynb       # End-to-end pipeline
├── checkpoints/                    # Saved model weights
├── results/                        # Logs, plots, outputs
├── docs/reports/                   # Documentation and reports
├── requirements.txt
└── README.md
```

---

## ⚙️ System Requirements

- Python 3.10+
- PyTorch (GPU recommended)
- CUDA-enabled GPU (for training)

---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

This project uses the HyperKvasir dataset.

Dataset is not included in this repository.

Place dataset locally as:

```text
Dataset/
├── labeled-images/
├── labeled-videos/
└── segmented-images/
```

Optional environment variable:

```bash
export HYPERKVASIR_ROOT=path_to_dataset
```

On Windows PowerShell:

```powershell
$env:HYPERKVASIR_ROOT = "D:\path\to\Dataset"
```

---

## ▶️ Pipeline Execution

Run:

- notebooks/ACDNet_Pipeline.ipynb

Execution flow:
1. Install dependencies
2. Configure paths and device
3. Data preprocessing and splitting
4. Train Anatomy CNN
5. Build ACDNet
6. Train multi-task model
7. Load best checkpoint
8. Evaluate on test set
9. Run inference (image + UI)

---

## 🧪 Training Details

Severity classes (3-class):
- Class 0: G0-G1 (low severity)
- Class 1: G2
- Class 2: G3

Optimization:
- Mixed Precision (AMP)
- AdamW optimizer
- Differential learning rates

Losses:
- Detection: Binary Cross Entropy
- Segmentation: Dice + Focal
- Severity: Cross Entropy
- Temporal: Video consistency loss

---

## 📈 Outputs

After training:

```text
checkpoints/
├── anatomy_cnn_best.pth
└── acdnet_best.pth

results/
├── training_log.csv
├── anatomy_cnn_curves.png
├── acdnet_training_curves.png
└── inference_example.png
```

---

## ⚠️ Common Issues

1. Checkpoint mismatch

Ensure compatibility with the current 3-class severity setup.

2. Temporal loss inactive

Verify video loader is enabled and video data is detected in the dataset step.

3. GPU not detected

Check:

```python
torch.cuda.is_available()
```

---

## 🔁 Reproducibility

- Fixed random seed
- Logged training metrics per epoch
- Deterministic split strategy with leakage audit

---

## 🔮 Future Work

- Real-time deployment refinement (Gradio/production serving)
- Advanced temporal modeling for long sequences
- External clinical validation pipeline
- Model optimization for edge devices

---

## 📜 License and Dataset

- Code: As per repository license
- Dataset: Follow HyperKvasir usage guidelines

---
