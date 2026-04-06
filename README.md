# Medical Image Segmentation Framework

An anatomy-conditioned deep learning pipeline for colonoscopy image analysis using HyperKvasir data.

This project trains and evaluates ACDNet for:
- Polyp detection
- Polyp segmentation
- Ulcerative colitis severity grading (3-class)
- Temporal consistency learning from video clips

## Project Structure

- `src/`: Core Python modules (`dataset.py`, `models.py`, `engine.py`)
- `notebooks/`: End-to-end notebook pipeline (`ACDNet_Pipeline.ipynb`)
- `Dataset/`: HyperKvasir data and segmentation assets
- `checkpoints/`: Saved model weights
- `results/`: Training curves, logs, and inference outputs
- `docs/reports/`: Archived implementation/progress reports

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for training
- Install dependencies from `requirements.txt`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Pipeline

Open and run `notebooks/ACDNet_Pipeline.ipynb` in order:

1. Cell 1: Install dependencies
2. Cell 2: Configure paths and device
3. Cell 3: Build datasets and splits
4. Cell 4: Train anatomy CNN
5. Cell 5: Build ACDNet
6. Cell 6: Train ACDNet
7. Cell 7: Load best checkpoint
8. Cell 8: Evaluate on test set
9. Cell 9/10: Inference and UI

## Notes

- Severity grading is configured as 3-class (merged low-severity group).
- Temporal loss is expected to be enabled during ACDNet training when video loader is available.
- Ensure checkpoint compatibility when switching between 4-class and 3-class model variants.
