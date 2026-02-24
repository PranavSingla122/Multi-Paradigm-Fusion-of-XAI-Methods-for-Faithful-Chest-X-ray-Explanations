# Multi-Paradigm Fusion of XAI Methods for Faithful Chest X-ray Explanations

> Official implementation of the MICCAI paper: *Multi-Paradigm Fusion of XAI Methods for Faithful Chest X-ray Explanations*

---

## Overview

This repository presents a modular framework for fusing complementary post-hoc XAI methods to produce more faithful and robust saliency explanations for chest X-ray classification. We integrate gradient-based, perturbation-based, and game-theoretic attribution methods through two fusion strategies:

- **Attention-Guided Fusion** — temperature-scaled softmax weighting for spatially adaptive integration
- **Hierarchical Multi-Scale Pyramid Fusion** — scale-specific fusion at resolutions {14, 28, 56, 112} with cross-scale weighted aggregation

Evaluated on the **COVID-19 Radiography Database** and **NIH ChestXray14** using insertion–deletion faithfulness and robustness metrics.

---

## Results

### COVID-19 Radiography

| Method | Insert ↑ | Delete ↓ | SSIM ↑ | RankCorr ↑ |
|---|---|---|---|---|
| Grad-CAM | 0.630 | 0.498 | 0.581 | 0.269 |
| Grad-CAM++ | 0.588 | 0.488 | 0.545 | 0.155 |
| Integrated Gradients | 0.682 | 0.419 | 0.495 | 0.119 |
| RISE | 0.603 | 0.420 | 0.352 | 0.073 |
| LIME | 0.664 | 0.474 | 0.323 | 0.332 |
| Attention Fusion | 0.670 | 0.507 | 0.454 | 0.449 |
| **Pyramid Fusion** | **0.690** | 0.524 | 0.563 | **0.492** |

### NIH ChestXray14

| Method | Insert ↑ | Delete ↓ | SSIM ↑ | RankCorr ↑ |
|---|---|---|---|---|
| Grad-CAM | 0.585 | 0.611 | 0.418 | 0.276 |
| Grad-CAM++ | 0.575 | 0.629 | 0.379 | 0.084 |
| **Integrated Gradients** | **0.637** | **0.411** | **0.465** | 0.231 |
| RISE | 0.450 | 0.309 | 0.328 | 0.056 |
| LIME | 0.508 | 0.435 | 0.357 | 0.285 |
| Attention Fusion | 0.562 | 0.375 | 0.375 | 0.286 |
| Pyramid Fusion | 0.589 | 0.408 | 0.458 | **0.317** |

---

## Repository Structure

```
├── xai_covid/               # XAI experiments on COVID-19 Radiography dataset
│   └── requirements.txt
├── xai_nih/                 # XAI experiments on NIH ChestXray14 dataset
│   └── requirements.txt
├── LICENSE
└── README.md
```

---

## Datasets

- **COVID-19 Radiography Database** — 21,165 images, 4 classes. Available at [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).
- **NIH ChestXray14** — 112,120 images, 15 pathology labels. Available at [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data).

Place datasets under a `data/` directory:
```
data/
├── covid19/
└── nih_chestxray14/
```

---

## Model Weights

| Dataset | Download |
|---|---|
| COVID-19 Radiography | [Google Drive](https://drive.google.com/file/d/1_BDhenYSgy-0fdhZnNkkYFR5dGn8_anh/view?usp=sharing) |
| NIH ChestXray14 | [Google Drive](https://drive.google.com/file/d/1ufUIHaLQSiFKf4ZvkiG3MKfqSXmmIcLs/view?usp=sharing) |

---

## Setup

```bash
git clone https://github.com/PranavSingla122/Multi-Paradigm-Fusion-of-XAI-Methods-for-Faithful-Chest-X-ray-Explanations.git
cd Multi-Paradigm-Fusion-of-XAI-Methods-for-Faithful-Chest-X-ray-Explanations
```

Install dependencies for the dataset you want to run:

```bash
# For COVID-19
pip install -r xai_covid/requirements.txt

# For NIH ChestXray14
pip install -r xai_nih/requirements.txt
```

---

## Usage

### COVID-19 Radiography

```bash
cd xai_covid
python main.py --mode xai_only --model_path outputs/models/best_model_auroc.pth --num_samples 100 --batch_size 1
```

### NIH ChestXray14

```bash
cd xai_nih
python main.py --mode xai_only --model_path outputs/models/best_model_auroc.pth --num_samples 100 --batch_size 1
```

---

## Implementation Details

| Setting | COVID-19 | NIH ChestXray14 |
|---|---|---|
| Batch size | 64 | 256 |
| Learning rate | 1×10⁻⁴ | 3×10⁻⁴ |
| Epochs | 50 | 50 |
| Input size | 224×224 | 224×224 |
| RISE masks | 6,000 | 8,000 |
| LIME samples | 300 | 300 |
| SHAP background | 50 | 50 |
| IG steps | 64 | 64 |
| Optimizer | AdamW (wd=1e-5) | AdamW (wd=1e-5) |
| Random seed | 42 | 42 |

---

## Citation

```bibtex
@inproceedings{xaifusion2025miccai,
  title     = {Multi-Paradigm Fusion of XAI Methods for Faithful Chest X-ray Explanations},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

---

## License

Apache-2.0
