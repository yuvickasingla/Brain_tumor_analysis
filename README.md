# Brain Tumor Classification — BRISC2025

Multi-model deep learning system for brain tumor classification using the **BRISC2025** dataset.  
Compares a **Custom CNN** (baseline) against **ResNet-50** and **EfficientNet-B3** (transfer learning).

---

## What's New vs. Previous Work

| Feature | Previous (Groupmate) | This Project |
|---|---|---|
| Dataset | Kaggle Nickparvar | **BRISC2025** (expert-annotated, multi-plane) |
| Model | Custom CNN only | Custom CNN **+ ResNet-50 + EfficientNet-B3** + MobileNet |
| Transfer Learning | ❌ None | ✅ 2-phase fine-tuning |
| LR Scheduler | ❌ None | ✅ ReduceLROnPlateau |
| Early Stopping | ❌ None | ✅ Patience-based |
| Metrics | Accuracy + CM | **+ AUC-ROC + Macro F1** |
| App | Single model | **Model selector dropdown** |
| Comparison | ❌ None | ✅ `compare.py` bar chart |

---

## Dataset — BRISC2025

**6,000** contrast-enhanced T1-weighted MRI scans  
**4 classes:** glioma · meningioma · pituitary · no_tumor  
**Planes:** axial, sagittal, coronal  
**Pre-split:** 5,000 train / 1,000 test (official)

Download: https://www.kaggle.com/datasets/briscdataset/brisc2025

Place extracted folder at:
```
data/brisc2025/
└── classification_task/
    ├── train/   (glioma/ meningioma/ no_tumor/ pituitary/)
    └── test/    (glioma/ meningioma/ no_tumor/ pituitary/)
```

---

## Setup

```bash
git clone <repo>
cd brain_tumor_brisc
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Workflow (Day-by-Day Plan)

### Day 1 — Train all models
```bash
# Baseline CNN
python train.py --model custom_cnn --epochs 25

# ResNet-50 (transfer learning, 2-phase automatic)
python train.py --model resnet50 --epochs 30

# EfficientNet-B3
python train.py --model efficientnet --epochs 30

#MobilNet
python train.py --model mobilenetv3    
```
Checkpoints saved to `models/<name>_best.pth`

### Day 2 — Evaluate and compare
```bash
python eval.py --model custom_cnn
python eval.py --model resnet50
python eval.py --model efficientnet
python eval.py --model mobilenetv3    
python compare.py        # generates comparison chart
```
Outputs saved to `outputs/`

### Day 3 — Streamlit demo
```bash
streamlit run app.py
```
Open http://localhost:8501 · select model · upload MRI · view Grad-CAM

---

## Project Structure
```
brain_tumor_brisc/
├── data/
│   └── brisc2025/          ← put the dataset here
├── models/                 ← saved checkpoints (.pth)
├── logs/                   ← training CSVs per model
├── outputs/                ← confusion matrices, ROC curves, summary CSV
├── config.py               ← all hyperparameters & paths
├── data_loader.py          ← BRISC2025 data pipeline
├── model.py                ← CustomCNN / ResNet50 / EfficientNet / MobileNet
├── train.py                ← training script (supports all 4 models)
├── eval.py                 ← evaluation: accuracy, F1, AUC-ROC
├── compare.py              ← side-by-side model comparison chart
├── gradcam.py              ← Grad-CAM (works with all models)
├── app.py                  ← Streamlit web app
├── utils.py                ← device, seeding
└── requirements.txt
```

---

## Transfer Learning Strategy

**Phase 1 (epochs 0 → N/2):**  Backbone frozen → only classifier head trains at `lr=1e-4`

**Phase 2 (epochs N/2 → N):**  Last 2-3 backbone blocks unfrozen → fine-tuned at `lr=1e-5`

This prevents catastrophic forgetting and converges faster than training from scratch.

---


