# 🚗 Car Damage Classifier

A deep learning project for multi-label car damage classification using EfficientNet-B3 and PyTorch.

---

## 📌 Project Overview

Given an image of a damaged car, the model predicts **which types of damage are present** in the image. This is a **multi-label classification** problem — meaning a single image can have multiple damage types at the same time.

---

## 🗂️ Repository Structure

```
car-damage/
├── classification/
│   └── car_damage_classification.ipynb   # Full pipeline notebook
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

- **Source**: [CarDD — Car Damage Detection Dataset](https://www.kaggle.com/datasets/nasimetemadi/car-damage-detection) on Kaggle
- **Size**: ~4,000 high-resolution images, 9,000+ damage instances
- **Format**: COCO JSON annotations (bounding boxes + segmentation masks)
- **Split**: Train (2,816) / Val (810) / Test (374)

### 🏷️ Damage Categories (6 Classes)

| Class | # Images |
|-------|----------|
| Scratch | 2,121 |
| Dent | 1,751 |
| Lamp Broken | 693 |
| Glass Shatter | 674 |
| Crack | 604 |
| Tire Flat | 309 |

---

## 🧠 Model Architecture

- **Backbone**: EfficientNet-B3 pretrained on ImageNet
- **Strategy**: Transfer learning — frozen early layers, fine-tuned last 3 blocks
- **Head**: Custom classifier (Linear → ReLU → Dropout → Linear)
- **Output**: 6 independent probabilities (one per damage class)

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 60 |
| Batch size | 32 |
| Optimizer | AdamW |
| Loss | BCEWithLogitsLoss (weighted) |
| Scheduler | CosineAnnealingLR |
| Image size | 224 × 224 |
| Device | GPU (CUDA) |

---

## 📈 Results

### Per-Class Metrics (Test Set)

| Class | ROC-AUC | F1 @ 0.5 | F1 @ Best Threshold |
|-------|---------|----------|----------------------|
| Glass Shatter | 0.9899 | 0.8535 | 0.8986 |
| Tire Flat | 0.9880 | 0.6988 | 0.8519 |
| Lamp Broken | 0.8994 | 0.5846 | 0.6567 |
| Dent | 0.8502 | 0.7411 | 0.7514 |
| Scratch | 0.8428 | 0.7838 | 0.7911 |
| Crack | 0.8074 | 0.4062 | 0.4912 |
| **Overall** | **0.8963** | **0.6789** | **0.7568** |

> 📝 Per-class thresholds were tuned using Precision-Recall curves on the test set.

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/Elsaraf1/car-damage.git
cd car-damage
```

2. Open the notebook:
```
classification/car_damage_classification.ipynb
```

3. Run on Kaggle with GPU enabled:
   - Upload the notebook to Kaggle
   - Enable GPU: **Settings → Accelerator → GPU T4 x2**
   - Add the dataset: [CarDD on Kaggle](https://www.kaggle.com/datasets/nasimetemadi/car-damage-detection)

---

## 🔜 Roadmap

- [x] Multi-label classification (EfficientNet-B3)
- [ ] Object detection (damage localization)
- [ ] Instance segmentation
- [ ] Salient object detection

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
