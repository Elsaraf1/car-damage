# 🚗 Car Damage Detection

A deep learning project for detecting and classifying car damage using computer vision. The project covers multiple tasks — classification, object detection, and instance segmentation.

---

## 📌 Project Overview

Given an image of a damaged car, the models can:
- **Classify** what types of damage are present (multi-label classification)
- **Locate** exactly where the damage is (object detection with bounding boxes)
- **Outline** the exact shape of each damage (instance segmentation with masks)

---

## 🗂️ Repository Structure

```
car-damage/
├── classification/
│   └── car_damage_classification.ipynb       # Multi-label classification pipeline
├── object_detection/
│   └── car_damage_detection.ipynb            # YOLOv8 object detection pipeline
├── instance_segmentation/
│   └── car_damage_segmentation.ipynb         # YOLOv8 instance segmentation pipeline
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

## 📁 Task 1 — Multi-Label Classification

### Model
- **Backbone**: EfficientNet-B3 pretrained on ImageNet
- **Strategy**: Transfer learning — frozen early layers, fine-tuned last 3 blocks
- **Head**: Custom classifier (Linear → ReLU → Dropout → Linear)
- **Output**: 6 independent probabilities (one per damage class)

### Training
| Parameter | Value |
|-----------|-------|
| Epochs | 60 |
| Batch size | 32 |
| Optimizer | AdamW |
| Loss | BCEWithLogitsLoss (weighted) |
| Scheduler | CosineAnnealingLR |
| Image size | 224 × 224 |
| Device | GPU (CUDA) |

### Results (Test Set)

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

## 📁 Task 2 — Object Detection

### Model
- **Architecture**: YOLOv8m pretrained on COCO
- **Strategy**: Fine-tuned on CarDD dataset
- **Input**: COCO annotations converted to YOLO format
- **Output**: Bounding boxes + class labels + confidence scores

### Training
| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Device | GPU (CUDA) |
| Early stopping | patience = 15 |

### Results (Test Set)

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| Glass Shatter | 0.986 | 0.937 |
| Tire Flat | 0.936 | 0.902 |
| Lamp Broken | 0.889 | 0.781 |
| Dent | 0.618 | 0.373 |
| Scratch | 0.585 | 0.336 |
| Crack | 0.499 | 0.262 |
| **Overall** | **0.752** | **0.599** |

---

## 📁 Task 3 — Instance Segmentation

### Model
- **Architecture**: YOLOv8m-seg pretrained on COCO
- **Strategy**: Fine-tuned on CarDD dataset
- **Input**: COCO polygon annotations converted to YOLO segmentation format
- **Output**: Bounding boxes + pixel-level masks per damage instance

### Training
| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Device | GPU (CUDA) |
| Early stopping | patience = 15 |

### Results (Test Set)

| Class | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|-------|-----------|--------------|------------|---------------|
| Glass Shatter | 0.991 | 0.940 | 0.991 | 0.920 |
| Tire Flat | 0.895 | 0.884 | 0.895 | 0.887 |
| Lamp Broken | 0.885 | 0.778 | 0.885 | 0.769 |
| Dent | 0.631 | 0.387 | 0.642 | 0.362 |
| Scratch | 0.611 | 0.360 | 0.598 | 0.297 |
| Crack | 0.589 | 0.349 | 0.566 | 0.237 |
| **Overall** | **0.767** | **0.617** | **0.763** | **0.578** |

---

## 📈 All Tasks Comparison

| Class | Classification F1 | Detection mAP50 | Segmentation mAP50 |
|-------|------------------|-----------------|-------------------|
| Glass Shatter | 0.90 | 0.986 | 0.991 |
| Tire Flat | 0.85 | 0.936 | 0.895 |
| Lamp Broken | 0.66 | 0.889 | 0.885 |
| Dent | 0.75 | 0.618 | 0.642 |
| Scratch | 0.79 | 0.585 | 0.598 |
| Crack | 0.49 | 0.499 | 0.566 |
| **Overall** | **0.757** | **0.752** | **0.763** |

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/Elsaraf1/car-damage.git
cd car-damage
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nasimetemadi/car-damage-detection) and place it in the root folder.

3. Open the notebook for the task you want:
   - `classification/car_damage_classification.ipynb`
   - `object_detection/car_damage_detection.ipynb`
   - `instance_segmentation/car_damage_segmentation.ipynb`

4. For best results run on Kaggle with GPU:
   - Enable GPU: **Settings → Accelerator → GPU T4 x2**
   - Add the dataset directly from Kaggle

---

## 🔜 Roadmap

- [x] Multi-label classification (EfficientNet-B3)
- [x] Object detection (YOLOv8m)
- [x] Instance segmentation (YOLOv8m-seg)
- [ ] Salient object detection

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
