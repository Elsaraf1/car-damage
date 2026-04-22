---
title: Car Damage Detector
emoji: 🚗
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🚗 Car Damage Detector

Upload a photo of a damaged car and get results from **4 AI models simultaneously**:

| Model | Task | Architecture |
|-------|------|-------------|
| 🏷️ Classification | What damage types are present | EfficientNet-B3 |
| 📦 Detection | Where is the damage (bounding boxes) | YOLOv8m |
| 🎭 Segmentation | Exact damage shape (pixel masks) | YOLOv8m-seg |
| 🌊 Saliency Map | Overall damage region highlight | U2-Net (lite) |

### Damage Classes
Dent · Scratch · Crack · Glass Shatter · Lamp Broken · Tire Flat

### Dataset
Trained on the [CarDD dataset](https://www.kaggle.com/datasets/nasimetemadi/car-damage-detection) — ~4,000 images, 9,000+ damage instances.
