

# 🌿 Plant Leaf Disease Detection (CNN + GLCM Hybrid Model)

## 📌 Overview

This repository contains the implementation of a **hybrid deep learning model** for plant leaf disease detection. The system combines **Convolutional Neural Networks (CNNs)** with **GLCM (Gray Level Co-occurrence Matrix) texture features**, enhancing the ability of the model to capture both **spatial patterns** (via CNN) and **texture-based features** (via GLCM).

This approach achieves **high accuracy** while remaining lightweight, making it suitable for deployment in **resource-constrained environments** like mobile or embedded devices.

---

## ⚡ Key Features

* 🌱 **Image Preprocessing**

  * Resizing and normalization
  * Data augmentation (rotation, flip, brightness adjustment)

* 🧮 **Feature Extraction**

  * GLCM features (contrast, correlation, energy, homogeneity) computed per leaf image
  * Features concatenated with CNN-learned embeddings

* 🤖 **Model Architecture**

  * Lightweight **CNN backbone** for image feature extraction
  * **GLCM features as additional input channels**
  * Fully connected layers for final classification

* 📊 **Performance Optimized**

  * Hybrid CNN+GLCM model outperforms CNN-only and GLCM-only models

---

## 📂 Dataset Structure

This project expects the dataset to be organized as:

```
dataset/
│── train/
│   ├── healthy/
│   ├── diseased/
│── val/
│   ├── healthy/
│   ├── diseased/
│── test/
    ├── healthy/
    ├── diseased/
```

* Compatible with **PlantVillage dataset** or custom leaf datasets.

---

## 📊 Results

| Model                 | Accuracy | Key Strengths                                         |
| --------------------- | -------- | ----------------------------------------------------- |
| **CNN-only**          | 91%      | Learns spatial patterns, but misses texture info      |
| **GLCM + MLP**        | 88%      | Lightweight, interpretable, uses handcrafted features |
| **Hybrid CNN + GLCM** | **95%**  | Best trade-off, combines CNN + texture features       |

---

## 🔮 Future Scope

* 📱 Deployment as a **mobile app** for real-time farmer usage
* ☁️ Integration with **cloud dashboards** for crop health monitoring
* 🖼️ Expansion to **multi-disease classification across crops**
* 🌍 Implementation on **edge devices (Raspberry Pi, Jetson Nano)**
* 🔎 Incorporate **explainability methods (Grad-CAM, SHAP)** for trust in predictions

---

Do you also want me to include a **training & inference workflow diagram** (like the fall detection one) that shows **Image → Preprocessing → GLCM + CNN → Hybrid Model → Prediction** for your README?
