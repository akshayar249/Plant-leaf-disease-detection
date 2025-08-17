

# 🌿 Plant Leaf Disease Detection (Streamlit App)

A web-based app to detect plant leaf diseases using a hybrid deep learning model. It analyzes an uploaded leaf image, compares it with a healthy reference, predicts the health status, and shows infected regions visually.

---

## 📱 App Features

* 📤 **Upload a leaf image** (JPEG, PNG)
* 🟢 **Select a healthy reference leaf** from a local dataset
* 🧠 **Model predicts**: Healthy or Diseased
* 📊 **Infection level (%)** estimated from texture deviation
* 🔬 **Infection heatmap overlay** on the original image (red/orange = more infection)
* 💡 **Plant care suggestions** based on severity
* 🎨 Clean and interactive **Streamlit UI** with blue theme

---

## 🧠 How the Model Works

The model is a **hybrid classifier** that combines:

### 1. 🧠 CNN (Convolutional Neural Network)

* Learns visual features from the **RGB leaf image** (e.g., color, shape, edges)
* Architecture includes convolution, ReLU, max pooling, and flatten layers
* Input image is resized to **64×64**

### 2. 📊 GLCM Texture Features

Extracted from the **grayscale version** of the leaf image using Gray-Level Co-occurrence Matrix (GLCM).

These 6 statistical features are used:

| Feature           | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| **Contrast**      | Measures local intensity variations (detects spotty patterns) |
| **Dissimilarity** | Highlights differences in neighboring pixel intensities       |
| **Homogeneity**   | Captures image uniformity (lower in diseased leaves)          |
| **Energy**        | Represents textural uniformity                                |
| **Correlation**   | Measures linear dependencies between pixels                   |
| **ASM**           | Angular Second Moment — another uniformity indicator          |

---

## 🔀 Model Pipeline

1. **Image Input**

   * RGB image passed to CNN
   * Grayscale image processed for GLCM features

2. **Feature Fusion**

   * CNN features and GLCM features are **concatenated**

3. **Fully Connected Layers**

   * Dense layers classify the input as:

     * `0`: Diseased
     * `1`: Healthy

---

## 📦 Required Libraries

Install all dependencies:

```bash
pip install streamlit torch torchvision opencv-python-headless scikit-image matplotlib pillow numpy
```

---

## 📁 Folder Setup

```
project/
├── app.py                  # Streamlit app
├── hybrid_cnn_glcm.pth     # Trained PyTorch model
└── dataset/
    └── test/
        └── healthy/        # Healthy reference leaf images (JPEG/PNG)
```

> The model will use one of the healthy leaves as a reference for infection comparison.

---

## 🚀 How to Run

1. Place your trained model file as `hybrid_cnn_glcm.pth` in the project folder
2. Add healthy images to `dataset/test/healthy/`
3. Launch the app:

```bash
streamlit run app.py
```

<img width="1772" height="968" alt="Screenshot (122)" src="https://github.com/user-attachments/assets/ee06fb20-2783-4793-a194-22a0dd140ab4" />
<img width="969" height="858" alt="Screenshot (123)" src="https://github.com/user-attachments/assets/657b962b-9ea7-4843-9fdd-624272401ddc" />
<img width="883" height="860" alt="Screenshot (124)" src="https://github.com/user-attachments/assets/97710726-662d-4882-baf1-dbebfcf60ba8" />
<img width="998" height="724" alt="Screenshot (125)" src="https://github.com/user-attachments/assets/3ecd50e8-fe29-4ea6-aeb8-fbf7c618b6a9" />





