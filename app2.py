import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import glob
import os

# === Configuration ===
IMAGE_SIZE = 64
DISPLAY_SIZE = 256
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === GLCM Feature Extractor ===
def compute_glcm_features(gray_img):
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ], dtype=np.float32)

# === Model Definition ===
class HybridCNNGLCM(torch.nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNGLCM, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten()
        )
        self.flattened_size = 32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.flattened_size + 6, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x, glcm_features):
        x = self.cnn(x)
        combined = torch.cat((x, glcm_features), dim=1)
        return self.fc(combined)

# === Load Trained Model ===
model = HybridCNNGLCM(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("hybrid_cnn_glcm.pth", map_location=DEVICE))
model.eval()

# === Load Healthy Leaf Dataset ===
healthy_folder = r"C:\Users\aksha\Downloads\dataset\test\healthy"
healthy_paths = glob.glob(os.path.join(healthy_folder, "*.jpg"))
healthy_images = []
healthy_features = []

for path in healthy_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    healthy_images.append((os.path.basename(path), img))
    healthy_features.append(compute_glcm_features(gray))

# === Streamlit UI Setup ===
st.set_page_config(page_title="Leaf Disease Detection", layout="centered")
st.markdown(
    """
    <style>
    body { background-color: #1E90FF; color: white; }
    .stButton>button { background-color: #32CD32; color: white; font-weight: bold; }
    .reportview-container .main { background-color: #1E90FF; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🌿 Leaf Disease Detection")

# === Upload Section ===
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_column_width=True)

    # === Healthy Reference Selector ===
    healthy_names = [name for name, _ in healthy_images]
    selected_name = st.selectbox("🟢 Select a reference healthy leaf", healthy_names)
    selected_index = healthy_names.index(selected_name)
    ref_img = healthy_images[selected_index][1]
    ref_features = healthy_features[selected_index]

    st.image(ref_img, caption="Reference Healthy Leaf", use_column_width=True)

    # === Preprocess for Model ===
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    glcm_features = compute_glcm_features(gray)
    glcm_tensor = torch.tensor(glcm_features).unsqueeze(0).to(DEVICE)

    # === Prediction ===
    with torch.no_grad():
        output = model(img_tensor, glcm_tensor)
        pred_class = output.argmax(1).item()

    class_names = ["Diseased", "Healthy"]
    st.subheader(f"📊 Predicted Class: **{class_names[pred_class]}**")

    # === Infection Level ===
    diff = np.abs(glcm_features - ref_features)
    infection_percent = np.mean(diff / (ref_features + 1e-6)) * 100
    infection_percent = np.clip(infection_percent, 0, 100)
    st.metric(label="🦠 Estimated Infection Level", value=f"{infection_percent:.2f}%")

    # === Infection Heatmap Overlay ===
    with st.expander("📸 View Infection Highlight"):
        ref_display_gray = cv2.cvtColor(cv2.resize(ref_img, (DISPLAY_SIZE, DISPLAY_SIZE)), cv2.COLOR_BGR2GRAY)
        input_display_rgb = cv2.resize(np.array(img), (DISPLAY_SIZE, DISPLAY_SIZE))
        input_display_gray = cv2.cvtColor(input_display_rgb, cv2.COLOR_RGB2GRAY)

        diff_img = cv2.absdiff(input_display_gray, ref_display_gray)
        heatmap = cv2.applyColorMap(cv2.convertScaleAbs(diff_img, alpha=3), cv2.COLORMAP_TURBO)

        overlay = cv2.addWeighted(input_display_rgb, 0.55, heatmap, 0.45, 0)
        st.image(overlay, caption="🔬 Infection Highlight (Red/Orange = Infected)", use_column_width=True)

    # === Suggestion Box ===
    with st.expander("🧪 Suggestions to Improve Leaf Health"):
        if infection_percent < 20:
            st.success("✅ Leaf is mostly healthy. Keep monitoring and maintain current care.")
        elif infection_percent < 50:
            st.warning("⚠️ Mild infection detected. Consider:")
            st.markdown("""
            - Use mild organic fungicide  
            - Ensure adequate sunlight and airflow  
            - Avoid water stagnation and overwatering  
            """)
        else:
            st.error("❌ Severe infection detected. Immediate action recommended:")
            st.markdown("""
            - Remove infected leaves to prevent spread  
            - Use appropriate chemical treatment (consult expert)  
            - Improve soil drainage and leaf dryness  
            - Monitor other plants nearby  
            """)
else:
    st.info("Please upload a leaf image to start the detection.")
