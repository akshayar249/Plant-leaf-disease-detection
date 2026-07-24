import os
import cv2
import torch
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# === Hyperparameters ===
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # change if needed

# === GLCM feature extractor ===
def compute_glcm_features(gray_img):
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    return np.array(features, dtype=np.float32)

# === Custom dataset with GLCM ===
class GLCMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        glcm_features = compute_glcm_features(gray_resized)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(glcm_features), label

# === Hybrid CNN + GLCM model ===
class HybridCNNGLCM(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNGLCM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Flatten()
        )
        self.flattened_size = 32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size + 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, glcm_features):
        x = self.cnn(x)
        combined = torch.cat((x, glcm_features), dim=1)
        out = self.fc(combined)
        return out

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# === Loaders ===
train_dataset = GLCMDataset('C:/Users/aksha/Downloads/dataset/train', transform)
val_dataset = GLCMDataset('C:/Users/aksha/Downloads/dataset/val', transform)
test_dataset = GLCMDataset('C:/Users/aksha/Downloads/dataset/test', transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === Print class order mapping to verify ===
print("Class to index mapping:", train_dataset.dataset.class_to_idx)
# This dictionary shows exactly which label corresponds to which folder name
# e.g. {'diseased': 0, 'healthy': 1}

# === Training setup ===
model = HybridCNNGLCM(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for images, glcm, labels in train_loader:
        images, glcm, labels = images.to(DEVICE), glcm.to(DEVICE), labels.to(DEVICE)
        outputs = model(images, glcm)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Train Acc: {acc:.4f}")

# === Evaluation function ===
def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, glcm, labels in loader:
            images, glcm, labels = images.to(DEVICE), glcm.to(DEVICE), labels.to(DEVICE)
            outputs = model(images, glcm)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)

print("Validation Accuracy:", evaluate(val_loader))
print("Test Accuracy:", evaluate(test_loader))

# === Save model function ===
def save_model(model, path='hybrid_cnn_glcm.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Save the model after training
save_model(model)

# === Prediction function on test data with correct class names ===
def predict(model, loader, class_names):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, glcm, _ in loader:
            images, glcm = images.to(DEVICE), glcm.to(DEVICE)
            outputs = model(images, glcm)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)

    # Print predictions for each test image
    for i, pred in enumerate(all_preds):
        print(f"Image {i+1}: Predicted class - {class_names[pred]}")

# Use the class names dynamically from dataset to keep the order consistent
class_names = train_dataset.dataset.classes  # this gets the folder names in correct order

# Run prediction on test data
predict(model, test_loader, class_names)
