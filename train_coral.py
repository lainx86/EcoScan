import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# CONFIG
# =========================
DATA_DIR = "samples/Dataset"
MODEL_NAME = "coral_bleaching_resnet18.pt"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
FINE_TUNE_EPOCHS = 5

LR = 1e-4
FINE_TUNE_LR = 1e-5
NUM_CLASSES = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS
# =========================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET & DATALOADER
# =========================
train_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_tf
)
val_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "valid"),
    transform=val_tf
)
test_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=val_tf
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Class mapping:", train_ds.class_to_idx)

# =========================
# MODEL
# =========================
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# =========================
# TRAINING (HEAD ONLY)
# =========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    running_loss /= len(train_ds)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f}")

# =========================
# FINE-TUNING (UNFREEZE layer4)
# =========================
print("\nStarting fine-tuning...")

for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=FINE_TUNE_LR
)

for epoch in range(FINE_TUNE_EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Fine-tune {epoch+1}/{FINE_TUNE_EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    running_loss /= len(train_ds)
    print(f"[Fine-tune] Epoch {epoch+1} | Loss: {running_loss:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), MODEL_NAME)
print(f"\nModel saved as {MODEL_NAME}")

# =========================
# TEST EVALUATION (ILMIAH)
# =========================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=train_ds.classes
))

# =========================
# SINGLE IMAGE INFERENCE
# =========================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    label = train_ds.classes[pred_idx]
    confidence = probs[pred_idx].item()
    return label, confidence

# Example:
# label, conf = predict_image("karang.jpg")
# print(label, conf)
