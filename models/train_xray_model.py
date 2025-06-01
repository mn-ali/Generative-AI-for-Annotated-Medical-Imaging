import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score
from data_loader_xray import get_dataloaders
from tqdm import tqdm


# Paths
CSV_PATH = "data/processed/filtered_metadata.csv"
IMAGE_DIR = "data/raw/xray_dataset/images"
MODEL_PATH = "models/xray_model.pt"

# Config
BATCH_SIZE = 128
IMAGE_SIZE = 224
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 15

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data loaders
train_loader, val_loader = get_dataloaders(CSV_PATH, IMAGE_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")

# Model setup
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)


# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Helper functions
def calculate_f1(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(int)
    return f1_score(y_true, y_pred_bin, average='micro')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # Wrap dataloader with tqdm for progress bar
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    f1 = calculate_f1(all_labels, all_outputs)

    return running_loss / len(dataloader.dataset), f1

# Training loop
best_val_f1 = 0
os.makedirs("models", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    print("Starting training loop...")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_PATH)
        print("  ✅ Saved best model!")

print("Training complete.")