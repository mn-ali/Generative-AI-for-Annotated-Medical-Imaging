import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121
from sklearn.metrics import f1_score
from tqdm import tqdm
from data_loader_xray import load_official_split

# Paths
CSV_PATH = "data/raw/xray_dataset/Data_Entry_2017.csv"
IMAGE_DIR = "data/raw/xray_dataset/images/"
TRAIN_VAL_LIST = "data/raw/xray_dataset/train_val_list.txt"
TEST_LIST = "data/raw/xray_dataset/test_list.txt"
MODEL_PATH = "models/densenet121_best.pt"
CHECKPOINT_PATH = "checkpoints/densenet121_latest.pth"

# Config
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 15

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data loaders
train_loader, val_loader = load_official_split(
    CSV_PATH,
    IMAGE_DIR,
    TRAIN_VAL_LIST,
    TEST_LIST,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    train_transform=transform_train,
    val_transform=transform_val
)

# Model setup
model = densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# Load checkpoint if exists
start_epoch = 0
best_val_f1 = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_f1 = checkpoint.get('best_val_f1', 0)
    print(f"✅ Loaded checkpoint from epoch {start_epoch}, best F1: {best_val_f1:.4f}")
else:
    print("🚀 Starting fresh training...")

# Helper functions
def calculate_f1(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(int)
    return f1_score(y_true, y_pred_bin, average='micro')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
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

# Ensure dirs exist
os.makedirs("models", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Training loop
for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\n📍 Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
    scheduler.step(val_f1)

    print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'best_val_f1': best_val_f1
    }, CHECKPOINT_PATH)
    print(f"💾 Checkpoint saved: {CHECKPOINT_PATH}")

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"🏅 New best model saved to: {MODEL_PATH}")

print("\n🎉 Training complete.")