import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# ---------- LABEL HANDLING ---------- #

VALID_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia", "No Finding"
]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(VALID_LABELS)}

def multi_hot_encode(labels):
    vector = torch.zeros(len(VALID_LABELS), dtype=torch.float32)
    for label in labels:
        if label in LABEL_TO_INDEX:
            vector[LABEL_TO_INDEX[label]] = 1.0
    return vector


# ---------- DATASET CLASS ---------- #

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None):
        self.df = dataframe
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, row["Image Index"])
        image = Image.open(image_path).convert("RGB")

        labels = row["Finding Labels"].split('|') if isinstance(row["Finding Labels"], str) else ["No Finding"]
        label_tensor = multi_hot_encode(labels)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor


# ---------- TRANSFORMS ---------- #

def get_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])


# ---------- LOADING OFFICIAL SPLITS ---------- #

def load_official_split(csv_path, images_dir, train_val_list, test_list, batch_size=32, image_size=224):
    df = pd.read_csv(csv_path)

    with open(train_val_list, 'r') as f:
        train_val_files = set(line.strip() for line in f)

    with open(test_list, 'r') as f:
        test_files = set(line.strip() for line in f)

    df_train_val = df[df["Image Index"].isin(train_val_files)].reset_index(drop=True)
    df_test = df[df["Image Index"].isin(test_files)].reset_index(drop=True)

    transform = get_transforms(image_size)

    train_dataset = ChestXrayDataset(df_train_val, images_dir, transform)
    test_dataset = ChestXrayDataset(df_test, images_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ---------- ENTRY POINT ---------- #

if __name__ == "__main__":
    csv_path = "data/raw/xray_dataset/Data_Entry_2017.csv"
    images_dir = "data/raw/xray_dataset/images/"
    train_val_list = "data/raw/xray_dataset/train_val_list.txt"
    test_list = "data/raw/xray_dataset/test_list.txt"

    train_loader, test_loader = load_official_split(csv_path, images_dir, train_val_list, test_list)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels example: {labels[0]}")
        break