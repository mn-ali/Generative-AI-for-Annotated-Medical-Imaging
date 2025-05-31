import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import torch

# Map class names to indices
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

VALID_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia", "No Finding"
]

def load_image_labels(csv_path, images_dir):
    df = pd.read_csv(csv_path)
    image_paths = []
    labels_list = []

    for idx, row in df.iterrows():
        image_name = row['Image Index']
        labels_raw = row['Finding Labels']
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            continue  # skip missing files

        # Convert '|' separated string to list
        labels = labels_raw.split('|') if isinstance(labels_raw, str) else ["No Finding"]

        image_paths.append(image_path)
        labels_list.append(labels)

        # Print progress every 1000 rows
        if idx % 1000 == 0:
            print(f"CSV rows read: {len(df)}")
            print(f"Images found in directory: {len(os.listdir(images_dir))}")
            print(f"Image paths collected: {len(image_paths)}")

    return image_paths, labels_list


class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label_list = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        label_tensor = multi_hot_encode(label_list)

        return image, label_tensor


def get_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])


def get_dataloaders(csv_path, images_dir, batch_size=32, image_size=224):
    image_paths, labels = load_image_labels(csv_path, images_dir)
    transform = get_transforms(image_size)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    train_dataset = ChestXrayDataset(train_paths, train_labels, transform)
    val_dataset = ChestXrayDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    csv_path = "data/processed/filtered_metadata.csv"
    images_dir = "data/raw/xray_dataset/images"  # all images merged here

    train_loader, val_loader = get_dataloaders(csv_path, images_dir)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Print example batch
    for images, labels in train_loader:
        print(f"Batch image tensor shape: {images.shape}")
        print(f"Batch labels example: {labels[0]}")
        break