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
LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
          'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
          'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}

def multi_hot_encode(labels):
    vector = torch.zeros(len(VALID_LABELS), dtype=torch.float32)
    for label in labels:
        if label in LABEL_TO_INDEX:
            vector[LABEL_TO_INDEX[label]] = 1.0
    return vector


# ---------- DATASET CLASS ---------- #

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['Image Index'])

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ Failed to load image at {image_path}: {e}")
            return None  # or raise an error

        if self.transform:
            image = self.transform(image)

        labels = row['Finding Labels'].split('|')
        label_vector = torch.zeros(len(LABELS), dtype=torch.float32)
        for label in labels:
            if label in LABEL_MAP:
                label_vector[LABEL_MAP[label]] = 1.0

        return image, label_vector


# ---------- TRANSFORMS ---------- #

def get_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])


# ---------- LOADING OFFICIAL SPLITS ---------- #

def load_official_split(csv_path, image_dir, train_val_list, test_list,
                        batch_size, image_size, train_transform=None, val_transform=None):
    
    df = pd.read_csv(csv_path)
    with open(train_val_list, 'r') as f:
        train_ids = set(line.strip() for line in f)
    with open(test_list, 'r') as f:
        test_ids = set(line.strip() for line in f)

    df_train_val = df[df['Image Index'].isin(train_ids)]
    df_test = df[df['Image Index'].isin(test_ids)]

    # Only assign default transforms if not provided
    if train_transform is None:
        train_transform = get_transforms(image_size)
    if val_transform is None:
        val_transform = get_transforms(image_size)

    train_dataset = ChestXrayDataset(df_train_val, image_dir, transform=train_transform)
    val_dataset = ChestXrayDataset(df_test, image_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------- ENTRY POINT ---------- #

if __name__ == "__main__":
    csv_path = "data/raw/xray_dataset/Data_Entry_2017.csv"
    images_dir = "data/raw/xray_dataset/images/"
    train_val_list = "data/raw/xray_dataset/train_val_list.txt"
    test_list = "data/raw/xray_dataset/test_list.txt"

    train_loader, test_loader = load_official_split(
        csv_path,
        images_dir,
        train_val_list,
        test_list,
        batch_size=32,
        image_size=224
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels example: {labels[0]}")
        break