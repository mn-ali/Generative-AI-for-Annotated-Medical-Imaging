from models.data_loader_xray import load_splits_and_labels, ChestXrayDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Paths
csv_path = "data/raw/Data_Entry_2017.csv"
train_val_list = "data/raw/train_val_list.txt"
test_list = "data/raw/test_list.txt"
image_dir = "data/raw/images/"

# Load official NIH split
train_val_df, train_val_labels, test_df, test_labels, classes = load_splits_and_labels(
    csv_path, train_val_list, test_list
)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets
train_dataset = ChestXrayDataset(train_val_df, train_val_labels, image_dir, transform)
test_dataset = ChestXrayDataset(test_df, test_labels, image_dir, transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Optional: print class list
print("Classes:", classes)