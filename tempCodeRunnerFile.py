import pandas as pd
import os

# Paths
image_folders = [f"data/raw/xray_dataset/images_{str(i).zfill(3)}" for i in range(1,13)]
metadata_path = "data/raw/xray_dataset/Data_Entry_2017.csv"

# Load metadata
metadata = pd.read_csv(metadata_path)

# Function to find which folder an image is in
def find_image_folder(image_name):
    for folder in image_folders:
        if os.path.exists(os.path.join(folder, image_name)):
            return folder
    return None  # Image not found

# Add a column for full relative path
metadata['Image Folder'] = metadata['Image Index'].apply(find_image_folder)

# Filter out images that are missing (folder not found)
filtered_metadata = metadata.dropna(subset=['Image Folder']).copy()

# Create a 'Full Path' column for easier image loading later
filtered_metadata['Full Path'] = filtered_metadata.apply(
    lambda row: os.path.join(row['Image Folder'], row['Image Index']),
    axis=1
)

# Save filtered metadata with full paths
filtered_metadata.to_csv("data/processed/filtered_metadata.csv", index=False)