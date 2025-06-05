import kagglehub

# Download latest version
path = kagglehub.dataset_download("awsaf49/brats2020-training-data")

print("data/raw/mri_dataset", path)