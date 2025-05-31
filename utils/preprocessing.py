import os
import pandas as pd
from tqdm import tqdm

# Valid labels (14 diseases + 'No Finding')
VALID_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia", "No Finding"
]

def clean_and_filter_metadata(csv_path, images_dir, output_path):
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Total rows before cleaning: {len(df)}")
    df = df.drop_duplicates(subset='Image Index')

    filtered_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning metadata"):
        image_name = row['Image Index']
        labels_raw = row['Finding Labels']

        # Standardize labels
        if pd.isna(labels_raw) or labels_raw.strip() == "":
            labels = ["No Finding"]
        else:
            labels = [label.strip() for label in labels_raw.split('|')]

        final_labels = []
        for label in labels:
            if label == "Nodule Mass":
                final_labels.extend(["Nodule", "Mass"])
            elif label == "Pleural Thickening":
                final_labels.append("Pleural_Thickening")
            elif label in VALID_LABELS:
                final_labels.append(label)
            elif label == "No Finding":
                final_labels.append("No Finding")
            else:
                # Unknown label, skip it
                pass

        # Verify image exists
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            # Skip missing images
            continue

        # Save cleaned row with final labels joined by '|'
        new_row = row.to_dict()
        new_row['Finding Labels'] = '|'.join(final_labels)
        filtered_rows.append(new_row)

    filtered_df = pd.DataFrame(filtered_rows)
    print(f"Rows after cleaning and filtering: {len(filtered_df)}")

    # Save to CSV
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered metadata saved to: {output_path}")


if __name__ == "__main__":
    csv_path = "data/raw/xray_dataset/Data_Entry_2017.csv"
    images_dir = "data/raw/xray_dataset/images"  # all images merged here
    output_path = "data/processed/filtered_metadata.csv"

    clean_and_filter_metadata(csv_path, images_dir, output_path)