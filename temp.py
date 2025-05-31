import pandas as pd

# Load the CSV
df = pd.read_csv("data/processed/filtered_metadata.csv")

# Preview
print(df.head())