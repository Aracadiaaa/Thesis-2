import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")

df = pd.read_csv(CSV_PATH)

LABEL_COL = "label"

print("\nTotal rows:", len(df))
print("\nLabel counts:")
print(df[LABEL_COL].value_counts().sort_index())

print("\nLabel percentages:")
print((df[LABEL_COL].value_counts(normalize=True).sort_index() * 100).round(2), "%")

# specifically malicious count (label 3)
malicious_count = (df[LABEL_COL] == 3).sum()
print(f"\nMalicious count: {malicious_count}")
