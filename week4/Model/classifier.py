import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")

df = pd.read_csv(DATA)

# ✅ YOU MUST SET THIS to your label column name
# Example options you might have elsewhere: "relevance", "label", "is_inappropriate", etc.
LABEL_COL = "relevance"  # <-- change this

if LABEL_COL not in df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found. Add it to final_training_data.csv first.")

y = df[LABEL_COL]
X = df.drop(columns=[LABEL_COL, "display_id"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for n in [100, 200, 500]:
    clf = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n=== n_estimators={n} ===")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
