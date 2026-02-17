import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")

df = pd.read_csv(DATA)

# ✅ encode ALL string columns automatically
for col in df.select_dtypes(include="object").columns:
    if col != "display_id":  # don't encode IDs
        df[col] = df[col].astype("category").cat.codes

LABEL_COL = "label"

y = df[LABEL_COL]
X = df.drop(columns=[LABEL_COL, "display_id"], errors="ignore")



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
        class_weight="balanced",   # fixes malicious imbalance
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    classes = clf.classes_

    # find index of malicious class (3)
    mal_idx = list(classes).index(3)

    THRESH = 0.30   # try 0.35, 0.30, 0.25

    preds = clf.predict(X_test)

    # force malicious if probability exceeds threshold
    preds = preds.copy()
    preds[proba[:, mal_idx] >= THRESH] = 3

    acc = accuracy_score(y_test, preds)

    print(f"\n=== n_estimators={n} ===")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
