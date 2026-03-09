import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

DATA = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")
MODEL_PKL = os.path.join(PROJECT_ROOT, "models", "rf_model.pkl")

df = pd.read_csv(DATA)

LABEL_COL = "label"
ID_COL = "display_id"

if LABEL_COL not in df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found in final_training_data.csv")

# Keep a copy BEFORE encoding/dropping so we can export original test rows with display_id
df_original = df.copy()

# Encode all string columns except display_id
for col in df.select_dtypes(include=["object", "string"]).columns:
    if col != ID_COL:
        df[col] = df[col].astype("category").cat.codes

y = df[LABEL_COL]
X = df.drop(columns=[LABEL_COL, ID_COL], errors="ignore")

# Keep row indices so we can recover original test rows with display_id
indices = df.index

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================= SAVE ORIGINAL TEST SETS =================
# These keep display_id so the UI can find images
test_rows = df_original.loc[idx_test].copy()

test_with_labels_path = os.path.join(PROJECT_ROOT, "Data", "original_test_with_labels.csv")
test_unlabeled_path = os.path.join(PROJECT_ROOT, "Data", "original_test_unlabeled.csv")

test_rows.to_csv(test_with_labels_path, index=False)
test_rows.drop(columns=[LABEL_COL], errors="ignore").to_csv(test_unlabeled_path, index=False)

print("✅ Saved original test sets:")
print("-", test_with_labels_path)
print("-", test_unlabeled_path)

best_clf = None
best_acc = -1
best_n = None

for n in [100, 200, 500]:
    clf = RandomForestClassifier(
        n_estimators=n,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    classes = clf.classes_

    # Find index of malicious class (3)
    mal_idx = list(classes).index(3)

    THRESH = 0.30

    preds = clf.predict(X_test).copy()

    # Force malicious if probability exceeds threshold
    preds[proba[:, mal_idx] >= THRESH] = 3

    acc = accuracy_score(y_test, preds)

    print(f"\n=== n_estimators={n} ===")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    if acc > best_acc:
        best_acc = acc
        best_clf = clf
        best_n = n

# ================= SAVE BEST MODEL =================
joblib.dump(
    {
        "model": best_clf,
        "feature_cols": X_train.columns.tolist()
    },
    MODEL_PKL
)

print(f"\n✅ Saved best model bundle: {MODEL_PKL}")
print(f"Best n_estimators: {best_n}")
print(f"Best accuracy: {best_acc:.4f}")