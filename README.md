# YouTube Inappropriate Video Detection (Multimodal Pipeline)

This project builds a multimodal classifier to detect inappropriate YouTube videos using:
- **RoBERTa** text embeddings (Title + Description)
- **VAE** reconstruction error as a visual anomaly score (Frames/thumbnail)
- **Metadata** (views, likes, category)
- **Random Forest** classifier + **probability threshold tuning** for improved malicious recall

---

## Folder Structure

week4/
Data/
Final_Ready_Dataset.csv
Images/ # flat folder of images (no class subfolders)
Processed/
roberta_embeddings.pt
roberta_video_ids.csv
vae_recon_scores.csv
final_training_data.csv
Model/
extract_data_features.py
infer_vae_scores.py
fusion_prep.py
classifier.py
vae_model.py
vae_trained.pth

yaml
Copy code

---

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install torch torchvision transformers pandas numpy scikit-learn tqdm pillow
Input Data Requirements
1) CSV: week4/Data/Final_Ready_Dataset.csv
Must include these columns:

display_id (video id)

title

description

label (1=Begin, 2=Borderline, 3=Malicious)

view_count_master

like_count_master

categories (string category name)

2) Images: week4/Data/Images/
Flat directory containing images named like:

php-template
Copy code
<video_id>_thumb.jpg
<video_id>_start.jpg
<video_id>_mid.jpg
<video_id>_end.jpg
Extra images are okay — the pipeline uses MAX reconstruction error per video.

Pipeline (Run Order)
Step 1 — Extract Text Embeddings (RoBERTa)
Creates:

week4/Data/Processed/roberta_embeddings.pt

week4/Data/Processed/roberta_video_ids.csv

Run:

bash
Copy code
python week4/Model/extract_data_features.py
Expected:

Embedding shape should match CSV rows (e.g., (4145, 768))

Step 2 — Compute Visual Anomaly Scores (VAE Inference)
Uses trained VAE weights and scores all images.

Creates:

week4/Data/vae_recon_scores.csv

Run:

bash
Copy code
python week4/Model/infer_vae_scores.py
Expected:

Total images scored: <some number> (can be > 4 per video)

Step 3 — Fusion (Build Final Training Table)
Merges per-video MAX visual score + embeddings + metadata + labels.

Creates:

week4/Data/final_training_data.csv

Run:

bash
Copy code
python week4/Model/fusion_prep.py
Expected:

Final shape around: (N, 773)

5 columns (id + views + likes + max_visual_score + label)

768 RoBERTa features

Step 4 — Train Random Forest Classifier + Threshold Tuning
Trains RF (80/20 split) and prints metrics.

Run:

bash
Copy code
python week4/Model/classifier.py
Notes:

Model uses category encoding internally.

Uses probability thresholding on class 3 (malicious) to improve recall.

Output / Metrics
The classifier prints:

Accuracy

Confusion matrix

Precision / Recall / F1 per class

You can tune:

n_estimators (100 / 200 / 500)

malicious probability threshold (THRESH, e.g., 0.30)

Notes / Common Issues
"could not convert string to float: Film & Animation"
You still have string columns in your feature matrix.
Fix: encode object columns to category codes in classifier.py.

Shape mismatch between RoBERTa and CSV
Make sure RoBERTa was extracted from the same CSV (Final_Ready_Dataset.csv)
in the same row order.

License
Add your preferred license here (MIT recommended).

pgsql
Copy code

If you want, paste your current `classifier.py` and I’ll add a “Configuration” section that lists the exact variables you can chan
