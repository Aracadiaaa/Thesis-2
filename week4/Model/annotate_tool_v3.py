import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
import ast # To safely parse the tags string

# --- CONFIGURATION ---
# EACH MEMBER MUST CHANGE THIS to their assigned file
INPUT_CSV = "week4\Data\jon_data.csv" 
IMAGE_DIR = "week4\Data\dataset_images"
OUTPUT_CSV = "labeled_dataset.csv"

# --- GUIDELINES FROM THESIS PDF ---
GUIDELINES = {
    "Entertainment": "BENIGN: Trailers, reviews, unboxings.\nBORDERLINE: Gossip, clickbait, exaggerated thumbnails.\nMALICIOUS: Dangerous stunts, sexual performances, violence.",
    "News & Politics": "BENIGN: Neutral reporting, educational.\nBORDERLINE: Sensational headlines, conspiracy theories.\nMALICIOUS: Hate speech, misinformation, violent propaganda.",
    "Gaming": "BENIGN: Walkthroughs. In-game violence is OK.\nBORDERLINE: 'RAGE QUIT' titles, edgy humor.\nMALICIOUS: Real-world violence, harassment, hacks/cheats.",
    "Film & Animation": "BENIGN: Trailers, shorts.\nBORDERLINE: Parody gore, clickbait.\nMALICIOUS: Realistic gore, Elsagate, extremist propaganda.",
    "People & Blogs": "BENIGN: Vlogs, travel, stories.\nBORDERLINE: Dramatic life updates, pranks.\nMALICIOUS: Bullying, self-harm stunts, dangerous challenges."
}

class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thesis Annotation Tool v3.0 (Full Metadata)")
        self.root.geometry("1200x900") # Made slightly taller for extra data

        # Load Data
        try:
            self.df = pd.read_csv(INPUT_CSV)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Could not find {INPUT_CSV}")
            root.destroy()
            return

        # Resume Logic
        if os.path.exists(OUTPUT_CSV):
            labeled = pd.read_csv(OUTPUT_CSV)
            processed_ids = set(labeled['display_id'])
            self.df = self.df[~self.df['display_id'].isin(processed_ids)].reset_index(drop=True)
        
        self.current_index = 0
        if len(self.df) == 0:
            messagebox.showinfo("Done", "All videos in this file are labeled!"); root.destroy(); return

        self.setup_ui()
        self.load_sample()

    def setup_ui(self):
        # --- SECTION 1: HEADER & GUIDELINES ---
        top_frame = tk.Frame(self.root, bg="#f0f0f0", pady=5)
        top_frame.pack(fill=tk.X)
        
        self.lbl_cat = tk.Label(top_frame, text="CATEGORY", font=("Arial", 12, "bold"), fg="blue", bg="#f0f0f0")
        self.lbl_cat.pack(anchor="w", padx=20)
        
        self.lbl_guide = tk.Label(top_frame, text="", font=("Arial", 9), bg="#e8e8e8", relief=tk.SUNKEN, padx=10, anchor="w")
        self.lbl_guide.pack(fill=tk.X, padx=20, pady=2)

        # --- SECTION 2: METADATA DASHBOARD (New!) ---
        meta_frame = tk.Frame(self.root, pady=5, padx=20)
        meta_frame.pack(fill=tk.X)

        # Metrics Row
        metrics_frame = tk.LabelFrame(meta_frame, text="Engagement Metrics", font=("Arial", 9, "bold"))
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_views = tk.Label(metrics_frame, text="Views: 0", font=("Arial", 10), fg="#333")
        self.lbl_views.pack(side=tk.LEFT, padx=20, pady=5)
        
        self.lbl_likes = tk.Label(metrics_frame, text="Likes: 0", font=("Arial", 10), fg="green")
        self.lbl_likes.pack(side=tk.LEFT, padx=20, pady=5)
        
        self.lbl_dislikes = tk.Label(metrics_frame, text="Dislikes: 0", font=("Arial", 10), fg="red")
        self.lbl_dislikes.pack(side=tk.LEFT, padx=20, pady=5)

        # Title
        self.lbl_title = tk.Label(meta_frame, text="VIDEO TITLE", font=("Arial", 13, "bold"), wraplength=1100, justify="left", anchor="w")
        self.lbl_title.pack(fill=tk.X, pady=(10, 5))

        # Description & Tags Split View
        text_frame = tk.Frame(meta_frame)
        text_frame.pack(fill=tk.X)

        # Description Box
        desc_frame = tk.Frame(text_frame)
        desc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        tk.Label(desc_frame, text="Description:", font=("Arial", 8, "bold")).pack(anchor="w")
        self.txt_desc = tk.Text(desc_frame, height=5, width=60, font=("Arial", 9), bg="#f9f9f9", wrap=tk.WORD)
        self.txt_desc.pack(fill=tk.BOTH)

        # Tags Box
        tags_frame = tk.Frame(text_frame)
        tags_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(tags_frame, text="Tags:", font=("Arial", 8, "bold")).pack(anchor="w")
        self.txt_tags = tk.Text(tags_frame, height=5, width=40, font=("Arial", 9), bg="#fffbe6", wrap=tk.WORD)
        self.txt_tags.pack(fill=tk.BOTH)

        # --- SECTION 3: IMAGES ---
        img_frame = tk.Frame(self.root, pady=10)
        img_frame.pack()
        
        # Grid layout for labels and images
        tk.Label(img_frame, text="Start (10%)", font=("Arial", 8)).grid(row=0, column=0)
        tk.Label(img_frame, text="Middle (50%)", font=("Arial", 8)).grid(row=0, column=1)
        tk.Label(img_frame, text="End (90%)", font=("Arial", 8)).grid(row=0, column=2)

        self.p_start = tk.Label(img_frame); self.p_start.grid(row=1, column=0, padx=5)
        self.p_mid = tk.Label(img_frame);   self.p_mid.grid(row=1, column=1, padx=5)
        self.p_end = tk.Label(img_frame);   self.p_end.grid(row=1, column=2, padx=5)

        # --- SECTION 4: CONTROLS ---
        btn_frame = tk.Frame(self.root, pady=15, bg="#ddd")
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Button(btn_frame, text="1. BENIGN [1]", bg="#90EE90", height=2, width=18, command=lambda: self.save("Benign")).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="2. BORDERLINE [2]", bg="#FFD700", height=2, width=18, command=lambda: self.save("Borderline")).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="3. MALICIOUS [3]", bg="#FF6347", height=2, width=18, command=lambda: self.save("Malicious")).pack(side=tk.RIGHT, padx=30)
        tk.Button(btn_frame, text="Skip [S]", bg="#bbb", height=2, width=10, command=lambda: self.save("Skip")).pack(side=tk.BOTTOM)

        # Hotkeys
        self.root.bind('1', lambda e: self.save("Benign"))
        self.root.bind('2', lambda e: self.save("Borderline"))
        self.root.bind('3', lambda e: self.save("Malicious"))
        self.root.bind('s', lambda e: self.save("Skip"))

    def load_sample(self):
        if self.current_index >= len(self.df):
            messagebox.showinfo("Done", "List Complete!"); self.root.destroy(); return

        row = self.df.iloc[self.current_index]
        cat_raw = str(row['categories']).strip()
        cat_folder = cat_raw.replace(" ", "_")
        vid_id = str(row['display_id'])

        # 1. Update Metrics (with comma formatting)
        try:
            views = "{:,}".format(int(row.get('view_count', 0)))
            likes = "{:,}".format(int(row.get('like_count', 0)))
            dislikes = "{:,}".format(int(row.get('dislike_count', 0)))
        except ValueError:
            views, likes, dislikes = "N/A", "N/A", "N/A"

        self.lbl_views.config(text=f"Views: {views}")
        self.lbl_likes.config(text=f"Likes: {likes}")
        self.lbl_dislikes.config(text=f"Dislikes: {dislikes}")

        # 2. Update Text Info
        self.lbl_title.config(text=str(row['title']))
        self.txt_desc.delete("1.0", tk.END); self.txt_desc.insert(tk.END, str(row['description'])[:500])

        # 3. Update Tags (Clean formatting)
        raw_tags = str(row.get('tags', ''))
        clean_tags = raw_tags.replace("'", "").replace("[", "").replace("]", "").replace('"', "")
        self.txt_tags.delete("1.0", tk.END); self.txt_tags.insert(tk.END, clean_tags)

        # 4. Update Guide
        self.lbl_cat.config(text=f"CATEGORY: {cat_raw}")
        matched_guide = next((v for k, v in GUIDELINES.items() if k in cat_raw), "General Rules apply.")
        self.lbl_guide.config(text=matched_guide)

        # 5. Load Images
        base_path = os.path.join(IMAGE_DIR, cat_folder)
        self.img_s = self.get_img(os.path.join(base_path, f"{vid_id}_start.jpg"))
        self.img_m = self.get_img(os.path.join(base_path, f"{vid_id}_mid.jpg"))
        self.img_e = self.get_img(os.path.join(base_path, f"{vid_id}_end.jpg"))
        
        self.p_start.config(image=self.img_s); self.p_mid.config(image=self.img_m); self.p_end.config(image=self.img_e)

    def get_img(self, path):
        if os.path.exists(path):
            img = Image.open(path).resize((220, 220)) # Resize for GUI
            return ImageTk.PhotoImage(img)
        return ImageTk.PhotoImage(Image.new('RGB', (220, 220), color='gray'))

    def save(self, label):
        row = self.df.iloc[self.current_index]
        data = {
            'display_id': row['display_id'], 
            'label': label, 
            'category': row['categories'],
            # Optional: Save these too if you want to analyze bias later
            'view_count': row.get('view_count', 0),
            'like_count': row.get('like_count', 0) 
        }
        pd.DataFrame([data]).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
        self.current_index += 1
        self.load_sample()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()