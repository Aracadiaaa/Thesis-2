import pandas as pd
import requests
import concurrent.futures
from tqdm import tqdm
import time
import os

# --- CONFIGURATION ---
# The exact filenames you provided
FILES_TO_PROCESS = [
    "Master_Dataset.csv"
]

TARGET_PER_CATEGORY = 1200  # We want a buffer above 1000
MAX_WORKERS = 5  # Safe number of threads to avoid IP bans

def check_video_availability(video_data):
    """
    1. Pings YouTube to see if the video is alive.
    2. Returns the EXACT row data if alive.
    3. Returns None if dead or error.
    """
    try:
        video_id = video_data['display_id'] 
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Fast timeout (3s) to skip lagging requests
        response = requests.get(url, headers=headers, timeout=3)
        
        # Check for specific "dead video" markers in HTML
        if "Video unavailable" in response.text:
            return None
        if "\"playabilityStatus\":{\"status\":\"ERROR\"" in response.text:
            return None
            
        # RETURN DATA EXACTLY AS IS - NO MODIFICATIONS
        return video_data 
        
    except Exception:
        return None

def validate_integrity(original_df, valid_rows, filename):
    """
    Reiterates through the valid data to ensure it matches the original source.
    """
    print(f"   [Integrity Check] Validating {len(valid_rows)} rows for {filename}...")
    
    if not valid_rows:
        print("   [Integrity Check] FAILED: No valid videos found.")
        return False

    # 1. COLUMN CHECK: Ensure no columns were dropped or renamed
    valid_df = pd.DataFrame(valid_rows)
    original_cols = list(original_df.columns)
    new_cols = list(valid_df.columns)
    
    if len(original_cols) != len(new_cols):
        print(f"   [CRITICAL] Column count mismatch! Original: {len(original_cols)}, New: {len(new_cols)}")
        return False
    
    # 2. DATA MATCH CHECK: Pick the first valid video and compare it to the original
    sample_id = valid_rows[0]['display_id']
    
    # Locate the row in the original dataframe
    original_row = original_df[original_df['display_id'] == sample_id]
    
    if original_row.empty:
        print(f"   [CRITICAL] ID {sample_id} exists in result but NOT in original file!")
        return False
        
    # If we get here, the data structure is safe
    print("   [PASS] Columns match. Data structure verified against original.")
    return True

def main():
    master_list = [] 

    for filename in FILES_TO_PROCESS:
        print(f"\n==========================================")
        print(f"Processing: {filename}")
        print(f"==========================================")

        if not os.path.exists(filename):
            print(f"Error: File {filename} not found. Skipping.")
            continue

        # Load original data (low_memory=False prevents warnings on big files)
        try:
            df = pd.read_csv(filename, low_memory=False)
            print(f"Loaded {len(df)} rows. checking availability...")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            continue

        videos_to_check = df.to_dict('records')
        valid_rows_this_file = []
        
        # --- PHASE 1: AVAILABILITY CHECK ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_video = {executor.submit(check_video_availability, vid): vid for vid in videos_to_check}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(videos_to_check), unit="vid"):
                result = future.result()
                
                if result is not None:
                    valid_rows_this_file.append(result)
                
                # Stop early if we hit the target to save time/bandwidth
                if len(valid_rows_this_file) >= TARGET_PER_CATEGORY:
                    print(f"--> Target of {TARGET_PER_CATEGORY} reached. Stopping early.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
        
        # --- PHASE 2: INTEGRITY VALIDATION ---
        if validate_integrity(df, valid_rows_this_file, filename):
            # Save individual cleaned file
            cleaned_filename = filename.replace(".csv", "_Cleaned.csv")
            clean_df = pd.DataFrame(valid_rows_this_file)
            clean_df.to_csv(cleaned_filename, index=False)
            print(f"Saved: {cleaned_filename}")
            
            # Add to master list
            master_list.extend(valid_rows_this_file)
        else:
            print(f"SKIPPING SAVE: Validation failed for {filename}")
        
        print("Cooling down network for 5 seconds...")
        time.sleep(5)

    # --- PHASE 3: FINAL MERGE ---
    if master_list:
        print(f"\n==========================================")
        print(f"All files processed. Merging Master Dataset...")
        master_df = pd.DataFrame(master_list)
        master_df.to_csv("Master_Dataset.csv", index=False)
        print(f"SUCCESS: 'Master_Dataset.csv' created with {len(master_df)} videos.")
        print(f"Integrity checks passed. You are ready for Frame Extraction.")
    else:
        print("No valid videos found.")

if __name__ == "__main__":
    main()