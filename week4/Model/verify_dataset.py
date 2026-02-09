import pandas as pd
import requests
import concurrent.futures
from tqdm import tqdm  # This creates a nice progress bar
import time

# --- CONFIGURATION ---
INPUT_FILE = "jon_data.csv"  # Replace with your actual file name
OUTPUT_FILE = "cleaned_dataset.csv"
TARGET_PER_CATEGORY = 1200  # We aim for 1200 to have a safe buffer above 1000
MAX_WORKERS = 5  # Number of simultaneous checks. Keep low to avoid IP bans.

def check_video_availability(video_data):
    """
    Checks if a YouTube video is available.
    Returns the row data if available, or None if unavailable.
    """
    video_id = video_data['display_id'] # Assuming 'display_id' holds the YouTube ID
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # We must pretend to be a real browser, or YouTube will block us immediately.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        # Check specific text markers in the response that indicate unavailability
        if "Video unavailable" in response.text:
            return None
        if "\"playabilityStatus\":{\"status\":\"ERROR\"" in response.text:
            return None
            
        return video_data # Return the row if valid
        
    except Exception as e:
        # If connection fails, assume broken to be safe
        return None

def main():
    print(f"Loading dataset from {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print("Error: File not found. Please check the INPUT_FILE name.")
        return

    print(f"Dataset loaded. Total rows: {len(df)}")
    print(f"Starting verification. Aiming for {TARGET_PER_CATEGORY} valid videos per category.")

    # List to hold our final clean data
    final_valid_data = []

    # Get list of unique categories
    categories = df['categories'].unique()

    for category in categories:
        print(f"\nProcessing Category: {category}")
        
        # Filter dataframe for just this category
        category_df = df[df['categories'] == category]
        
        # Convert to a list of dictionaries for easier processing
        videos_to_check = category_df.to_dict('records')
        
        valid_count = 0
        category_valid_rows = []

        # Use ThreadPoolExecutor to check multiple videos at once
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a queue of future tasks
            future_to_video = {executor.submit(check_video_availability, video): video for video in videos_to_check}
            
            # Process as they complete
            # tqdm creates the progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(videos_to_check), unit="vid"):
                result = future.result()
                
                if result is not None:
                    category_valid_rows.append(result)
                    valid_count += 1
                
                # OPTIMIZATION: Stop early if we hit our target!
                if valid_count >= TARGET_PER_CATEGORY:
                    print(f"--> Target reached for {category}. Stopping early to save time.")
                    # Cancel remaining tasks to save bandwidth
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
        
        print(f"Found {len(category_valid_rows)} valid videos for {category}.")
        final_valid_data.extend(category_valid_rows)
        
        # Small sleep between categories to be nice to the server
        time.sleep(2)

    # Convert back to DataFrame and save
    print(f"\nSaving {len(final_valid_data)} validated videos to {OUTPUT_FILE}...")
    clean_df = pd.DataFrame(final_valid_data)
    clean_df.to_csv(OUTPUT_FILE, index=False)
    print("Done! You are ready for the next step.")

if __name__ == "__main__":
    main()