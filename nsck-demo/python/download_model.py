
import os
import requests
import sys

# Configuration
TARGET_FILENAME = "phi-3-mini-4k-instruct.Q4_K_M.gguf"
DEST_DIR = os.path.join("..", "..", "models") # Relative to nsck-demo/python
DEST_FILE = os.path.join(DEST_DIR, TARGET_FILENAME)

# Candidate URLs (Primary -> Backups)
MODEL_URLS = [
    "https://huggingface.co/QuantFactory/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct.Q4_K_M.gguf",
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    "https://huggingface.co/lmstudio-community/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
]

def download_file(url, dest_path):
    print(f"Attempting download from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        wrote = 0
        
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                wrote += len(data)
                f.write(data)
                # Simple progress bar
                if total_size > 0:
                    percent = (wrote / total_size) * 100
                    sys.stdout.write(f"\rProgress: {percent:.1f}% ({wrote // (1024*1024)}MB)")
                    sys.stdout.flush()
        
        print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"\nError downloading model: {e}")
        return False

if __name__ == "__main__":
    # Ensure directory exists
    if not os.path.exists(DEST_DIR):
        try:
            os.makedirs(DEST_DIR)
            print(f"Created directory: {DEST_DIR}")
        except OSError as e:
            print(f"Error creating directory {DEST_DIR}: {e}")
            sys.exit(1)

    if os.path.exists(DEST_FILE):
        print(f"Model already exists at {DEST_FILE}")
        sys.exit(0)
    
    success = False
    for url in MODEL_URLS:
        if download_file(url, DEST_FILE):
             success = True
             break
    
    if not success:
        print("All download attempts failed.")
        sys.exit(1)
