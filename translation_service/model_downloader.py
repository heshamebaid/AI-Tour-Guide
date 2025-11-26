"""
Script to download required model files
"""

import os
import urllib.request
import hashlib
from pathlib import Path


def download_file(url: str, filename: str, expected_hash: str = None):
    """Download a file with progress indication."""
    print(f"Downloading {filename}...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\nDownloaded {filename}")
        
        # Verify hash if provided
        if expected_hash:
            with open(filename, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != expected_hash:
                print(f"Warning: Hash mismatch for {filename}")
            else:
                print(f"Hash verified for {filename}")
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        raise


def setup_models():
    """Download all required model files."""
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # SAM model
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    sam_path = "models/sam_vit_b.pth"
    
    if not os.path.exists(sam_path):
        download_file(sam_url, sam_path)
    else:
        print(f"SAM model already exists at {sam_path}")
    
    print("Model setup complete!")


def main():
    """Main function."""
    print("Setting up required models...")
    setup_models()


if __name__ == "__main__":
    main()
