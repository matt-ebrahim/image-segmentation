import os
import tarfile
import requests
from tqdm import tqdm
import config

def download_file(url, dest_path):
    """Download file from URL with progress bar."""
    print(f"Downloading from {url}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    print("Download complete.")

def extract_tar(tar_path, dest_path):
    """Extract tar file."""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=dest_path)
    print("Extraction complete.")

def check_data_exists():
    """Check if data already exists."""
    if os.path.exists(config.RAW_DATA_DIR):
        print("Data already exists.")
        return True
    return False

def main():
    """Main function to download and extract data."""
    # Check if data already exists
    if check_data_exists():
        return
    
    # Create data directory
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Download data if tar file doesn't exist
    if not os.path.exists(config.DATASET_TAR_FILE):
        download_file(config.DOWNLOAD_URL, config.DATASET_TAR_FILE)
    else:
        print("Tar file already exists.")
        
    # Extract data if not already extracted
    if not os.path.exists(config.RAW_DATA_DIR):
        extract_tar(config.DATASET_TAR_FILE, config.DATA_DIR)
    else:
        print("Data already extracted.")

if __name__ == "__main__":
    main()