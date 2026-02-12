import os
import tarfile
import requests
import config

def download_file(url, dest_path):
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size=block_size):
            file.write(data)
    print("Download complete.")

def extract_tar(tar_path, dest_path):
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=dest_path)
    print("Extraction complete.")

def main():
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    
    tar_filename = os.path.join(config.DATA_DIR, "Task04_Hippocampus.tar")
    
    if not os.path.exists(tar_filename):
        download_file(config.DOWNLOAD_URL, tar_filename)
    else:
        print("Tar file already exists.")
        
    # Extract if the folder inside doesn't exist yet
    if not os.path.exists(os.path.join(config.DATA_DIR, "Task04_Hippocampus")):
        extract_tar(tar_filename, config.DATA_DIR)
    else:
        print("Data already extracted.")

if __name__ == "__main__":
    main()
