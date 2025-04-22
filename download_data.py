import gdown
import zipfile
import os

# --- Configuration ---
file_id = '1MZU1tu5sSnzwhd-M6iwn0hqOkZVoUEE0'  # File ID from Google Drive
zip_path = 'data/data-raw/motor-data.zip'     # Path to save the .zip file
extract_to = 'data/data-raw/'       # Folder where the files will be extracted

# --- Step 1: Download the .zip file ---
print("Downloading .zip file...")
gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)

# --- Step 2: Extract the .zip file ---
print("Extracting .zip file...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(path=extract_to)
    print("Extracted files:", zip_ref.namelist())

# --- Step 3: Delete the .zip file ---
if os.path.exists(zip_path):
    os.remove(zip_path)
    print(f"File {zip_path} deleted.")
else:
    print("The .zip file was not found for deletion.")
