import kagglehub
import shutil
import os

# 1. Define where you actually want the data
target_dir = "../data"

# 2. Download latest version (this downloads to a local cache)
print("Downloading from Kaggle (this may take a moment)...")
cache_path = kagglehub.dataset_download("wurmplekuljit/paris6k")
print("Downloaded to cache:", cache_path)

# 3. Copy files to your desired target directory
print(f"Copying dataset to {target_dir}...")
os.makedirs(target_dir, exist_ok=True)
shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)

print("Done! Dataset is ready at:", target_dir)