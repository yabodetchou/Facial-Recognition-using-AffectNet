import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import kagglehub

# This script downloads the AffectNet dataset from Kaggle and organizes it into
# separate "images" and "labels" folders under the raw data directory.


def main():
    """
    Main function to download the dataset from Kaggle, organize images and labels,
    and clean up temporary cache files.
    """
    
    # Define project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    IMG_DIR = DATA_DIR / "images"
    LABEL_DIR = DATA_DIR / "labels"

    print("Downloading dataset from Kaggle...")

    # Download the dataset using KaggleHub; returns the local path to the extracted dataset
    kaggle_path = kagglehub.dataset_download("fatihkgg/affectnet-yolo-format")
    print(f"Dataset downloaded to: {kaggle_path}")

    # Create directories for images and labels if they do not exist
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    # Gather all image files (png/jpg) recursively from the Kaggle dataset folder
    all_image_files = []
    for root, _, files in os.walk(kaggle_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg")):
                all_image_files.append((root, file))

    print("Copying images and labels...")

    # Copy images and corresponding label files
    for root, file in tqdm(all_image_files, desc="Processing files"):
        # Construct label filename
        label_name = os.path.splitext(file)[0] + ".txt"

        # Determine corresponding label path
        label_path = os.path.join(root.replace("images", "labels"), label_name)

        # Copy image to IMG_DIR
        shutil.copy(os.path.join(root, file), IMG_DIR / file)

        # Copy label if it exists
        if os.path.exists(label_path):
            shutil.copy(label_path, LABEL_DIR / label_name)

    print(f"Import complete. Images and labels are now in {DATA_DIR}")

    # Clear KaggleHub cache to free space
    cache_path = os.path.expanduser("~/.cache/kagglehub/datasets/fatihkgg")
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        print("KaggleHub cache cleared.")


if __name__ == "__main__":
    main()