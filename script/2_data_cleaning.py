import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm
import random


def is_corrupt(path: Path) -> bool:
    """
    Check if an image file is corrupt.
    
    Args:
        path (Path): Path to the image file.
        
    Returns:
        bool: True if image can be opened and verified, False otherwise.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # Verify image integrity
        return False  # Not corrupt
    except Exception:
        return True  # Corrupt image


def is_blurry(img: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Determine if an image is blurry using the Laplacian variance method.
    
    Args:
        img (np.ndarray): Grayscale image array.
        threshold (float): Variance threshold below which image is considered blurry.
        
    Returns:
        bool: True if blurry, False otherwise.
    """
    img_pil = Image.fromarray(img)
    # Apply a Laplacian-like filter using PIL's built-in kernel
    laplacian_kernel = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, 1, 0,
                1,-4, 1,
                0, 1, 0],
        scale=1,
        offset=0
    )
    laplacian_img = img_pil.filter(laplacian_kernel)

    # Convert to numpy array
    laplacian_array = np.array(laplacian_img)

    # If variance is low, the image is likely blurry
    return laplacian_array.var() < threshold


def is_overexposed(img: np.ndarray, low: int = 30, high: int = 220) -> bool:
    """
    Determine if an image is overexposed based on mean pixel intensity.
    
    Args:
        img (np.ndarray): Grayscale image array.
        low (int): Minimum acceptable mean intensity.
        high (int): Maximum acceptable mean intensity.
        
    Returns:
        bool: True if overexposed or underexposed, False otherwise.
    """
    mean_intensity = img.mean()
    return not (low <= mean_intensity <= high)


def resize_image(path: Path, size: tuple[int, int] = (96, 96)) -> np.ndarray:
    """
    Open, convert to grayscale, resize, and convert image to a NumPy array.
    
    Args:
        path (Path): Path to image file.
        size (tuple): Desired output size (width, height).
        
    Returns:
        np.ndarray: Processed grayscale image array.
    """
    img = Image.open(path).convert("L").resize(size)
    return np.array(img)


def build_image_dict(img_dir: Path) -> dict[str, np.ndarray]:
    """
    Process all images in a directory: remove corrupt, blurry, overexposed, 
    and duplicate images.
    
    Args:
        img_dir (Path): Directory containing image files.
        
    Returns:
        dict: Dictionary mapping valid image filenames to NumPy arrays.
    """
    img_dict = {}
    seen_hashes = set()
    skipped_files = 0

    for file in tqdm(os.listdir(img_dir), desc="Processing images"):
        file_path = img_dir / file

        # Skip corrupt images
        if is_corrupt(file_path):
            skipped_files += 1
            continue

        # Resize and convert image to grayscale
        img = resize_image(file_path)

        # Hash image for duplicate detection
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        if img_hash in seen_hashes:
            skipped_files += 1
            continue  # Skip exact duplicates
        seen_hashes.add(img_hash)

        # Skip blurry or overexposed images
        if is_blurry(img) or is_overexposed(img):
            skipped_files += 1
            continue

        # Keep the valid image
        img_dict[file] = img

    print(f"Images processed successfully. {skipped_files} images were flagged as corrupt, duplicates, overexposed or blurry and skipped.")

    return img_dict


from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

def save_cleaned_images(output_dir: Path, img_dict: dict[str, np.ndarray], 
                        split_ratios=(0.7, 0.15, 0.15), seed=345):
    """
    Save cleaned images to train/test/validation folders.
    
    Args:
        output_dir (Path): Directory to save images.
        img_dict (dict): Dictionary of image filenames and arrays.
        split_ratios (tuple): Fractions for (train, val, test). Must sum to 1.
        seed (int): Random seed for reproducibility.
    """
    assert len(split_ratios) == 3, "split_ratios must be a tuple of 3 values."
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1."
    
    train_ratio, val_ratio, test_ratio = split_ratios
    
    random.seed(seed)
    filenames = list(img_dict.keys())
    random.shuffle(filenames)
    
    n_total = len(filenames)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    splits = {
        "train": filenames[:n_train],
        "val": filenames[n_train:n_train+n_val],
        "test": filenames[n_train+n_val:]
    }
    
    for split_name, split_files in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for filename in tqdm(split_files, desc=f"Saving {split_name} images"):
            img_array = img_dict[filename]
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(split_dir / filename)
    
    print(f"Saved {n_train} train, {n_val} validation, and {n_test} test images to {output_dir}.")


def create_annotation_csv(label_dir: Path, output_dir: Path, img_dict: dict[str, np.ndarray]):
    """
    Create a CSV annotation file for the cleaned images based on corresponding label files.
    
    Args:
        label_dir (Path): Directory containing label text files.
        output_dir (Path): Directory to save the CSV file.
        img_dict (dict): Dictionary of cleaned image filenames.
    """
    print("Creating annotation CSV...")
    data = []

    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / 'images' /split_name
        if not split_dir.exists():
            continue
        
        for img_file in tqdm(list(split_dir.iterdir()), desc=f"Processing {split_name}"):
            if img_file.suffix.lower() not in [".jpg", ".png"]:
                continue

            label_file = label_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                continue

            with open(label_file, "r") as f:
                parts = f.readline().strip().split()
                if len(parts) == 5:
                    data.append([img_file.name] + parts + [split_name])

    # Convert to DataFrame and save CSV
    df = pd.DataFrame(data, columns=["file_name", "label", "x_center", "y_center", "width", "height", "split"])
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "affectnet_annotations.csv", index=False)
    print(f"CSV created successfully with {len(df)} entries.")

def main():
    # Define project directories
    PROJECT_ROOT = Path(__file__).parent.parent
    IMG_DIR = PROJECT_ROOT / "data" / "raw" / "images"
    LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "labels"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

    # Process images
    img_dict = build_image_dict(IMG_DIR)

    # Save cleaned images
    save_cleaned_images(OUTPUT_DIR / "images", img_dict)

    # Create CSV annotations
    df_with_splits = create_annotation_csv(LABEL_DIR, OUTPUT_DIR, img_dict)


if __name__ == "__main__":
    main()