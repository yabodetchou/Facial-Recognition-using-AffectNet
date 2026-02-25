import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm


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
    # Apply a Laplacian-like filter using PIL's built-in kernel
    laplacian_kernel = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, 1, 0,
                1,-4, 1,
                0, 1, 0],
        scale=1,
        offset=0
    )
    laplacian_img = img.filter(laplacian_kernel)

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
    print("Processing images...")
    img_dict = {}
    seen_hashes = set()

    for file in tqdm(os.listdir(img_dir), desc="Processing images"):
        file_path = img_dir / file

        # Skip corrupt images
        if is_corrupt(file_path):
            continue

        # Resize and convert image to grayscale
        img = resize_image(file_path)

        # Hash image for duplicate detection
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        if img_hash in seen_hashes:
            continue  # Skip exact duplicates
        seen_hashes.add(img_hash)

        # Skip blurry or overexposed images
        if is_blurry(img) or is_overexposed(img):
            continue

        # Keep the valid image
        img_dict[file] = img

    print("Images processed successfully.")

    return img_dict


def save_cleaned_images(output_dir: Path, img_dict: dict[str, np.ndarray]):
    """
    Save cleaned images to the specified directory.
    
    Args:
        output_dir (Path): Directory to save images.
        img_dict (dict): Dictionary of image filenames and arrays.
    """
    print("Saving images...")
    output_dir.mkdir(parents=True, exist_ok=True)
    num_img = 0
    for filename, img_array in tqdm(img_dict.items(), desc="Saving images"):
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(output_dir / filename)
        num_img +=1

    print(f"{num_img} images saved to {output_dir}.")


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

    for img_name in tqdm(img_dict.keys(), desc="Processing labels"):
        label_file = label_dir / (os.path.splitext(img_name)[0] + ".txt")

        if not label_file.exists():
            continue

        with open(label_file, "r") as f:
            parts = f.readline().strip().split()
            if len(parts) == 5:
                # Add filename + label info
                data.append([img_name] + parts)

    # Convert to DataFrame and save CSV
    df = pd.DataFrame(data, columns=["file_name", "label", "x_center", "y_center", "width", "height"])
    df.to_csv(output_dir / "affectnet_annotations.csv", index=False)
    print("CSV created successfully.")


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
    create_annotation_csv(LABEL_DIR, OUTPUT_DIR, img_dict)


if __name__ == "__main__":
    main()