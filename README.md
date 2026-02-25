# Project Setup and Usage

This project processes the AffectNet dataset for machine learning purposes. It includes scripts for downloading and cleaning images, generating labels, and preparing data for training models.

---

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

## 2. Create a Python Virtual Environment

It's recommended to use a virtual environment to isolate dependencies.

```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

---

## 3. Install Required Packages

Install all Python dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Required Packages:**

- `numpy`
- `pandas`
- `Pillow`
- `opencv-python`
- `tqdm`
- `kagglehub`

> All versions are specified in `requirements.txt` to ensure reproducibility.

---

## 4. Run the Scripts

All scripts are located in the `scripts/` folder. You need to run them in the following order:

### **Step 1: Download and organize raw dataset**

```bash
python scripts/download_dataset.py
```

- Downloads the AffectNet dataset from Kaggle using `kagglehub`.
- Organizes files into `data/raw/images` and `data/raw/labels`.
- Cleans temporary cache used by `kagglehub`.

### **Step 2: Clean and process images**

```bash
python scripts/clean_dataset.py
```

- Removes corrupt, blurry, overexposed, or duplicate images.
- Converts images to grayscale and resizes them to 96x96 pixels.
- Saves cleaned images to `data/cleaned/images`.
- Generates a CSV annotation file `data/cleaned/affectnet_annotations.csv`.

---

## 5. Notes

- Make sure the virtual environment is active whenever you run the scripts.
- Do **not** commit large image files or the `venv` folder to GitHub.
- Use the `.gitignore` provided to keep the repository clean.

---

## 6. Directory Structure

After setup and running scripts, your folder structure should look like this:

```
project-root/
│
├─ data/
│  ├─ raw/
│  │  ├─ images/
│  │  └─ labels/
│  └─ cleaned/
│     ├─ images/
│     └─ affectnet_annotations.csv
│
├─ scripts/
│  ├─ download_dataset.py
│  └─ clean_dataset.py
│
├─ requirements.txt
└─ README.md
```

This setup ensures your dataset is clean and ready for model training.

