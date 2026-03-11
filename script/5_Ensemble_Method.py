import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# --- CONFIGURATION & PATHS ---

PROJECT_ROOT = Path(__file__).resolve().parents[1] # parents[1] is the same as .parent.parent
CLEANED_DIR     = PROJECT_ROOT.joinpath("data", "cleaned")
IMAGES_DIR      = CLEANED_DIR.joinpath("images")
ANNOTATION_FILE = CLEANED_DIR.joinpath("affectnet_annotations.csv")
PCA_CACHE_FILE  = PROJECT_ROOT.joinpath("script", "cache", "pca.pkl")




# HELPER FUNCTIONS

def load_split(split_name):
    """Loads image data and labels from the cleaned directory."""
    df = pd.read_csv(ANNOTATION_FILE)
    df = df[df["split"] == split_name].copy()
    img_directory = IMAGES_DIR / split_name
    matrix_of_pixel = np.zeros((len(df), 96 * 96), dtype=np.float64)
    vector_of_categories = df["label"].astype(int).to_numpy()
    filenames = df["file_name"].tolist()
    for i, filename in enumerate(tqdm(filenames, desc=f"Loading {split_name}")):
        img = Image.open(img_directory / filename).convert("L")
        matrix_of_pixel[i] = np.array(img, dtype=np.float64).flatten()
    return matrix_of_pixel, vector_of_categories


def subsample(features, labels, max_per_class, seed=42):
    """Ensures balanced classes for training."""
    random_number_generator = np.random.default_rng(seed)
    row_numbers_img = []
    for category in np.unique(labels):
        index = np.where(labels == category)[0]
        if len(index) > max_per_class:
            index = random_number_generator.choice(index, max_per_class, replace=False)
        row_numbers_img.extend(index.tolist())
    return features[np.array(sorted(row_numbers_img))], labels[np.array(sorted(row_numbers_img))]


# KNN FROM SCRATCH

class KNearestNeighbors:
    def __init__(self, num_of_neighbors=5):
        self.num_of_neighbors = num_of_neighbors # number of neighbors
        self.X_training = None # stores faces from training ser
        self.y_training = None # stores the emotion (0,1,2..) for every face stored in X_training

    def fit(self, pixels_training_set, num_code_emotion):
        self.X_training = np.asarray(pixels_training_set, dtype=np.float32)
        self.y_training = np.asarray(num_code_emotion)

    def predict_probability(self, new_faces): # compares new faces against old faces
        """Calculates confidence based on neighbor frequency."""
        new_faces = np.asarray(new_faces, dtype=np.float32)
        # Vectorized Euclidean Distance
        dists = np.sqrt(np.sum(new_faces ** 2, axis=1, keepdims=True) +
                        np.sum(self.X_training ** 2, axis=1) -
                        2.0 * (new_faces @ self.X_training.T))
        knn_indices = np.argsort(dists, axis=1)[:, :self.num_of_neighbors]
        knn_labels = self.y_training[knn_indices]

        classes = np.unique(self.y_training)
        probs = []
        for i in range(len(new_faces)):
            counts = [np.sum(knn_labels[i] == c) for c in classes]
            probs.append(np.array(counts) / self.num_of_neighbors)
        return np.array(probs), classes


# ENSEMBLE LAYER

class FacialExpressionEnsemble:
    """Combines SVM and KNN outputs using weighted voting."""

    def __init__(self, svm_model, knn_model, svm_weight=0.6):
        self.svm = svm_model
        self.knn = knn_model
        self.svm_weight = svm_weight
        self.knn_weight = 1.0 - svm_weight

    def predict(self, new_faces):
        # SVM Confidence (Softmax of decision scores)
        svm_scores = self.svm.decision_function(new_faces)
        svm_probs = np.exp(svm_scores) / np.sum(np.exp(svm_scores), axis=1, keepdims=True)

        # KNN Confidence
        knn_probs, classes = self.knn.predict_probability(new_faces)

        # Weighted Decision
        final_probs = (self.svm_weight * svm_probs) + (self.knn_weight * knn_probs)
        return classes[np.argmax(final_probs, axis=1)]


# EVALUATION METRICS

def calculate_metrics(y_true, predicted_y, labels):
    """Calculates Precision, Recall, and F1 manually."""
    label_map = {label: i for i, label in enumerate(labels)}
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, y_predicted in zip(y_true, predicted_y):
        confusion_matrix[label_map[yt], label_map[y_predicted]] += 1

    results = {}
    for i, label in enumerate(labels):
        true_positive = confusion_matrix[i, i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results[label] = (precision, recall, f1_score)

    macro_f1_score = np.mean([v[2] for v in results.values()]) # the average of all the averages of the f1 scotes
    return confusion_matrix, results, macro_f1_score


# MAIN PIPELINE
from PCA_3 import PCA

def main():
    # Load and Prepare Data
    X_train_raw, y_train_raw = load_split("train")
    X_test_raw, y_test_raw = load_split("test")
    X_training, y_training= subsample(X_train_raw, y_train_raw, 200)
    X_test, y_test = subsample(X_test_raw, y_test_raw, 100)

    # FIXED PATH LOGIC
    if not PCA_CACHE_FILE.exists():
        raise FileNotFoundError(f"Could not find the PCA file at {PCA_CACHE_FILE}. "
                                f"Please run PCA_3.py first to generate it.")

    with open(PCA_CACHE_FILE, "rb") as f:
        pca = pickle.load(f)

    # Use the 'pca' object you just loaded to project the data
    Z_train = pca.apply_projection(X_training, n_components=150)
    Z_test = pca.apply_projection(X_test, n_components=150)



    print("Projection data using loaded PCA model")
    Z_train = pca.apply_projection(X_training, n_components=150)
    Z_test = pca.apply_projection(X_test, n_components=150)

    # 3. Standardization
    z_mean, z_std = Z_train.mean(axis=0), Z_train.std(axis=0)
    z_std[z_std == 0] = 1.0
    Z_train = (Z_train - z_mean) / z_std
    Z_test = (Z_test - z_mean) / z_std

    # Initialize Classifiers

    # Import your OneVsAllKernelSVM class here or from SVM_4.py
    from SVM_4 import OneVsAllKernelSVM
    svm = OneVsAllKernelSVM(kernel="rbf", C=10.0, gamma=0.01)
    svm.fit(Z_train, y_training)

    knn = KNearestNeighbors(num_of_neighbors=7)
    knn.fit(Z_train, y_training)

    # Ensemble & Evaluate
    ensemble = FacialExpressionEnsemble(svm, knn, svm_weight=0.7)
    y_predicted = ensemble.predict(Z_test)

    confusion_matrix, per_class, macro_f1_score = calculate_metrics(y_test, y_predicted, np.unique(y_test))

    print(f"Confusion Matrix:\n{confusion_matrix}")
    print(f"\n")
    print(f"Overall Macro F1-Score: {macro_f1_score:.4f}")


if __name__ == "__main__":
    main()

