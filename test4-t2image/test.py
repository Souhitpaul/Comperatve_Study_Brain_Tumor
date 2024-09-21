import os
import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import cupy as cp  # CUDA acceleration
from scipy import stats  # Use scipy.stats for kurtosis and skewness

# Preprocessing
def preprocess_image(image):
    # Resize and normalize the image
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image

def preprocess_nii_image(nii_file_path):
    # Load NIfTI file
    img = nib.load(nii_file_path)
    img_data = img.get_fdata()
    # Take a slice and preprocess it
    image_slice = img_data[:, :, img_data.shape[2] // 2]  # Taking the middle slice
    image_slice = cv2.resize(image_slice, (256, 256))
    image_slice = np.uint8(cv2.normalize(image_slice, None, 0, 255, cv2.NORM_MINMAX))
    return image_slice

# Gaussian Blur and Morphological Processing
def apply_histogram_equalization_and_adaptive_threshold(image):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)
    
    # Adaptive Gaussian Thresholding
    adaptive_thresh_image = cv2.adaptiveThreshold(
        equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return adaptive_thresh_image

# K-means Clustering (using GPU)
def apply_kmeans(image, clusters=2):
    pixel_values = image.reshape((-1, 1)).astype(np.float32)
    
    # Apply KMeans using GPU
    pixel_values_gpu = cp.array(pixel_values)
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(pixel_values_gpu.get())
    labels = kmeans.labels_
    
    segmented_image = labels.reshape(image.shape)
    return segmented_image

# Feature Extraction
def extract_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    mean = np.mean(image)
    std_dev = np.std(image)
    entropy = -np.sum(image * np.log2(image + 1e-7))
    
    # Convert image to NumPy array for stats calculation
    image_np = cp.asnumpy(image)
    
    # Use scipy.stats for kurtosis and skewness
    kurtosis = stats.kurtosis(image_np.flatten())
    skewness = stats.skew(image_np.flatten())
    
    features = [contrast, correlation, energy, homogeneity, mean, std_dev, entropy, kurtosis, skewness]
    return features

# PCA for Feature Reduction
def apply_pca(features):
    pca = PCA(n_components=5)
    reduced_features = pca.fit_transform(features)
    return reduced_features

# Load images
def load_images(yes_folder, no_folder):
    features = []
    labels = []
    
    for file_name in os.listdir(yes_folder):
        if file_name.endswith(".nii"):
            image = preprocess_nii_image(os.path.join(yes_folder, file_name))
            image = apply_histogram_equalization_and_adaptive_threshold(image)
            image = apply_kmeans(image)
            feature = extract_features(image)
            features.append(feature)
            labels.append(1)  # Tumor class
    
    for file_name in os.listdir(no_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = preprocess_image(cv2.imread(os.path.join(no_folder, file_name)))
            image = apply_histogram_equalization_and_adaptive_threshold(image)
            image = apply_kmeans(image)
            feature = extract_features(image)
            features.append(feature)
            labels.append(0)  # Non-tumor class
    
    return np.array(features), np.array(labels)


# Main function
def main():
    yes_folder = "newdataset/yes"  # Folder containing .nii files
    no_folder = "newdataset/no"    # Folder containing .jpg/.jpeg files
    
    features, labels = load_images(yes_folder, no_folder)
    features = apply_pca(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train SVM
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
