from src.dataset import load_dataset_with_mask_offset
from src.segmentation import segment_leaf_image
from src.evaluation import compute_iou_dataset
from src.features import extract_features
from src.train_classifier import train_random_forest
import os
import cv2
import pandas as pd

# Paths
IMAGES_DIR = "data/images"
MASKS_DIR = "data/masks"
PRED_DIR = "predicted_masks"

# Load dataset
dataset = load_dataset_with_mask_offset(IMAGES_DIR, MASKS_DIR)

# Segment leaves and save masks
os.makedirs(PRED_DIR, exist_ok=True)
for sample in dataset:
    img = cv2.imread(sample['img_path'])
    pred_mask = segment_leaf_image(img)
    save_folder = os.path.join(PRED_DIR, sample['class'])
    os.makedirs(save_folder, exist_ok=True)
    filename = os.path.basename(sample['img_path'])
    cv2.imwrite(os.path.join(save_folder, filename), pred_mask)

# Compute IoU
compute_iou_dataset(dataset, PRED_DIR)

# Extract features
records = []
for class_name in os.listdir(PRED_DIR):
    class_folder = os.path.join(PRED_DIR, class_name)
    for fname in os.listdir(class_folder):
        mask = cv2.imread(os.path.join(class_folder, fname), cv2.IMREAD_GRAYSCALE)
        features = extract_features(mask)
        if features:
            records.append([class_name] + features)

columns = ["class", "area", "perimeter", "aspect_ratio", "circularity",
           "compactness", "convexity", "hu1","hu2","hu3","hu4","hu5","hu6","hu7"]
df_features = pd.DataFrame(records, columns=columns)
df_features.to_csv("leaf_features.csv", index=False)

# Train classifier
train_random_forest(df_features)
