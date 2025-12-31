import cv2
import numpy as np
import os

def jaccard_index(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union != 0 else 0.0

def compute_iou_dataset(dataset, pred_base_dir):
    iou_list = []
    for sample in dataset:
        gt_mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        pred_mask_path = os.path.join(pred_base_dir, sample['class'], os.path.basename(sample['img_path']))
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        iou = jaccard_index(pred_mask > 0, gt_mask > 0)
        iou_list.append(iou)
    avg_iou = np.mean(iou_list)
    print(f"Average IoU over dataset: {avg_iou:.4f}")
    return iou_list, avg_iou
