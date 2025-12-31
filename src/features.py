import cv2
import math
import numpy as np
from skimage.measure import label, regionprops

def extract_features(mask):
    mask = (mask > 0).astype(int)
    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return None
    leaf = max(regions, key=lambda r: r.area)

    area = leaf.area
    perimeter = leaf.perimeter
    convex_area = leaf.convex_area
    minr, minc, maxr, maxc = leaf.bbox
    height = maxr - minr
    width = maxc - minc
    aspect_ratio = width / (height + 1e-6)
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)
    compactness = area / (perimeter**2 + 1e-6)
    convexity = area / (convex_area + 1e-6)

    moments = cv2.moments(mask.astype(np.uint8))
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_log = [-1 * math.copysign(1.0, h) * math.log10(abs(h)) if h != 0 else 0 for h in hu_moments]

    return [area, perimeter, aspect_ratio, circularity, compactness, convexity, *hu_log]
