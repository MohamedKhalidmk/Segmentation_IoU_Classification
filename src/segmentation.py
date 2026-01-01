import cv2
import numpy as np

def clean_and_extract_leaf(img):
    """
    Standard HSV cleaning to remove noise and isolate the green leaf.
    """
    # Denoise
    denoised = cv2.medianBlur(img, 7)
    denoised = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)

    # HSV Selection
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Return the clean image (Green leaf, Black background)
    return cv2.bitwise_and(denoised, denoised, mask=mask)

def segment_leaf_image(img):

    # === THE FIX: Clean the image before doing anything else ===
    img = clean_and_extract_leaf(img)

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)

    # Step 3: Denoise (median filter)
    denoised = cv2.medianBlur(contrast, 5)

    # Step 4: Thresholding (Otsu)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Inverse
    h, w = binary.shape
    corners = [binary[0,0], binary[0, w-1], binary[h-1, 0], binary[h-1, w-1]]
    if np.mean(corners) > 127:
        binary = cv2.bitwise_not(binary)

    # Step 6: Contour-Based Masking
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predicted_mask = np.zeros_like(binary)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(predicted_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return predicted_mask