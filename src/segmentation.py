import cv2
import numpy as np

def segment_leaf_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    denoised = cv2.medianBlur(contrast, 5)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape
    corners = [binary[0,0], binary[0, w-1], binary[h-1, 0], binary[h-1, w-1]]
    if np.mean(corners) > 127:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predicted_mask = np.zeros_like(binary)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(predicted_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return predicted_mask
