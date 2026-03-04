# roi_diff_analyzer_internal_config.py
import cv2
import numpy as np
# import os # May be needed if image paths are relative to this script

# --- ★★★★★ Modify your test parameters here ★★★★★ ---

# 1. Specify the paths of the two images to compare
IMAGE_PATH_1 = "D:/Github/beach-volleyball-tracker/output_data/tracking_output/segment_069/frames/frame_000100.jpg"  # <<--- Edit here
IMAGE_PATH_2 = "D:/Github/beach-volleyball-tracker/output_data/tracking_output/segment_170/frames/frame_000001.jpg" # <<--- Edit here

# 2. Precisely define two score ROI coordinates (x, y, width, height)
#    These values should match the scoreboard positions in your images
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # Example: Team 1 / upper score <<--- Edit here
SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # Example: Team 2 / lower score <<--- Edit here

# 3. Whether to preprocess ROI (True or False)
APPLY_PREPROCESSING = True # <<--- Edit here (True: grayscale + blur, False: grayscale only)

# 4. (If APPLY_PREPROCESSING is True) Gaussian blur kernel size for preprocessing (must be odd positive numbers)
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5) # <<--- Edit here (e.g., (3,3) or (5,5))

# --- ★★★★★ End of parameter section ★★★★★ ---


def calculate_sad(image1_gray, image2_gray):
    """Calculate the Sum of Absolute Differences (SAD) for two grayscale images."""
    if image1_gray is None or image2_gray is None:
        return float('inf')
    if image1_gray.shape != image2_gray.shape:
        print("Warning: ROI image sizes do not match, cannot calculate SAD.")
        return float('inf')
    diff = cv2.absdiff(image1_gray, image2_gray)
    sad = np.sum(diff)
    return sad

def calculate_mse(image1_gray, image2_gray):
    """Calculate Mean Squared Error (MSE) for two grayscale images."""
    if image1_gray is None or image2_gray is None:
        return float('inf')
    if image1_gray.shape != image2_gray.shape:
        print("Warning: ROI image sizes do not match, cannot calculate MSE.")
        return float('inf')
    err = np.sum((image1_gray.astype("float") - image2_gray.astype("float")) ** 2)
    mse = err / float(image1_gray.shape[0] * image1_gray.shape[1])
    return mse

def get_roi_from_image(image, roi_coords, image_name_for_error=""):
    """Safely extract ROI from an image."""
    x, y, w, h = roi_coords
    if image is None: 
        print(f"Error: Image passed to get_roi_from_image ({image_name_for_error}) is None.")
        return None
    frame_height, frame_width = image.shape[:2]
    if not (0 <= x < frame_width and 0 <= y < frame_height and x + w <= frame_width and y + h <= frame_height and w > 0 and h > 0):
        print(f"Warning: ROI {roi_coords} for image '{image_name_for_error}' is out of bounds ({frame_width}x{frame_height}) or invalid.")
        return None
    return image[y:y+h, x:x+w]

def preprocess_roi(roi_image):
    """Preprocess ROI image (grayscale + optional blur)."""
    if roi_image is None: return None
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    if APPLY_PREPROCESSING: # Apply blur based on global flag
        # print("Applying ROI preprocessing (Gaussian blur)...")
        blurred_roi = cv2.GaussianBlur(gray_roi, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        return blurred_roi
    return gray_roi


def main():
    # Load the two images
    image1 = cv2.imread(IMAGE_PATH_1)
    image2 = cv2.imread(IMAGE_PATH_2)

    if image1 is None:
        print(f"Error: Unable to read first image '{IMAGE_PATH_1}'")
        return
    if image2 is None:
        print(f"Error: Unable to read second image '{IMAGE_PATH_2}'")
        return

    print(f"Comparing images: '{IMAGE_PATH_1}' and '{IMAGE_PATH_2}'")
    print(f"Using ROI Team 1: {SCORE_ROI_TEAM1}")
    print(f"Using ROI Team 2: {SCORE_ROI_TEAM2}")
    if APPLY_PREPROCESSING:
        print(f"ROI preprocessing enabled: grayscale + Gaussian blur (kernel: {GAUSSIAN_BLUR_KERNEL_SIZE})")
    else:
        print(f"ROI preprocessing enabled: grayscale only")


    # Extract and preprocess ROIs
    roi1_img1_orig = get_roi_from_image(image1, SCORE_ROI_TEAM1, IMAGE_PATH_1)
    roi1_img2_orig = get_roi_from_image(image2, SCORE_ROI_TEAM1, IMAGE_PATH_2)
    roi2_img1_orig = get_roi_from_image(image1, SCORE_ROI_TEAM2, IMAGE_PATH_1)
    roi2_img2_orig = get_roi_from_image(image2, SCORE_ROI_TEAM2, IMAGE_PATH_2)

    roi1_img1_processed = preprocess_roi(roi1_img1_orig)
    roi1_img2_processed = preprocess_roi(roi1_img2_orig)
    roi2_img1_processed = preprocess_roi(roi2_img1_orig)
    roi2_img2_processed = preprocess_roi(roi2_img2_orig)


    if roi1_img1_processed is None or roi1_img2_processed is None:
        print("Error: Unable to process Team 1 ROI. Please check ROI coordinates and images.")
        # return # Even if one ROI fails, we can still try to compute the other
    if roi2_img1_processed is None or roi2_img2_processed is None:
        print("Error: Unable to process Team 2 ROI. Please check ROI coordinates and images.")
        # return

    # Compute differences
    sad_roi1 = calculate_sad(roi1_img1_processed, roi1_img2_processed)
    mse_roi1 = calculate_mse(roi1_img1_processed, roi1_img2_processed)

    sad_roi2 = calculate_sad(roi2_img1_processed, roi2_img2_processed)
    mse_roi2 = calculate_mse(roi2_img1_processed, roi2_img2_processed)

    print("\n--- Difference Calculation Results ---")
    if sad_roi1 != float('inf'):
        print(f"ROI 1 (Team 1 score area):")
        print(f"  Sum of Absolute Differences (SAD): {sad_roi1}")
        print(f"  Mean Squared Error (MSE):        {mse_roi1:.2f}")
    else:
        print(f"ROI 1 (Team 1 score area): Unable to compute difference (ROI extraction may have failed)")

    if sad_roi2 != float('inf'):
        print(f"\nROI 2 (Team 2 score area):")
        print(f"  Sum of Absolute Differences (SAD): {sad_roi2}")
        print(f"  Mean Squared Error (MSE):        {mse_roi2:.2f}")
    else:
        print(f"\nROI 2 (Team 2 score area): Unable to compute difference (ROI extraction may have failed)")


    # Display ROI images used for comparison (optional)
    if roi1_img1_processed is not None: cv2.imshow("Img1 - ROI1 (Processed)", roi1_img1_processed)
    if roi1_img2_processed is not None: cv2.imshow("Img2 - ROI1 (Processed)", roi1_img2_processed)
    if roi2_img1_processed is not None: cv2.imshow("Img1 - ROI2 (Processed)", roi2_img1_processed)
    if roi2_img2_processed is not None: cv2.imshow("Img2 - ROI2 (Processed)", roi2_img2_processed)
    
    # Display diff images (optional)
    if roi1_img1_processed is not None and roi1_img2_processed is not None and roi1_img1_processed.shape == roi1_img2_processed.shape:
        diff_display_roi1 = cv2.absdiff(roi1_img1_processed, roi1_img2_processed)
        cv2.imshow("Diff ROI1 (Img1 vs Img2)", diff_display_roi1)
    if roi2_img1_processed is not None and roi2_img2_processed is not None and roi2_img1_processed.shape == roi2_img2_processed.shape:
        diff_display_roi2 = cv2.absdiff(roi2_img1_processed, roi2_img2_processed)
        cv2.imshow("Diff ROI2 (Img1 vs Img2)", diff_display_roi2)

    print("\nDisplaying images... Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
