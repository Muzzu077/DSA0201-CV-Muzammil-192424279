"""
Experiment 26: Morphological operations based on OpenCV using Opening technique
Description: Opening is erosion followed by dilation. Useful in removing background noise.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_26_Opening.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Create structured noise (small dots) on the original image for demonstration
    noisy_img = img.copy()
    h, w, _ = noisy_img.shape
    noise_mask = np.random.rand(h, w) < 0.05
    noisy_img[noise_mask] = (255, 255, 255)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(noisy_img, cv2.MORPH_OPEN, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, opened)
    print(f"Opening morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Noisy Image (Input)", noisy_img)
            cv2.imshow("Opened Image (Noise Removed)", opened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
