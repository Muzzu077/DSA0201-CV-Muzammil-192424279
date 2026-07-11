"""
Experiment 20: Sharpening of Image using High-Boost Masks
Description: Boost edges using High-Boost filtering (Formula: g = f + k * mask, where k > 1).
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_20_HighBoost.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    img_float = np.float32(img)
    
    # 1. Blur the image
    blurred = cv2.GaussianBlur(img_float, (9, 9), 0)
    
    # 2. Generate mask (High-pass filter)
    mask = img_float - blurred
    
    # 3. Apply High-Boost filtering with boost factor k > 1 (e.g., k = 2.5)
    k = 2.5
    high_boost = img_float + k * mask
    
    # Post-process
    high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, high_boost)
    print(f"High-boost filtered image saved to: {output_path} (k = {k})")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("High-Boost Sharpened (k=2.5)", high_boost)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
