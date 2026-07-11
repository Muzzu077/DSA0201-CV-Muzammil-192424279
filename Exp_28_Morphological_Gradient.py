"""
Experiment 28: Morphological operations based on OpenCV using Morphological Gradient technique
Description: Morphological Gradient is the difference between dilation and erosion.
             It outlines boundaries of objects.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_28_Morphological_Gradient.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, gradient)
    print(f"Morphological Gradient result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Morphological Gradient (Object Outlines)", gradient)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
