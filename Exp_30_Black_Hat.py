"""
Experiment 30: Morphological operations based on OpenCV using Black Hat technique
Description: Black Hat is the difference between closing and input image.
             It highlights dark structures on bright backgrounds.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_30_BlackHat.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, blackhat)
    print(f"Black Hat morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Black Hat (Dark Elements)", blackhat)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
