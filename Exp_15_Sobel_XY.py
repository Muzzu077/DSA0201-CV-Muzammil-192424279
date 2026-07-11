"""
Experiment 15: Perform Edge detection using Sobel Matrix along XY axis
Description: Detect edges in both dimensions by combining Sobel X and Sobel Y results.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_15_SobelXY.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute X and Y Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine X and Y gradients using L2 norm: magnitude = sqrt(sobelx^2 + sobely^2)
    sobel_xy = cv2.magnitude(sobel_x, sobel_y)
    sobel_xy_abs = np.uint8(np.absolute(sobel_xy))
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sobel_xy_abs)
    print(f"Combined Sobel XY edge detection image saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Sobel XY Edges", sobel_xy_abs)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
