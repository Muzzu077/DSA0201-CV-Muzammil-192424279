"""
Experiment 13: Perform Edge detection using Sobel Matrix along X axis
Description: Detect vertical edges using the Sobel operator in the X direction.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_13_SobelX.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel operator along X axis (dx=1, dy=0)
    # Use CV_64F to capture negative gradients, then take absolute value
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_abs = np.uint8(np.absolute(sobel_x))
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sobel_x_abs)
    print(f"Sobel X edge detection image saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Sobel X Edges", sobel_x_abs)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
