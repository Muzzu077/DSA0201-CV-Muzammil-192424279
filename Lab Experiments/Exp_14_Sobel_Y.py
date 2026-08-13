"""
Experiment 14: Perform Edge detection using Sobel Matrix along Y axis
Description: Detect horizontal edges using the Sobel operator in the Y direction.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_14_SobelY.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel operator along Y axis (dx=0, dy=1)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_abs = np.uint8(np.absolute(sobel_y))
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sobel_y_abs)
    print(f"Sobel Y edge detection image saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Sobel Y Edges", sobel_y_abs)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
