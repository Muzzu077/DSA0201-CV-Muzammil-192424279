"""
Experiment 01d: Dilate an Image using Dilate function
Description: Read an image and apply morphological dilation.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_01d_Dilation.jpg")
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Define kernel for dilation
    kernel = np.ones((5, 5), np.uint8)
    
    # Dilate the image
    dilated = cv2.dilate(img, kernel, iterations=1)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, dilated)
    print(f"Dilated image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Dilated Image", dilated)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
