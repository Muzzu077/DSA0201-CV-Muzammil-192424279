"""
Experiment 23: Find the boundary of the image using Convolution kernel for the given image
Description: Convolve the image with a custom boundary detection kernel (Laplacian of Gaussian or edge kernel).
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_23_Boundary_Convolution.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)
                       
    boundary = cv2.filter2D(blurred, -1, kernel)
    _, boundary_thresh = cv2.threshold(boundary, 30, 255, cv2.THRESH_BINARY)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, boundary_thresh)
    print(f"Boundary image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Convolved Boundary Response", boundary)
            cv2.imshow("Thresholded Boundary Output", boundary_thresh)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
