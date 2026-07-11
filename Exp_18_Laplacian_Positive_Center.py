"""
Experiment 18: Sharpening of Image using Laplacian mask with positive center coefficient
Description: Apply standard positive center Laplacian kernel and add it to the original image.
Mask:
 0  -1   0
-1   4  -1
 0  -1   0
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_18_Laplacian_PosCenter.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Define positive center Laplacian kernel
    kernel = np.array([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]], dtype=np.float32)
                       
    # Filter the image using 2D convolution
    img_float = np.float32(img)
    laplacian = cv2.filter2D(img_float, -1, kernel)
    
    # For a positive center, Sharpened = Original + Laplacian
    sharpened = img_float + laplacian
    
    # Post-process
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    laplacian_display = np.clip(np.absolute(laplacian), 0, 255).astype(np.uint8)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sharpened)
    print(f"Sharpened image (Laplacian Pos-Center) saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Laplacian Pos-Center Edges", laplacian_display)
            cv2.imshow("Sharpened Image (Pos-Center)", sharpened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
