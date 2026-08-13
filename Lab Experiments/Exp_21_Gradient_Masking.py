"""
Experiment 21: Sharpening of Image using Gradient masking
Description: Generate a Sobel gradient magnitude mask and blend it with the original image
             to selectively sharpen edges.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_21_GradientMask.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to grayscale for gradient estimation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute Sobel gradients in X and Y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalize gradient magnitude to [0, 1] range to act as a mask
    cv2.normalize(grad_mag, grad_mag, 0, 1, cv2.NORM_MINMAX)
    
    # Expand dims to 3 channels to match BGR shape
    grad_mask = np.dstack([grad_mag, grad_mag, grad_mag])
    
    # Perform Laplacian filtering for sharpening data
    laplacian_edges = cv2.Laplacian(img, cv2.CV_32F)
    
    # Apply gradient mask to selective sharpening
    img_float = np.float32(img)
    sharpened = img_float - 1.5 * (grad_mask * laplacian_edges)
    
    # Post-process
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    mask_display = np.uint8(grad_mag * 255)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sharpened)
    print(f"Gradient-masked sharpened image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Gradient Magnitude Mask", mask_display)
            cv2.imshow("Gradient-Masked Sharpened", sharpened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
