"""
Experiment 19: Sharpening of Image using unsharp masking
Description: Subtract a blurred version of the image from the original to create an unsharp mask,
             and add this mask back to the original image to sharpen it (Formula: fs = f - fb, g = f + fs).
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_19_UnsharpMask.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    img_float = np.float32(img)
    
    # 1. Blur the original image
    blurred = cv2.GaussianBlur(img_float, (9, 9), 0)
    
    # 2. Subtract blurred image from original to get the unsharp mask (high-frequency detail)
    mask = img_float - blurred
    
    # 3. Add mask back to the original image
    sharpened = img_float + 1.0 * mask
    
    # Post-process
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    mask_display = np.clip(np.absolute(mask), 0, 255).astype(np.uint8)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sharpened)
    print(f"Unsharp masked sharpened image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Unsharp Mask (Edges)", mask_display)
            cv2.imshow("Sharpened Image (Unsharp Masking)", sharpened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
