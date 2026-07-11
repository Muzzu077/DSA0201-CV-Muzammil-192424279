"""
Experiment 06: Perform moving of an image from one place to another (Translation)
Description: Translate an image horizontally by 100 pixels and vertically by 50 pixels.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_06_Translation.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    (h, w) = img.shape[:2]
    
    # Shift parameters: tx (horizontal), ty (vertical)
    tx, ty = 100, 50
    
    # Translation Matrix M = [[1, 0, tx], [0, 1, ty]]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    translated = cv2.warpAffine(img, M, (w, h))
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, translated)
    print(f"Translated image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Translated Image (dx=100, dy=50)", translated)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
