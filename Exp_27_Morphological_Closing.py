"""
Experiment 27: Morphological operations based on OpenCV using Closing technique
Description: Closing is dilation followed by erosion. Useful in closing small holes inside objects.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_27_Closing.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Simulate internal holes (black spots) inside solid areas of image
    noisy_img = img.copy()
    h, w, _ = noisy_img.shape
    noise_mask = np.random.rand(h, w) < 0.05
    noisy_img[noise_mask] = (0, 0, 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(noisy_img, cv2.MORPH_CLOSE, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, closed)
    print(f"Closing morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Image with Holes (Input)", noisy_img)
            cv2.imshow("Closed Image (Holes Filled)", closed)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
