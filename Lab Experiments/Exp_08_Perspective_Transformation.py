"""
Experiment 08: Perform Perspective Transformation on the image
Description: Warp perspective of the image based on four point correspondences.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_08_Perspective.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    (rows, cols) = img.shape[:2]
    
    # Define four source corners of the image and their destination coordinates
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    pts2 = np.float32([[100, 80], [cols - 80, 50], [50, rows - 100], [cols - 120, rows - 60]])
    
    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Apply transformation
    perspective_result = cv2.warpPerspective(img, M, (cols, rows))
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, perspective_result)
    print(f"Perspective warped image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Perspective Warp Result", perspective_result)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
