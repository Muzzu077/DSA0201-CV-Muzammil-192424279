"""
Experiment 07: Perform Affine Transformation on the image
Description: Apply affine warp based on three point correspondences.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_07_Affine.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    (rows, cols) = img.shape[:2]
    
    # Define three points in original image and their corresponding positions in output
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    
    # Calculate affine transformation matrix
    M = cv2.getAffineTransform(pts1, pts2)
    
    # Apply transformation
    affine_result = cv2.warpAffine(img, M, (cols, rows))
    
    # Draw reference points on original image for visualization
    img_points = img.copy()
    for pt in pts1:
        cv2.circle(img_points, tuple(map(int, pt)), 5, (0, 0, 255), -1)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, affine_result)
    print(f"Affine warped image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original with Points", img_points)
            cv2.imshow("Affine Warp Result", affine_result)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
