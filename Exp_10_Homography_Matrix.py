"""
Experiment 10: Perform transformation using Homography matrix
Description: Map the coordinates of an image using cv2.findHomography and cv2.warpPerspective.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_10_Homography.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    h, w = img.shape[:2]
    
    # Define source coordinates (four corners of the image)
    pts_src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    
    # Define target coordinates (skewed layout)
    pts_dst = np.float32([[50, 80], [w - 100, 40], [w - 50, h - 100], [80, h - 50]])
    
    # Find Homography Matrix
    H, status = cv2.findHomography(pts_src, pts_dst)
    print("Estimated Homography Matrix:\n", H)
    
    # Warp perspective using Homography
    homography_result = cv2.warpPerspective(img, H, (w, h))
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, homography_result)
    print(f"Homography warped image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Homography Result", homography_result)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
