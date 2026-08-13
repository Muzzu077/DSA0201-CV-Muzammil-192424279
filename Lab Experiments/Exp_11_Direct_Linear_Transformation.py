"""
Experiment 11: Perform transformation using Direct Linear Transformation (DLT)
Description: Formulate and solve DLT equations manually using Singular Value Decomposition (SVD)
             to estimate the Homography matrix, warp the image, and compare with cv2.findHomography.
"""
import cv2
import numpy as np
import os
import sys

def compute_dlt_homography(pts_src, pts_dst):
    """
    Computes Homography matrix using Direct Linear Transformation (DLT) algorithm.
    Each correspondence (x, y) -> (x', y') yields 2 linear equations:
    -x * h11 - y * h12 - h13 + 0*h21 + 0*h22 + 0*h23 + x*x'*h31 + y*x'*h32 + x'*h33 = 0
    0*h11 + 0*h12 + 0*h13 - x*h21 - y*h22 - h23 + x*y'*h31 + y*y'*h32 + y'*h33 = 0
    """
    A = []
    for i in range(len(pts_src)):
        x, y = pts_src[i][0], pts_src[i][1]
        xp, yp = pts_dst[i][0], pts_dst[i][1]
        
        row1 = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        row2 = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        
        A.append(row1)
        A.append(row2)
        
    A = np.array(A)
    
    # Perform Singular Value Decomposition (SVD) on A
    U, S, Vt = np.linalg.svd(A)
    
    # The solution h is the last row of Vt (last column of V) corresponding to smallest singular value
    h = Vt[-1]
    
    # Reshape to 3x3 matrix and normalize so H[2,2] is 1.0
    H = h.reshape((3, 3))
    H = H / H[2, 2]
    return H

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_11_DLT_Homography.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    h, w = img.shape[:2]
    
    # Define source coordinates (four corners of the image)
    pts_src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    # Define target coordinates (skewed layout)
    pts_dst = np.float32([[80, 50], [w - 60, 100], [w - 100, h - 40], [40, h - 80]])
    
    # 1. Compute H using manual DLT
    H_dlt = compute_dlt_homography(pts_src, pts_dst)
    print("Homography Matrix estimated via DLT:\n", H_dlt)
    
    # 2. Compute H using OpenCV's built-in algorithm for comparison
    H_cv, _ = cv2.findHomography(pts_src, pts_dst)
    print("Homography Matrix estimated via OpenCV:\n", H_cv)
    
    # Warp the image using manual DLT homography
    dlt_result = cv2.warpPerspective(img, H_dlt, (w, h))
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, dlt_result)
    print(f"DLT warped image saved to: {output_path}")
    
    # Display results
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("DLT Warp Result", dlt_result)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
