"""
Experiment 05: Perform Rotation of an image to clockwise and counter-clockwise direction
Description: Rotate an image by 90 degrees clockwise and 90 degrees counter-clockwise.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_cw = os.path.join("outputs", "Exp_05_Rotation_CW.jpg")
    output_ccw = os.path.join("outputs", "Exp_05_Rotation_CCW.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Get dimensions
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotate 90 degrees clockwise (angle = -90)
    matrix_cw = cv2.getRotationMatrix2D(center, -90, 1.0)
    rotated_cw = cv2.warpAffine(img, matrix_cw, (w, h))
    
    # Rotate 90 degrees counter-clockwise (angle = 90)
    matrix_ccw = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated_ccw = cv2.warpAffine(img, matrix_ccw, (w, h))
    
    # Save the outputs
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_cw, rotated_cw)
    cv2.imwrite(output_ccw, rotated_ccw)
    print(f"Clockwise rotated image saved to: {output_cw}")
    print(f"Counter-clockwise rotated image saved to: {output_ccw}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Clockwise Rotated (-90 deg)", rotated_cw)
            cv2.imshow("Counter-Clockwise Rotated (90 deg)", rotated_ccw)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
