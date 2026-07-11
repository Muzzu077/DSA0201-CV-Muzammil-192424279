"""
Experiment 22b: Do Cropping, Copying and pasting image inside another image using OpenCV
Description: Crop a Region of Interest (ROI) from an image, copy it, and paste it back at a different location.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_22b_Cropped_Pasted.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    img_copy = img.copy()
    
    ymin, ymax, xmin, xmax = 70, 230, 70, 230
    roi = img[ymin:ymax, xmin:xmax]
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    
    roi_h, roi_w = roi.shape[:2]
    target_ymin, target_xmin = 350, 350
    
    img_copy[target_ymin:target_ymin+roi_h, target_xmin:target_xmin+roi_w] = roi
    cv2.rectangle(img_copy, (target_xmin, target_ymin), (target_xmin+roi_w, target_ymin+roi_h), (0, 255, 0), 2)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, img_copy)
    print(f"Cropped and pasted image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Source Area Highlighted (Red)", img)
            cv2.imshow("Pasted Copy Highlighted (Green)", img_copy)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
