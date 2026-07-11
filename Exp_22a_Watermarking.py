"""
Experiment 22a: Insert water marking to the image using OpenCV
Description: Draw a semi-transparent text watermark onto an image using overlay blending.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_22a_Watermark.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    overlay = img.copy()
    (h, w) = img.shape[:2]
    
    # Add text to overlay
    watermark_text = "CONFIDENTIAL - OPENCV"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    color = (255, 255, 255)
    
    text_size, _ = cv2.getTextSize(watermark_text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    text_x = (w - text_w) // 2
    text_y = (h + text_h) // 2
    
    cv2.rectangle(overlay, (0, text_y - text_h - 20), (w, text_y + 20), (0, 0, 0), -1)
    cv2.putText(overlay, watermark_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    alpha = 0.3
    watermarked = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, watermarked)
    print(f"Watermarked image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Watermarked Image", watermarked)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
