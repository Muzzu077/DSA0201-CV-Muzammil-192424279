"""
Experiment 04: Scaling an image to its Bigger and Smaller sizes
Description: Scale an image to 2.0x size (bigger) and 0.5x size (smaller).
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_bigger = os.path.join("outputs", "Exp_04_Scaled_Bigger.jpg")
    output_smaller = os.path.join("outputs", "Exp_04_Scaled_Smaller.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Scale Image
    # Smaller using INTER_AREA (preferred for shrinking)
    smaller = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # Bigger using INTER_LINEAR (preferred for zooming/enlarging)
    bigger = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    
    # Save the outputs
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_bigger, bigger)
    cv2.imwrite(output_smaller, smaller)
    print(f"Bigger scaled image saved to: {output_bigger} (Shape: {bigger.shape})")
    print(f"Smaller scaled image saved to: {output_smaller} (Shape: {smaller.shape})")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Smaller Image (0.5x)", smaller)
            cv2.imshow("Bigger Image (2.0x)", bigger)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
