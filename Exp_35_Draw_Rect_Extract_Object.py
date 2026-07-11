"""
Experiment 35: Draw Rectangular shape and extract objects
Description: Draw a bounding box around a specific coordinate region of interest,
             and crop/extract that object.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_rect_img = os.path.join("outputs", "Exp_35_Rectangle_Drawn.jpg")
    output_cropped_obj = os.path.join("outputs", "Exp_35_Extracted_Object.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    drawn_img = img.copy()
    
    # 1. Define object coordinates to extract
    # The bounding box is x = [70, 230], y = [70, 230]
    x, y, w, h = 70, 70, 160, 160
    
    cv2.rectangle(drawn_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.putText(drawn_img, "Target Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 2. Extract (crop) the object
    extracted_obj = img[y:y+h, x:x+w]
    
    # Save the output images
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_rect_img, drawn_img)
    cv2.imwrite(output_cropped_obj, extracted_obj)
    
    print(f"Rectangle image saved to: {output_rect_img}")
    print(f"Extracted object image saved to: {output_cropped_obj}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Image with Rectangle", drawn_img)
            cv2.imshow("Extracted Object (Cropped)", extracted_obj)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
