"""
Experiment 33: Face Detection using OpenCV
Description: Read an image, load the Haar Cascade frontal face classifier,
             detect faces, and draw bounding boxes around detected faces.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "face_sample.jpg")
    output_path = os.path.join("outputs", "Exp_33_Face_Detection.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained Haar Cascade classifier from OpenCV data folder
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {cascade_path}")
        return
        
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Number of faces detected: {len(faces)}")
    
    # Create copy to draw detection boxes
    result_img = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 3) # Blue box
        cv2.putText(result_img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, result_img)
    print(f"Face detection result saved to: {output_path}")
    
    # Display results
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Detected Faces", result_img)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
