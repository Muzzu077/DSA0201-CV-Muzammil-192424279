"""
Experiment 09: Perform Perspective Transformation on the Video
Description: Apply perspective warp to each frame of a video.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample_video.mp4")
    output_path = os.path.join("outputs", "Exp_09_Perspective_Video.mp4")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {input_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    os.makedirs("outputs", exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Set up points for perspective transformation (e.g. slight tilt)
    pts1 = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    pts2 = np.float32([[80, 50], [width - 80, 50], [30, height - 40], [width - 30, height - 40]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    print("Processing video frames and warping perspective...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        warped_frame = cv2.warpPerspective(frame, M, (width, height))
        out.write(warped_frame)
        
        # Display the video frame
        if not headless:
            try:
                cv2.imshow("Original Video Frame", frame)
                cv2.imshow("Warped Video Frame", warped_frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            except:
                pass
            
    cap.release()
    out.release()
    if not headless:
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
    print(f"Perspective warped video saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
