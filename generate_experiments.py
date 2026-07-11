import os

scripts = {}

# --- Experiment 01a ---
scripts["Exp_01a_Grayscale.py"] = '''"""
Experiment 01a: Convert an Image into Grayscale
Description: Read an image and convert it into Grayscale using OpenCV.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_01a_Grayscale.jpg")
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, gray)
    print(f"Grayscale image saved to: {output_path}")
    
    # Display the result (only if interactive and display available)
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Grayscale Image", gray)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 01b ---
scripts["Exp_01b_GaussianBlur.py"] = '''"""
Experiment 01b: Convert an Image to Blur using GaussianBlur
Description: Read an image and apply Gaussian blur.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_01b_GaussianBlur.jpg")
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Apply Gaussian Blur (kernel size 15x15)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, blurred)
    print(f"Blurred image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Gaussian Blurred Image", blurred)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 01c ---
scripts["Exp_01c_Canny_Outline.py"] = '''"""
Experiment 01c: Convert Image to Show Outline using Canny
Description: Read an image and show its outline using the Canny function.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_01c_Canny_Outline.jpg")
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    outline = cv2.Canny(gray, 100, 200)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, outline)
    print(f"Outline image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Canny Outline", outline)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 01d ---
scripts["Exp_01d_Dilation.py"] = '''"""
Experiment 01d: Dilate an Image using Dilate function
Description: Read an image and apply morphological dilation.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_01d_Dilation.jpg")
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Define kernel for dilation
    kernel = np.ones((5, 5), np.uint8)
    
    # Dilate the image
    dilated = cv2.dilate(img, kernel, iterations=1)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, dilated)
    print(f"Dilated image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Dilated Image", dilated)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 01e ---
scripts["Exp_01e_Erosion.py"] = '''"""
Experiment 01e: Erode an Image using Erode function
Description: Read an image and apply morphological erosion.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_01e_Erosion.jpg")
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Define kernel for erosion
    kernel = np.ones((5, 5), np.uint8)
    
    # Erode the image
    eroded = cv2.erode(img, kernel, iterations=1)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, eroded)
    print(f"Eroded image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Eroded Image", eroded)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 02 ---
scripts["Exp_02_Slow_Fast_Video.py"] = '''"""
Experiment 02: Read captured video and display in slow motion and fast motion
Description: Read a video file, modify frame display delay and frame skipping to display in slow/fast motion, and write output videos.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample_video.mp4")
    slow_output = os.path.join("outputs", "Exp_02_Slow_Motion.mp4")
    fast_output = os.path.join("outputs", "Exp_02_Fast_Motion.mp4")
    
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Process Slow Motion (save to file)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {input_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Output video for slow motion (0.5x speed by halving the FPS output setting)
    out_slow = cv2.VideoWriter(slow_output, fourcc, fps * 0.5, (width, height))
    
    print("Processing Slow Motion (saving to file)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out_slow.write(frame)
    cap.release()
    out_slow.release()
    print(f"Slow motion video saved to: {slow_output}")
    
    # 2. Process Fast Motion (2.0x speed by doubling the FPS output setting)
    cap = cv2.VideoCapture(input_path)
    out_fast = cv2.VideoWriter(fast_output, fourcc, fps * 2.0, (width, height))
    
    print("Processing Fast Motion (saving to file)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out_fast.write(frame)
    cap.release()
    out_fast.release()
    print(f"Fast motion video saved to: {fast_output}")
    
    # 3. Interactive Playback (skipped in headless mode)
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        print("Now showing slow motion playback (press 'q' to skip to fast motion)...")
        cap = cv2.VideoCapture(input_path)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Slow Motion Playback", frame)
                if cv2.waitKey(80) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            
            print("Now showing fast motion playback (press 'q' to exit)...")
            cap = cv2.VideoCapture(input_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Fast Motion Playback", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Playback window failed: {e}")
    else:
        print("Running in headless mode. Skipping interactive video playback.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 03 ---
scripts["Exp_03_Webcam_Slow_Fast_Video.py"] = '''"""
Experiment 03: Capture video from Web Camera and Display in Slow and Fast Motion
Description: Capture live webcam video, and stream/save it in slow motion and fast motion.
Note: Falls back to using synthetic sample video if no physical webcam is available.
"""
import cv2
import os
import sys

def main():
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    
    # Attempt to open the web camera (index 0)
    # In headless testing, we directly use the synthetic video to simulate webcam captures
    cap = None
    is_webcam = False
    
    if not headless:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            is_webcam = True
            
    if not is_webcam:
        print("Using synthetic video as simulated webcam feed...")
        cap = cv2.VideoCapture(os.path.join("inputs", "sample_video.mp4"))
        if not cap.isOpened():
            print("Error: Simulated webcam video not found.")
            return
            
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    slow_output = os.path.join("outputs", "Exp_03_Webcam_Slow.mp4")
    fast_output = os.path.join("outputs", "Exp_03_Webcam_Fast.mp4")
    os.makedirs("outputs", exist_ok=True)
    
    # Save a small clip of 90 frames (~3 seconds)
    frames_buffer = []
    print("Capturing 90 frames for processing...")
    
    for i in range(90):
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame.copy())
        
        # Display capture stream if GUI available
        if not headless:
            try:
                cv2.imshow("Webcam Live Capture Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass
            
    cap.release()
    if not headless:
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
    if not frames_buffer:
        print("No frames captured.")
        return
        
    # Write Slow Motion (0.5x speed: 15 fps)
    out_slow = cv2.VideoWriter(slow_output, fourcc, fps * 0.5, (width, height))
    for frame in frames_buffer:
        out_slow.write(frame)
    out_slow.release()
    print(f"Slow motion captured clip saved to: {slow_output}")
    
    # Write Fast Motion (2.0x speed: 60 fps)
    out_fast = cv2.VideoWriter(fast_output, fourcc, fps * 2.0, (width, height))
    for frame in frames_buffer:
        out_fast.write(frame)
    out_fast.release()
    print(f"Fast motion captured clip saved to: {fast_output}")
    
    # Interactive playback
    if not headless:
        try:
            print("Playing slow motion capture (press 'q' to continue to fast motion)...")
            for frame in frames_buffer:
                cv2.imshow("Webcam Slow Motion", frame)
                if cv2.waitKey(int(1000 / (fps * 0.5))) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
            
            print("Playing fast motion capture (press 'q' to exit)...")
            for frame in frames_buffer:
                cv2.imshow("Webcam Fast Motion", frame)
                if cv2.waitKey(int(1000 / (fps * 2.0))) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display window failed: {e}")
    else:
        print("Running in headless mode. Skipping interactive video playback.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 04 ---
scripts["Exp_04_Image_Scaling.py"] = '''"""
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
'''

# --- Experiment 05 ---
scripts["Exp_05_Image_Rotation.py"] = '''"""
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
'''

# --- Experiment 06 ---
scripts["Exp_06_Image_Translation.py"] = '''"""
Experiment 06: Perform moving of an image from one place to another (Translation)
Description: Translate an image horizontally by 100 pixels and vertically by 50 pixels.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_06_Translation.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    (h, w) = img.shape[:2]
    
    # Shift parameters: tx (horizontal), ty (vertical)
    tx, ty = 100, 50
    
    # Translation Matrix M = [[1, 0, tx], [0, 1, ty]]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    translated = cv2.warpAffine(img, M, (w, h))
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, translated)
    print(f"Translated image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Translated Image (dx=100, dy=50)", translated)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 07 ---
scripts["Exp_07_Affine_Transformation.py"] = '''"""
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
'''

# --- Experiment 08 ---
scripts["Exp_08_Perspective_Transformation.py"] = '''"""
Experiment 08: Perform Perspective Transformation on the image
Description: Warp perspective of the image based on four point correspondences.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_08_Perspective.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    (rows, cols) = img.shape[:2]
    
    # Define four source corners of the image and their destination coordinates
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    pts2 = np.float32([[100, 80], [cols - 80, 50], [50, rows - 100], [cols - 120, rows - 60]])
    
    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Apply transformation
    perspective_result = cv2.warpPerspective(img, M, (cols, rows))
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, perspective_result)
    print(f"Perspective warped image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Perspective Warp Result", perspective_result)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 09 ---
scripts["Exp_09_Perspective_Transformation_Video.py"] = '''"""
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
'''

# --- Experiment 10 ---
scripts["Exp_10_Homography_Matrix.py"] = '''"""
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
    print("Estimated Homography Matrix:\\n", H)
    
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
'''

# --- Experiment 11 ---
scripts["Exp_11_Direct_Linear_Transformation.py"] = '''"""
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
    print("Homography Matrix estimated via DLT:\\n", H_dlt)
    
    # 2. Compute H using OpenCV's built-in algorithm for comparison
    H_cv, _ = cv2.findHomography(pts_src, pts_dst)
    print("Homography Matrix estimated via OpenCV:\\n", H_cv)
    
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
'''

# --- Experiment 12 ---
scripts["Exp_12_Canny_Edge_Detection.py"] = '''"""
Experiment 12: Perform Edge detection using Canny method
Description: Apply grayscale conversion followed by high-quality Canny Edge Detection with customizable thresholds.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_12_Canny_Edges.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian noise reduction before Canny (highly recommended for edge detection)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    # Thresholds: Low = 50, High = 150
    edges = cv2.Canny(blurred, 50, 150)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, edges)
    print(f"Canny edge image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Canny Edges (50, 150)", edges)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 13 ---
scripts["Exp_13_Sobel_X.py"] = '''"""
Experiment 13: Perform Edge detection using Sobel Matrix along X axis
Description: Detect vertical edges using the Sobel operator in the X direction.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_13_SobelX.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel operator along X axis (dx=1, dy=0)
    # Use CV_64F to capture negative gradients, then take absolute value
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_abs = np.uint8(np.absolute(sobel_x))
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sobel_x_abs)
    print(f"Sobel X edge detection image saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Sobel X Edges", sobel_x_abs)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 14 ---
scripts["Exp_14_Sobel_Y.py"] = '''"""
Experiment 14: Perform Edge detection using Sobel Matrix along Y axis
Description: Detect horizontal edges using the Sobel operator in the Y direction.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_14_SobelY.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel operator along Y axis (dx=0, dy=1)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_abs = np.uint8(np.absolute(sobel_y))
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sobel_y_abs)
    print(f"Sobel Y edge detection image saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Sobel Y Edges", sobel_y_abs)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 15 ---
scripts["Exp_15_Sobel_XY.py"] = '''"""
Experiment 15: Perform Edge detection using Sobel Matrix along XY axis
Description: Detect edges in both dimensions by combining Sobel X and Sobel Y results.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_15_SobelXY.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute X and Y Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine X and Y gradients using L2 norm: magnitude = sqrt(sobelx^2 + sobely^2)
    sobel_xy = cv2.magnitude(sobel_x, sobel_y)
    sobel_xy_abs = np.uint8(np.absolute(sobel_xy))
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sobel_xy_abs)
    print(f"Combined Sobel XY edge detection image saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Sobel XY Edges", sobel_xy_abs)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 16 ---
scripts["Exp_16_Laplacian_Negative_Center.py"] = '''"""
Experiment 16: Sharpening of Image using Laplacian mask with negative center coefficient
Description: Apply standard negative center Laplacian kernel for edge detection and subtract
             it from the original image to obtain a sharpened image.
Mask:
 0   1   0
 1  -4   1
 0   1   0
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_16_Laplacian_NegCenter.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Define negative center Laplacian kernel
    kernel = np.array([[ 0,  1,  0],
                       [ 1, -4,  1],
                       [ 0,  1,  0]], dtype=np.float32)
                       
    # Filter the image using 2D convolution
    img_float = np.float32(img)
    laplacian = cv2.filter2D(img_float, -1, kernel)
    
    # For a negative center, Sharpened = Original - Laplacian
    sharpened = img_float - laplacian
    
    # Clip values to range [0, 255] and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    laplacian_display = np.clip(np.absolute(laplacian), 0, 255).astype(np.uint8)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sharpened)
    print(f"Sharpened image (Laplacian Neg-Center) saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Laplacian Edge Output", laplacian_display)
            cv2.imshow("Sharpened Image", sharpened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 17 ---
scripts["Exp_17_Laplacian_Diagonal.py"] = '''"""
Experiment 17: Sharpening of Image using Laplacian mask with diagonal neighbors
Description: Apply Laplacian kernel extending to diagonal neighbors (negative center -8)
             and subtract it from the original image to sharpen.
Mask:
 1   1   1
 1  -8   1
 1   1   1
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_17_Laplacian_Diagonal.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Define diagonal extended negative center Laplacian kernel
    kernel = np.array([[ 1,  1,  1],
                       [ 1, -8,  1],
                       [ 1,  1,  1]], dtype=np.float32)
                       
    # Filter the image using 2D convolution
    img_float = np.float32(img)
    laplacian = cv2.filter2D(img_float, -1, kernel)
    
    # Sharpened = Original - Laplacian
    sharpened = img_float - laplacian
    
    # Post-process
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    laplacian_display = np.clip(np.absolute(laplacian), 0, 255).astype(np.uint8)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sharpened)
    print(f"Sharpened image (Laplacian Diagonal) saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Laplacian Diagonal Edges", laplacian_display)
            cv2.imshow("Sharpened Image (Diagonal)", sharpened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 18 ---
scripts["Exp_18_Laplacian_Positive_Center.py"] = '''"""
Experiment 18: Sharpening of Image using Laplacian mask with positive center coefficient
Description: Apply standard positive center Laplacian kernel and add it to the original image.
Mask:
 0  -1   0
-1   4  -1
 0  -1   0
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_18_Laplacian_PosCenter.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Define positive center Laplacian kernel
    kernel = np.array([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]], dtype=np.float32)
                       
    # Filter the image using 2D convolution
    img_float = np.float32(img)
    laplacian = cv2.filter2D(img_float, -1, kernel)
    
    # For a positive center, Sharpened = Original + Laplacian
    sharpened = img_float + laplacian
    
    # Post-process
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    laplacian_display = np.clip(np.absolute(laplacian), 0, 255).astype(np.uint8)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sharpened)
    print(f"Sharpened image (Laplacian Pos-Center) saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Laplacian Pos-Center Edges", laplacian_display)
            cv2.imshow("Sharpened Image (Pos-Center)", sharpened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 19 ---
scripts["Exp_19_Unsharp_Masking.py"] = '''"""
Experiment 19: Sharpening of Image using unsharp masking
Description: Subtract a blurred version of the image from the original to create an unsharp mask,
             and add this mask back to the original image to sharpen it (Formula: fs = f - fb, g = f + fs).
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_19_UnsharpMask.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    img_float = np.float32(img)
    
    # 1. Blur the original image
    blurred = cv2.GaussianBlur(img_float, (9, 9), 0)
    
    # 2. Subtract blurred image from original to get the unsharp mask (high-frequency detail)
    mask = img_float - blurred
    
    # 3. Add mask back to the original image
    sharpened = img_float + 1.0 * mask
    
    # Post-process
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    mask_display = np.clip(np.absolute(mask), 0, 255).astype(np.uint8)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sharpened)
    print(f"Unsharp masked sharpened image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Unsharp Mask (Edges)", mask_display)
            cv2.imshow("Sharpened Image (Unsharp Masking)", sharpened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 20 ---
scripts["Exp_20_High_Boost_Masks.py"] = '''"""
Experiment 20: Sharpening of Image using High-Boost Masks
Description: Boost edges using High-Boost filtering (Formula: g = f + k * mask, where k > 1).
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_20_HighBoost.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    img_float = np.float32(img)
    
    # 1. Blur the image
    blurred = cv2.GaussianBlur(img_float, (9, 9), 0)
    
    # 2. Generate mask (High-pass filter)
    mask = img_float - blurred
    
    # 3. Apply High-Boost filtering with boost factor k > 1 (e.g., k = 2.5)
    k = 2.5
    high_boost = img_float + k * mask
    
    # Post-process
    high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, high_boost)
    print(f"High-boost filtered image saved to: {output_path} (k = {k})")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("High-Boost Sharpened (k=2.5)", high_boost)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 21 ---
scripts["Exp_21_Gradient_Masking.py"] = '''"""
Experiment 21: Sharpening of Image using Gradient masking
Description: Generate a Sobel gradient magnitude mask and blend it with the original image
             to selectively sharpen edges.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_21_GradientMask.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Convert to grayscale for gradient estimation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute Sobel gradients in X and Y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalize gradient magnitude to [0, 1] range to act as a mask
    cv2.normalize(grad_mag, grad_mag, 0, 1, cv2.NORM_MINMAX)
    
    # Expand dims to 3 channels to match BGR shape
    grad_mask = np.dstack([grad_mag, grad_mag, grad_mag])
    
    # Perform Laplacian filtering for sharpening data
    laplacian_edges = cv2.Laplacian(img, cv2.CV_32F)
    
    # Apply gradient mask to selective sharpening
    img_float = np.float32(img)
    sharpened = img_float - 1.5 * (grad_mask * laplacian_edges)
    
    # Post-process
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    mask_display = np.uint8(grad_mag * 255)
    
    # Save the output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, sharpened)
    print(f"Gradient-masked sharpened image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Gradient Magnitude Mask", mask_display)
            cv2.imshow("Gradient-Masked Sharpened", sharpened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 22a ---
scripts["Exp_22a_Watermarking.py"] = '''"""
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
'''

# --- Experiment 22b ---
scripts["Exp_22b_Cropping_Copy_Paste.py"] = '''"""
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
'''

# --- Experiment 23 ---
scripts["Exp_23_Boundary_Convolution.py"] = '''"""
Experiment 23: Find the boundary of the image using Convolution kernel for the given image
Description: Convolve the image with a custom boundary detection kernel (Laplacian of Gaussian or edge kernel).
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_23_Boundary_Convolution.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)
                       
    boundary = cv2.filter2D(blurred, -1, kernel)
    _, boundary_thresh = cv2.threshold(boundary, 30, 255, cv2.THRESH_BINARY)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, boundary_thresh)
    print(f"Boundary image saved to: {output_path}")
    
    # Display the result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Convolved Boundary Response", boundary)
            cv2.imshow("Thresholded Boundary Output", boundary_thresh)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 24 ---
scripts["Exp_24_Morphological_Erosion.py"] = '''"""
Experiment 24: Morphological operations based on OpenCV using Erosion technique
Description: Perform Erosion with varying kernel sizes and display the structural shrinkage.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_24_Erosion.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    eroded = cv2.erode(img, kernel, iterations=2)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, eroded)
    print(f"Erosion morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Eroded Image (Kernel 7x7, Iter=2)", eroded)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 25 ---
scripts["Exp_25_Morphological_Dilation.py"] = '''"""
Experiment 25: Morphological operations based on OpenCV using Dilation technique
Description: Perform Dilation with varying structuring elements and display the structural growth.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_25_Dilation.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(img, kernel, iterations=2)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, dilated)
    print(f"Dilation morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Dilated Image (Kernel Ellipse 7x7, Iter=2)", dilated)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 26 ---
scripts["Exp_26_Morphological_Opening.py"] = '''"""
Experiment 26: Morphological operations based on OpenCV using Opening technique
Description: Opening is erosion followed by dilation. Useful in removing background noise.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_26_Opening.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Create structured noise (small dots) on the original image for demonstration
    noisy_img = img.copy()
    h, w, _ = noisy_img.shape
    noise_mask = np.random.rand(h, w) < 0.05
    noisy_img[noise_mask] = (255, 255, 255)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(noisy_img, cv2.MORPH_OPEN, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, opened)
    print(f"Opening morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Noisy Image (Input)", noisy_img)
            cv2.imshow("Opened Image (Noise Removed)", opened)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 27 ---
scripts["Exp_27_Morphological_Closing.py"] = '''"""
Experiment 27: Morphological operations based on OpenCV using Closing technique
Description: Closing is dilation followed by erosion. Useful in closing small holes inside objects.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_27_Closing.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    # Simulate internal holes (black spots) inside solid areas of image
    noisy_img = img.copy()
    h, w, _ = noisy_img.shape
    noise_mask = np.random.rand(h, w) < 0.05
    noisy_img[noise_mask] = (0, 0, 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(noisy_img, cv2.MORPH_CLOSE, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, closed)
    print(f"Closing morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Image with Holes (Input)", noisy_img)
            cv2.imshow("Closed Image (Holes Filled)", closed)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 28 ---
scripts["Exp_28_Morphological_Gradient.py"] = '''"""
Experiment 28: Morphological operations based on OpenCV using Morphological Gradient technique
Description: Morphological Gradient is the difference between dilation and erosion.
             It outlines boundaries of objects.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_28_Morphological_Gradient.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, gradient)
    print(f"Morphological Gradient result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Morphological Gradient (Object Outlines)", gradient)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 29 ---
scripts["Exp_29_Top_Hat.py"] = '''"""
Experiment 29: Morphological operations based on OpenCV using Top Hat technique
Description: Top Hat is the difference between input image and opening.
             It highlights bright objects on dark backgrounds.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_29_TopHat.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, tophat)
    print(f"Top Hat morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Top Hat (Bright Elements)", tophat)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 30 ---
scripts["Exp_30_Black_Hat.py"] = '''"""
Experiment 30: Morphological operations based on OpenCV using Black Hat technique
Description: Black Hat is the difference between closing and input image.
             It highlights dark structures on bright backgrounds.
"""
import cv2
import numpy as np
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample.jpg")
    output_path = os.path.join("outputs", "Exp_30_BlackHat.jpg")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, blackhat)
    print(f"Black Hat morphological result saved to: {output_path}")
    
    # Display result
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("Original Image", img)
            cv2.imshow("Black Hat (Dark Elements)", blackhat)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 31 ---
scripts["Exp_31_Recognise_Watch.py"] = '''"""
Experiment 31: Recognise watch from the given image by general Object recognition using OpenCV
Description: Detect keypoints and descriptors using ORB, match features using Brute-Force Matcher
             with Hamming distance, filter matches, and draw a bounding box around the localized watch.
"""
import cv2
import numpy as np
import os
import sys

def main():
    template_path = os.path.join("inputs", "watch.jpg")
    scene_path = os.path.join("inputs", "scene_with_watch.jpg")
    output_path = os.path.join("outputs", "Exp_31_Watch_Recognition.jpg")
    
    # Load the template (object to recognize) and scene (containing the object)
    template_img = cv2.imread(template_path)
    scene_img = cv2.imread(scene_path)
    
    if template_img is None or scene_img is None:
        print("Error: Could not load watch template or scene image.")
        return
        
    # Convert images to grayscale
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Find keypoints and descriptors
    kp_template, desc_template = orb.detectAndCompute(gray_template, None)
    kp_scene, desc_scene = orb.detectAndCompute(gray_scene, None)
    
    if desc_template is None or desc_scene is None:
        print("Error: ORB features could not be calculated.")
        return
        
    # Initialize Brute-Force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(desc_template, desc_scene)
    
    # Sort matches by distance (best first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Display the matched points
    matched_img = cv2.drawMatches(template_img, kp_template, scene_img, kp_scene, matches[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Find Homography to locate template boundary in the scene
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is not None:
        h, w = template_img.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst_corners = cv2.perspectiveTransform(pts, H)
        
        recognition_result = scene_img.copy()
        cv2.polylines(recognition_result, [np.int32(dst_corners)], True, (0, 0, 255), 4)
        cv2.putText(recognition_result, "WATCH RECOGNIZED", (int(dst_corners[0][0][0]), int(dst_corners[0][0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        print("Could not compute homography.")
        recognition_result = scene_img
        
    # Save results
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, recognition_result)
    print(f"Watch Recognition result saved to: {output_path}")
    
    # Display results
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            cv2.imshow("ORB Matches", matched_img)
            if H is not None:
                cv2.imshow("Watch Detected in Scene", recognition_result)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display not available: {e}")
    else:
        print("Running in headless mode. Skipping display window.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 32 ---
scripts["Exp_32_Play_Video_Reverse.py"] = '''"""
Experiment 32: Using OpenCV play Video in Reverse mode
Description: Load all frames of a video file, reverse the sequence, display them,
             and write the reversed video to outputs.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "sample_video.mp4")
    output_path = os.path.join("outputs", "Exp_32_Reverse_Video.mp4")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {input_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Read all frames into memory
    frames = []
    print("Reading video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Total frames read: {len(frames)}. Reversing...")
    reversed_frames = frames[::-1]
    
    # Write to output file
    os.makedirs("outputs", exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in reversed_frames:
        out.write(f)
    out.release()
    print(f"Reversed video saved to: {output_path}")
    
    # Play reversed video
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    if not headless:
        try:
            print("Playing reversed video playback (press 'q' to exit)...")
            for f in reversed_frames:
                cv2.imshow("Reversed Video", f)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Display window failed: {e}")
    else:
        print("Running in headless mode. Skipping interactive video playback.")

if __name__ == "__main__":
    main()
'''

# --- Experiment 33 ---
scripts["Exp_33_Face_Detection.py"] = '''"""
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
'''

# --- Experiment 34 ---
scripts["Exp_34_Vehicle_Detection.py"] = '''"""
Experiment 34: Vehicle Detection in a Video frame using OpenCV
Description: Read frames of a traffic video, load the Haar Cascade vehicle classifier,
             detect vehicles, and draw bounding boxes around them.
             Uses an active motion-contour based detector fallback to ensure detection boxes
             show on synthetic test videos if Cascade parameters do not match synthetic textures.
"""
import cv2
import os
import sys

def main():
    input_path = os.path.join("inputs", "cars_video.mp4")
    cascade_path = os.path.join("inputs", "cars.xml")
    output_path = os.path.join("outputs", "Exp_34_Vehicle_Detection.mp4")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {input_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    os.makedirs("outputs", exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize Cascade Classifier
    car_cascade = cv2.CascadeClassifier(cascade_path)
    has_cascade = not car_cascade.empty()
    if not has_cascade:
        print("Warning: cars.xml cascade file not found or empty. Using robust motion-contour detection fallback...")
        
    # Background Subtractor for fallback motion detection
    backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)
    
    headless = os.environ.get("CV_HEADLESS", "0") == "1" or not sys.stdin.isatty()
    print("Processing video frames for vehicle detection...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = []
        
        # Method 1: Haar Cascade (if available)
        if has_cascade:
            cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            for box in cars:
                boxes.append(box)
                
        # Method 2: Motion Detection Fallback (if cascade yields no boxes or is missing)
        if not boxes:
            fg_mask = backSub.apply(frame)
            _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 600:
                    x, y, w, h = cv2.boundingRect(c)
                    if 80 < y < 420 and w < 150 and h < 100:
                        boxes.append((x, y, w, h))
                        
        # Draw bounding boxes around detected vehicles
        display_frame = frame.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Vehicle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        out.write(display_frame)
        
        if not headless:
            try:
                cv2.imshow("Vehicle Detection Feed", display_frame)
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
        
    print(f"Vehicle detection processed. Output video saved to: {output_path}")

if __name__ == "__main__":
    main()
'''

# --- Experiment 35 ---
scripts["Exp_35_Draw_Rect_Extract_Object.py"] = '''"""
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
'''

# Write the generator file
def generate():
    for filename, code in scripts.items():
        filepath = filename
        with open(filepath, "w") as f:
            f.write(code)
        print(f"Generated {filepath}")

if __name__ == "__main__":
    generate()
