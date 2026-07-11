"""
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
