"""
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
