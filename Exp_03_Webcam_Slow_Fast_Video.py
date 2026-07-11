"""
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
