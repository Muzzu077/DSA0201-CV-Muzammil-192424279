import cv2
import numpy as np
import os
import urllib.request

def create_directory_structure():
    os.makedirs('inputs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    print("Created 'inputs' and 'outputs' directories.")

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"{local_path} already exists. Skipping download.")
        return True
    try:
        print(f"Downloading {url} to {local_path}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response, open(local_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully downloaded {local_path}.")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def generate_sample_image():
    path = os.path.join('inputs', 'sample.jpg')
    if os.path.exists(path):
        return
    print("Generating synthetic sample.jpg...")
    # Create a 600x600 colorful image
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Draw a gradient background
    for y in range(600):
        img[y, :, 0] = int(y / 600 * 128)  # Blue gradient
        img[y, :, 1] = int((600 - y) / 600 * 128)  # Green gradient
        img[y, :, 2] = 64  # Constant red

    # Draw shapes
    cv2.circle(img, (150, 150), 80, (0, 0, 255), -1)  # Solid Red circle
    cv2.rectangle(img, (350, 80), (520, 250), (0, 255, 0), 10)  # Green rectangle border
    
    # Triangle
    points = np.array([[300, 350], [150, 500], [450, 500]], np.int32)
    cv2.polylines(img, [points], True, (255, 255, 0), 5)  # Cyan triangle outline
    cv2.fillPoly(img, [points - [0, 50]], (0, 255, 255))  # Yellow filled shifted triangle
    
    # Draw text
    cv2.putText(img, "OpenCV Lab Sample", (50, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    cv2.imwrite(path, img)
    print("Created inputs/sample.jpg")

def generate_watch_images():
    watch_path = os.path.join('inputs', 'watch.jpg')
    scene_path = os.path.join('inputs', 'scene_with_watch.jpg')
    
    if os.path.exists(watch_path) and os.path.exists(scene_path):
        return
        
    print("Generating synthetic watch and scene images...")
    
    # 1. Create a watch image (200x200)
    watch = np.zeros((200, 200, 3), dtype=np.uint8)
    # Fill with white background
    watch.fill(240)
    
    # Draw watch strap
    cv2.rectangle(watch, (80, 0), (120, 200), (80, 80, 80), -1) # Dark grey strap
    # Draw watch outer ring
    cv2.circle(watch, (100, 100), 70, (40, 40, 40), 10) # dark outer circle
    # Draw watch face (white inside)
    cv2.circle(watch, (100, 100), 65, (255, 255, 255), -1)
    
    # Draw tick marks and watch hands
    cv2.circle(watch, (100, 100), 3, (0, 0, 0), -1) # Center pin
    cv2.line(watch, (100, 100), (100, 50), (0, 0, 255), 3) # Hour hand pointing up (red)
    cv2.line(watch, (100, 100), (140, 100), (0, 0, 0), 2) # Minute hand pointing right (black)
    
    # Add dial numbers
    cv2.putText(watch, "12", (90, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(watch, "6", (95, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(watch, "3", (145, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(watch, "9", (45, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imwrite(watch_path, watch)
    print("Created inputs/watch.jpg")
    
    # 2. Create a scene containing the watch (800x600)
    scene = np.zeros((600, 800, 3), dtype=np.uint8)
    # Background texture/gradient
    for y in range(600):
        scene[y, :, 0] = int(y / 600 * 50) + 100
        scene[y, :, 1] = int(y / 600 * 30) + 100
        scene[y, :, 2] = int(y / 600 * 10) + 100
        
    # Draw other objects in the scene
    cv2.rectangle(scene, (50, 50), (300, 250), (180, 180, 150), -1) # A book
    cv2.putText(scene, "My Notebook", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    # Place the watch in the scene at a rotated and shifted position
    # Let's rotate the watch image by 30 degrees and scale it to 150x150
    center = (100, 100)
    rot_matrix = cv2.getRotationMatrix2D(center, 30, 0.75) # 30 deg rotation, 0.75 scale
    rotated_watch = cv2.warpAffine(watch, rot_matrix, (200, 200), borderValue=(120, 120, 120))
    
    # Overlay the rotated watch onto the scene at coordinates (x=450, y=300)
    h, w, _ = rotated_watch.shape
    x_offset, y_offset = 450, 300
    scene[y_offset:y_offset+h, x_offset:x_offset+w] = rotated_watch
    
    cv2.imwrite(scene_path, scene)
    print("Created inputs/scene_with_watch.jpg")

def generate_sample_video():
    path = os.path.join('inputs', 'sample_video.mp4')
    if os.path.exists(path):
        return
    print("Generating synthetic sample_video.mp4...")
    # 150 frames, 30 fps (5 seconds), 640x480 resolution
    width, height = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    x, y = 50, 50
    vx, vy = 6, 4
    radius = 30
    
    for frame_idx in range(150):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background color: gradient
        frame[:, :, 0] = 50 # dark blueish
        
        # Bouncing ball physics
        x += vx
        y += vy
        if x - radius < 0 or x + radius > width:
            vx = -vx
            x += vx * 2
        if y - radius < 0 or y + radius > height:
            vy = -vy
            y += vy * 2
            
        # Draw bouncing ball
        cv2.circle(frame, (x, y), radius, (0, 255, 255), -1) # Yellow ball
        
        # Draw text
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
        
    out.release()
    print("Created inputs/sample_video.mp4")

def generate_cars_video():
    path = os.path.join('inputs', 'cars_video.mp4')
    if os.path.exists(path):
        return
    print("Generating synthetic cars_video.mp4...")
    # 150 frames, 30 fps (5 seconds), 640x480 resolution
    width, height = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    # We will simulate 3 cars moving down road lanes
    car1_y = 100
    car2_y = 250
    car3_y = 380
    
    car1_x = -100
    car2_x = 700
    car3_x = -150
    
    for frame_idx in range(150):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw road
        cv2.rectangle(frame, (0, 80), (width, 420), (50, 50, 50), -1)
        # Lane dividers
        for lx in range(0, width, 40):
            cv2.line(frame, (lx, 200), (lx + 20, 200), (255, 255, 255), 2)
            cv2.line(frame, (lx, 320), (lx + 20, 320), (255, 255, 255), 2)
            
        # Move cars
        car1_x += 5
        car2_x -= 4
        car3_x += 6
        
        # Draw Car 1 (Red rectangle)
        if car1_x < width + 100:
            cv2.rectangle(frame, (car1_x, car1_y), (car1_x + 80, car1_y + 40), (0, 0, 255), -1)
            cv2.circle(frame, (car1_x + 20, car1_y + 40), 8, (10, 10, 10), -1)
            cv2.circle(frame, (car1_x + 60, car1_y + 40), 8, (10, 10, 10), -1)
            
        # Draw Car 2 (Green rectangle)
        if car2_x > -100:
            cv2.rectangle(frame, (car2_x, car2_y), (car2_x + 80, car2_y + 40), (0, 255, 0), -1)
            cv2.circle(frame, (car2_x + 20, car2_y + 40), 8, (10, 10, 10), -1)
            cv2.circle(frame, (car2_x + 60, car2_y + 40), 8, (10, 10, 10), -1)
            
        # Draw Car 3 (Blue rectangle)
        if car3_x < width + 100:
            cv2.rectangle(frame, (car3_x, car3_y), (car3_x + 70, car3_y + 35), (255, 0, 0), -1)
            cv2.circle(frame, (car3_x + 15, car3_y + 35), 7, (10, 10, 10), -1)
            cv2.circle(frame, (car3_x + 55, car3_y + 35), 7, (10, 10, 10), -1)
            
        out.write(frame)
        
    out.release()
    print("Created inputs/cars_video.mp4")

if __name__ == "__main__":
    create_directory_structure()
    generate_sample_image()
    generate_watch_images()
    generate_sample_video()
    generate_cars_video()
    
    # Download real face image and car cascade for realistic tests
    download_file("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg", os.path.join('inputs', 'face_sample.jpg'))
    download_file("https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml", os.path.join('inputs', 'cars.xml'))
    
    print("\nSample assets generated successfully!")
