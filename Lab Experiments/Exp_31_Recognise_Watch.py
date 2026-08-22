"""
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
