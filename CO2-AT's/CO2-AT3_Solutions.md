~~***SIMATS ENGINEERING **

~~***ASSESSMENT TOOL 1 **

~~***COURSE NAME: Computer Vision with OpenCV COURSE CODE: DSA02 **

~~***COURSE OUTCOME COVERED **

~~***CO2: Apply image processing and feature extraction techniques, including edge  detection, motion estimation, and optical flow, for analyzing images.. (BL3) ***

~~***Assessment Tool Weightage: 50% ***

~~***QUESTIONS: 5 Total Marks:50 Annexure A **

~~***Analytical Problem Solving **

~~***Code of Conduct: ***

~~***I \_G.MD Muzammil-192424279\_\_ (Name / Reg No) certify that this submission is my original work and  that I have adhered to the guidelines specified for this assessment. ***

~~***I understand that any violation of academic integrity rules will result in disciplinary action. Signature of the student: Muzammil…!!***

## Question 1: Image Preprocessing and Feature Extraction

> **Scenario:** A drone is used to capture aerial images of a farmland to identify pest-affected crop regions. Due to sensor limitations, wind movement, and changing sunlight, the captured images contain noise, shadows, and uneven illumination. Pest-affected areas usually show visible texture and color variations compared to healthy crop regions.

### a) Preprocessing Pipeline — 4 Marks

**Answer:**

1. **Noise Reduction:** Apply **Gaussian Smoothing Filter** to remove sensor/atmospheric noise while preserving general spatial structure.

2. **Illumination Correction:** Apply **Homomorphic Filtering / Background Division** to eliminate non-uniform sunlight gradients and cloud shadows.

3. **Contrast Enhancement:** Apply **CLAHE** in LAB color space ($L^\*$ channel) to enhance subtle boundary contrast between healthy and damaged leaves.

4. **Image Normalization:** Perform **Min-Max Scaling** to map pixel intensities linearly into standard dynamic range $\[0, 255\]$.


### b) Feature Extraction — 3 Marks

**Answer:**  
Select **Local Binary Patterns (LBP)** combined with **ExG (Excess Green Vegetation Index)**. LBP captures fine micro-texture variations caused by pest nibbling, while ExG highlights leaf discoloration.


### c) Rotation and Scale Variations — 3 Marks

**Answer:**  
Drone flight height changes alter crop scale, while camera angle shifts alter rotation.

- **Mitigation:** Use **SIFT** or **Rotation-Invariant Uniform LBP ($LBP\_\{P,R\}^\{	ext\{riu2\}\}$)** constructed over multi-scale image pyramids to maintain invariant detection.


## Question 2: Stereo Vision and Motion Estimation

> **Scenario:** A self-driving car is equipped with stereo cameras to navigate through a busy street. The system must estimate the distance of nearby vehicles and pedestrians in real time. It should also track moving objects accurately to avoid collisions, especially when pedestrians or vehicles change speed suddenly.

### a) Stereo Disparity and Depth Estimation — 4 Marks

**Answer:**  
Stereo vision uses triangulation from left and right camera views separated by baseline distance $B$: $$	ext\{Depth \} Z = rac\{f \\cdot B\}\{d\}$$ where $f$ is focal length and $d = x\_l - x\_r$ is disparity.

- **Nearby Objects:** Produce **large disparity $d$** (easy to resolve accurately).

- **Distant Objects:** Produce **small disparity $d$** (higher depth error sensitivity).


### b) Optical Flow for Motion Tracking — 3 Marks

**Answer:**  
Optical flow calculates motion field $\[u, v\]$ between consecutive video frames for each pedestrian pixel. Vector trajectories provide real-time velocity magnitude and direction, enabling forward path forecasting and collision avoidance.


### c) Errors and Mitigation Techniques — 3 Marks

**Answer:**

- **Error Sources:** Camera miscalibration, textureless surfaces (road asphalt), sudden movement, and lighting changes.

- **Mitigation:** Epipolar Image Rectification, Pyramidal Lucas-Kanade optical flow, and **Sensor Fusion (Stereo Camera + LiDAR)**.

