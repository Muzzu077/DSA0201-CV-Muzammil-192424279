# SIMATS ENGINEERING
### DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING
**COURSE:** Computer Vision with OpenCV (DSA02)  
**COURSE OUTCOME COVERED:** CO2: Apply image processing and feature extraction techniques, including edge detection, motion estimation, and optical flow, for analyzing images.. (BL3)  
**ASSESSMENT TOOL 2:** Scenario-Based  
**Weightage:** 30% | **Total Marks:** 30

---

## Question 1: Crack Detection on Concrete Surfaces Using Drone Images

> **Scenario:** A civil inspection team is developing a computer vision system to detect thin cracks on concrete bridges and building surfaces. Images are captured using a drone under outdoor conditions. Due to changing sunlight, shadows, and camera vibration, the raw images contain uneven illumination, low contrast, and Gaussian noise.

### A. Propose a suitable image preprocessing pipeline to prepare the drone images for crack detection. Include methods for illumination correction, noise reduction, contrast enhancement, and image normalization. Justify each step.
**Answer:**  
1. **Illumination Correction (Top-Hat Transform / Background Division):**  
   *Method:* Subtract morphological opening or low-pass filtered illumination background from original image.  
   *Justification:* Eliminates non-uniform outdoor sunlight gradients and drone shadows so cracks have uniform background thresholds across the entire bridge surface.
2. **Noise Reduction (Bilateral Filter):**  
   *Method:* Applies spatial and range Gaussian weighting.  
   *Justification:* Suppresses drone camera Gaussian noise in flat concrete regions while strictly preserving fine high-contrast crack edges.
3. **Contrast Enhancement (CLAHE):**  
   *Method:* Contrast Limited Adaptive Histogram Equalization computed over local $8 	imes 8$ contextual tiles.  
   *Justification:* Boosts local contrast of faint crack boundaries without amplifying background concrete texture noise.
4. **Image Normalization (Min-Max Scaling):**  
   *Method:* Linearly maps pixel intensities to $[0, 255]$.  
   *Justification:* Standardizes dynamic intensity range across varying drone flight lighting conditions.

---

### B. Select an appropriate edge detection technique for identifying thin surface cracks. Compare it with at least one other edge detector and explain why your selected technique is more suitable.
**Answer:**  
- **Selected Method:** **Canny Edge Detector**.
- **Comparison with Sobel Operator:**  
  - *Sobel:* Computes simple first-order intensity gradients without edge thinning. It produces thick, noisy, double-line edge responses around single crack lines and is highly susceptible to background concrete roughness.
  - *Canny:* Uses **Non-Maximum Suppression (NMS)** to thin candidate edges down to accurate 1-pixel-wide lines. It also uses **Hysteresis Dual Thresholding** ($T_{	ext{high}}, T_{	ext{low}}$) to track continuous weak crack segments that are connected to strong edge pixels, significantly outperforming Sobel for fine crack tracing.

---

### C. After edge detection, some crack segments appear broken or disconnected. Suggest a suitable post-processing technique to connect the crack regions.
**Answer:**  
Apply **Directional Morphological Closing** using oriented line structuring elements (or Hessian-based **Frangi Vesselness Filtering** followed by Connected Component Graph Linking) to bridge broken crack gaps along dominant crack directions.

---

## Question 2: Human Motion Tracking in a Security Camera Video

> **Scenario:** A security camera is installed in a hallway to monitor people moving in different directions. The system must estimate the motion of each person across video frames for tracking and activity analysis. The hallway contains plain walls, smooth floors, and occasional shadows.

### A. Explain how the Lucas-Kanade optical flow method can be used to estimate the motion of people between consecutive video frames.
**Answer:**  
- **Brightness Constancy Assumption:** Assumes pixel intensity $I(x,y,t)$ remains constant along motion trajectory:
  $$I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t) \implies I_x u + I_y v + I_t = 0$$
- **Spatial Coherence:** Assumes all pixels within a local $N 	imes N$ window move with identical velocity vector $\mathbf{v} = [u, v]^T$.
- **Least Squares Optimization:** Solves over-constrained linear system $\mathbf{A} \mathbf{v} = \mathbf{b}$ via:
  $$\mathbf{v} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b}, \quad 	ext{where } \mathbf{A}^T \mathbf{A} = egin{bmatrix} \sum I_x^2 & \sum I_x I_y \ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$$

---

### B. Discuss how the aperture effect can create errors in motion estimation, especially on uniform regions such as plain walls, floors, and clothing surfaces.
**Answer:**  
- **Aperture Effect & Flat Regions:** On plain walls, smooth floors, or uniform clothing, spatial gradients are nearly zero ($I_x pprox 0, I_y pprox 0$).
- **Singularity of Structure Tensor:** Matrix $\mathbf{A}^T \mathbf{A}$ becomes rank-deficient (singular), rendering Lucas-Kanade unable to compute motion vectors. Along 1D straight edges, only normal velocity component can be determined, causing parallel motion errors.

---

### C. If a person suddenly changes speed or direction, describe how optical flow vectors will be affected. Suggest a method to improve tracking accuracy.
**Answer:**  
- **Effect:** Sudden velocity changes violate small motion assumptions of 1st-order Taylor expansion, causing tracking loss or spurious vector spikes.
- **Improvement Method:** Use **Pyramidal Lucas-Kanade (Coarse-to-Fine Multi-Scale Optical Flow)** combined with **Kalman Filtering / DeepSORT** motion state prediction.

---

## Question 3: Vehicle Tracking at a Traffic Junction Using Video Frames

> **Scenario:** A smart traffic monitoring system uses CCTV footage to track vehicles at a busy road junction. Vehicles move at different speeds, stop suddenly, turn, and experience shadows, reflections, and partial occlusion.

### A. Explain how optical flow can be used to estimate the direction and speed of moving vehicles.
**Answer:**  
Dense optical flow (e.g. Farnebäck algorithm) calculates motion vectors $[u, v]$ for all vehicle pixels:
- **Instantaneous Speed:** $	ext{Speed} = \sqrt{u^2 + v^2} 	imes 	ext{Scale Factor} 	imes 	ext{FPS}$
- **Motion Direction:** $	heta = rctan2(v, u)$

---

### B. Identify two challenges affecting motion estimation (occlusion, shadows, sudden braking) and explain their effects.
**Answer:**  
1. **Partial Occlusion:** When one vehicle overlaps another, brightness constancy and spatial coherence are violated, causing optical flow vectors to merge or fail.
2. **Moving Shadows & Reflections:** Moving vehicle shadows change road pixel intensities, generating false optical flow vectors on stationary asphalt.

---

### C. Suggest a suitable improvement method to make vehicle tracking more reliable during overlaps or sudden directional changes.
**Answer:**  
Implement a **Hybrid Deep Learning & Multi-Object Tracking Pipeline**: Combine **YOLOv8 Object Detection** (bounding boxes) with **ByteTRACK / DeepSORT** (Kalman Filter + Deep Re-ID embeddings) and **HSV Shadow Suppression**.
