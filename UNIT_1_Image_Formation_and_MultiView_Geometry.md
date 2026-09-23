# UNIT 1: Image Formation, Camera Models, & Multi-View Geometry (CO1 - BL2/BL3)

---

## 1. High-Yield Concept Masterclass

### 1.1 Fundamentals: Computer Vision vs. Image Processing vs. Computer Graphics
Understanding the input-output relationships between visual computing domains is a classic exam favorite:

```
                  +-------------------------------------------------+
                  |                   INPUT DATA                    |
                  +-----------------------+-------------------------+
                  |         IMAGE         |     NON-IMAGE / MODEL   |
+--------+--------+-----------------------+-------------------------+
| OUTPUT | IMAGE  |   IMAGE PROCESSING    |    COMPUTER GRAPHICS    |
|  DATA  |        | (Enhancement, Filters)|  (3D Models -> 2D Image)|
+--------+--------+-----------------------+-------------------------+
|        | HIGH-  |    COMPUTER VISION    |     DATA PROCESSING /   |
|        | LEVEL/ | (Images -> Meaning,   |     MACHINE LEARNING    |
|        | SCENE  |  3D Depth, Objects)   |   (Non-visual Analysis) |
+--------+--------+-----------------------+-------------------------+
```

* **Image Processing (Image $\to$ Image):** Takes a 2D image as input and outputs a transformed/enhanced 2D image (e.g., histogram equalization, Gaussian blur, morphological dilation, contrast adjustment).
* **Computer Vision (Image $\to$ Model/Description/Action):** Takes 2D images as input and recovers 3D geometric structure, semantic understanding, object classifications, or control decisions (e.g., detecting tumors in MRI, estimating vehicle distance, face recognition).
* **Computer Graphics (Model $\to$ Image):** Takes abstract mathematical descriptions/3D polygon meshes and renders a 2D visual scene (e.g., Unreal Engine, OpenGL rendering, Ray tracing).
* **Augmented Reality (AR):** Merges **Computer Vision** (tracking real-world camera pose) with **Computer Graphics** (rendering synthetic 3D assets aligned with real objects).

---

### 1.2 The Three Levels of Computer Vision
Vision processing pipelines operate in three distinct hierarchical tiers:

| Level | Input $\to$ Output | Typical Operations / Algorithms | Example Real-World Task |
| :--- | :--- | :--- | :--- |
| **Low-Level Vision** | Pixel Image $\to$ Pixel Image / Feature Map | Noise filtering, contrast enhancement, thresholding, edge detection (Sobel, Canny). No scene understanding. | Grayscale conversion, smoothing sensor noise, edge extraction. |
| **Mid-Level Vision** | Feature Map $\to$ Segmented Entities / Regions / Descriptors | Contour extraction, region segmentation, shape description (Fourier Descriptors, MAT), feature extraction (SIFT, ORB, HOG). | Grouping edge pixels into bounding lines, extracting connected components. |
| **High-Level Vision** | Extracted Features $\to$ Semantic Knowledge / Decisions | Object classification, 3D scene reconstruction, trajectory prediction, visual SLAM, decision making. | Identifying a pedestrian from a crowd, calculating autonomous braking distance. |

---

### 1.3 Physics of Image Formation & Sensor Geometry

#### The Radiometric Model
An image function $I(x,y)$ captured by an optical sensor is the product of two physical components:
$$I(x,y) = i(x,y) \cdot r(x,y)$$
Where:
* $i(x,y) \in (0, \infty)$ is the **Illumination component** (amount of incident light from the source).
* $r(x,y) \in [0, 1]$ is the **Reflectance component** (intrinsic material property of the object surface).
* **Spatial Resolution:** Determined by the number of sampling pixels ($M \times N$).
* **Grayscale / Radiometric Resolution:** Determined by quantization bit-depth $L = 2^k$ (for an 8-bit image, $k=8 \implies 256$ intensity levels from 0 to 255).

---

### 1.4 Pinhole Camera Model & Perspective Projection

#### The Geometry of Projection
Let the Camera Optical Center (Pin-hole) be at the origin $(0, 0, 0)$, looking down the optical Z-axis (principal axis). The image projection plane is placed at focal distance $f$ from the optical center.

```
       3D Scene Point (X, Y, Z)
              *
             / \
            /   \
           /     \
    ------/-------* [Projected Image Point (x, y)]
         /   |
        /    | f (focal length)
       /     |
      O------*--- Optical Axis (Z)
   Camera Center (0, 0, 0)
```

By similar triangles:
$$\frac{x}{f} = \frac{X}{Z} \implies x = f \frac{X}{Z}$$
$$\frac{y}{f} = \frac{Y}{Z} \implies y = f \frac{Y}{Z}$$

#### Critical Mathematical Insights:
1. **Non-Linearity:** Perspective projection is inherently non-linear in Euclidean Cartesian coordinates due to division by depth $Z$.
2. **Loss of Depth:** Any point on the 3D ray $(kX, kY, kZ)$ projects to the exact same 2D pixel $(x, y)$. A single camera loses absolute depth scale.
3. **Horizon & Vanishing Points:** Parallel 3D lines that are not parallel to the camera plane converge at a single **Vanishing Point** in the image plane.

---

### 1.5 Homogeneous Coordinates & Projective Transformations

To make perspective projection a **linear matrix multiplication**, we convert standard Cartesian coordinates to Projective/Homogeneous coordinates by adding an extra dimension:
$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix}_{\text{Cartesian}} \longrightarrow \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}_{\text{Homogeneous}}$$

Converting a 2D homogeneous vector $[x_w, y_w, w]^T$ back to Euclidean 2D coordinates:
$$x = \frac{x_w}{w}, \quad y = \frac{y_w}{w} \quad (\text{for } w \neq 0)$$
*(If $w=0$, the point represents an ideal point at infinity / directional vector).*

---

### 1.6 Camera Calibration: Intrinsic & Extrinsic Parameters

A full camera mapping from a 3D World Coordinate system to 2D Pixel Image Coordinates $(u, v)$ is expressed as:
$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{P} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Where:
* $\mathbf{P}$ is the **Full Camera Projection Matrix** ($3 \times 4$, rank 3, **11 Degrees of Freedom** up to scale).
* $\mathbf{K}$ is the **Intrinsic Calibration Matrix** ($3 \times 3$ upper-triangular, **5 internal parameters**).
* $[\mathbf{R} \mid \mathbf{t}]$ is the **Extrinsic Matrix** ($3 \times 4$, **6 external parameters**: 3 for Rotation $\mathbf{R}$, 3 for Translation $\mathbf{t}$).

```
       +----------------------- INTRINSIC MATRIX K (5 DOF) -----------------------+
       |                                                                          |
       |  [ f_x    s    c_x ]       f_x = f * m_x (focal length in horizontal pixels) |
       |  [  0    f_y   c_y ]  where f_y = f * m_y (focal length in vertical pixels)   |
       |  [  0     0     1  ]       c_x, c_y   = Principal Point (optical center) |
       |                            s          = Skew coefficient (often 0)       |
       +--------------------------------------------------------------------------+
```

```
       +---------------------- EXTRINSIC MATRIX [R | t] (6 DOF) -------------------+
       |                                                                          |
       |  [ r11  r12  r13 | t_x ]   R = 3x3 Orthogonal Rotation Matrix             |
       |  [ r21  r22  r23 | t_y ]       (R^T * R = I, det(R) = +1 -> 3 Euler angles)|
       |  [ r31  r32  r33 | t_z ]   t = 3x1 Translation Vector (3 position coords)|
       +--------------------------------------------------------------------------+
```

#### Lens Distortion Models
Real-world optical lenses deviate from the ideal pinhole model, creating distortions:
1. **Radial Distortion (Barrel & Pincushion):** Rays bend more near lens edges than at the optical center.
   $$x_{\text{corrected}} = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
   $$y_{\text{corrected}} = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
   where $r^2 = x^2 + y^2$. ($k_1, k_2, k_3$ are radial distortion coefficients).
   * $k_1 < 0 \implies$ **Barrel Distortion** (wide-angle / fisheye lenses, lines bulge outwards).
   * $k_1 > 0 \implies$ **Pincushion Distortion** (telephoto lenses, lines bow inwards).
2. **Tangential Distortion:** Occurs when the lens sensor assembly is not perfectly parallel to the image sensor plane ($p_1, p_2$ coefficients).

---

### 1.7 Epipolar Geometry & Binocular / Stereo Vision

When two cameras view the same 3D point $\mathbf{X}$, multi-view geometric constraints simplify correspondence search:

```
        3D World Point X
             *
            / \
           /   \
          /     \
  Image 1/       \ Image 2
   [x1] *         * [x2]  (x2 lies on Epipolar Line l2)
       /  \     /  \
      /    \   /    \
     /      \ /      \
   C1 *------*--------* C2 (Baseline B)
    (e1)             (e2)
  Epipole 1        Epipole 2
```

#### Key Geometric Definitions:
* **Baseline ($B$):** The 3D distance between camera optical centers $C_1$ and $C_2$.
* **Epipolar Plane:** The 2D plane formed by the 3D world point $\mathbf{X}$ and the two optical centers $C_1, C_2$.
* **Epipoles ($e_1, e_2$):** The point where the baseline intersects each image plane (the projection of camera center $C_1$ onto image 2, and vice versa).
* **Epipolar Line:** The intersection of the epipolar plane with the image plane. **The match for any point in Image 1 MUST lie on the corresponding epipolar line in Image 2 (reduces a 2D search to a 1D search line!).**

#### Essential Matrix ($\mathbf{E}$) vs. Fundamental Matrix ($\mathbf{F}$)

| Characteristic | Essential Matrix ($\mathbf{E}$) | Fundamental Matrix ($\mathbf{F}$) |
| :--- | :--- | :--- |
| **Coordinate Space** | Calibrated / Normalized Camera Coordinates ($\hat{x}$) | Raw Uncalibrated Pixel Coordinates ($p$) |
| **Relationship** | $\hat{x}_2^T \mathbf{E} \hat{x}_1 = 0$ | $p_2^T \mathbf{F} p_1 = 0$ |
| **Dependence** | Depends **ONLY** on Extrinsics $[\mathbf{R} \mid \mathbf{t}]$: $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ | Depends on Intrinsics $\mathbf{K}_1, \mathbf{K}_2$ AND Extrinsics: $\mathbf{F} = \mathbf{K}_2^{-T} \mathbf{E} \mathbf{K}_1^{-1}$ |
| **Matrix Size & Rank** | $3 \times 3$, **Rank = 2** | $3 \times 3$, **Rank = 2** |
| **Degrees of Freedom** | **5 DOF** (3 Rotation + 2 Translation direction up to scale) | **7 DOF** (9 elements - 1 scale - 1 rank constraint $\det(\mathbf{F})=0$) |
| **Singular Values** | Two equal non-zero singular values: $(\sigma, \sigma, 0)$ | Two arbitrary non-zero singular values: $(\sigma_1, \sigma_2, 0)$ |
| **Estimation Algorithm** | 5-point algorithm (Nistér) | 8-point algorithm (Longuet-Higgins) / 7-point algorithm |

---

### 1.8 Stereo Disparity & Depth Estimation Formula

In a rectified stereo setup (parallel optical axes separated by baseline $B$, identical focal length $f$):

```
                       3D Point (X, Y, Z)
                              *
                             / \
                            /   \
                           /  |  \
                          /   |   \  Z (Depth to recover)
                         /    |    \
   Image Left Plane     /     |     \    Image Right Plane
     [--*--|-----]     /      |      \     [-----|--*--]
       x_L   Optical  /       |       \           x_R   Optical
             Axis    *--------+--------*                Axis
                    C_Left    |      C_Right
                     <------- B -------> (Baseline)
```

1. **Disparity ($d$):** The difference in horizontal pixel coordinates between corresponding points:
   $$d = x_L - x_R$$
2. **Depth ($Z$) Calculation:**
   $$Z = \frac{f \cdot B}{d}$$
   Where:
   * $Z$ is perpendicular depth (in meters/millimeters).
   * $f$ is focal length (in pixels or mm).
   * $B$ is stereo baseline distance between cameras (in meters/mm).
   * $d$ is disparity (in pixels or mm).

#### Critical Exam Properties:
* **Inverse Relationship:** Depth $Z$ is inversely proportional to disparity $d$ ($Z \propto \frac{1}{d}$).
* **Close objects:** Have **LARGE disparity** (huge pixel shift between left and right views).
* **Far objects:** Have **SMALL / NEAR-ZERO disparity** ($d \to 0 \implies Z \to \infty$).
* If disparity $d = 0$, the depth is at optical infinity.

---

### 1.9 Multi-View 3D Reconstruction Pipeline (Drone/Aerial Mapping)

The complete end-to-end photogrammetry pipeline converts multiple 2D overlapping drone photos into metric 3D digital twins:

```
+--------------------------+
| 1. Image Acquisition     | Drone flies with 70-80% forward/side overlap
+------------+-------------+
             |
+------------v-------------+
| 2. Feature Extraction    | Extract SIFT / ORB invariant keypoints & descriptors
+------------+-------------+
             |
+------------v-------------+
| 3. Feature Matching      | Match descriptors using FLANN / Ratio Test (Lowe's < 0.75)
+------------+-------------+
             |
+------------v-------------+
| 4. Outlier Rejection     | RANSAC with Fundamental Matrix (F) / Homography (H)
+------------+-------------+
             |
+------------v-------------+
| 5. Structure from Motion | Solve Bundle Adjustment (SBA) -> Camera Poses + Sparse Point Cloud
+------------+-------------+
             |
+------------v-------------+
| 6. Dense Stereo Matching | Multi-View Stereo (MVS) -> Dense 3D Point Cloud
+------------+-------------+
             |
+------------v-------------+
| 7. Meshing & Texturing   | Delaunay Triangulation / Poisson Reconstruction -> 3D Mesh
+--------------------------+
```

---

## 2. Essential OpenCV Functions Cheat Sheet for Unit 1

```python
import cv2
import numpy as np

# 1. Camera Calibration
ret, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=obj_points, # 3D real-world chessboard corners (Z=0)
    imagePoints=img_points,   # 2D detected pixel corners
    imageSize=(w, h),
    cameraMatrix=None,
    distCoeffs=None
)

# 2. Undistort an Image
undistorted_img = cv2.undistort(img, K, dist_coeff)

# 3. Find Fundamental Matrix (with RANSAC outlier rejection)
F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0)

# 4. Find Essential Matrix (requires intrinsic matrix K)
E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC)

# 5. Recover Relative Camera Pose (Rotation R, Translation t from Essential Matrix)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)

# 6. Stereo Disparity Map Computation (Semi-Global Block Matching)
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*5, blockSize=5)
disparity = stereo.compute(gray_left, gray_right) # Disparity = disp_map / 16.0
```

---

## 3. High-Yield Exam MCQs with Step-by-Step Solutions

### [Bloom's Level 1: Remember]

#### Q1. What is the rank and degrees of freedom (DOF) of the Fundamental Matrix ($F$)?
- (A) Rank 3, 9 DOF
- (B) Rank 2, 7 DOF
- (C) Rank 2, 5 DOF
- (D) Rank 3, 8 DOF
>**Answer:** **(B)**
>**Explanation:** The Fundamental Matrix $F$ is a $3 \times 3$ singular matrix with **Rank 2** (since $\det(F)=0$). It has 9 raw entries, but loses 1 DOF due to arbitrary global scale and 1 DOF due to the rank constraint ($\det(F)=0$), leaving **7 Degrees of Freedom**.

#### Q2. Which matrix maps normalized/calibrated camera coordinates between two views?
- (A) Homography Matrix
- (B) Intrinsic Matrix
- (C) Essential Matrix
- (D) Fundamental Matrix
>**Answer:** **(C)**
>**Explanation:** The Essential Matrix ($E = [t]_\times R$) relates points in normalized/calibrated camera space ($\hat{x}_2^T E \hat{x}_1 = 0$), while the Fundamental Matrix relates raw uncalibrated pixel coordinates ($p_2^T F p_1 = 0$).

#### Q3. In an optical camera, which type of lens distortion causes straight lines near image borders to bow outward like a barrel?
- (A) Tangential distortion
- (B) Chromatic aberration
- (C) Negative radial distortion ($k_1 < 0$)
- (D) Positive radial distortion ($k_1 > 0$)
>**Answer:** **(C)**
>**Explanation:** Radial distortion with negative coefficient ($k_1 < 0$) causes magnification to decrease with distance from optical center, resulting in **Barrel Distortion** (common in wide-angle/fisheye lenses). Positive $k_1 > 0$ causes Pincushion distortion.

#### Q4. The point where the baseline connecting two camera centers intersects the image plane is defined as the:
- (A) Principal point
- (B) Vanishing point
- (C) Focal point
- (D) Epipole
>**Answer:** **(D)**
>**Explanation:** The intersection of the baseline with an image plane is the **Epipole** ($e_1, e_2$). It represents the projection of the other camera's optical center.

---

### [Bloom's Level 2: Understand]

#### Q5. A computer vision system classifies medical X-rays into normal vs. pneumonia. Which level of computer vision does this belong to?
- (A) Low-level vision
- (B) Mid-level vision
- (C) High-level vision
- (D) Computer graphics
>**Answer:** **(C)**
>**Explanation:** High-level vision involves semantic interpretation, cognitive reasoning, diagnostic decision-making, and object/condition classification from visual features.

#### Q6. Why can a single stationary monocular pinhole camera NOT measure the absolute metric scale/distance of an unknown object?
- (A) The lens has spherical distortion
- (B) Perspective projection causes an inherent depth-scale ambiguity where $(kX, kY, kZ)$ projects to the same $(x, y)$
- (C) Monocular cameras only capture grayscale luminance
- (D) The intrinsic matrix $K$ cannot be inverted
>**Answer:** **(B)**
>**Explanation:** In perspective projection $x = f \frac{X}{Z}$ and $y = f \frac{Y}{Z}$, scaling all 3D coordinates $(X, Y, Z)$ by any constant factor $k$ yields identical pixel coordinates $(x, y)$. Hence, a small object close to the lens looks identical to a large object far away.

#### Q7. In epipolar geometry, what does the epipolar line $l_2 = F p_1$ physically represent in the second image?
- (A) The trajectory of the camera motion
- (B) The projection of the entire 3D viewing ray passing through $p_1$ and optical center $C_1$ onto Image 2
- (C) The boundary of the object
- (D) The horizon line
>**Answer:** **(B)**
>**Explanation:** The point $p_1$ in image 1 corresponds to a 3D ray extending from camera center $C_1$ to infinity. The projection of this continuous 3D line onto the image plane of Camera 2 forms the 2D **Epipolar Line** $l_2$.

---

### [Bloom's Level 3: Apply - Numerical Problems]

#### Q8. A stereo camera system has a focal length $f = 800\text{ pixels}$ and a baseline $B = 0.2\text{ meters}$ ($20\text{ cm}$). A vehicle detected in the stereo pair has a disparity of $d = 16\text{ pixels}$. What is the depth $Z$ of the vehicle from the camera plane?
- (A) $5.0\text{ meters}$
- (B) $10.0\text{ meters}$
- (C) $2.5\text{ meters}$
- (D) $20.0\text{ meters}$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>Using the stereo triangulation formula:
>$$Z = \frac{f \cdot B}{d}$$
>Given: $f = 800\text{ px}$, $B = 0.2\text{ m}$, $d = 16\text{ px}$.
>$$Z = \frac{800 \cdot 0.2}{16} = \frac{160}{16} = 10.0\text{ meters}$$

#### Q9. For the stereo camera in Q8 ($f=800\text{ px}$, $B=0.2\text{ m}$), if an obstacle approaches and its depth reduces from $10\text{ m}$ to $4\text{ m}$, what is the new disparity?
- (A) $8\text{ pixels}$
- (B) $20\text{ pixels}$
- (C) $40\text{ pixels}$
- (D) $64\text{ pixels}$
>**Answer:** **(C)**
>**Calculation / Step-by-Step:**
>$$d = \frac{f \cdot B}{Z} = \frac{800 \cdot 0.2}{4} = \frac{160}{4} = 40\text{ pixels}$$
>(Notice that as the object gets closer, disparity increases proportionally!).

#### Q10. A 2D point is given in homogeneous coordinates as $P_h = [150, 300, 3]^T$. What are its Euclidean Cartesian coordinates $(x, y)$?
- (A) $(150, 300)$
- (B) $(50, 100)$
- (C) $(450, 900)$
- (D) $(50, 300)$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>To convert from homogeneous $[x_w, y_w, w]^T$ to Cartesian $(x, y)$:
>$$x = \frac{x_w}{w} = \frac{150}{3} = 50$$
>$$y = \frac{y_w}{w} = \frac{300}{3} = 100$$
>So $(x, y) = (50, 100)$.

#### Q11. A pinhole camera has a physical focal length of $f = 12\text{ mm}$. A sensor pixel has dimensions $s_x = 0.006\text{ mm/pixel}$. What is the focal length $f_x$ in pixel units in the intrinsic calibration matrix $K$?
- (A) $72\text{ pixels}$
- (B) $200\text{ pixels}$
- (C) $2000\text{ pixels}$
- (D) $1200\text{ pixels}$
>**Answer:** **(C)**
>**Calculation / Step-by-Step:**
>$$f_x = \frac{f}{s_x} = \frac{12\text{ mm}}{0.006\text{ mm/pixel}} = 2000\text{ pixels}$$

---

### [Bloom's Level 4: Analyze]

#### Q12. In a stereo vision system, what happens to depth estimation error ($\Delta Z$) as the true object depth $Z$ increases?
- (A) Depth error remains constant
- (B) Depth error grows linearly with depth ($\Delta Z \propto Z$)
- (C) Depth error grows quadratically with depth ($\Delta Z \propto Z^2$)
- (D) Depth error drops to zero
>**Answer:** **(C)**
>**Explanation:**
>Differentiating $Z = \frac{f B}{d}$ with respect to disparity $d$:
>$$\left|\frac{\partial Z}{\partial d}\right| = \frac{f B}{d^2} = \frac{Z^2}{f B}$$
>Thus, depth measurement error is proportional to the **square of the distance** ($\Delta Z \approx \frac{Z^2}{f B} \Delta d$). At large distances, stereo depth resolution degrades drastically!

#### Q13. An engineer applies the 8-point algorithm on unnormalized pixel coordinates with values around $(1500, 2000)$ and finds the calculated Fundamental Matrix is wildly inaccurate. What is the fundamental cause (Hartley's normalization insight)?
- (A) The Fundamental matrix requires at least 50 points
- (B) The data matrix has gigantic condition numbers because $x^2 \approx 10^6$ while the constant term is $1$, leading to numerical instability in SVD
- (C) The images must be captured by infrared cameras
- (D) Epipolar lines must always be vertical
>**Answer:** **(B)**
>**Explanation:** In the unnormalized 8-point algorithm, terms like $x x'$ reach $10^6$ while scale terms are $1$. The resulting linear system is extremely ill-conditioned. **Hartley's Normalized 8-Point Algorithm** translates the points to centroid $(0,0)$ and scales them so the average distance is $\sqrt{2}$, ensuring numerical stability.

---

### [Bloom's Level 5: Evaluate]

#### Q14. Compare Essential Matrix $E$ and Fundamental Matrix $F$. If camera intrinsics $K_1$ and $K_2$ are UNKNOWN, which of the following statements is strictly TRUE?
- (A) We can compute $E$ directly using the 5-point algorithm
- (B) We cannot compute $E$, but we CAN still estimate $F$ directly from pixel correspondences
- (C) Neither $E$ nor $F$ can be computed without prior calibration
- (D) Depth $Z$ can be computed without knowing $K$ or baseline $B$
>**Answer:** **(B)**
>**Explanation:** $F$ operates directly on raw uncalibrated pixel coordinates ($p_2^T F p_1 = 0$) and can be solved using matching pixels alone. In contrast, $E = K_2^T F K_1$ requires knowing the intrinsic matrices $K_1, K_2$.

#### Q15. When deploying a real-time drone photogrammetry 3D reconstruction system, what is the primary benefit of performing Stereo Rectification as a preprocessing step?
- (A) It converts color images to grayscale
- (B) It aligns epipolar lines horizontally to the same scanline ($y_1 = y_2$), transforming the 2D correspondence search problem into a 1D horizontal search along single rows
- (C) It increases the drone flight speed
- (D) It eliminates all radial lens distortions completely without calibration
>**Answer:** **(B)**
>**Explanation:** Rectification warps images so that epipolar lines become collinear and parallel to image rows. Corresponding pixels are guaranteed to have the same y-coordinate ($y_L = y_R$), reducing computational search complexity from $O(W \times H)$ to $O(W)$.
