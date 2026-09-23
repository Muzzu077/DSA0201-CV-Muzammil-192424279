# UNIT 5: Real-World Applications, Motion Estimation, & Real-Time Tracking (CO5 - BL3/BL6)

---

## 1. High-Yield Concept Masterclass

### 1.1 Face Recognition: Eigenfaces (PCA Subspace Modeling)

Introduced by Matthew Turk and Alex Pentland (1991), **Eigenfaces** is a foundational 2D appearance-based face recognition method based on Principal Component Analysis (PCA).

```
   TRAINING FACES (M)               MEAN FACE (Psi)                EIGENFACES (U_K)             WEIGHT VECTOR (Omega)
+-----------------------+        +-------------------+          +--------------------+        +---------------------+
| [Face 1]  [Face 2] ...|  --->  | [Average of all]  |  --->    | [u_1]  [u_2] ...   |  --->  | [omega_1, omega_2,  |
| (Each N x N = 4096-D) |        |    M faces        |          | (Top K eigenvectors|        |  ..., omega_K]^T    |
+-----------------------+        +-------------------+          |  e.g., K = 30-50)  |        | (Compressed to 50-D)|
                                                                +--------------------+        +---------------------+
```

#### 1. Mathematical Pipeline & Formulation:
1. **Vectorization:** Convert each $N \times N$ grayscale face image into a $D$-dimensional column vector $\Gamma_i$ (where $D = N^2$, e.g., $64 \times 64 \implies D = 4096$).
2. **Mean Face Computation:**
   $$\Psi = \frac{1}{M} \sum_{i=1}^M \Gamma_i$$
3. **Mean-Subtracted Difference Vectors:**
   $$\Phi_i = \Gamma_i - \Psi$$
   Form data matrix $\mathbf{A} = [\Phi_1, \Phi_2, \dots, \Phi_M]$ of size $D \times M$.
4. **Covariance Matrix:**
   $$\mathbf{C} = \frac{1}{M} \mathbf{A} \mathbf{A}^T \quad (\text{Size } D \times D = 4096 \times 4096 \implies 16.7\text{ million entries!})$$
5. **The Snapshots / Matrix Trick (Sirovich & Kirby):**
   Directly solving eigenvectors of $4096 \times 4096$ matrix $\mathbf{C}$ is computationally impossible on embedded devices. Instead, solve the smaller $M \times M$ surrogate matrix $\mathbf{L}$:
   $$\mathbf{L} = \mathbf{A}^T \mathbf{A} \quad (\text{Size } M \times M, \text{e.g., } 100 \times 100)$$
   Let $v_k$ be the eigenvectors of $\mathbf{L}$ with eigenvalues $\lambda_k$:
   $$\mathbf{A}^T \mathbf{A} v_k = \lambda_k v_k \implies \mathbf{A} \mathbf{A}^T (\mathbf{A} v_k) = \lambda_k (\mathbf{A} v_k)$$
   Therefore, the true eigenvectors (Eigenfaces) $\mathbf{u}_k$ of $\mathbf{C}$ are computed directly as:
   $$\mathbf{u}_k = \frac{\mathbf{A} v_k}{\|\mathbf{A} v_k\|}$$
6. **Feature Projection:**
   Project a test face $\Gamma$ into the low-dimensional eigenspace:
   $$\mathbf{\Omega} = \mathbf{U}_K^T (\Gamma - \Psi) = [\omega_1, \omega_2, \dots, \omega_K]^T$$
7. **Classification & Verification (Two-Threshold Rule):**
   * **Subspace Reconstruction Error ($\epsilon_d$):**
     $$\Phi_{\text{rec}} = \sum_{k=1}^K \omega_k \mathbf{u}_k, \quad \epsilon_d = \|\Phi - \Phi_{\text{rec}}\|$$
     * If $\epsilon_d > \theta_{\text{face}} \implies$ **NOT A FACE** (non-face object/background).
   * **Nearest Neighbor Distance ($\epsilon_k$):**
     $$\epsilon_k = \min_j \|\mathbf{\Omega} - \mathbf{\Omega}_j\|$$
     * If $\epsilon_d \le \theta_{\text{face}}$ and $\epsilon_k \le \theta_{\text{known}} \implies$ **RECOGNIZED KNOWN PERSON**.
     * If $\epsilon_d \le \theta_{\text{face}}$ and $\epsilon_k > \theta_{\text{known}} \implies$ **UNKNOWN FACE** (registered face absent from database).

#### 2. Illumination Subspace Trick:
In raw pixels, lighting variations dominate identity differences. Empirical studies show that the first 3 leading eigenfaces ($\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$) capture $\sim 85\%$ of lighting changes (shadows, highlights). **Discarding the first 3 eigenfaces and projecting onto $\mathbf{u}_4 \dots \mathbf{u}_K$ makes Eigenfaces illumination invariant!**

---

### 1.2 Motion Estimation & Optical Flow

Optical flow is the 2D vector field of apparent pixel velocities $(u, v)$ between consecutive video frames.

```
FRAME t                     FRAME t + dt                    OPTICAL FLOW FIELD
+------------------+        +------------------+            +------------------+
|      [Ball]      |  --->  |           [Ball] |    --->    |      ----->      |
|     (x, y)       |        |     (x+dx, y+dy) |            |     (u, v)       |
+------------------+        +------------------+            +------------------+
```

#### 1. Brightness Constancy Assumption:
The intensity of a physical scene point remains constant across small time intervals:
$$I(x, y, t) = I(x + \delta x, y + \delta y, t + \delta t)$$

#### 2. Derivation of the Optical Flow Constraint Equation (OFCE):
Applying 1st-order Taylor Series expansion:
$$I(x + \delta x, y + \delta y, t + \delta t) \approx I(x, y, t) + \frac{\partial I}{\partial x}\delta x + \frac{\partial I}{\partial y}\delta y + \frac{\partial I}{\partial t}\delta t$$
Subtracting $I(x, y, t)$ and dividing by $\delta t$:
$$\frac{\partial I}{\partial x}\frac{\delta x}{\delta t} + \frac{\partial I}{\partial y}\frac{\delta y}{\delta t} + \frac{\partial I}{\partial t} = 0$$
$$\mathbf{I_x u + I_y v + I_t = 0} \quad \text{or} \quad \nabla I \cdot \mathbf{v} + I_t = 0$$
Where:
* $I_x = \frac{\partial I}{\partial x}$, $I_y = \frac{\partial I}{\partial y}$ are spatial image gradients.
* $I_t = \frac{\partial I}{\partial t}$ is the temporal frame-to-frame derivative.
* $u = \frac{dx}{dt}, v = \frac{dy}{dt}$ are the horizontal and vertical velocity components.

#### 3. The Aperture Problem:
The OFCE is **1 linear equation with 2 unknowns $(u, v)$ at each pixel**. We can only determine velocity **perpendicular to the edge** (normal flow); velocity parallel to the edge is ambiguous when viewed through a small aperture!

```
+-------------------------------------------------------------------------------------------------+
|                                 LUCAS-KANADE vs. HORN-SCHUNCK                                   |
+-------------------+---------------------------------------------+-------------------------------+
| PARAMETER         | LUCAS-KANADE (Local Method)                 | HORN-SCHUNCK (Global Method)  |
+-------------------+---------------------------------------------+-------------------------------+
| Core Principle    | Assumes flow (u, v) is constant in a small  | Minimizes a global energy     |
|                   | local spatial window Omega (e.g., 3x3).     | functional over whole image.  |
+-------------------+---------------------------------------------+-------------------------------+
| System Solved     | Overdetermined system via Least Squares:    | Solves coupled Euler-Lagrange |
|                   | [Sum(Ix^2)   Sum(IxIy)] [u] = - [Sum(IxIt)] | equations iteratively via     |
|                   | [Sum(IxIy)   Sum(Iy^2)] [v]   - [Sum(IyIt)] | Gauss-Seidel relaxation.      |
+-------------------+---------------------------------------------+-------------------------------+
| Solvability       | Solvable ONLY where matrix has two large    | Solvable everywhere because   |
|                   | eigenvalues (i.e. CORNERS / TEXTURED EDGES) | smoothness constraint spreads |
|                   | Fails on flat, textureless regions.         | motion into flat regions.     |
+-------------------+---------------------------------------------+-------------------------------+
| Flow Field Type   | **Sparse Optical Flow**                     | **Dense Optical Flow**        |
+-------------------+---------------------------------------------+-------------------------------+
| Large Displacements| Handled using Coarse-to-Fine Pyramids       | Handled using Pyramids        |
+-------------------+---------------------------------------------+-------------------------------+
```

---

### 1.3 Object Tracking Algorithms

```
+-------------------------------------------------------------------------------------------------+
|                                    OBJECT TRACKING TAXONOMY                                     |
+-------------------+-----------------------------------------------+-----------------------------+
| TRACKER           | CORE PRINCIPLE                                | STRENGTHS & LIMITATIONS     |
+-------------------+-----------------------------------------------+-----------------------------+
| Kalman Filter     | Recursive Bayesian linear Gaussian estimator. | Optimal for Gaussian noise; |
|                   | Predicts state x_k = [x, y, v_x, v_y]^T and   | handles temporary occlusions|
|                   | updates using noisy measurements z_k.         | bridges measurement dropouts|
+-------------------+-----------------------------------------------+-----------------------------+
| Meanshift         | Non-parametric mode-seeking gradient ascent   | Fast; fixed window size     |
|                   | on target color histogram probability map.    | (fails when target scales). |
+-------------------+-----------------------------------------------+-----------------------------+
| CamShift (Continu-| Continuously updates tracking window size and | Invariant to object scaling |
| ously Adaptive MS)| orientation using 0th, 1st, 2nd moments.      | and 2D rotation.            |
+-------------------+-----------------------------------------------+-----------------------------+
| DeepSORT          | Kalman Filter motion prediction + Hungarian   | Current state-of-the-art    |
|                   | algorithm for IoU association + Deep CNN Re-ID| for Multi-Object Tracking   |
|                   | appearance embedding matching.                | (MOT) through long occlusions|
+-------------------+-----------------------------------------------+-----------------------------+
```

#### The Kalman Filter Two-Phase Recursive Cycle:

```
        PREDICT PHASE                                             UPDATE / CORRECT PHASE
+-----------------------------+                             +-----------------------------------+
| 1. State Prediction:        |                             | 1. Compute Kalman Gain:           |
|    x_k^- = F * x_{k-1}      |                             |    K_k = P_k^- * H^T *            |
|                             |                             |          (H * P_k^- * H^T + R)^(-1)   |
| 2. Covariance Prediction:   | --------> (Measurement) ---->                                   |
|    P_k^- = F * P_{k-1} * F^T|             z_k             | 2. Update State Estimate:         |
|            + Q              |                             |    x_k = x_k^- + K_k*(z_k - H*x_k)|
+-----------------------------+                             |                                   |
                                                            | 3. Update Error Covariance:       |
              ^                                             |    P_k = (I - K_k * H) * P_k^-    |
              |                                             +-----------------------------------+
              +---------------------------------------------------------------|
```

---

### 1.4 Autonomous Driving Perception Stack (Lane Extraction & ADAS)

```
RAW CAMERA VIDEO ---> MULTI-SPACE COLOR FILTER ---> DIRECTIONAL SOBEL X ---> INVERSE PERSPECTIVE MAPPING (IPM)
                      (HLS L + LAB B Fusion)         (Suppresses horizontal)  (Bird's-Eye Top-Down View)
                                                                                          |
                                                                                          v
SAFETY DASHBOARD <--- KALMAN TRAJECTORY FILTER <--- RADIUS OF CURVATURE & <--- SLIDING WINDOW 2nd-ORDER
(LDW Alert State)     (Temporal Smoothing)           LATERAL OFFSET              POLYNOMIAL FIT (x = Ay^2+By+C)
```

#### 1. Color Space Fusion for Adverse Lighting:
* Standard RGB fails under shadows and glare.
* **HLS L-Channel (Lightness):** Isolates high-contrast **White Lanes**.
* **LAB B-Channel (Blue-Yellow Chromaticity):** Isolates **Yellow Lanes** even under direct afternoon sun glare and tree shadows.

#### 2. Inverse Perspective Mapping (IPM / Bird's Eye View):
Applies a perspective Homography matrix $\mathbf{H}_{\text{IPM}}$ to warp the driver's perspective view into an orthographic top-down ground plane. In IPM space:
* Parallel road lanes become strictly **parallel vertical lines**.
* Perspective convergence to a vanishing point is eliminated.
* Metric pixel-to-meter scaling factors ($y_{\text{m\_per\_pix}}, x_{\text{m\_per\_pix}}$) can be directly applied!

#### 3. Mathematical Road Curvature & Vehicle Lateral Offset:
In real-world meter space, the road boundary is modeled as a 2nd-order parabola:
$$x(y) = A y^2 + B y + C$$
* **Radius of Curvature ($R_{\text{curve}}$) via Differential Geometry:**
  $$R_{\text{curve}} = \frac{\left(1 + \left(\frac{dx}{dy}\right)^2\right)^{3/2}}{\left|\frac{d^2x}{dy^2}\right|} = \frac{\left(1 + (2 A y + B)^2\right)^{3/2}}{|2 A|}$$
  *(Evaluated at the vehicle's front bumper plane $y = y_{\text{bottom}}$)*.
* **Vehicle Lateral Center Offset ($d_{\text{offset}}$):**
  $$x_{\text{lane\_center}} = \frac{x_{\text{left\_base}} + x_{\text{right\_base}}}{2}$$
  $$d_{\text{offset}} = (x_{\text{lane\_center}} - x_{\text{vehicle\_center}}) \times x_{\text{m\_per\_pix}}$$
* **Lane Departure Warning (LDW) State Machine:**
  * $|d_{\text{offset}}| < 0.30\text{ m} \implies$ **SAFE (Green)**
  * $0.30\text{ m} \le |d_{\text{offset}}| \le 0.50\text{ m} \implies$ **CAUTION (Yellow)**
  * $|d_{\text{offset}}| > 0.50\text{ m} \implies$ **CRITICAL DEPARTURE (Red Alert)**

---

## 2. Essential OpenCV Functions Cheat Sheet for Unit 5

```python
import cv2
import numpy as np

# 1. Lucas-Kanade Optical Flow (Sparse with Pyramids)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

# 2. Farneback Dense Optical Flow
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# 3. Kalman Filter Initialization
kf = cv2.KalmanFilter(dynamParams=4, measureParams=2)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32) # State: [x, y, vx, vy]
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32) # Measure: [x, y]
# Predict & Correct
predicted = kf.predict()
corrected = kf.correct(measurement)

# 4. CamShift Tracking
track_window = (x, y, w, h)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
ret, track_window = cv2.CamShift(dst_prob_map, track_window, term_crit)

# 5. Inverse Perspective Mapping (IPM)
H_ipm = cv2.getPerspectiveTransform(src_trapezoid, dst_rectangle)
birdseye = cv2.warpPerspective(frame, H_ipm, (w, h))
```

---

## 3. High-Yield Exam MCQs with Step-by-Step Solutions

### [Bloom's Level 1: Remember]

#### Q1. What is the fundamental Brightness Constancy assumption equation in Optical Flow?
- (A) $I(x, y, t) = I(x, y, 0)$
- (B) $I(x, y, t) = I(x + \delta x, y + \delta y, t + \delta t)$
- (C) $\nabla I = 0$
- (D) $I_x^2 + I_y^2 = I_t^2$
>**Answer:** **(B)**
>**Explanation:** Optical flow assumes that the illumination and reflectance of a moving physical point remain constant between closely spaced consecutive frames: $I(x, y, t) = I(x + \delta x, y + \delta y, t + \delta t)$.

#### Q2. In the Snapshots method for Eigenfaces with $M=100$ face images of size $64 \times 64$ ($D = 4096$), what are the dimensions of the surrogate matrix $\mathbf{L} = \mathbf{A}^T \mathbf{A}$ whose eigenvalues are calculated?
- (A) $4096 \times 4096$
- (B) $64 \times 64$
- (C) $100 \times 100$
- (D) $4096 \times 100$
>**Answer:** **(C)**
>**Explanation:** The data matrix $\mathbf{A}$ has size $D \times M$ ($4096 \times 100$). The surrogate covariance matrix $\mathbf{L} = \mathbf{A}^T \mathbf{A}$ has size $M \times M = \mathbf{100 \times 100}$, making eigenvalue decomposition extremely fast compared to $4096 \times 4096$.

#### Q3. In the Lucas-Kanade optical flow formulation, under what mathematical condition can the local $2 \times 2$ matrix $A^T A = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$ be reliably inverted?
- (A) When both eigenvalues $\lambda_1$ and $\lambda_2$ are zero
- (B) When the matrix has two large positive eigenvalues (i.e. at sharp corners and high-texture regions)
- (C) When the image is completely flat and smooth
- (D) When only vertical edges are present
>**Answer:** **(B)**
>**Explanation:** The matrix $A^T A$ is identical to the Harris Corner structure tensor. It is well-conditioned and invertible only when both eigenvalues $\lambda_1, \lambda_2$ are significantly greater than zero, which occurs at corners. In flat regions, $\lambda_1 \approx \lambda_2 \approx 0$, causing inversion failure.

---

### [Bloom's Level 2: Understand]

#### Q4. Why does the standard Lucas-Kanade optical flow algorithm fail when tracking rapid, large object motions between video frames?
- (A) Lucas-Kanade can only process colored images
- (B) The 1st-order Taylor series approximation assumes infinitesimal displacements ($\delta x, \delta y \to 0$), failing when motion exceeds a few pixels
- (C) Inverting $A^T A$ takes too long
- (D) The camera frame rate is too high
>**Answer:** **(B)**
>**Explanation:** The Optical Flow Constraint Equation is derived from a 1st-order Taylor series truncation. When pixel motion is large ($>2\text{--}3\text{ pixels}$), higher-order non-linear terms dominate. This is resolved in practice using **Coarse-to-Fine Gaussian Pyramids** (`calcOpticalFlowPyrLK`).

#### Q5. In an autonomous vehicle lane departure warning (LDW) system, what is the primary benefit of converting the camera feed using Inverse Perspective Mapping (IPM)?
- (A) It converts color frames to binary format
- (B) It warps the perspective trapezoid into an orthographic top-down bird's-eye view, making parallel road lanes strictly parallel and eliminating perspective vanishing point distortions
- (C) It increases the camera frame rate to 120 FPS
- (D) It acts as a low-pass Gaussian blur filter
>**Answer:** **(B)**
>**Explanation:** In camera view, lane lines converge to a horizon vanishing point due to projective geometry. IPM removes projective foreshortening, mapping the ground plane so lanes are parallel vertical lines, enabling accurate metric polynomial fitting and curvature calculation.

---

### [Bloom's Level 3: Apply - Calculations & Formulations]

#### Q6. A lane detection system fits a 2nd-order polynomial to lane coordinates in meters: $x(y) = 0.0005 y^2 - 0.02 y + 3.5$. What is the instantaneous Radius of Curvature $R$ at the vehicle bumper ($y = 0\text{ meters}$)?
- (A) $500\text{ meters}$
- (B) $1000\text{ meters}$
- (C) $200\text{ meters}$
- (D) $50\text{ meters}$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>Given: $x(y) = A y^2 + B y + C$ with $A = 0.0005$, $B = -0.02$, $C = 3.5$.
>1. 1st derivative: $\frac{dx}{dy} = 2 A y + B$. At $y = 0$: $\frac{dx}{dy} = B = -0.02$.
>2. 2nd derivative: $\frac{d^2x}{dy^2} = 2 A = 2(0.0005) = 0.001$.
>3. Curvature Formula:
>   $$R = \frac{(1 + (dx/dy)^2)^{3/2}}{|d^2x/dy^2|} = \frac{(1 + (-0.02)^2)^{3/2}}{|0.001|} = \frac{(1 + 0.0004)^{3/2}}{0.001} \approx \frac{1.0006}{0.001} \approx \mathbf{1000\text{ meters}}$$

#### Q7. In an autonomous vehicle camera image ($1280 \times 720$ resolution), the vehicle optical centerline is at $x_{\text{vehicle}} = 640\text{ pixels}$. The detected left and right lane bases at the bottom of the image are $x_{\text{left}} = 300\text{ pixels}$ and $x_{\text{right}} = 1020\text{ pixels}$. Given horizontal resolution factor $x_{\text{m\_per\_pix}} = 3.7 / 720\text{ meters/pixel}$ ($\approx 0.00514\text{ m/pixel}$), what is the lateral offset of the vehicle from the lane center?
- (A) $+0.103\text{ meters}$ (Vehicle is $10.3\text{ cm}$ left of center)
- (B) $-0.500\text{ meters}$
- (C) $0.000\text{ meters}$
- (D) $+0.800\text{ meters}$
>**Answer:** **(A)**
>**Calculation / Step-by-Step:**
>1. Compute lane center:
>   $$x_{\text{lane\_center}} = \frac{x_{\text{left}} + x_{\text{right}}}{2} = \frac{300 + 1020}{2} = \frac{1320}{2} = 660\text{ pixels}$$
>2. Pixel difference:
>   $$\Delta x = x_{\text{lane\_center}} - x_{\text{vehicle}} = 660 - 640 = +20\text{ pixels}$$
>3. Metric Offset:
>   $$\text{Offset} = +20\text{ pixels} \times \left(\frac{3.7\text{ m}}{720\text{ px}}\right) = +20 \times 0.005138 \approx \mathbf{+0.103\text{ meters } (10.3\text{ cm})}$$

#### Q8. A linear 1D Kalman filter has prior variance estimate $P_k^- = 4.0$ and measurement noise variance $R = 1.0$. If measurement matrix $H = 1$, what is the Kalman Gain $K_k$?
- (A) $0.20$
- (B) $0.80$
- (C) $0.50$
- (D) $1.00$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>$$K_k = \frac{P_k^- H^T}{H P_k^- H^T + R} = \frac{4.0 \times 1}{(1)(4.0)(1) + 1.0} = \frac{4.0}{5.0} = \mathbf{0.80}$$

---

### [Bloom's Level 4: Analyze]

#### Q9. In an automated face recognition attendance system deployed near an office entrance, recognition accuracy collapses drastically between 8:00 AM (bright sunlight from side window) and 6:00 PM (artificial fluorescent ceiling lighting). Which algorithmic modification to the Eigenfaces pipeline provides the highest resilience against this variation?
- (A) Increasing image resolution from $64 \times 64$ to $1024 \times 1024$
- (B) Preprocessing with Contrast Limited Adaptive Histogram Equalization (CLAHE) and discarding the first 3 leading Eigenfaces ($u_1, u_2, u_3$) during projection
- (C) Using simple thresholding
- (D) Reducing the database size to 1 image
>**Answer:** **(B)**
>**Explanation:** CLAHE normalizes local illumination gradients across facial patches. Furthermore, the first 3 principal components in PCA capture diffuse lighting directions rather than identity. Discarding them eliminates environmental lighting shifts while preserving biometric features.

#### Q10. Why does CamShift outperform standard MeanShift when tracking a human face that approaches and recedes from a webcam?
- (A) MeanShift only works on binary images
- (B) Standard MeanShift uses a fixed-size search window, causing tracking loss when the face grows larger or smaller, whereas CamShift continuously recalculates the 0th spatial moment (area) and 2nd moments to dynamically resize and rotate the tracking bounding box
- (C) CamShift uses deep learning whereas MeanShift does not
- (D) MeanShift cannot calculate histograms
>**Answer:** **(B)**
>**Explanation:** Standard MeanShift assumes a rigid, fixed search window. **CamShift** (Continuously Adaptive Mean Shift) uses the 0th moment to scale the search window proportionally to the target's visual size, and 2nd moments to adjust bounding box orientation.

---

### [Bloom's Level 5: Evaluate]

#### Q11. In an ISO 26262 ASIL-B autonomous driving perception pipeline, the vehicle encounters a section of road where lane markings are temporarily occluded by an overtaking truck for $300\text{ ms}$ (9 frames). Which architectural component is responsible for maintaining smooth lateral trajectory guidance without triggering sudden dangerous steering jerks?
- (A) Canny edge detector
- (B) Recursive Kalman State-Space Filter maintaining momentum and temporal polynomial state propagation during measurement dropouts
- (C) Otsu thresholding
- (D) Bilateral filter
>**Answer:** **(B)**
>**Explanation:** An isolated per-frame vision detector collapses during occlusions. The **discrete Kalman filter** maintains velocity and state covariances, seamlessly propagating the estimated polynomial trajectory ($\hat{x}_k = F \hat{x}_{k-1}$) through measurement gaps.
