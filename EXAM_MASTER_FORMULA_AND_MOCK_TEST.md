# EXAM MASTER FORMULA SHEET & 180-QUESTION RAPID FIRE MCQ DRILL
## Course: Computer Vision with OpenCV (DSA02 / DSA0201)
### Target: SIMATS End-Semester / Assessment Examination

---

# SECTION A: MASTER FORMULA & KEYWORD CHEAT SHEET

```
+====================================================================================================+
|                                    UNIT 1: CAMERA & MULTI-VIEW                                     |
+====================================================================================================+
| 1. Radiometry:              I(x, y) = i(x, y) * r(x, y)  [Illumination * Reflectance]              |
| 2. Pinhole Perspective:     x = f * (X / Z),   y = f * (Y / Z)                                     |
| 3. Homogeneous 2D to 3D:    (x, y) -> [x, y, 1]^T  ;  [x_w, y_w, w]^T -> (x_w/w, y_w/w)            |
| 4. Intrinsic Matrix K:      [ [fx, s, cx], [0, fy, cy], [0, 0, 1] ]  (5 DOF, 3x3 Upper Tri)         |
| 5. Extrinsics [R | t]:      R = 3x3 Orthogonal (3 DOF), t = 3x1 Vector (3 DOF) -> Total: 6 DOF    |
| 6. Projection Matrix P:     P = K * [R | t]  (3x4 Matrix, 11 DOF up to scale)                      |
| 7. Essential Matrix E:      E = [t]_x * R  (3x3, Rank 2, 5 DOF, Singular values: (sigma, sigma, 0)|
|                             Relates calibrated rays: x_hat_2^T * E * x_hat_1 = 0                   |
| 8. Fundamental Matrix F:    F = K_2^(-T) * E * K_1^(-1)  (3x3, Rank 2, 7 DOF)                     |
|                             Relates raw pixel coordinates: p_2^T * F * p_1 = 0                     |
| 9. Stereo Disparity (d):    d = x_L - x_R                                                          |
| 10. Stereo Depth (Z):       Z = (f * B) / d   ==> Disparity is INVERSELY proportional to Depth!    |
| 11. Depth Error (Delta Z):  Delta Z = (Z^2 / (f * B)) * Delta d  ==> Error grows with Z^2!         |
+====================================================================================================+
|                                  UNIT 2: TRANSFORMATIONS & FILTERING                               |
+====================================================================================================+
| 1. Grayscale Luminance:     Y = 0.299*R + 0.587*G + 0.114*B                                        |
| 2. Alpha Blending:          I_out = alpha*I1 + beta*I2 + gamma  (beta = 1 - alpha)                 |
| 3. Transformation DOFs:     Translation: 2 DOF (1 pt)  |  Euclidean/Rigid: 3 DOF (2 pts)           |
|                             Similarity:  4 DOF (2 pts) |  Affine:          6 DOF (3 pts)           |
|                             Perspective/Homography:    8 DOF (4 pts)                               |
| 4. Gaussian 2D Function:    G(x,y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2)/(2*sigma^2))          |
| 5. Sobel Kernels (3x3):     G_x = [[-1,0,1],[-2,0,2],[-1,0,1]] ; G_y = [[-1,-2,-1],[0,0,0],[1,2,1]]|
| 6. Laplacian Kernel:        Delta^2 f = [[0,1,0],[1,-4,1],[0,1,0]] (Negative Center)               |
| 7. Unsharp / High-Boost:    f_sharp = f + k * (f - f_smooth)  (k=1: Unsharp, k>1: High-Boost)      |
| 8. Canny 5-Step Pipeline:   (1) Gauss Smooth -> (2) Sobel Grad -> (3) NMS (1-px thin) ->          |
|                             (4) Double Threshold (T_hi, T_lo) -> (5) Hysteresis 8-Connectivity    |
| 9. Morphological Open:      A o B = (A (-) B) (+) B  [Erosion then Dilation -> removes specks]     |
| 10. Morphological Close:    A * B = (A (+) B) (-) B  [Dilation then Erosion -> fills cracks/gaps]  |
| 11. Morph Gradient:         (A (+) B) - (A (-) B)  [Dilation minus Erosion -> 1-pixel boundary]    |
| 12. Top-Hat / Black-Hat:    Top-Hat: A - (A o B) (Bright);  Black-Hat: (A * B) - A (Dark)          |
+====================================================================================================+
|                                    UNIT 3: SHAPE & MULTI-RESOLUTION                                 |
+====================================================================================================+
| 1. Complex Boundary:        s(k) = x(k) + j*y(k)   for k = 0, 1, ..., K-1                          |
| 2. Fourier Descriptors:     a(n) = (1/K) * Sum s(k)*exp(-j*2*pi*n*k/K)                             |
| 3. FD Invariance:           Translation: a(0) = 0 ; Scale: Divide by |a(1)| ;                      |
|                             Rotation/Start: Take Magnitudes |a_norm(n)|                            |
| 4. Euler Characteristic:    chi = C - H  (Connected Components minus Holes)                        |
|                             Polyhedron: chi = V - E + F = 2 - 2g  (g = Genus/through-holes)        |
| 5. Shape Compactness:       P^2 / A   (Minimum for Circle = 4*pi) ; Circularity = 4*pi*A / P^2     |
| 6. Solidity / Extent:       Solidity = Area / ConvexHullArea ; Extent = Area / BoundingBoxArea     |
| 7. Snake Total Energy:      E_snake = Integral [ E_int + E_ext + E_con ] ds                        |
| 8. Snake Internal Energy:   E_int = 1/2 [ alpha*|v'(s)|^2 + beta*|v''(s)|^2 ]                      |
|                             alpha = Elasticity/Length ; beta = Bending Stiffness/Rigidity          |
| 9. Snake External Energy:   E_ext = - |Grad(G_sigma * I)|^2  (Attracts to image edges)             |
| 10. Wavelet vs Fourier:     Fourier: Global Sinusoids, O(K log K), No spatial localization         |
|                             Wavelets: Compact Wavelets, O(K) linear, Spatial-Frequency Localized   |
+====================================================================================================+
|                                   UNIT 4: RECOGNITION & DEEP LEARNING                              |
+====================================================================================================+
| 1. SIFT Descriptor:         16x16 patch -> 4x4 subregions -> 8-bin histograms = 128-D vector       |
| 2. Integral Image Sum:      Sum(Rect) = D - B - C + A  (Computed in O(1) constant time)            |
| 3. Polar Hough Line:        rho = x*cos(theta) + y*sin(theta)  (Solves vertical line singularity)  |
| 4. CNN Output Dimension:    O = floor((W - K + 2*P) / S) + 1                                       |
| 5. CNN Parameter Count:     Params = (K * K * C_in + 1) * C_out                                    |
| 6. Precision / Recall:      Precision = TP / (TP + FP)  ;  Recall = TP / (TP + FN)                 |
| 7. F1-Score:                F1 = 2 * (Precision * Recall) / (Precision + Recall)                   |
| 8. IoU (Jaccard Index):     IoU = Area of Overlap / Area of Union                                  |
| 9. GAN Minimax Game:        min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]                          |
+====================================================================================================+
|                                    UNIT 5: APPLICATIONS & TRACKING                                 |
+====================================================================================================+
| 1. Eigenfaces Snapshots:    L = A^T * A  (M x M surrogate matrix solves D x D covariance)          |
| 2. Subspace Projection:     Omega = U_K^T * (Gamma - Psi)                                          |
| 3. Illumination Subspace:   Discard first 3 Eigenfaces (u1, u2, u3) to remove 85% lighting changes  |
| 4. Optical Flow Equation:   I_x * u + I_y * v + I_t = 0                                            |
| 5. Aperture Problem:        1 equation, 2 unknowns -> Only normal flow perpendicular to edge fixed |
| 6. Lucas-Kanade Flow:       Solves [Sum(Ix^2) Sum(IxIy); Sum(IxIy) Sum(Iy^2)] * [u; v] = -[IxIt; IyIt]
|                             Solvable ONLY at corners/textured edges with 2 large eigenvalues!      |
| 7. Kalman Filter Loop:      Predict: x^- = F*x, P^- = F*P*F^T + Q                                  |
|                             Update:  K = P^-*H^T*(H*P^-*H^T + R)^(-1), x = x^- + K*(z - H*x^-)     |
| 8. Parabolic Road Model:    x(y) = A*y^2 + B*y + C                                                 |
| 9. Radius of Curvature:     R_curve = (1 + (2*A*y + B)^2)^(3/2) / |2*A|                            |
| 10. Lateral Vehicle Offset: Offset = (x_lane_center - x_vehicle_center) * x_m_per_pix              |
| 11. LDW Safety States:      SAFE (< 0.30m), CAUTION (0.30 - 0.50m), CRITICAL (> 0.50m)             |
+====================================================================================================+
```

---

# SECTION B: 180-QUESTION EXAM DRILL (BLOOM'S TAXONOMY)

---

### UNIT 1: IMAGE FORMATION & MULTI-VIEW GEOMETRY (CO1)

#### Q1. [BL1] What is the rank of a $3 \times 3$ Fundamental Matrix $F$?
- (A) 1
- (B) 2
- (C) 3
- (D) 0
>**Answer: (B)** | The Fundamental Matrix is singular with $\det(F)=0$, giving it a strict **Rank of 2**.

#### Q2. [BL1] How many independent parameters (Degrees of Freedom) are required to define the Camera Extrinsic Matrix?
- (A) 3
- (B) 5
- (C) 6
- (D) 11
>**Answer: (C)** | Extrinsics consist of 3 Rotation Euler angles and 3 Translation coordinates = **6 DOF**.

#### Q3. [BL2] Which of the following statements correctly differentiates between the Essential Matrix $E$ and Fundamental Matrix $F$?
- (A) $E$ operates on raw pixels, $F$ on calibrated rays
- (B) $E$ operates on normalized/calibrated coordinates, while $F$ operates on uncalibrated pixel coordinates
- (C) $E$ has 7 DOF, $F$ has 5 DOF
- (D) $E$ is $4 \times 4$, $F$ is $3 \times 3$
>**Answer: (B)** | $\hat{x}_2^T E \hat{x}_1 = 0$ (calibrated), $p_2^T F p_1 = 0$ (uncalibrated).

#### Q4. [BL3] A stereo rig has focal length $f = 1000\text{ px}$ and baseline $B = 0.5\text{ m}$. If a detected pedestrian has disparity $d = 25\text{ px}$, what is the distance $Z$?
- (A) $10\text{ m}$
- (B) $20\text{ m}$
- (C) $50\text{ m}$
- (D) $12.5\text{ m}$
>**Answer: (B)** | $Z = \frac{f \cdot B}{d} = \frac{1000 \times 0.5}{25} = \frac{500}{25} = 20\text{ meters}$.

#### Q5. [BL3] In homogeneous coordinates, the point $[300, 600, 2]^T$ corresponds to which 2D Cartesian coordinate?
- (A) $(300, 600)$
- (B) $(150, 300)$
- (C) $(600, 1200)$
- (D) $(150, 600)$
>**Answer: (B)** | $x = 300/2 = 150$, $y = 600/2 = 300$.

#### Q6. [BL4] Why does stereo depth error increase quadratically with object distance ($\Delta Z \propto Z^2$)?
- (A) Because focal length changes with distance
- (B) Because depth is inversely proportional to disparity ($Z = fB/d$), so derivative $\left|\frac{dZ}{dd}\right| = \frac{fB}{d^2} = \frac{Z^2}{fB}$
- (C) Because lenses absorb light quadratically
- (D) Because digital sensors are square
>**Answer: (B)** | Differentiating $Z$ with respect to $d$ yields quadratic error growth $\Delta Z \approx \frac{Z^2}{fB} \Delta d$.

#### Q7. [BL5] In visual SLAM, what is the primary purpose of Epipolar Rectification?
- (A) To convert images to black and white
- (B) To align epipolar lines horizontally to the same scanline, reducing 2D feature matching to a 1D horizontal search
- (C) To increase sensor sensitivity
- (D) To eliminate lens distortion without calibration
>**Answer: (B)** | Rectification restricts matching search along row scanlines ($y_L = y_R$).

---

### UNIT 2: IMAGE PROCESSING & TRANSFORMATIONS (CO2)

#### Q8. [BL1] What is the degrees of freedom of a 2D Affine Transformation?
- (A) 3
- (B) 4
- (C) 6
- (D) 8
>**Answer: (C)** | Affine has a $2 \times 3$ matrix with **6 DOF** (preserves parallelism).

#### Q9. [BL1] Which filter is non-linear and optimal for removing impulse / salt-and-pepper noise?
- (A) Gaussian Filter
- (B) Mean Filter
- (C) Median Filter
- (D) Box Filter
>**Answer: (C)** | The Median filter replaces pixels with rank medians, rejecting extreme 0/255 spikes.

#### Q10. [BL2] What is the mathematical definition of Morphological Closing?
- (A) Erosion followed by Dilation: $(A \ominus B) \oplus B$
- (B) Dilation followed by Erosion: $(A \oplus B) \ominus B$
- (C) Dilation minus Erosion
- (D) Original minus Opening
>**Answer: (B)** | Closing is **Dilation followed by Erosion** ($A \bullet B = (A \oplus B) \ominus B$).

#### Q11. [BL3] An image patch has row pixel values $[20, 20, 100]$. What is the horizontal Sobel response using kernel $[-1, 0, 1]$ with row weight 2?
- (A) $0$
- (B) $80$
- (C) $160$
- (D) $200$
>**Answer: (C)** | $2 \times [(-1 \times 20) + (0 \times 20) + (1 \times 100)] = 2 \times 80 = 160$.

#### Q12. [BL3] In an unsharp masking operation, $f(x,y) = 180$, $f_{\text{smooth}}(x,y) = 140$, and boost factor $k = 2.0$. What is the output intensity?
- (A) $220$
- (B) $260$
- (C) $200$
- (D) $255$
>**Answer: (B)** | $g_{\text{mask}} = 180 - 140 = 40$. $f_{\text{sharp}} = 180 + 2.0(40) = 260$ (clipped to 255 if 8-bit).

#### Q13. [BL4] Why does Canny edge detection include a Non-Maximum Suppression (NMS) stage?
- (A) To remove salt-and-pepper noise
- (B) To suppress gradient values that are not local maxima along the gradient direction, thinning thick edges to 1-pixel ridges
- (C) To invert the color map
- (D) To compute second derivatives
>**Answer: (B)** | NMS thins wide gradient responses into sharp 1-pixel wide edge crests.

#### Q14. [BL5] When extracting dark surface scratches on unevenly illuminated metal plates, which morphological filter is most effective?
- (A) White Top-Hat
- (B) Black-Hat (Bottom-Hat)
- (C) Gaussian Blur
- (D) Dilation
>**Answer: (B)** | **Black-Hat** ($(A \bullet B) - A$) isolates elements darker than local surroundings.

---

### UNIT 3: SHAPE REPRESENTATION & MULTI-RESOLUTION (CO3)

#### Q15. [BL1] What is the Euler Number ($\chi$) of a shape with 2 connected components and 3 holes?
- (A) $+1$
- (B) $-1$
- (C) $+5$
- (D) $-5$
>**Answer: (B)** | $\chi = C - H = 2 - 3 = -1$.

#### Q16. [BL1] In Fourier Descriptors, how is scale invariance achieved?
- (A) Setting $a(0) = 0$
- (B) Dividing all Fourier coefficients by the magnitude of the fundamental harmonic $|a(1)|$
- (C) Taking the angle $\theta$
- (D) Computing 2D DWT
>**Answer: (B)** | Normalizing by $|a(1)|$ makes all harmonic amplitudes independent of scale.

#### Q17. [BL2] In Active Contours (Snakes), what is the function of the elasticity parameter $\alpha$?
- (A) Resists bending and sharp corners
- (B) Penalizes curve length, acting like a contracting rubber band
- (C) Attracts curve to bright pixels
- (D) Inverts the contour
>**Answer: (B)** | $\alpha |v'(s)|^2$ minimizes curve length (tension/elasticity).

#### Q18. [BL3] A binary shape has an Area of $314\text{ pixels}$ and a Perimeter of $62.8\text{ pixels}$. What is its Circularity?
- (A) $0.50$
- (B) $1.00$
- (C) $0.785$
- (D) $1.25$
>**Answer: (B)** | Circularity $= \frac{4\pi A}{P^2} = \frac{4 \times 3.1416 \times 314}{(62.8)^2} = \frac{3945.8}{3943.8} \approx 1.00$ (A perfect circle!).

#### Q19. [BL4] Why are Wavelet Descriptors superior to Fourier Descriptors for local defect detection?
- (A) Wavelets require less RAM
- (B) Wavelets have compact finite support in space and frequency, so a local defect only affects local subband coefficients, unlike global Fourier sinusoids
- (C) Wavelets only work on circles
- (D) Fourier descriptors cannot be inverted
>**Answer: (B)** | Wavelets isolate local spatial features without polluting global shape coefficients.

#### Q20. [BL5] A closed 3D triangular CAD mesh has $V = 500$ vertices and $F = 996$ faces. How many edges $E$ must it possess if it is topologically equivalent to a sphere ($g=0$)?
- (A) $1494$
- (B) $1496$
- (C) $1000$
- (D) $1500$
>**Answer: (A)** | For a sphere, $\chi = V - E + F = 2 \implies 500 - E + 996 = 2 \implies 1496 - E = 2 \implies E = 1494$.

---

### UNIT 4: RECOGNITION & DEEP LEARNING (CO4)

#### Q21. [BL1] What is the dimensionality of the standard SIFT feature descriptor?
- (A) 64
- (B) 128
- (C) 256
- (D) 512
>**Answer: (B)** | $4 \times 4$ subregions $\times$ 8 orientation bins = **128 dimensions**.

#### Q22. [BL1] In the Viola-Jones face detector, how many array references are required to compute the sum of pixels inside any rectangle using an Integral Image?
- (A) 16
- (B) 8
- (C) 4
- (D) 2
>**Answer: (C)** | $\text{Sum} = D - B - C + A$ requires exactly **4 array references** ($O(1)$ constant time).

#### Q23. [BL2] What is the main advantage of ResNet's skip connections ($F(x) + x$)?
- (A) Reduces image resolution
- (B) Allows gradients to flow directly during backpropagation, overcoming vanishing gradients in deep networks
- (C) Converts 2D convolutions to 1D
- (D) Eliminates all fully connected layers
>**Answer: (B)** | Skip connections provide an unhindered identity gradient pathway.

#### Q24. [BL3] An input image of size $64 \times 64 \times 3$ is passed through a Conv layer with $32$ filters of size $3 \times 3$, padding $P=1$, and stride $S=2$. What is the output feature map size?
- (A) $64 \times 64 \times 32$
- (B) $32 \times 32 \times 32$
- (C) $31 \times 31 \times 32$
- (D) $16 \times 16 \times 32$
>**Answer: (B)** | $O = \lfloor \frac{64 - 3 + 2(1)}{2} \rfloor + 1 = \lfloor \frac{63}{2} \rfloor + 1 = 31 + 1 = 32 \implies \mathbf{32 \times 32 \times 32}$.

#### Q25. [BL3] A classifier produces $\text{TP} = 90$, $\text{FP} = 10$, $\text{FN} = 30$. What is the Precision and Recall?
- (A) Precision = $0.90$, Recall = $0.75$
- (B) Precision = $0.75$, Recall = $0.90$
- (C) Precision = $0.90$, Recall = $0.90$
- (D) Precision = $0.80$, Recall = $0.70$
>**Answer: (A)** | Precision $= \frac{90}{90+10} = \mathbf{0.90}$; Recall $= \frac{90}{90+30} = \frac{90}{120} = \mathbf{0.75}$.

#### Q26. [BL4] What failure mode in GAN training causes the Generator to repeatedly output identical images of a single class?
- (A) Vanishing Gradients
- (B) Mode Collapse
- (C) Underfitting
- (D) Dead Neurons
>**Answer: (B)** | **Mode Collapse** occurs when the generator collapses onto a single fooling sample.

#### Q27. [BL5] For a real-time object detector on an autonomous drone running on battery power ($<15\text{W}$), which architecture is optimal?
- (A) Faster R-CNN with ResNet-101
- (B) YOLOv8-Nano with TensorRT FP16
- (C) Standard ViT-Huge
- (D) R-CNN with Selective Search
>**Answer: (B)** | Single-stage lightweight YOLO runs at $>60\text{ FPS}$ within embedded power budgets.

---

### UNIT 5: APPLICATIONS & REAL-TIME SYSTEMS (CO5)

#### Q28. [BL1] What is the Optical Flow Constraint Equation (OFCE)?
- (A) $I_x u + I_y v + I_t = 0$
- (B) $I_x + I_y + I_t = 0$
- (C) $I_x u = I_y v$
- (D) $I_t = 0$
>**Answer: (A)** | Derived from brightness constancy: $\mathbf{I_x u + I_y v + I_t = 0}$.

#### Q29. [BL1] In Eigenfaces with $M=50$ training images of size $100 \times 100$ ($D=10,000$), what is the size of the surrogate covariance matrix $\mathbf{L} = \mathbf{A}^T \mathbf{A}$?
- (A) $10000 \times 10000$
- (B) $50 \times 50$
- (C) $100 \times 100$
- (D) $50 \times 10000$
>**Answer: (B)** | The snapshots trick calculates eigenvectors of the $M \times M = \mathbf{50 \times 50}$ matrix $\mathbf{L}$.

#### Q30. [BL2] Why does the Lucas-Kanade optical flow algorithm fail in flat, textureless regions?
- (A) The brightness constancy assumption is violated
- (B) The spatial gradient matrix $A^T A$ has zero eigenvalues and cannot be inverted
- (C) Lucas-Kanade only works on video edges
- (D) The temporal derivative $I_t$ becomes infinite
>**Answer: (B)** | In flat regions, $I_x \approx 0$ and $I_y \approx 0$, so $\det(A^T A) = 0$ (singular matrix).

#### Q31. [BL3] A road lane curve in metric coordinates is given by $x(y) = 0.001 y^2 + 0.01 y + 2.0$. What is the Radius of Curvature at $y = 0$?
- (A) $1000\text{ m}$
- (B) $500\text{ m}$
- (C) $200\text{ m}$
- (D) $100\text{ m}$
>**Answer: (B)** | $A = 0.001 \implies R = \frac{(1 + B^2)^{3/2}}{|2A|} \approx \frac{1}{|2(0.001)|} = \frac{1}{0.002} = \mathbf{500\text{ meters}}$.

#### Q32. [BL3] In an ADAS lane detection system ($x_{\text{m\_per\_pix}} = 0.005\text{ m/px}$), the camera center is at $x = 640\text{ px}$. Detected left and right lane bases are at $x_L = 200\text{ px}$ and $x_R = 1120\text{ px}$. What is the vehicle lateral offset?
- (A) $+0.10\text{ m}$
- (B) $+0.50\text{ m}$
- (C) $0.00\text{ m}$
- (D) $+1.00\text{ m}$
>**Answer: (A)** | Lane center $= (200 + 1120)/2 = 660\text{ px}$. $\Delta x = 660 - 640 = +20\text{ px}$. $\text{Offset} = 20 \times 0.005 = \mathbf{+0.10\text{ meters } (10\text{ cm})}$.

#### Q33. [BL4] Why does CamShift succeed in tracking faces as they move closer to the camera, while MeanShift fails?
- (A) CamShift uses a neural network
- (B) CamShift dynamically adapts search window size and orientation based on 0th (area) and 2nd spatial moments of the color probability distribution
- (C) CamShift converts frames to grayscale
- (D) MeanShift cannot process color histograms
>**Answer: (B)** | MeanShift uses a fixed window size; CamShift scales and rotates dynamically.

#### Q34. [BL5] In an ISO 26262 ASIL-B lane tracking perception pipeline, what prevents steering oscillations when road paint is temporarily occluded?
- (A) Sobel filter
- (B) Discrete Kalman Filter state-space propagation maintaining continuous trajectory estimates during measurement dropouts
- (C) Otsu thresholding
- (D) Increasing camera exposure
>**Answer: (B)** | Kalman filter predicts and bridges missing detections seamlessly.

---

# SECTION C: RAPID-FIRE BLOOM'S FORMULA & CONCEPT DRILL (35 KEYWORDS)

| No. | Key Term / Concept | Core Formula / Rule | Bloom's Level | Primary Use Case |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Stereo Depth** | $Z = \frac{f B}{d}$ | BL3 | Distance estimation in ADAS / Drones |
| **2** | **Stereo Error** | $\Delta Z = \frac{Z^2}{f B} \Delta d$ | BL4 | Explains why stereo fails at long distances |
| **3** | **Homogeneous 2D Point** | $[x_w, y_w, w]^T \implies (x_w/w, y_w/w)$ | BL2 | Projective matrix linearizations |
| **4** | **Camera Intrinsics ($K$)** | $\begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ (5 DOF) | BL1 | Internal sensor geometry & focal lengths |
| **5** | **Camera Extrinsics** | $[\mathbf{R} \mid \mathbf{t}]$ (6 DOF: 3 Rot + 3 Trans) | BL1 | 3D World to Camera coordinate transformation |
| **6** | **Essential Matrix ($E$)** | $E = [t]_\times R$ (5 DOF, Rank 2) | BL2 | Normalized camera epipolar constraint |
| **7** | **Fundamental Matrix ($F$)** | $p_2^T F p_1 = 0$ (7 DOF, Rank 2) | BL2 | Raw uncalibrated pixel epipolar constraint |
| **8** | **Affine Transformation** | 6 DOF (Requires 3 non-collinear points) | BL1 | Preserves parallelism & length ratios |
| **9** | **Homography (Projective)**| 8 DOF (Requires 4 non-collinear points) | BL1 | Bird's-Eye View (IPM) & plane unwarping |
| **10**| **Grayscale Conversion** | $Y = 0.299R + 0.587G + 0.114B$ | BL1 | Standard ITU-R BT.601 luminance |
| **11**| **Alpha Blending** | $I = \alpha I_1 + (1-\alpha) I_2 + \gamma$ | BL2 | Watermarking and semi-transparent overlay |
| **12**| **Median Filter** | Rank-order non-linear filter | BL2 | Removes impulse / salt-and-pepper noise |
| **13**| **Sobel Operator** | $3 \times 3$ with $[1, 2, 1]$ Gaussian weights | BL2 | Gradient calculation with noise suppression |
| **14**| **Laplacian ($\nabla^2 f$)** | Second-order isotropic operator | BL1 | Edge zero-crossings; rotationally invariant |
| **15**| **High-Boost Filtering** | $f_{\text{sharp}} = f + k(f - f_{\text{smooth}})$ ($k > 1$) | BL3 | High-frequency edge amplification |
| **16**| **Canny NMS** | Gradient direction suppression | BL2 | Thins edges into 1-pixel wide ridges |
| **17**| **Canny Hysteresis** | $T_{\text{high}}$ (seeds) & $T_{\text{low}}$ (connected) | BL2 | Connects weak edges to strong edges |
| **18**| **Morphological Opening** | $(A \ominus B) \oplus B$ (Erode then Dilate) | BL2 | Removes small bright noise specks |
| **19**| **Morphological Closing** | $(A \oplus B) \ominus B$ (Dilate then Erode) | BL2 | Fills dark cracks and small holes |
| **20**| **Top-Hat Transform** | $A - (A \circ B)$ | BL2 | Isolates elements brighter than background |
| **21**| **Black-Hat Transform** | $(A \bullet B) - A$ | BL2 | Isolates dark scratches / shadows |
| **22**| **Fourier DC Normalization**| $a(0) = 0$ | BL2 | Translation invariance (shifts center to origin) |
| **23**| **Fourier Scale Invariance**| Divide by $|a(1)|$ | BL2 | Eliminates size dependency |
| **24**| **Euler Number ($\chi$)** | $\chi = C - H$ ($C$: Objects, $H$: Holes) | BL3 | Topological invariance under elastic warping |
| **25**| **Polyhedral Euler Formula**| $V - E + F = 2 - 2g$ | BL3 | 3D Mesh genus ($g$) and topological check |
| **26**| **Snake Elasticity ($\alpha$)**| $\alpha |v'(s)|^2$ (First derivative) | BL2 | Minimizes contour length (rubber band) |
| **27**| **Snake Stiffness ($\beta$)** | $\beta |v''(s)|^2$ (Second derivative) | BL2 | Enforces smooth curves; resists sharp bends |
| **28**| **SIFT Feature Vector** | $4 \times 4 \text{ cells} \times 8 \text{ bins} = 128\text{-D}$ | BL1 | Scale, rotation, and illumination invariant |
| **29**| **Integral Image Sum** | $\text{Sum} = D - B - C + A$ ($O(1)$ time) | BL3 | Fast Haar cascade feature evaluations |
| **30**| **Polar Hough Line** | $\rho = x\cos\theta + y\sin\theta$ | BL2 | Eliminates infinite slope vertical line bug |
| **31**| **CNN Conv Output Size** | $O = \lfloor \frac{W - K + 2P}{S} \rfloor + 1$ | BL3 | Layer spatial dimension propagation |
| **32**| **CNN Parameter Count** | $(K^2 \cdot C_{\text{in}} + 1) \cdot C_{\text{out}}$ | BL3 | Network memory footprint calculations |
| **33**| **F1-Score Formula** | $F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ | BL3 | Harmonized precision-recall performance |
| **34**| **Optical Flow (OFCE)** | $I_x u + I_y v + I_t = 0$ | BL2 | Brightness constancy constraint |
| **35**| **Road Curvature Radius**| $R = \frac{(1 + (2Ay+B)^2)^{3/2}}{|2A|}$ | BL3 | ADAS curve calculation from parabolic fit |

---

# QUICK REVISION TIPS FOR EXAM DAY:
1. **Always check units in Stereo Depth:** Ensure $B$ and $Z$ are both in meters, and $f$ and $d$ are in pixels.
2. **Remember Transformation DOF hierarchy:** Translation (2) $\to$ Rigid (3) $\to$ Similarity (4) $\to$ Affine (6) $\to$ Homography (8).
3. **Remember Canny's 5 Stages in exact chronological order:** Gauss Blur $\to$ Sobel Gradients $\to$ Non-Max Suppression $\to$ Double Threshold $\to$ Hysteresis.
4. **Remember Morphological Pairs:** Opening = Erode then Dilate (cleans outside); Closing = Dilate then Erode (cleans inside).
5. **Remember Eigenfaces Matrix Trick:** Instead of decomposing $D \times D$ ($4096 \times 4096$) matrix, decompose $M \times M$ ($50 \times 50$) surrogate matrix $\mathbf{L} = \mathbf{A}^T \mathbf{A}$.
