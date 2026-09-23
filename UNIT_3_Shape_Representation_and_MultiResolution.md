# UNIT 3: Shape Representation, Deformable Models, & Multi-Resolution Analysis (CO3 - BL4)

---

## 1. High-Yield Concept Masterclass

### 1.1 Contour-Based Representation & Fourier Descriptors

A closed 2D object boundary of $K$ ordered coordinate points $(x_k, y_k)$ for $k = 0, 1, \dots, K-1$ can be represented as a 1D complex spatial sequence:
$$s(k) = x(k) + j \cdot y(k)$$

```
          (x_2, y_2)
             *--------* (x_1, y_1)
            /          \
           /  Object    \
          *    Region    * (x_0, y_0) [Starting Point]
           \            /
            *----------*
          (x_3, y_3)  (x_4, y_4)
```

#### 1. Discrete Fourier Transform (DFT) of Boundary
Applying the 1D DFT to the complex sequence $s(k)$ yields the **Fourier Descriptors (FDs)** $a(n)$:
$$a(n) = \frac{1}{K} \sum_{k=0}^{K-1} s(k) \cdot e^{-j \frac{2\pi n k}{K}}, \quad \text{for } n = 0, 1, \dots, K-1$$

#### 2. Inverse DFT & Boundary Reconstruction (Harmonic Filtering)
To reconstruct and smooth the shape using only the first $N$ harmonic coefficients ($N < K$):
$$s_{\text{rec}}(k) = \sum_{n=0}^{N-1} a(n) \cdot e^{j \frac{2\pi n k}{K}}$$
* **Low-Frequency Harmonics ($n$ near 0):** Capture the macro-geometry, general shape envelope, and global silhouette.
* **High-Frequency Harmonics ($n$ near $K/2$):** Capture micro-structure, sharp corners, fine boundary texture, and sensor noise.

#### 3. Derivation of Geometric Invariance Transformations
To recognize shapes regardless of where they appear, how large they are, or how they are oriented:

| Transformation | Effect on Coordinates | Effect on Fourier Descriptors $a(n)$ | Standard Normalization Technique |
| :--- | :--- | :--- | :--- |
| **Translation** by $(\Delta x, \Delta y)$ | $s'(k) = s(k) + \Delta z$ | $a'(0) = a(0) + \Delta z$, but $a'(n) = a(n)$ for $n \neq 0$ | **Set DC Component $a(0) = 0$** (centers shape at origin) |
| **Uniform Scale** by factor $\alpha$ | $s'(k) = \alpha \cdot s(k)$ | $a'(n) = \alpha \cdot a(n)$ | **Divide all coefficients by $|a(1)|$:** $a_{\text{norm}}(n) = \frac{a(n)}{\|a(1)\|}$ |
| **Rotation** by angle $\theta$ | $s'(k) = s(k) \cdot e^{j\theta}$ | $a'(n) = a(n) \cdot e^{j\theta}$ (pure phase shift) | **Take the absolute magnitude:** $\|a_{\text{norm}}(n)\|$ |
| **Starting Point Shift** by $k_0$ | $s'(k) = s(k - k_0)$ | $a'(n) = a(n) \cdot e^{-j \frac{2\pi n k_0}{K}}$ (phase shift) | **Take the absolute magnitude:** $\|a_{\text{norm}}(n)\|$ |

> [!IMPORTANT]
> A feature vector formed by the magnitude of normalized Fourier descriptors:
> $$\mathbf{f} = \left[ \frac{|a(2)|}{|a(1)|}, \frac{|a(3)|}{|a(1)|}, \dots, \frac{|a(N)|}{|a(1)|} \right]$$
> is **simultaneously invariant to Translation, Scale, Rotation, and Starting Point!**

---

### 1.2 Region-Based Representation & Medial Axis Transform (MAT)

Region-based analysis evaluates the interior spatial topology and mass distribution rather than just the perimeter boundary.

```
       CONTOUR REPRESENTATION                 MEDIAL AXIS TRANSFORM (MAT)
       (Boundary outline only)               (Internal topological skeleton)
       +--------------------+                     +--------------------+
       |  /--------------\  |                     |        /---\       |
       | /                \ |                     |       /  |  \      |
       | |     Region     | |                     |      *---*---*     |
       | \                / |                     |       \  |  /      |
       |  \--------------/  |                     |        \---/       |
       +--------------------+                     +--------------------+
```

#### 1. Mathematical Formulation of Medial Axis Transform (Skeleton)
The **Medial Axis (Skeleton)** $S(R)$ of a 2D region $R$ with boundary $\partial R$ is defined as the locus of centers of **maximal inscribed open disks (balls)** $B(p, r)$ that touch the boundary in at least two points:
$$S(R) = \{ p \in R \mid \nexists q \in R \text{ such that } B(p, D(p)) \subset B(q, D(q)) \}$$
Where $D(p) = \min_{y \in \partial R} \|p - y\|$ is the Euclidean Distance Transform (quench distance).

#### 2. Industrial Quality Control & Defect Detection via MAT:
* **Missing Gear Teeth:** A flawless gear with $N$ teeth yields a central circular hub with exactly $N$ radial branch arms. If a tooth is missing, branch count reduces to $N-1$.
* **Surface Burrs / Spores:** Unwanted protrusions introduce local distance peaks, spawning **extra spurious skeletal branches**. Detected by counting vertex branch degrees $\text{deg}(v) \ge 3$.
* **Internal Air Bubbles / Voids:** Internal holes introduce closed loops (cycles) into the skeletal graph, detected via the Euler Characteristic.

---

### 1.3 Geometric & Topological Descriptors

#### 1. Shape Compactness & Circularity
* **Compactness:** $\text{Compactness} = \frac{P^2}{A}$ (Minimum for a circle: $4\pi \approx 12.57$).
* **Circularity / Form Factor:** $C = \frac{4\pi A}{P^2}$ (Equals $1.0$ for a perfect circle; $<1.0$ for elongated/irregular shapes).
* **Solidity:** $\text{Solidity} = \frac{\text{Area}}{\text{Convex Hull Area}}$ (Detects surface indentations and broken teeth).
* **Extent:** $\text{Extent} = \frac{\text{Area}}{\text{Bounding Box Area}}$.

#### 2. Statistical Moments & Hu's 7 Invariant Moments
* **Raw 2D Spatial Moments:** $m_{pq} = \sum_x \sum_y x^p y^q f(x, y)$
  * Total Mass / Area: $A = m_{00}$
  * Centroid: $\bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$
* **Central Moments (Translation Invariant):** $\mu_{pq} = \sum_x \sum_y (x - \bar{x})^p (y - \bar{y})^q f(x, y)$
* **Normalized Central Moments (Scale Invariant):** $\eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^\gamma}$, where $\gamma = \frac{p+q}{2} + 1$
* **Hu's 7 Moments:** Non-linear combinations of $\eta_{pq}$ that are **invariant to Translation, Scale, and 2D Rotation**.

#### 3. Topological Descriptors & Euler Number
Topological properties remain invariant under continuous elastic deformations ("rubber-sheet" transformations: stretching, bending, twisting, rotation, scaling).
$$\text{Euler Number / Characteristic } \chi = C - H$$
Where:
* $C$ = Number of connected components (objects)
* $H$ = Number of internal holes / voids

```
   Letter 'A' -> C = 1, H = 1  ==>  Euler Number = 1 - 1 = 0
   Letter 'B' -> C = 1, H = 2  ==>  Euler Number = 1 - 2 = -1
   Letter 'C' -> C = 1, H = 0  ==>  Euler Number = 1 - 0 = +1
   Letter '8' -> C = 1, H = 2  ==>  Euler Number = 1 - 2 = -1
```

For 3D polygonal polyhedra meshes:
$$\chi = V - E + F = 2 - 2g$$
Where $V$ = vertices, $E$ = edges, $F$ = faces, and $g$ = genus (number of through-holes / handles; $g=0$ for a sphere/cube $\implies V-E+F=2$).

---

### 1.4 Deformable Models: Active Contours (Snakes)

Introduced by Kass, Witkin, and Terzopoulos (1988), an Active Contour (Snake) is an energy-minimizing parametric spline curve $v(s) = [x(s), y(s)]^T$ for $s \in [0, 1]$ driven by internal and external forces.

```
+-----------------------------------------------------------------------------------+
|                        TOTAL SNAKE ENERGY FUNCTIONAL                              |
|          E_snake = Integral_0^1 [ E_int(v(s)) + E_ext(v(s)) + E_con(v(s)) ] ds    |
+-----------------------------------------+-----------------------------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
+-------v-------------------------------+         +-------------------------v---------------------+
| 1. INTERNAL ENERGY (Curve Geometry)   |         | 2. EXTERNAL ENERGY (Image Attraction)         |
| E_int = 1/2 [ alpha*|v'(s)|^2         |         | E_ext = - |Grad(I(x,y))|^2                   |
|             + beta*|v''(s)|^2 ]       |         | Attracts the snake toward high-gradient edges.|
| - alpha: Elasticity (minimizes length)|         | Or: - |Grad(G_sigma * I)|^2 (blurred edges)   |
| - beta:  Stiffness (resists bending)  |         +-----------------------------------------------+
+---------------------------------------+
```

#### Energy Term Breakdown:
1. **Elasticity / Tension ($\alpha$):** Controlled by 1st derivative $|v'(s)|^2 = \left(\frac{dx}{ds}\right)^2 + \left(\frac{dy}{ds}\right)^2$. Acts like a contracting rubber band; prevents curve from stretching.
2. **Bending Stiffness / Rigidity ($\beta$):** Controlled by 2nd derivative $|v''(s)|^2 = \left(\frac{d^2x}{ds^2}\right)^2 + \left(\frac{d^2y}{ds^2}\right)^2$. Penalizes sharp corners and jagged kinks; enforces smooth continuous curvature.
3. **External Potential Energy ($E_{\text{ext}}$):** Attracts snake to salient image features (edges/valleys):
   $$E_{\text{ext}}(x,y) = -\|\nabla [G_\sigma(x,y) * I(x,y)]\|^2$$

---

### 1.5 Multi-Resolution Analysis: Fourier vs. Wavelet Descriptors

```
+-----------------------------------------------------------------------------------+
|                           IMAGE PYRAMID HIERARCHY                                 |
|                                                                                   |
|           Level 2 (128x128)      /\                                               |
|                                 /  \    GAUSSIAN PYRAMID:                         |
|           Level 1 (256x256)    /    \   Smooth with Gaussian & Subsample by 2     |
|                               /      \                                            |
|           Level 0 (512x512)  /________\ LAPLACIAN PYRAMID:                        |
|                                         L_i = G_i - pyrUp(G_{i+1}) (Detail band)  |
+-----------------------------------------------------------------------------------+
```

#### Fourier Descriptors vs. Wavelet Descriptors

| Comparison Parameter | Fourier Descriptors (FDs) | Wavelet Descriptors (WDs) |
| :--- | :--- | :--- |
| **Basis Functions** | Sinusoids ($\sin, \cos, e^{-j\omega t}$) with **infinite spatial support** | Localized wavelets (Daubechies, Haar) with **compact, finite support** |
| **Spatial Localization** | **None.** Spectral coefficients are global across the entire boundary | **Excellent.** Simultaneous time/space and frequency localization |
| **Sensitivity to Local Defects** | **Catastrophic.** A single tiny notch or broken gear tooth corrupts **ALL** Fourier coefficients | **Isolated.** A local scratch or broken tooth only alters the local wavelet subband |
| **Computational Complexity** | Fast Fourier Transform (FFT): $\mathcal{O}(K \log K)$ | Fast Wavelet Transform (DWT): **$\mathcal{O}(K)$ linear time** |
| **Multi-Scale Analysis** | Requires manual multi-frequency grouping | Natural dyadic octave subband tree decomposition ($LL, LH, HL, HH$) |

---

## 2. Essential OpenCV & Python Functions Cheat Sheet for Unit 3

```python
import cv2
import numpy as np

# 1. Contour Extraction & Properties
contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, closed=True)
circularity = (4 * np.pi * area) / (perimeter ** 2)

# 2. Moments & Hu Moments (Rotation, Scale, Translation Invariant)
M = cv2.moments(cnt)
cx = int(M['m10'] / M['m00']) # Centroid X
cy = int(M['m01'] / M['m00']) # Centroid Y
hu_moments = cv2.HuMoments(M) # 7 invariant numbers
log_hu = -np.sign(hu_moments) * np.log10(np.abs(hu_moments)) # Log scale

# 3. Polygonal Approximation (Douglas-Peucker algorithm)
epsilon = 0.02 * cv2.arcLength(cnt, True)
approx_polygon = cv2.approxPolyDP(cnt, epsilon, closed=True)

# 4. Convex Hull & Solidity
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area) / hull_area

# 5. Distance Transform & Medial Axis Skeleton
dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
_, skeleton = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# 6. Gaussian & Laplacian Pyramids
G1 = cv2.pyrDown(img) # Gaussian Level 1 (Half resolution)
G0_reconstructed = cv2.pyrUp(G1) # Upsample back
L0 = cv2.subtract(img, G0_reconstructed) # Laplacian Bandpass Detail
```

---

## 3. High-Yield Exam MCQs with Step-by-Step Solutions

### [Bloom's Level 1: Remember]

#### Q1. In Fourier Descriptors of a closed boundary, which normalization step achieves translation invariance?
- (A) Dividing all coefficients by $|a(1)|$
- (B) Setting the DC coefficient $a(0) = 0$
- (C) Taking the complex conjugate of $a(n)$
- (D) Multiplying by $e^{j\theta}$
>**Answer:** **(B)**
>**Explanation:** The DC component $a(0) = \frac{1}{K}\sum s(k)$ is the centroid $(\bar{x}, \bar{y})$ of the contour. Setting $a(0) = 0$ shifts the coordinate center to the origin, making the representation strictly **translation invariant**.

#### Q2. What is the Euler Number ($\chi$) of a binary image containing 3 isolated solid discs where 2 of the discs each have a single interior hole?
- (A) $3$
- (B) $1$
- (C) $-1$
- (D) $2$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>$$\chi = C - H$$
>Where $C$ = Number of connected components = $3$.
>$H$ = Total number of holes = $2$ ($1 + 1$).
>$$\chi = 3 - 2 = 1$$

#### Q3. In the energy functional of an Active Contour (Snake), what physical property does the parameter $\beta$ controlling the second derivative $|v''(s)|^2$ enforce?
- (A) Minimization of curve length (elasticity/tension)
- (B) Resistance to sharp bending (stiffness/rigidity)
- (C) Attraction to dark pixel valleys
- (D) Dilation of the internal area
>**Answer:** **(B)**
>**Explanation:** The second derivative measures curvature. Weighting it with $\beta$ penalizes high curvature oscillations and sharp corners, enforcing **bending stiffness / smooth rigidity**. (The first derivative with $\alpha$ enforces elasticity/length contraction).

---

### [Bloom's Level 2: Understand]

#### Q4. Why are Wavelet Descriptors superior to Fourier Descriptors for detecting localized surface burrs on manufactured industrial parts?
- (A) Fourier descriptors require integer inputs only
- (B) Wavelet descriptors have compact spatial-frequency support, so a local surface defect alters only local subband coefficients, whereas Fourier basis functions are global sinusoids that corrupt every coefficient
- (C) Fourier descriptors cannot represent closed curves
- (D) Wavelet transform cannot be computed on computers
>**Answer:** **(B)**
>**Explanation:** Fourier basis functions ($e^{-j\omega t}$) have infinite support, spreading local boundary perturbations across all harmonics. Wavelet basis functions have compact finite support, isolating local shape anomalies directly in time-space.

#### Q5. What is the computational complexity of computing Fourier Descriptors versus Wavelet Descriptors for a boundary of $K$ points?
- (A) Fourier: $\mathcal{O}(K^2)$, Wavelet: $\mathcal{O}(K \log K)$
- (B) Fourier: $\mathcal{O}(K \log K)$ via FFT, Wavelet: $\mathcal{O}(K)$ linear time via Fast Wavelet Transform
- (C) Both are $\mathcal{O}(K^3)$
- (D) Fourier: $\mathcal{O}(K)$, Wavelet: $\mathcal{O}(K^2)$
>**Answer:** **(B)**
>**Explanation:** The 1D Discrete Fourier Transform via FFT requires $\mathcal{O}(K \log K)$ operations, whereas the Discrete Wavelet Transform (Mallat's pyramid filter-bank algorithm) runs in strict **linear time $\mathcal{O}(K)$**.

---

### [Bloom's Level 3: Apply - Calculations & Formulations]

#### Q6. A shape boundary of $K=4$ points has coordinates $(0,0), (2,0), (2,2), (0,2)$. Represented as complex sequence $s(k) = [0+0j, 2+0j, 2+2j, 0+2j]$. What is the DC Fourier Descriptor $a(0)$?
- (A) $1 + 1j$
- (B) $4 + 4j$
- (C) $2 + 2j$
- (D) $0 + 0j$
>**Answer:** **(A)**
>**Calculation / Step-by-Step:**
>$$a(0) = \frac{1}{K} \sum_{k=0}^{K-1} s(k)$$
>$$a(0) = \frac{1}{4} [(0+0j) + (2+0j) + (2+2j) + (0+2j)] = \frac{1}{4} [4 + 4j] = 1 + 1j$$
>(The DC term represents the centroid $(\bar{x}, \bar{y}) = (1, 1)$).

#### Q7. A binary object has an area $A = 100\text{ pixels}$ and its convex hull has an area of $125\text{ pixels}$. What is the Solidity of this object?
- (A) $1.25$
- (B) $0.80$
- (C) $0.25$
- (D) $0.50$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>$$\text{Solidity} = \frac{\text{Area}}{\text{Convex Hull Area}} = \frac{100}{125} = 0.80$$

#### Q8. A geometric region has an Area $A = 50$ and Perimeter $P = 20$. What is its Circularity (Form Factor)?
- (A) $\frac{\pi}{2} \approx 1.57$
- (B) $\frac{4\pi \times 50}{400} = \frac{200\pi}{400} = \frac{\pi}{2} \approx 1.57$
- (C) $0.50$
- (D) $1.00$
>**Answer:** **(A)**
>**Calculation / Step-by-Step:**
>$$\text{Circularity} = \frac{4 \pi A}{P^2} = \frac{4 \pi \times 50}{(20)^2} = \frac{200\pi}{400} = \frac{\pi}{2} \approx 1.57$$

---

### [Bloom's Level 4: Analyze]

#### Q9. During Medial Axis Transform (MAT) computation on camera images of metal gears, fine sensor noise creates jagged single-pixel boundary roughness. How does this affect the resulting skeleton?
- (A) The skeleton remains completely unchanged
- (B) It creates hundreds of false, spurious skeletal branch arms pointing toward every boundary noise point, destroying the graph topology
- (C) The skeleton shrinks to a single center point
- (D) The Euler number increases by 1000
>**Answer:** **(B)**
>**Explanation:** MAT is notoriously sensitive to boundary perturbations. Every local boundary protrusion acts as a local maximal disk center, spawning a new skeletal branch arm. Preprocessing with Gaussian/Median filtering and morphological closing is mandatory!

#### Q10. In Active Contours (Snakes), if the elasticity weight $\alpha$ is set extremely high ($\alpha \to \infty$) while image gradient forces are moderate, what will happen to the snake?
- (A) It will snap onto intricate boundary corners perfectly
- (B) It will contract continuously like an overtightened rubber band, collapsing to a single point and ignoring the true object boundary
- (C) It will become a straight line
- (D) It will expand infinitely
>**Answer:** **(B)**
>**Explanation:** Elasticity ($\alpha |v'|^2$) minimizes total perimeter length. If $\alpha$ dominates external image forces, the snake minimizes length by shrinking to a point without adhering to object edges.

---

### [Bloom's Level 5: Evaluate]

#### Q11. A 3D CAD mesh model of an industrial turbine blade has $V = 1200\text{ vertices}$, $E = 3600\text{ edges}$, and $F = 2402\text{ faces}$. Is this closed mesh topologically equivalent to a standard sphere ($g=0$) or does it contain a through-hole/handle ($g=1$)?
- (A) Genus $g=0$ (Sphere-like, $\chi = 2$)
- (B) Genus $g=1$ (Torus-like, $\chi = 0$)
- (C) Genus $g=2$ ($\chi = -2$)
- (D) Invalid mesh topology
>**Answer:** **(A)**
>**Calculation / Step-by-Step:**
>Using the polyhedral Euler-Poincaré formula:
>$$\chi = V - E + F = 1200 - 3600 + 2402 = 2$$
>Since $\chi = 2 - 2g \implies 2 = 2 - 2g \implies 2g = 0 \implies g = 0$.
>The mesh is topologically simple (equivalent to a sphere with zero through-holes).
