# UNIT 2: Image Processing, Spatial Filtering, & Transformations (CO2 - BL3)

---

## 1. High-Yield Concept Masterclass

### 1.1 Color Spaces & Image Arithmetic

#### Standard Grayscale Conversion (ITU-R BT.601 / CIE Luminance)
Human eyes have unequal sensitivity to primary colors (highest sensitivity to Green, moderate to Red, lowest to Blue). The standard luminance formula is:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

* **OpenCV Color Default:** OpenCV loads images in **BGR order** by default (not RGB).
* **Color Conversions:** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`, `cv2.COLOR_BGR2HSV`, `cv2.COLOR_BGR2HLS`, `cv2.COLOR_BGR2LAB`.

#### Alpha Blending & Watermarking Equation
Superimposing a watermark/overlay image $I_2$ onto a base image $I_1$ with transparency weight $\alpha \in [0, 1]$:
$$I_{\text{out}}(x, y) = \alpha \cdot I_1(x, y) + \beta \cdot I_2(x, y) + \gamma$$
Where $\beta = 1 - \alpha$ and $\gamma$ is a scalar brightness offset. (Implemented via `cv2.addWeighted()`).

---

### 1.2 Hierarchy of 2D Geometric Transformations

Geometric transformations map spatial coordinate $(x, y)$ to transformed coordinate $(x', y')$. They form a strict mathematical hierarchy:

```
+-----------------------------------------------------------------------------+
| PROJECTIVE / HOMOGRAPHY (8 DOF) - Preserves straight lines / collinearity    |
|   +-----------------------------------------------------------------------+ |
|   | AFFINE (6 DOF) - Preserves parallelism, parallel length ratios         | |
|   |   +-----------------------------------------------------------------+ | |
|   |   | SIMILARITY (4 DOF) - Preserves angles, shape aspect ratios      | | |
|   |   |   +-----------------------------------------------------------+ | | |
|   |   |   | EUCLIDEAN / RIGID (3 DOF) - Preserves absolute Euclidean  | | | |
|   |   |   |                             lengths and angles            | | | |
|   |   |   |   +-----------------------------------------------------+ | | | |
|   |   |   |   | TRANSLATION (2 DOF) - Pure coordinate shift (t_x,t_y)| | | | |
|   |   |   |   +-----------------------------------------------------+ | | | |
|   |   |   +-----------------------------------------------------------+ | | |
|   |   +-----------------------------------------------------------------+ | |
|   +-----------------------------------------------------------------------+ |
+-----------------------------------------------------------------------------+
```

#### Mathematical Forms & Degrees of Freedom:

| Transformation Type | Transformation Matrix $\mathbf{M}$ | DOF | Invariant Geometric Properties | Minimum Point Correspondences Required |
| :--- | :--- | :--- | :--- | :--- |
| **Translation** | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}$ | **2** | Orientation, lengths, angles, areas | **1 point** $(x_1 \to x_1')$ |
| **Euclidean / Rigid** | $\begin{bmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \end{bmatrix}$ | **3** | Lengths (isometry), angles, areas | **2 points** |
| **Similarity** | $\begin{bmatrix} s\cos\theta & -s\sin\theta & t_x \\ s\sin\theta & s\cos\theta & t_y \end{bmatrix}$ | **4** | Angles, ratio of distances, shapes | **2 points** |
| **Affine** | $\begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \end{bmatrix}$ | **6** | **Parallelism**, ratio of parallel lengths/areas | **3 non-collinear points** (`cv2.getAffineTransform`) |
| **Perspective / Homography** | $\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix}$ | **8** | **Collinearity / Straight lines**, Cross-ratio | **4 non-collinear points** (`cv2.getPerspectiveTransform`, `cv2.findHomography`) |

#### Direct Linear Transformation (DLT) for Homography
Given 4 point pairs $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$, each pair yields 2 linear equations:
$$\begin{bmatrix} -x_i & -y_i & -1 & 0 & 0 & 0 & x'_i x_i & x'_i y_i & x'_i \\ 0 & 0 & 0 & -x_i & -y_i & -1 & y'_i x_i & y'_i y_i & y'_i \end{bmatrix} \mathbf{h} = \mathbf{0}$$
Stacking 4 points gives an $8 \times 9$ matrix $\mathbf{A}$. The solution $\mathbf{h}$ is the singular vector corresponding to the smallest singular value in Singular Value Decomposition ($\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \implies \mathbf{h} = \text{last column of } \mathbf{V}$).

---

### 1.3 2D Spatial Filtering & Smoothing

#### Convolution vs. Correlation
* **2D Correlation:**
  $$g(x, y) = f(x, y) \otimes h(x, y) = \sum_{s=-a}^a \sum_{t=-b}^b h(s, t) \cdot f(x+s, y+t)$$
* **2D Convolution:** Flips the filter kernel by $180^\circ$ horizontally and vertically before sliding:
  $$g(x, y) = f(x, y) * h(x, y) = \sum_{s=-a}^a \sum_{t=-b}^b h(s, t) \cdot f(x-s, y-t)$$
  *(If the kernel is symmetric, like Gaussian, Convolution $\equiv$ Correlation).*

#### Comparison of Primary Smoothing Filters

| Filter Type | Linear / Non-Linear | Mathematical Principle | Primary Application / Noise Removed | Edge Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Box / Mean Filter** | **Linear** | Unweighted local average: $\frac{1}{K^2} \sum f(i,j)$ | General blur / uniform noise | **Severely blurs sharp edges** |
| **Gaussian Filter** | **Linear** | 2D Gaussian bell-curve weights: $G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$ | Natural Gaussian sensor noise | Smooth blur; separable ($G_{2D} = G_x \cdot G_y$) |
| **Median Filter** | **Non-Linear** | Selects the statistical median value in window | **Salt-and-pepper / impulse noise** | **Preserves sharp edges perfectly** without blurring |
| **Bilateral Filter** | **Non-Linear** | Fuses spatial Gaussian distance + photometric intensity similarity Gaussian | Edge-preserving cartoon smoothing | **Smoothes flat regions while leaving step edges intact** |

---

### 1.4 Spatial Sharpening & Edge Detection Operators

```
IMAGE INTENSITY PROFILE:     ____/----\____  (Step / Ramp Edge)
1st DERIVATIVE (Gradient):   ____/\---/\___  (Produces wide peak at edge)
2nd DERIVATIVE (Laplacian):  ___/\_v_/\_v__  (Produces ZERO-CROSSING at edge center)
```

#### 1. First-Order Derivative Operators (Gradient)
Gradient Vector: $\nabla f = \begin{bmatrix} G_x \\ G_y \end{bmatrix} = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}$
* **Gradient Magnitude:** $M(x,y) = \sqrt{G_x^2 + G_y^2} \approx |G_x| + |G_y|$
* **Gradient Direction (Orientation):** $\theta(x,y) = \arctan\left(\frac{G_y}{G_x}\right)$ *(perpendicular to edge direction)*

Standard Gradient Kernels ($3 \times 3$):
* **Sobel Operators** (incorporates Gaussian smoothing weights $[1, 2, 1]$ to suppress noise):
  $$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$
* **Prewitt Operators** (uniform weights):
  $$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -1 & 0 & +1 \\ -1 & 0 & +1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ +1 & +1 & +1 \end{bmatrix}$$
* **Roberts Cross Operators** ($2 \times 2$ diagonal):
  $$G_x = \begin{bmatrix} +1 & 0 \\ 0 & -1 \end{bmatrix}, \quad G_y = \begin{bmatrix} 0 & +1 \\ -1 & 0 \end{bmatrix}$$

#### 2. Second-Order Derivative: The Laplacian Operator ($\nabla^2 f$)
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$
* **Isotropic (Rotationally Invariant):** Detects edges equally in all directions with a single convolution!
* **Standard Laplacian Kernels:**
  $$\text{Negative Center (4-neighbor): } \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}, \quad \text{Negative Center (8-neighbor): } \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$
  $$\text{Positive Center (4-neighbor): } \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \text{Positive Center (8-neighbor): } \begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$

#### 3. Unsharp Masking & High-Boost Filtering
* **Step 1:** Blur original image: $f_{\text{smooth}}(x, y) = \text{GaussianBlur}(f(x, y))$
* **Step 2:** Generate mask: $g_{\text{mask}}(x, y) = f(x, y) - f_{\text{smooth}}(x, y)$
* **Step 3:** Add weighted mask back to original:
  $$f_{\text{sharp}}(x, y) = f(x, y) + k \cdot g_{\text{mask}}(x, y)$$
  * $k = 1 \implies$ **Standard Unsharp Masking**
  * $k > 1 \implies$ **High-Boost Filtering** (amplifies high-frequency details while boosting base image by $A = 1+k$).

---

### 1.5 The Canny Edge Detector (5-Stage Optimal Pipeline)

John Canny (1986) formulated three optimal criteria: (1) Low error rate, (2) Precise edge localization, (3) Single response to a single edge.

```
+-------------------------------------------------------------------------------+
| STAGE 1: Gaussian Smoothing                                                   |
| Convolve image with 2D Gaussian kernel to suppress sensor noise.              |
+---------------------------------------+---------------------------------------+
                                        |
+---------------------------------------v---------------------------------------+
| STAGE 2: Gradient Intensity & Direction Computation                           |
| Apply Sobel kernels G_x, G_y to compute magnitude M(x,y) and direction theta.  |
+---------------------------------------+---------------------------------------+
                                        |
+---------------------------------------v---------------------------------------+
| STAGE 3: Non-Maximum Suppression (NMS)                                        |
| Round gradient angle theta to 4 sectors: 0 deg (Horizontal), 45 deg,          |
| 90 deg (Vertical), 135 deg. Suppress pixel if not strictly local maximum      |
| along its gradient direction -> THINS THICK EDGES TO 1-PIXEL WIDE RIDGES!     |
+---------------------------------------+---------------------------------------+
                                        |
+---------------------------------------v---------------------------------------+
| STAGE 4: Double Thresholding (Hysteresis Bounds)                              |
| Define T_high and T_low (typically ratio 2:1 or 3:1).                          |
| - If M(x,y) >= T_high -> STRONG EDGE PIXEL (confirmed edge)                   |
| - If T_low <= M(x,y) < T_high -> WEAK EDGE PIXEL (candidate)                  |
| - If M(x,y) < T_low -> SUPPRESSED (noise)                                     |
+---------------------------------------+---------------------------------------+
                                        |
+---------------------------------------v---------------------------------------+
| STAGE 5: Edge Tracking by Hysteresis                                          |
| Weak edge pixels are retained IF AND ONLY IF they are 8-connected to at      |
| least one Strong Edge Pixel. Isolated weak edge pixels are discarded.        |
+-------------------------------------------------------------------------------+
```

---

### 1.6 Morphological Image Processing

Morphological operations probe binary (or grayscale) shapes with a small geometric probe called a **Structuring Element (SE)** $B$:

```
  EROSION (A (-) B)         DILATION (A (+) B)           OPENING (A o B)            CLOSING (A * B)
 (Shrinks Foreground)      (Expands Foreground)        (Erosion then Dilation)    (Dilation then Erosion)
+--------------------+    +--------------------+    +--------------------+    +--------------------+
|  ###        ###    |    | #######  #######   |    |  ###        ###    |    |  ################  |
|  ###        ###    |    | #######  #######   |    |  ###        ###    |    |  ################  |
|                    |    |   ##############   |    |                    |    |  ################  |
|  ###        ###    |    | #######  #######   |    |  ###        ###    |    |  ################  |
| (Noise specks die) |    | (Fills holes/gaps) |    | (Smooths contours) |    | (Fills internal gap|
+--------------------+    +--------------------+    +--------------------+    +--------------------+
```

#### Comprehensive Morphological Operation Guide:

| Operation | Mathematical Definition | Physical Geometric Effect | Practical Real-World Use Case |
| :--- | :--- | :--- | :--- |
| **Erosion** | $A \ominus B = \{z \mid (B)_z \subseteq A\}$ | Shrinks object boundaries; completely removes objects smaller than SE | Removing tiny salt noise specks; separating touching cells |
| **Dilation** | $A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$ | Expands object boundaries; fills small internal holes | Bridging broken character strokes in OCR; joining fragmented edges |
| **Opening** | $A \circ B = (A \ominus B) \oplus B$ | **Erosion followed by Dilation**. Eliminates thin protrusions and specks | Removing noisy background flecks without changing object size |
| **Closing** | $A \bullet B = (A \oplus B) \ominus B$ | **Dilation followed by Erosion**. Fills narrow holes and cracks | Bridging gaps in broken lane lines; sealing enclosed cavities |
| **Morphological Gradient** | $(A \oplus B) - (A \ominus B)$ | Difference between Dilation and Erosion | **Extracting sharp 1-pixel boundary outlines** of binary objects |
| **Top-Hat (White Top-Hat)** | $A - (A \circ B)$ | Original image minus its morphological opening | **Isolating bright elements** that are smaller than SE on uneven dark backgrounds |
| **Black-Hat (Bottom-Hat)** | $(A \bullet B) - A$ | Morphological closing minus original image | **Isolating dark elements / shadows** on bright backgrounds |
| **Boundary Extraction** | $A - (A \ominus B)$ | Original minus eroded version | Fast interior boundary tracing |

---

## 2. Essential OpenCV Functions Cheat Sheet for Unit 2

```python
import cv2
import numpy as np

# 1. Color Conversion & Alpha Blending
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
watermarked = cv2.addWeighted(img1, 0.7, img2, 0.3, 0.0) # alpha=0.7, beta=0.3, gamma=0

# 2. Geometric Transformations
# 2a. Affine (3 points)
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M_affine = cv2.getAffineTransform(pts1, pts2) # 2x3 matrix
affine_out = cv2.warpAffine(img, M_affine, (width, height))

# 2b. Perspective / Homography (4 points)
pts_src = np.float32([[100,100],[400,100],[100,400],[400,400]])
pts_dst = np.float32([[0,0],[300,0],[50,300],[250,300]])
H = cv2.getPerspectiveTransform(pts_src, pts_dst) # 3x3 matrix
persp_out = cv2.warpPerspective(img, H, (300, 300))

# 3. Spatial Smoothing
blur_box = cv2.blur(img, (5, 5))
blur_gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
blur_median = cv2.medianBlur(img, 5) # Ideal for salt-and-pepper noise
blur_bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# 4. Derivative Edge Detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
canny_edges = cv2.Canny(gray, threshold1=50, threshold2=150) # Hysteresis thresholds

# 5. Morphological Operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
erosion  = cv2.erode(binary_img, kernel, iterations=1)
dilation = cv2.dilate(binary_img, kernel, iterations=1)
opening  = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closing  = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, kernel)
tophat   = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
```

---

## 3. High-Yield Exam MCQs with Step-by-Step Solutions

### [Bloom's Level 1: Remember]

#### Q1. How many minimum non-collinear corresponding point pairs are mathematically required to compute an Affine Transformation and a Perspective Homography respectively?
- (A) 2 points for Affine, 3 points for Homography
- (B) 3 points for Affine, 4 points for Homography
- (C) 4 points for Affine, 8 points for Homography
- (D) 6 points for Affine, 8 points for Homography
>**Answer:** **(B)**
>**Explanation:** An Affine matrix has 6 degrees of freedom (each point provides 2 equations: $x', y'$), requiring **3 point pairs** ($3 \times 2 = 6$ equations). A Projective Homography matrix has 8 degrees of freedom, requiring **4 point pairs** ($4 \times 2 = 8$ equations).

#### Q2. Which filtering technique is non-linear, sorts pixel neighborhood values, and is the gold standard for removing impulse ("salt-and-pepper") noise without blurring edges?
- (A) Gaussian Filter
- (B) Box Filter
- (C) Median Filter
- (D) Laplacian of Gaussian
>**Answer:** **(C)**
>**Explanation:** The Median filter replaces the center pixel with the statistical median of neighborhood intensities. Outlier noise spikes (0 or 255) are pushed to the sorted extremes and discarded, leaving the true edge intact.

#### Q3. In the Canny edge detection algorithm, which stage is specifically responsible for thinning thick gradient contours down to 1-pixel wide continuous ridges?
- (A) Gaussian Blurring
- (B) Double Thresholding
- (C) Non-Maximum Suppression (NMS)
- (D) Hysteresis Tracking
>**Answer:** **(C)**
>**Explanation:** Non-Maximum Suppression compares each pixel's gradient magnitude against its two neighbors along the gradient orientation direction. If the pixel is not strictly the maximum, it is suppressed to 0, producing 1-pixel thin ridges.

#### Q4. What is the mathematical operation for Morphological Opening of image $A$ by structuring element $B$?
- (A) Dilation followed by Erosion: $(A \oplus B) \ominus B$
- (B) Erosion followed by Dilation: $(A \ominus B) \oplus B$
- (C) Dilation minus Erosion: $(A \oplus B) - (A \ominus B)$
- (D) Original minus Opening: $A - (A \circ B)$
>**Answer:** **(B)**
>**Explanation:** Opening is defined as **Erosion followed by Dilation** ($A \circ B = (A \ominus B) \oplus B$). Closing is Dilation followed by Erosion ($A \bullet B = (A \oplus B) \ominus B$).

---

### [Bloom's Level 2: Understand]

#### Q5. Why does the Sobel operator produce cleaner, less noisy edge responses compared to the Prewitt operator?
- (A) Sobel uses a $5 \times 5$ kernel while Prewitt uses a $3 \times 3$ kernel
- (B) Sobel incorporates Gaussian smoothing coefficients $[1, 2, 1]$ orthogonal to the derivative direction, reducing high-frequency noise sensitivity
- (C) Sobel calculates second derivatives whereas Prewitt calculates first derivatives
- (D) Sobel uses non-linear median averaging
>**Answer:** **(B)**
>**Explanation:** In the horizontal Sobel kernel $G_x = \begin{bmatrix}-1 & 0 & 1\end{bmatrix}^T * \begin{bmatrix}1 & 2 & 1\end{bmatrix}$, the vector $[1, 2, 1]$ applies triangular/Gaussian smoothing across adjacent rows, suppressing camera sensor noise.

#### Q6. What is the fundamental difference between an Affine transformation and a Perspective (Homography) transformation?
- (A) Affine preserves angles; Homography does not
- (B) Affine preserves parallelism of lines; Homography allows parallel lines to converge at a vanishing point
- (C) Affine requires 4 points; Homography requires 3 points
- (D) Homography is a linear $2 \times 2$ matrix, while Affine is $3 \times 3$
>**Answer:** **(B)**
>**Explanation:** Under Affine transformations, parallel lines remain strictly parallel (affine geometry). Under Projective/Perspective transformations, parallel 3D lines project to lines that intersect at a vanishing point on the horizon line.

---

### [Bloom's Level 3: Apply - Calculations & Convolutions]

#### Q7. A $3 \times 3$ image patch $I$ is convolved with a horizontal Sobel kernel $G_x$:
$$I = \begin{bmatrix} 10 & 10 & 80 \\ 10 & 10 & 80 \\ 10 & 10 & 80 \end{bmatrix}, \quad G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}$$
What is the resulting gradient value $G_x$ at the central pixel?
- (A) $0$
- (B) $70$
- (C) $280$
- (D) $320$
>**Answer:** **(C)**
>**Calculation / Step-by-Step:**
>$$G_x = \sum (I \cdot G_x)$$
>$$G_x = [(-1 \times 10) + (0 \times 10) + (+1 \times 80)] + [(-2 \times 10) + (0 \times 10) + (+2 \times 80)] + [(-1 \times 10) + (0 \times 10) + (+1 \times 80)]$$
>$$G_x = [-10 + 80] + [-20 + 160] + [-10 + 80] = 70 + 140 + 70 = 280$$

#### Q8. For the image patch in Q7, what is the vertical Sobel gradient $G_y$ using kernel $G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$?
- (A) $280$
- (B) $0$
- (C) $-140$
- (D) $140$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>Row 1 and Row 3 are identical ($[10, 10, 80]$).
>Row 1 multiplied by $[-1, -2, -1]$ yields $-(10 + 20 + 80) = -110$.
>Row 2 multiplied by $[0, 0, 0] = 0$.
>Row 3 multiplied by $[+1, +2, +1]$ yields $+(10 + 20 + 80) = +110$.
>Total $G_y = -110 + 0 + 110 = 0$.
>(Since intensity does not change vertically, vertical gradient is 0!).

#### Q9. A $3 \times 3$ neighborhood around a noisy pixel contains values: $[12, 14, 15, 13, 255, 14, 16, 12, 15]$. What is the output intensity after applying a $3 \times 3$ Median Filter?
- (A) $40$
- (B) $255$
- (C) $14$
- (D) $15$
>**Answer:** **(C)**
>**Calculation / Step-by-Step:**
>Sort the 9 intensity values in ascending order:
>$$\{12, 12, 13, 14, \mathbf{14}, 15, 15, 16, 255\}$$
>The 5th element (median) is **14**.
>The impulse noise spike (255) is completely eliminated!

#### Q10. An unsharp masking filter uses original pixel value $f(x,y) = 150$, smoothed value $f_{\text{smooth}}(x,y) = 110$, and high-boost factor $k = 1.5$. What is the sharpened output pixel value?
- (A) $190$
- (B) $210$
- (C) $160$
- (D) $250$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>1. Compute mask: $g_{\text{mask}} = f - f_{\text{smooth}} = 150 - 110 = 40$
>2. Apply formula: $f_{\text{sharp}} = f + k \cdot g_{\text{mask}} = 150 + 1.5 \times 40 = 150 + 60 = 210$.

---

### [Bloom's Level 4: Analyze]

#### Q11. A binary document image has printed text where fine characters (like 'e' and 'a') have small broken gaps in their strokes, and the background contains tiny isolated 1-pixel noise specks. Which sequence of morphological operations will simultaneously fix both issues?
- (A) Dilation followed by Closing
- (B) Opening (removes background specks) followed by Closing (bridges stroke gaps)
- (C) Erosion followed by Opening
- (D) Top-hat followed by Black-hat
>**Answer:** **(B)**
>**Explanation:** Morphological **Opening** ($A \circ B$) eliminates isolated background noise specks without affecting overall letter geometry. Subsequent **Closing** ($A \bullet B$) fills internal stroke breaks and fuses fractured gaps in the characters.

#### Q12. In Canny edge detection, setting $T_{\text{high}} = 200$ and $T_{\text{low}} = 50$ results in discontinuous dashed road lane edges. How should the engineer tune the thresholds to obtain solid continuous lane boundaries without amplifying background noise?
- (A) Increase $T_{\text{high}}$ to 250
- (B) Lower $T_{\text{low}}$ to 25 while keeping $T_{\text{high}} = 200$ so hysteresis can track faint connecting edges
- (C) Set $T_{\text{low}} = T_{\text{high}} = 200$
- (D) Remove Gaussian filtering
>**Answer:** **(B)**
>**Explanation:** Strong edges that seed the detection are captured by $T_{\text{high}}$. The continuity of edges across faint or faded segments depends on $T_{\text{low}}$ via hysteresis tracking. Lowering $T_{\text{low}}$ allows weak connecting edge pixels to bridge the dashed segments.

---

### [Bloom's Level 5: Evaluate]

#### Q13. An automated quality inspection system on a conveyor belt must inspect metallic parts under severe non-uniform shadow gradients. Which morphological transform should be selected to extract dark defect scratches while suppressing the uneven illumination background?
- (A) Morphological Dilation
- (B) White Top-Hat Transform
- (C) Black-Hat (Bottom-Hat) Transform
- (D) Median Blur
>**Answer:** **(C)**
>**Explanation:** The **Black-Hat Transform** ($(A \bullet B) - A$) isolates elements that are **darker than their local surroundings** and smaller than the structuring element, effectively flattening slow-varying uneven illumination and highlighting dark scratches. (White top-hat isolates bright elements).
