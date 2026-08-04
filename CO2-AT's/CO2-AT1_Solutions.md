# SIMATS ENGINEERING
### DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING
**COURSE:** Computer Vision with OpenCV (DSA02)  
**COURSE OUTCOME COVERED:** CO2: Apply image processing and feature extraction techniques, including edge detection, motion estimation, and optical flow, for analyzing images.. (BL3)  
**ASSESSMENT TOOL 1:** Analytical Problem Solving  
**Weightage:** 50% | **Total Marks:** 50

---

## Question 1: Salt-and-Pepper Noise Removal Before Edge Detection

> **Scenario:** A grayscale image of size $1024 \times 1024$ is captured using a surveillance camera under poor lighting conditions. The image is corrupted by salt-and-pepper noise with a noise density of $0.05$. This means that some pixels are randomly changed to either black or white intensity values. The image has to be preprocessed before applying an edge detection algorithm. Since edge detection is sensitive to abrupt intensity variations, the selected filter should remove noise effectively while preserving important edge details.

### a. Identify the most suitable filtering technique for removing salt-and-pepper noise without causing significant edge blurring.
**Answer:**  
The most suitable filtering technique for removing salt-and-pepper (impulse) noise without blurring structural edges is the **Median Filter** (a non-linear order-statistic spatial filter).

---

### b. Explain why the selected filter performs better than an averaging filter for impulse noise removal.
**Answer:**  
- **Limitations of Linear Averaging Filter:**  
  An averaging filter computes the arithmetic mean of pixel values in a neighborhood. When an impulse noise value (such as $0$ for pepper or $255$ for salt) occurs within the filter window, its extreme magnitude heavily distorts the calculated average. Consequently, the noise value is smeared across all neighboring pixels, producing visible gray smudges and severely blurring sharp intensity transitions (edges).
  
- **Superiority of Non-Linear Median Filter:**  
  The median filter ranks all intensity values in the spatial neighborhood in ascending numerical order and replaces the center pixel with the median (middle) value. Because salt ($255$) and pepper ($0$) impulses occupy the extreme ends of the sorted array, they are completely filtered out from the median selection (provided noisy pixels do not constitute a majority in the window). Consequently, impulse noise is completely removed while step edges remain crisp and unblurred.

---

### c. If a $3 \times 3$ median filter is applied, estimate the probability that a noisy center pixel will be corrected. Clearly state your assumptions.

#### Assumptions:
1. **Noise Independence:** Each pixel in the $1024 \times 1024$ image is corrupted by salt-and-pepper noise independently with probability $p = 0.05$.
2. **Neighborhood Size:** A $3 \times 3$ filtering window contains $N = 9$ pixels ($1$ center pixel + $8$ spatial neighbors).
3. **Correction Criterion:** A median filter successfully replaces a noisy central pixel with an uncorrupted intensity if at least $5$ out of the $9$ pixels in the window are clean (uncorrupted).

#### Mathematical Derivation:
Since the center pixel is **given to be noisy**, the median filter will select an uncorrupted value if the remaining $8$ neighboring pixels contain **at least 4 uncorrupted pixels** (which implies **at most 3 noisy pixels** among the 8 neighbors).

Let $X \sim 	ext{Binomial}(n=8, p=0.05)$ represent the number of noisy pixels among the 8 neighbors.

$$P(	ext{Correction} \mid 	ext{Center Noisy}) = P(X \le 3) = \sum_{x=0}^{3} inom{8}{x} (0.05)^x (0.95)^{8-x}$$

- $P(X=0) = inom{8}{0} (0.05)^0 (0.95)^8 = 0.6634204$
- $P(X=1) = inom{8}{1} (0.05)^1 (0.95)^7 = 8 	imes 0.05 	imes 0.698337 = 0.2793349$
- $P(X=2) = inom{8}{2} (0.05)^2 (0.95)^6 = 28 	imes 0.0025 	imes 0.735092 = 0.0514564$
- $P(X=3) = inom{8}{3} (0.05)^3 (0.95)^5 = 56 	imes 0.000125 	imes 0.773781 = 0.0054165$

$$	ext{Sum} = 0.6634204 + 0.2793349 + 0.0514564 + 0.0054165 = 0.9996282 \quad (	ext{or } 99.9628\%)$$

**Conclusion:**  
There is an **estimated $99.9628\%$ probability** that a noisy center pixel will be successfully corrected by a $3 	imes 3$ median filter.

---

### d. Estimate the total number of noisy pixels present in the original image.
**Answer:**  
- Total Pixels: $N_{	ext{total}} = 1024 	imes 1024 = 1,048,576$ pixels.
- Noise Density: $p = 0.05$.
- **Expected Noisy Pixels:**  
  $$E[	ext{Noisy Pixels}] = N_{	ext{total}} 	imes p = 1,048,576 	imes 0.05 = 52,428.8 pprox 52,429 	ext{ pixels}$$

---

### e. Estimate the expected number of noisy pixels remaining after median filtering.
**Answer:**  
A pixel in the filtered image remains noisy if $5$ or more pixels out of the $9$ pixels in its $3 	imes 3$ window are noisy.

Let $K \sim 	ext{Binomial}(n=9, p=0.05)$ be the total noisy pixels in a $3 	imes 3$ window.

$$P(	ext{Failure}) = P(K \ge 5) = \sum_{k=5}^{9} inom{9}{k} (0.05)^k (0.95)^{9-k}$$

- $P(K=5) = 126 	imes (0.05)^5 	imes (0.95)^4 = 3.207 	imes 10^{-5}$
- $P(K=6) = 84 	imes (0.05)^6 	imes (0.95)^3 = 1.125 	imes 10^{-6}$
- $P(K=7) = 36 	imes (0.05)^7 	imes (0.95)^2 = 2.538 	imes 10^{-8}$
- $P(K=8) = 9 	imes (0.05)^8 	imes (0.95)^1 = 3.34 	imes 10^{-10}$
- $P(K=9) = 1 	imes (0.05)^9 = 1.95 	imes 10^{-12}$

$$P(	ext{Failure}) pprox 3.322 	imes 10^{-5} \quad (	ext{or } 0.003322\%)$$

- **Expected Remaining Noisy Pixels:**  
  $$E[	ext{Remaining Noisy}] = 1,048,576 	imes 3.322 	imes 10^{-5} pprox 34.84 pprox 35 	ext{ pixels}$$

*(Noise reduction efficiency: $rac{52,429 - 35}{52,429} 	imes 100\% pprox 99.93\%$)*

---

### f. Explain how the preprocessing strategy would change if the same image was affected by Gaussian noise instead of salt-and-pepper noise.
**Answer:**  
- **Nature of Noise:** Gaussian noise is additive, zero-mean noise distributed continuously across all image pixels (unlike sparse, high-amplitude impulse spikes).
- **Ineffectiveness of Median Filter:** The median filter is inefficient for Gaussian noise because it does not perform spatial averaging needed to smooth zero-mean continuous Gaussian fluctuations.
- **Recommended Strategy:**  
  1. **Gaussian Smoothing Filter:** Computes spatial weighted averaging using a 2D Gaussian kernel $G(x,y) = rac{1}{2\pi\sigma^2} e^{-rac{x^2+y^2}{2\sigma^2}}$, effectively reducing noise variance.
  2. **Bilateral Filter (Preferred for Edge Detection):** Combines a spatial domain Gaussian with an intensity range Gaussian:
     $$BF[I]_p = rac{1}{W_p} \sum_{q \in S} I_q \cdot g_{\sigma_s}(\|p - q\|) \cdot g_{\sigma_r}(|I_p - I_q|)$$
     This removes Gaussian noise in uniform background areas while preventing smoothing across high-intensity contrast edges.

---

## Question 2: Discretization of a Continuous Image

> **Scenario:** An analog grayscale image has a smooth sinusoidal intensity variation along the horizontal direction. The intensity changes continuously with respect to the spatial coordinate $x$, while it remains constant along the $y$-direction. The image is defined over a spatial range of $0$ to $255$ in both directions. This continuous image has to be converted into a digital image for storage and further processing in a computer vision system.

### a. Explain how the continuous image can be converted into a digital image using sampling and quantization.
**Answer:**  
Image digitization converts continuous space $(x,y)$ and continuous intensity $f(x,y)$ into discrete representation:
1. **Spatial Sampling:** Discretizes spatial coordinates by taking intensity values at discrete spatial grid points $(m\Delta x, n\Delta y)$, partitioning the 2D plane into a grid of discrete pixels.
2. **Intensity Quantization:** Discretizes real-valued continuous intensity amplitudes $f(x,y)$ into a finite set of discrete digital numbers (e.g. 256 levels for an 8-bit image).

---

### b. Identify the spatial variation pattern of the image in the x-direction.
**Answer:**  
The spatial variation pattern along the $x$-direction is a **1D Continuous Sinusoidal Function**:
$$f(x,y) = A \sin(2\pi f_0 x + \phi) + C$$
where $A$ is amplitude, $f_0$ is spatial frequency (cycles per spatial unit), $\phi$ is phase shift, and $C$ is DC offset. Along the $y$-direction, intensity is invariant ($rac{\partial f}{\partial y} = 0$).

---

### c. Determine the minimum sampling requirement needed to avoid aliasing in the x-direction.
**Answer:**  
According to the **Nyquist-Shannon Sampling Theorem**, the spatial sampling frequency $f_s$ must be at least twice the maximum spatial frequency $f_{\max} = f_0$ of the continuous image:
$$f_s \ge 2 f_0 \quad 	ext{(samples per spatial unit)}$$
The maximum spatial sampling interval $\Delta x$ is:
$$\Delta x \le rac{1}{2 f_0}$$

---

### d. Explain what will happen if the image is sampled below the required sampling rate.
**Answer:**  
Sampling below the Nyquist rate ($f_s < 2 f_0$) causes **spatial aliasing**. High-frequency sinusoidal oscillations fold back (overlap) into lower frequencies, generating false low-frequency wave patterns (**moiré patterns** and false brightness ripples) that do not exist in the original scene.

---

### e. Suggest a suitable 8-bit quantization scheme for storing the image.
**Answer:**  
Use **Uniform Scalar 8-Bit Quantization**:
- Quantization Levels: $L = 2^8 = 256$ discrete levels ($0$ to $255$).
- Mapping Function:
  $$Q(f) = 	ext{round}\left( rac{f - f_{\min}}{f_{\max} - f_{\min}} 	imes 255 ight)$$
- Quantization Step Size: $\Delta q = rac{f_{\max} - f_{\min}}{256}$.

---

### f. Discuss the effect of quantization on image quality, storage size, and intensity accuracy.
**Answer:**  
- **Image Quality:** 8-bit quantization (256 levels) provides smooth visual shading that matches human visual perception. Bit depths below 6 bits introduce false contouring (posterization).
- **Storage Size:** For an $N 	imes N$ image quantized at 8 bits/pixel, storage required is $N^2$ bytes (e.g. $256 	imes 256 = 64 	ext{ KB}$).
- **Intensity Accuracy:** Quantization introduces bounded error $|e| \le rac{\Delta q}{2}$ with Mean Squared Error $	ext{MSE} = rac{\Delta q^2}{12}$, giving a high Peak Signal-to-Noise Ratio (PSNR $pprox 48 	ext{ dB}$).

---

## Question 3: Edge Detection in Thin Line Images

> **Scenario:** A $512 	imes 512$ grayscale image contains very thin vertical and horizontal white lines of 1-pixel thickness on a dark uniform background. The objective is to detect the line boundaries accurately without losing thin edge information.

### a. Compare Sobel, Prewitt, and Canny edge detectors for detecting 1-pixel-thick vertical and horizontal lines.
**Answer:**  
- **Sobel and Prewitt Operators:** First-order differential operators with built-in 1D smoothing. For a 1-pixel line, spatial averaging reduces the peak line gradient. Moreover, Sobel/Prewitt yield **double responses** (two parallel edges on both sides of a 1-pixel line) rather than a single thin edge.
- **Canny Edge Detector:** Incorporates **Non-Maximum Suppression (NMS)**, which thins double gradient responses down to single 1-pixel-wide edge lines. However, if the initial Gaussian smoothing parameter $\sigma$ is too large, the 1-pixel line intensity gets excessively smoothed into the background, causing line loss.

---

### b. For a horizontal edge located near pixel position (256, 256), compute the Sobel gradient response using the standard Sobel operator.
**Answer:**  
Consider a step edge transition near $(256, 256)$ where upper row is background ($0$) and lower rows are white line ($255$):

$$I = egin{bmatrix} 0 & 0 & 0 \ 255 & 255 & 255 \ 255 & 255 & 255 \end{bmatrix}$$

Standard Sobel Kernels:
$$S_x = egin{bmatrix} -1 & 0 & 1 \ -2 & 0 & 2 \ -1 & 0 & 1 \end{bmatrix}, \quad S_y = egin{bmatrix} -1 & -2 & -1 \ 0 & 0 & 0 \ 1 & 2 & 1 \end{bmatrix}$$

**Calculations:**
- $G_x = \sum (S_x \odot I) = 0$
- $G_y = (-1)(0) + (-2)(0) + (-1)(0) + (0)(255) + (0)(255) + (0)(255) + (1)(255) + (2)(255) + (1)(255) = 4 	imes 255 = 1020$
- **Gradient Magnitude:** $G = \sqrt{G_x^2 + G_y^2} = \sqrt{0^2 + 1020^2} = 1020$
- **Gradient Direction:** $	heta = rctan2(G_y, G_x) = rctan2(1020, 0) = 90^\circ$ (pointing vertically, normal to horizontal edge).

---

### c. Explain how the gradient magnitude and gradient direction help in identifying the strength and orientation of an edge.
**Answer:**  
- **Gradient Magnitude ($G$):** Quantifies edge strength/contrast. High $G$ values correspond to prominent intensity steps.
- **Gradient Direction ($	heta$):** Specifies the direction of steepest intensity change. The physical edge orientation is orthogonal to $	heta$ ($	heta \pm 90^\circ$).

---

### d. Discuss the role of Gaussian smoothing in the Canny edge detection process.
**Answer:**  
Gaussian smoothing filters high-frequency noise spikes, ensuring that local image derivatives represent true scene structural edges rather than random noise fluctuations.

---

### e. Explain how Gaussian smoothing may affect edge localization, especially for thin lines.
**Answer:**  
For 1-pixel-thick lines, Gaussian kernels spread the line's energy into neighboring background pixels, dampening peak gradient amplitude below detection thresholds and shifting exact line boundaries.

---

### f. If the Canny threshold is reduced to half of its original value, explain the expected changes in detected edges, weak edges, and false edges.
**Answer:**  
- **Weak Edges:** Lowering thresholds allows faint/thin line segments to pass $T_{	ext{low}}$, reducing false negatives and recovering broken line segments.
- **False Edges:** Increases vulnerability to background noise spikes, generating false edge detections (higher false positive rate).

---

## Question 4: Feature Extraction for Handwritten Digit Recognition

> **Scenario:** A computer vision system is being developed for recognizing handwritten digits from scanned images. The handwritten digits vary in size, orientation, stroke thickness, and writing style.

### a. Explain the working principles of SIFT, SURF, and ORB feature extraction methods.
**Answer:**  
- **SIFT (Scale-Invariant Feature Transform):** Uses Difference-of-Gaussians (DoG) across scale space pyramid to locate scale-invariant keypoints. Assigns orientation from gradient histograms and computes a 128-dimensional descriptor.
- **SURF (Speeded-Up Robust Features):** Uses Fast-Hessian detector with integral images and box filters. Assigns orientation via 2D Haar wavelets and builds a 64-dimensional descriptor.
- **ORB (Oriented FAST and Rotated BRIEF):** Uses oFAST (FAST corner detector + Intensity Centroid orientation) and rBRIEF (steered binary descriptors) matched via bitwise XOR Hamming distance.

---

### b. Compare SIFT, SURF, and ORB in terms of scale invariance, rotation invariance, computational cost, and real-time suitability.
**Answer:**  

| Feature | SIFT | SURF | ORB |
| :--- | :--- | :--- | :--- |
| **Scale Invariance** | Excellent (DoG Octaves) | Good (Box Filters) | Fair (Image Pyramids) |
| **Rotation Invariance**| Excellent (Gradient Hist) | Good (Haar Wavelets) | Good (Intensity Centroid) |
| **Computational Cost**| High (Floating Point) | Moderate (Integral Img) | Very Low (Binary Operations) |
| **Real-Time Suitability**| Poor (Slow) | Moderate | Excellent (Ultra-Fast) |

---

### c. If a digit image is rotated by 45 degrees, identify which feature extraction methods can still produce reliable matching features.
**Answer:**  
All three methods (**SIFT, SURF, and ORB**) can extract reliable matching features under 45-degree rotation due to their explicit orientation assignment algorithms. SIFT and SURF exhibit highest stability for off-axis continuous rotations.

---

### d. If the average number of keypoints detected per image is 120 and 50 images are processed, compute the total number of keypoints detected.
**Answer:**  
$$	ext{Total Keypoints} = 120 	ext{ keypoints/image} 	imes 50 	ext{ images} = 6,000 	ext{ keypoints}$$

---

### e. Suggest a suitable dimensionality reduction technique to reduce the feature size while retaining 95% of the useful information.
**Answer:**  
**Principal Component Analysis (PCA)** is the ideal technique. It projects feature descriptors onto orthogonal eigenvectors, selecting the top $k$ components that capture $95\%$ cumulative variance.

---

### f. Explain how dimensionality reduction improves classification efficiency in handwritten digit recognition.
**Answer:**  
- **Mitigates Curse of Dimensionality:** Prevents model overfitting and improves classification generalization.
- **Reduces Computational Overhead:** Shrinks feature vector dimensions, accelerating training and real-time prediction speeds.

---

## Question 5: Morphological Preprocessing for Character Recognition

> **Scenario:** A binary image of size $256 	imes 256$ contains scanned handwritten characters. Due to poor scanning quality, the image contains small white noise dots in the background and small black gaps inside the character strokes.

### a. Identify the morphological operation suitable for removing small white noise dots from the background.
**Answer:**  
**Morphological Opening** ($A \circ B = (A \ominus B) \oplus B$), which performs Erosion followed by Dilation.

---

### b. Identify the morphological operation suitable for filling small gaps or holes inside the character strokes.
**Answer:**  
**Morphological Closing** ($A ullet B = (A \oplus B) \ominus B$), which performs Dilation followed by Erosion.

---

### c. Explain the difference between opening and closing operations using simple image processing terminology.
**Answer:**  
- **Opening (Erosion $ightarrow$ Dilation):** Removes small foreground white noise specks, disconnects thin touching objects, and smooths outer object boundaries.
- **Closing (Dilation $ightarrow$ Erosion):** Fills small background black holes/gaps inside foreground objects, bridges broken stroke lines, and smooths inner object boundaries.

---

### d. If a $3 	imes 3$ square structuring element is used, explain how it affects thin character strokes.
**Answer:**  
Erosion shrinks foreground character boundaries by $1$ pixel on all sides (reducing stroke width by $2$ pixels). If a character stroke is only $1$ or $2$ pixels thick, erosion during opening will **completely erase** the stroke.

---

### e. Discuss the risk of using a large structuring element during preprocessing.
**Answer:**  
- **Over-Erosion:** Erases fine stroke details and small characters completely.
- **Over-Dilation:** Fuses adjacent distinct characters together, destroying topological character structures.

---

### f. Propose a complete preprocessing pipeline for handwritten character recognition, starting from image acquisition to feature extraction.
**Answer:**  
1. **Image Acquisition & Grayscale Conversion:** Capture digit image and convert to 8-bit single-channel grayscale.
2. **Adaptive Binarization:** Apply **Otsu’s Thresholding** to create a clean binary foreground/background image.
3. **Noise Elimination:** Apply **Morphological Opening** ($3 	imes 3$ cross SE) to remove background white dots.
4. **Stroke Hole Filling:** Apply **Morphological Closing** ($3 	imes 3$ cross SE) to bridge stroke breaks.
5. **Bounding Box Normalization & Deskewing:** Extract character contours, crop bounding box, deskew via affine shear transform, and resize to $28 	imes 28$ pixels.
6. **Feature Extraction:** Compute HOG (Histogram of Oriented Gradients) or normalized pixel intensities for OCR classification.
