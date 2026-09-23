# UNIT 4: Recognition, Machine Learning, & Deep Learning for Vision (CO4 - BL5)

---

## 1. High-Yield Concept Masterclass

### 1.1 Classical Feature Detectors & Descriptors

```
+-------------------------------------------------------------------------------------------------+
|                               CLASSICAL FEATURE DESCRIPTOR LANDSCAPE                            |
+-------------------+----------------------------------------------------+------------------------+
| METHOD            | CORE MATHEMATICAL PRINCIPLE                        | KEY PROPERTIES         |
+-------------------+----------------------------------------------------+------------------------+
| SIFT (Lowe, 2004) | Difference of Gaussians (DoG) scale-space extrema; | 128-D Descriptor.      |
|                   | 16x16 neighborhood -> 4x4 cells -> 8-bin gradient  | Invariant to scale,    |
|                   | histograms. Taylor expansion for sub-pixel accuracy| rotation, lighting.    |
+-------------------+----------------------------------------------------+------------------------+
| SURF (Bay, 2006)  | Box filter approximation of Hessian matrix using   | 64-D or 128-D.         |
|                   | Integral Images. Haar wavelets for orientation.    | Faster than SIFT.      |
+-------------------+----------------------------------------------------+------------------------+
| ORB (Rublee, 2011)| FAST corner test + Intensity Centroid orientation  | Binary 256-bit string. |
|                   | + steered rBRIEF descriptor. Hamming distance.     | Real-time, patent-free |
+-------------------+----------------------------------------------------+------------------------+
| HOG (Dalal, 2005) | 8x8 cells -> 9-bin unsigned gradient histograms    | Dense grid descriptor. |
|                   | (0-180 deg) -> 2x2 cell overlapping block L2 norm  | Pedestrian detection.  |
+-------------------+----------------------------------------------------+------------------------+
| Viola-Jones Haar  | Rectangular Haar-like features computed in O(1)    | Extremely fast;        |
| Cascades (2001)   | via Integral Images + AdaBoost Attentive Cascade.  | Rigid frontal faces.   |
+-------------------+----------------------------------------------------+------------------------+
```

#### 1. SIFT 128-Dimensional Vector Structure:
* Keypoint neighborhood ($16 \times 16$ pixels) is divided into **$4 \times 4$ sub-regions (16 cells)**.
* Each cell computes an **8-bin gradient orientation histogram** ($0^\circ, 45^\circ, \dots, 315^\circ$).
* Total feature vector size: $4 \times 4 \times 8 = \mathbf{128\text{ dimensions}}$.

#### 2. Integral Images ($O(1)$ Constant Time Summation):
The Integral Image $II(x, y)$ at $(x, y)$ stores the sum of all pixels above and to the left:
$$II(x, y) = \sum_{x' \le x, y' \le y} I(x', y')$$
Any arbitrary rectangle sum with top-left $A$, top-right $B$, bottom-left $C$, and bottom-right $D$ is computed with **only 4 array lookups**:
$$\text{Sum}(\text{Rect}) = D - B - C + A$$

```
   (0,0)----+---------+
        |   |    A    |    B
        |---+---------+
        |   |         |
        |   |  RECT   |
        |---+---------+
        |   |    C    |    D
        +---+---------+
```

---

### 1.2 The Hough Transform (Linear & Circular Feature Extraction)

```
        IMAGE SPACE (Cartesian)                    PARAMETER SPACE (Polar Normal)
        y ^                                          rho ^
          |      Line: y = m*x + c                       |            / Intersection (rho*, theta*)
          |        * (x1, y1)                            |   ~~~~~~~ / ~~~~~~~ (Point 1 sinusoid)
          |       /                                      |  /       *       \
          |      /                                       | /         \       \
          |     /  * (x2, y2)                            |/~~~~~~~~~~~\~~~~~~~\ (Point 2 sinusoid)
          +-------------> x                              +-------------------------> theta
                                                            0       theta*     180 deg
```

#### 1. Polar Normal Parameterization:
To avoid the mathematical infinity singularity of vertical lines ($m \to \infty$ in $y = mx + c$), Duda and Hart parameterized lines as:
$$\rho = x \cos\theta + y \sin\theta$$
Where:
* $\rho$ is the perpendicular distance from the origin to the line ($\rho \ge 0$).
* $\theta$ is the angle of the normal vector relative to the horizontal X-axis ($\theta \in [0^\circ, 180^\circ)$).

#### 2. Duality Principle:
* A single isolated edge point $(x_i, y_i)$ in image space maps to a **continuous sinusoidal curve** in $(\rho, \theta)$ accumulator space.
* Multiple collinear edge points lying along the same line generate **concurrent sinusoidal curves that intersect at a single accumulator bin $(\rho^*, \theta^*)$**.
* Peak detection in the 2D accumulator array $A(\rho, \theta)$ discovers straight lines.

#### 3. Circular Hough Transform:
Equation of a circle: $(x - a)^2 + (y - b)^2 = r^2$.
* If radius $r$ is **known**: 2D accumulator array $A(a, b)$ is used.
* If radius $r$ is **unknown**: **3D accumulator array $A(a, b, r)$** is required.

---

### 1.3 Machine Learning Classifiers for Vision

```
+-------------------------------------------------------------------------------------------------+
|                                MACHINE LEARNING CLASSIFIERS IN CV                               |
+-------------------+---------------------------------------------------+-------------------------+
| CLASSIFIER        | CORE FORMULATION / WORKING                        | KEY COMPUTER VISION USE |
+-------------------+---------------------------------------------------+-------------------------+
| Support Vector    | Finds optimal separating hyperplane w^T*x + b = 0 | HOG + Linear SVM for    |
| Machines (SVM)    | maximizing margin 2 / ||w||. Kernel trick for     | pedestrian detection;   |
|                   | non-linear data: RBF K(x,x') = exp(-gamma||x-x'||^2) image classification.   |
+-------------------+---------------------------------------------------+-------------------------+
| K-Nearest         | Non-parametric, instance-based lazy learner.      | Bag of Visual Words     |
| Neighbors (KNN)   | Classifies based on majority vote of k neighbors. | (BoVW) patch matching.  |
+-------------------+---------------------------------------------------+-------------------------+
| Hidden Markov     | Generative probabilistic model for sequential data| Video gesture and       |
| Models (HMM)      | with hidden states, transition A, emission B.     | human action recognition|
|                   | Solved by Forward, Viterbi, Baum-Welch algorithms | over temporal frames.   |
+-------------------+---------------------------------------------------+-------------------------+
| Random Forest     | Ensemble of de-correlated decision trees trained  | Semantic body part      |
|                   | via Bagging (Bootstrap Aggregation) + random      | segmentation (Kinect);  |
|                   | feature subsets. Reduces variance / overfitting.  | robust to noise.        |
+-------------------+---------------------------------------------------+-------------------------+
```

---

### 1.4 Convolutional Neural Networks (CNN) Architecture & Mechanics

```
INPUT IMAGE       CONVOLUTION + RELU         MAX POOLING (2x2)         FULLY CONNECTED       OUTPUT LOGITS
[32x32x3]  --->  [Conv 3x3, 32 filters] --->    [16x16x32]      --->     [Dense 512]   --->   [10 Classes]
                 [Output: 32x32x32]
```

#### 1. Convolutional Layer Spatial Output Dimension Formula:
For an input of spatial width $W$, filter kernel size $K$, zero-padding $P$, and stride $S$:
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

#### 2. Trainable Parameter Count:
For a Conv layer with $C_{\text{in}}$ input channels, $C_{\text{out}}$ filters of size $K \times K$, with additive bias:
$$\text{Total Parameters} = (K \times K \times C_{\text{in}} + 1) \times C_{\text{out}}$$

#### 3. Landmark CNN Architectural Evolutions:

| Architecture | Year | Key Architectural Innovations | Impact on Computer Vision |
| :--- | :--- | :--- | :--- |
| **LeNet-5** | 1998 | 2 Conv + 2 Subsampling + 3 FC layers; Sigmoid/Tanh | Pioneered digit recognition (MNIST) |
| **AlexNet** | 2012 | 8 layers (5 Conv + 3 FC); **ReLU activations, Dropout (0.5), GPU training** | Sparked the Deep Learning revolution on ImageNet |
| **VGG-16 / 19** | 2014 | Replaced large filters with **homogeneous stacks of small $3 \times 3$ filters** | Proved that depth + small receptive fields is superior |
| **Inception (GoogLeNet)**| 2014 | Multi-scale parallel convs ($1 \times 1, 3 \times 3, 5 \times 5$) + **$1 \times 1$ bottleneck convs** | Reduced parameters to 5M; multi-scale feature capture |
| **ResNet (50/101/152)**| 2015 | **Residual Skip Connections ($F(x) + x$)**; Identity mapping | Solved vanishing gradients; enabled training >1000 layers |
| **MobileNet** | 2017 | **Depthwise Separable Convolutions** (Depthwise + Pointwise $1 \times 1$) | $8\times$ fewer computations; real-time on edge mobile |
| **Vision Transformer (ViT)**| 2020 | Non-convolutional; $16 \times 16$ patch projection + **Self-Attention** | Global receptive field from layer 1; state-of-the-art |

> [!TIP]
> **Why stack two $3 \times 3$ Conv layers instead of one $5 \times 5$ layer?**
> 1. **Same Receptive Field:** Two $3 \times 3$ convs cover an effective $5 \times 5$ spatial area ($3 + (3-1) = 5$).
> 2. **Fewer Parameters:** $2 \times (3 \times 3 \times C^2) = 18 C^2$ parameters vs. $1 \times (5 \times 5 \times C^2) = 25 C^2$ (**$28\%$ reduction!**).
> 3. **More Non-Linearity:** Two ReLU activation functions instead of one, enabling the network to learn more complex decision boundaries.

---

### 1.5 Object Detection Paradigms (Two-Stage vs. One-Stage)

```
TWO-STAGE (Faster R-CNN):   Image -> Backbone CNN -> Region Proposal Network (RPN) -> RoI Pooling -> Fast R-CNN Head (High accuracy, slower)
ONE-STAGE (YOLO / SSD):     Image -> Backbone CNN -> Single Dense Grid Regression (x, y, w, h, confidence, class) -> NMS (Real-time, ultra-fast)
```

#### Comparison: Two-Stage vs. One-Stage Detectors

| Feature | Two-Stage (Faster R-CNN) | One-Stage (YOLO / SSD) |
| :--- | :--- | :--- |
| **Pipeline Workflow** | Stage 1: Generate ~300 region proposals (RPN); Stage 2: Classify and refine each proposal | Single forward pass treats detection as a direct spatial regression problem |
| **Inference Speed** | Slower (5–15 FPS on GPU) | **Real-Time (>30–120 FPS on GPU / Edge SoCs)** |
| **Detection Accuracy** | Slightly higher localization precision on small/dense objects | Highly competitive with modern YOLOv8/v10 architectures |
| **Loss Formulation** | Separate RPN loss + ROI classification & regression loss | Unified multi-task loss (CIoU bounding box + objectness + focal class loss) |

---

### 1.6 Generative Models: GANs & Autoencoders

#### Generative Adversarial Networks (GANs) - Goodfellow (2014)
A zero-sum two-player minimax game between **Generator $G$** and **Discriminator $D$**:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]$$

```
   Random Latent Noise (z) ---> [ GENERATOR G ] ---> Synthetic Image G(z) ---\
                                                                             +--> [ DISCRIMINATOR D ] ---> Real (1) vs. Fake (0)
   Real Dataset Image (x) ---------------------------------------------------/
```

* **Generator $G(z)$:** Tries to create photorealistic synthetic images to fool the Discriminator.
* **Discriminator $D(x)$:** Tries to correctly classify images as Real ($1$) or Fake ($0$).
* **Common GAN Failure Modes:**
  * **Mode Collapse:** Generator produces only a single repetitive sample type that successfully fools $D$, failing to capture data diversity.
  * **Vanishing Gradient:** If $D$ becomes too strong too early, $D(G(z)) \to 0$, causing gradients to vanish and stopping $G$'s learning.

---

### 1.7 Performance Evaluation Metrics in Computer Vision

```
                          PREDICTED POSITIVE          PREDICTED NEGATIVE
 ACTUAL POSITIVE       |  True Positive (TP)      |   False Negative (FN)   |  <- Sensitivity / Recall = TP / (TP+FN)
 ACTUAL NEGATIVE       |  False Positive (FP)     |   True Negative (TN)    |  <- Specificity = TN / (TN+FP)
                       +--------------------------+-------------------------+
                                ^                          ^
                       Precision = TP/(TP+FP)
```

#### Core Metric Formulas:

1. **Precision (Exactness):**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
2. **Recall / Sensitivity (Completeness):**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
3. **F1-Score (Harmonic Mean):**
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
4. **Intersection over Union (IoU / Jaccard Index):**
   $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|\text{Box}_A \cap \text{Box}_B|}{|\text{Box}_A \cup \text{Box}_B|}$$
   *(A predicted bounding box is counted as a **True Positive (TP)** if $\text{IoU} \ge 0.50$ (or threshold) and matches ground-truth class; otherwise it is a **False Positive (FP)**).*
5. **Mean Average Precision (mAP):** Area under the Precision-Recall curve averaged across all object classes.

---

## 2. Essential OpenCV & ML Functions Cheat Sheet for Unit 4

```python
import cv2
import numpy as np

# 1. Feature Detection (ORB & SIFT)
orb = cv2.ORB_create(nfeatures=1000)
kp_orb, des_orb = orb.detectAndCompute(gray_img, None) # Binary descriptors (uint8)

sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(gray_img, None) # 128-D float32 descriptors

# 2. Descriptor Matching (FLANN & Brute-Force)
# For ORB (Binary): Use NORM_HAMMING
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_orb1, des_orb2)

# 3. Hough Line Transform (Standard & Probabilistic)
edges = cv2.Canny(gray_img, 50, 150)
lines_polar = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100) # Returns (rho, theta)
lines_segments = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=30, maxLineGap=10) # Returns [x1, y1, x2, y2]

# 4. Hough Circles
circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=30, minRadius=10, maxRadius=100)

# 5. Haar Cascade Object Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
```

---

## 3. High-Yield Exam MCQs with Step-by-Step Solutions

### [Bloom's Level 1: Remember]

#### Q1. How many feature dimensions does the standard Lowe's SIFT descriptor vector contain?
- (A) 64
- (B) 128
- (C) 256
- (D) 512
>**Answer:** **(B)**
>**Explanation:** SIFT partitions a $16 \times 16$ keypoint neighborhood into $4 \times 4 = 16$ sub-regions, each generating an 8-bin gradient orientation histogram: $16 \times 8 = \mathbf{128\text{ dimensions}}$.

#### Q2. What distance metric is mathematically optimal and fastest for matching binary feature descriptors (such as ORB and BRIEF)?
- (A) Euclidean distance ($L_2$)
- (B) Cosine similarity
- (C) Hamming distance (bitwise XOR followed by bit population count)
- (D) Mahalanobis distance
>**Answer:** **(C)**
>**Explanation:** Binary descriptors represent features as strings of 0s and 1s. The **Hamming distance** counts the number of differing bits via a hardware-accelerated bitwise XOR (`POPCNT`), executing orders of magnitude faster than floating-point $L_2$ Euclidean distance.

#### Q3. In a Convolutional Neural Network, what is the primary structural purpose of a Residual Skip Connection ($F(x) + x$) in ResNet?
- (A) To reduce the image spatial resolution
- (B) To enable unhindered gradient flow directly to earlier layers during backpropagation, eliminating the vanishing gradient problem in very deep networks
- (C) To perform max-pooling
- (D) To convert 2D convolutions into 1D convolutions
>**Answer:** **(B)**
>**Explanation:** The shortcut connection transmits the identity gradient ($\frac{\partial (F(x)+x)}{\partial x} = \frac{\partial F}{\partial x} + 1$) directly back through the network, preventing gradients from vanishing even when networks exceed 100+ layers.

---

### [Bloom's Level 2: Understand]

#### Q4. Why is the polar normal equation $\rho = x\cos\theta + y\sin\theta$ preferred over the slope-intercept equation $y = mx + c$ in the standard Hough Transform?
- (A) Polar equations run faster on GPUs
- (B) Slope-intercept $y = mx + c$ suffers from a mathematical singularity where the slope $m \to \infty$ for vertical lines, requiring an unbounded accumulator space
- (C) Polar equations cannot detect curved lines
- (D) Hough transform only works in polar coordinates
>**Answer:** **(B)**
>**Explanation:** In Cartesian slope-intercept form, nearly vertical lines have slopes approaching infinity ($m \to \pm\infty$), which cannot be mapped onto a finite bounded computer accumulator array. Polar parameterization bounds $\theta \in [0^\circ, 180^\circ)$ and $\rho \in [0, \text{diagonal length}]$.

#### Q5. What is the fundamental difference in region extraction between Fast R-CNN and Faster R-CNN?
- (A) Fast R-CNN uses deep learning; Faster R-CNN uses classical Haar cascades
- (B) Fast R-CNN relies on slow CPU-based Selective Search for region proposals, whereas Faster R-CNN replaces Selective Search with a fully convolutional, GPU-accelerated Region Proposal Network (RPN)
- (C) Faster R-CNN does not use bounding boxes
- (D) Fast R-CNN is a one-stage detector; Faster R-CNN is a three-stage detector
>**Answer:** **(B)**
>**Explanation:** Fast R-CNN solved the CNN feature re-computation problem via RoI pooling, but was bottlenecked by CPU Selective Search (~2 seconds/image). **Faster R-CNN** introduced the **Region Proposal Network (RPN)** sharing convolutional features with the detection head, making the entire pipeline end-to-end differentiable and fast.

---

### [Bloom's Level 3: Apply - Calculations & Formulations]

#### Q6. An input image has dimensions $32 \times 32 \times 3$. A Convolutional layer applies $16$ filters of size $5 \times 5$ with padding $P = 2$ and stride $S = 1$. What is the spatial output feature map dimension, and how many total trainable parameters (including biases) does this layer have?
- (A) Output: $28 \times 28 \times 16$; Parameters: $1,200$
- (B) Output: $32 \times 32 \times 16$; Parameters: $1,216$
- (C) Output: $32 \times 32 \times 16$; Parameters: $400$
- (D) Output: $16 \times 16 \times 16$; Parameters: $1,216$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>1. Spatial Output Size:
>   $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1 = \left\lfloor \frac{32 - 5 + 2(2)}{1} \right\rfloor + 1 = \left\lfloor \frac{32 - 5 + 4}{1} \right\rfloor + 1 = 31 + 1 = 32$$
>   Output shape = $32 \times 32 \times 16$.
>2. Trainable Parameter Count:
>   $$\text{Parameters} = (K \times K \times C_{\text{in}} + 1) \times C_{\text{out}} = (5 \times 5 \times 3 + 1) \times 16 = (75 + 1) \times 16 = 76 \times 16 = \mathbf{1,216\text{ parameters}}$$

#### Q7. In an Integral Image, the four corners of a rectangular region are stored at coordinates: $A(20,20) = 45$, $B(20,60) = 135$, $C(80,20) = 185$, and $D(80,60) = 540$. What is the exact sum of the pixels inside this rectangle?
- (A) $265$
- (B) $310$
- (C) $355$
- (D) $220$
>**Answer:** **(A)**
>**Calculation / Step-by-Step:**
>$$\text{Sum} = D - B - C + A$$
>$$\text{Sum} = 540 - 135 - 185 + 45 = 540 - 320 + 45 = 220 + 45 = 265$$

#### Q8. A vehicle detection model produces the following confusion matrix on a test set of 100 images: $\text{True Positives (TP)} = 80$, $\text{False Positives (FP)} = 20$, $\text{False Negatives (FN)} = 10$, $\text{True Negatives (TN)} = 90$. What are the Precision, Recall, and F1-Score of this model?
- (A) Precision = $0.80$, Recall = $0.889$, F1 = $0.842$
- (B) Precision = $0.889$, Recall = $0.80$, F1 = $0.842$
- (C) Precision = $0.80$, Recall = $0.80$, F1 = $0.80$
- (D) Precision = $0.75$, Recall = $0.90$, F1 = $0.818$
>**Answer:** **(A)**
>**Calculation / Step-by-Step:**
>$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{80}{80 + 20} = \frac{80}{100} = \mathbf{0.80}$$
>$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{80}{80 + 10} = \frac{80}{90} \approx \mathbf{0.889}$$
>$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.80 \times 0.889}{0.80 + 0.889} = \frac{1.4224}{1.689} \approx \mathbf{0.842}$$

#### Q9. Two bounding boxes for a detected traffic light have coordinates: Ground Truth $A = [0, 0, 10, 10]$ (Area = $100$) and Prediction $B = [5, 0, 15, 10]$ (Area = $100$). The intersection area between them is $5 \times 10 = 50$. What is the Intersection over Union (IoU)?
- (A) $0.50$
- (B) $0.333$
- (C) $0.25$
- (D) $0.667$
>**Answer:** **(B)**
>**Calculation / Step-by-Step:**
>$$\text{Area of Union} = \text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B) = 100 + 100 - 50 = 150$$
>$$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}} = \frac{50}{150} = \frac{1}{3} \approx \mathbf{0.333}$$
>(Since $\text{IoU} = 0.333 < 0.50$, at standard $0.50$ threshold, this prediction is classified as a False Positive!).

---

### [Bloom's Level 4: Analyze]

#### Q10. During the training of a Generative Adversarial Network (GAN) for synthesizing rare road hazard images, the Generator starts outputting identical images of a single yellow traffic cone regardless of what random latent vector $z$ is provided. What failure phenomenon has occurred?
- (A) Exploding Gradients
- (B) Mode Collapse
- (C) Overfitting on Discriminator
- (D) Under-segmentation
>**Answer:** **(B)**
>**Explanation:** **Mode Collapse** occurs when the Generator discovers a small subset of outputs (a single "mode") that easily fools the Discriminator, repeatedly outputting only that sample while completely ignoring the full diversity of the real training distribution.

#### Q11. Compare Vision Transformers (ViTs) and CNNs for high-resolution satellite imagery inspection. Why do ViTs require much larger training datasets (like ImageNet-21k or JFT-300M) than CNNs to achieve competitive accuracy?
- (A) ViTs cannot process 3-channel RGB images
- (B) CNNs possess built-in inductive biases (translation equivariance and local spatial locality), whereas ViTs have no prior knowledge of 2D spatial grid topology and must learn all spatial relationships from data via self-attention
- (C) ViTs do not have loss functions
- (D) ViTs can only be trained with unsupervised clustering
>**Answer:** **(B)**
>**Explanation:** Convolutions inherently assume local pixel connectivity and translation invariance (inductive bias). Transformers treat image patches as arbitrary tokens without assuming spatial locality, requiring massive datasets to learn the geometry of the physical visual world from scratch.

---

### [Bloom's Level 5: Evaluate]

#### Q12. You are tasked with designing an embedded computer vision system on an autonomous delivery robot with a strict 15W power budget and a mandatory latency budget of $<25\text{ ms}$ per frame ($>40\text{ FPS}$). Which object detection model architecture should be selected?
- (A) Faster R-CNN with ResNet-152 backbone
- (B) Standard Vision Transformer (ViT-Huge)
- (C) Single-stage MobileNet-YOLO / YOLOv8-Nano with TensorRT FP16 quantization
- (D) R-CNN with Selective Search
>**Answer:** **(C)**
>**Explanation:** Two-stage detectors (Faster R-CNN) and heavy Vision Transformers exceed the 25ms embedded latency budget. **YOLO-Nano / MobileNet** with single-stage regression and depthwise separable convolutions optimized in FP16 executes in $<10\text{ ms}$ within a $<15\text{W}$ TDP envelope.
