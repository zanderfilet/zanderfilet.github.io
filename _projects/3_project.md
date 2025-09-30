---
layout: page
title: Fun with Filters and Frequencies
description: Cyprian Zander
importance: 1
category: CS180
related_publications: false
img: assets/img/cs180/p2/cover.png
---

### Overview

In this project, I explored edge detection with finite differences and Gaussian smoothing, image sharpening with unsharp masking, hybrid images by mixing high and low frequencies, and multi-resolution blending with Gaussian and Laplacian stacks for seamless composites.

---

### Part 1: Fun with Filters

#### 1.1 Convolutions from Scratch

Convolutions are mathematical operations applied to images to extract features, such as edges. In this context, a kernel (or filter) slides across the image, performing element-wise multiplication and summation with the overlapping region at each position.

I implemented two convolution methods from scratch in NumPy:

- **`convolve2d_np_four`**: For each pixel, the kernel is flipped and multiplied with the corresponding image patch using four nested loops, directly computing the convolution sum.
- **`convolve2d_np_two`**: This method improves efficiency by extracting a region from the padded image and computing the convolution sum with the flipped kernel using only two loops.

```
def convolve2d_np_four(i, k):
    h, w = i.shape
    kh, kw = k.shape

    i_pad = np.pad(i, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode='constant', constant_values=0)
    
    k_flip = np.flip(k)

    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            for ki in range(kh):
                for kj in range(kw):
                    out[i, j] += i_pad[i + ki, j + kj] * k_flip[ki, kj]

    return out

def convolve2d_np_two(i, k):
    h, w = i.shape
    kh, kw = k.shape

    i_pad = np.pad(i, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode='constant', constant_values=0)
    
    k_flip = np.flip(k)

    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            roi = i_pad[i:i+kh, j:j+kw] # full pixel-wise convolution
            out[i, j] = np.sum(roi * k_flip)

    return out
```

For clarity, padding is applied to the image to ensure that the convolution operation can be performed at the edges. Adding a border of zeros around the image allows the algorithm to apply the convolution at every pixel in the image. Flipping the kernel is necessary to perform convolution, rather than cross-correlation.


##### Filters

Below are three sample kernels I used to convolve some images. 

<div class="row">
    <div class="col-sm">
        <strong>D<sub>x</sub></strong>
        <p>This finite difference filter detects vertical edges by computing the difference between horizontal pixel intensities.</p>
        $$
        D_x = \begin{bmatrix} 1 & 0 & -1 \end{bmatrix}
        $$
    </div>
    <div class="col-sm">
        <strong>D<sub>y</sub></strong>
        <p>This finite difference filter detects horizontal edges by computing the difference between vertical pixel intensities.</p>
        $$
        D_y = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}
        $$
    </div>
    <div class="col-sm">
        <strong>Box Filter</strong>
        <p>This filter replaces each pixel with the average of its neighbors (blur).</p>
        $$
        	ext{Box Filter} = \frac{1}{81}
        \begin{bmatrix}
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
        \end{bmatrix}
        $$
    </div>
</div>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/cyprian.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/dx_filter_cyp.png" title="dx" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/dy_filter_cyp.png" title="dy" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/box_filter_cyp.png" title="box filter" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">From left to right, the original, D<sub>x</sub> applied to it, then D<sub>y</sub>, then the box filter.</p>


#### Efficiency and Accuracy of Convolution Implementations

Here is a summary of the compute time I observed when performing the 9x9 box filter convolution on the above image, compared to scipy.signal.convolve2d as a benchmark, along with a comparison of differences to the output provided by the SciPy method.

| Implementation | Time Taken (s) | Max Difference to SciPy |
|-----------------|----------------|--------------------------|
| Four loops      | 10.100        | 1.44e-15   |
| Two loops       | 0.956         | 5.55e-16    |
| SciPy           | 0.041        | 0                        |

As can be seen, the maximum differences from the reference function are extremely small, which can be attributed to minor variations in floating-point precision.

---

#### 1.2 Finite Difference Operator

I used finite difference operators to compute the derivative of an image, highlighting regions with large intensity changes (what our eyes perceive as edges). By calculating the gradient magnitude from the $$D_x$$ and $$D_y$$ filters (by simply computing the magnitude of both filtered images combined), I produced a composite edge-detection image.

For example, I first applied the $$D_x$$ and $$D_y$$ filters to the reference image shown on the left.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_dx.png" title="dx" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_dy.png" title="dy" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Original, D<sub>x</sub>, D<sub>y</sub>.</p>

Next, I computed the gradient magnitude between the two convolutions, as seen in the first image below, and manually set a threshold to isolate edges from noise.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_gradient.png" title="Grad mag" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_edges.png" title="Binarized edges" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Gradient magnitude image, binarized edge image.</p>

I selected a pixel intensity threshold of 0.3 to classify edges. My reasoning was that I preferred to preserve edge continuity, even if it meant retaining some noise in the image. This threshold struck a healthy balance, eliminating most of the noise in the grass while almost perfectly maintaining the outline of the subject.

Nevertheless, these edge detections still appear somewhat noisy, so in the next section I explore the use of Gaussian blurs to clean up the output.

---

#### 1.3 Derivative of Gaussian (DoG) Filter

To smooth our binarized edge image, I first applied a 2D Gaussian filter (outer product of a 1D Gaussian) to the original image before applying the $$D_x$$ and $$D_y$$ filters. Applying a Gaussian filter reduces high-frequency noise in the image, which helps the finite difference filters produce cleaner and more continuous edge detections. Below is each step in the procedure.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_blurred.png" title="Blurred" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: original, Right: Gaussian blur applied ($$kernel_size = 15$$, $$\sigma=2$$).</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_dx_blurred.png" title="Dx blurred" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_dy_blurred.png" title="Dy blurred" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: $$D_x$$ filter applied on blurred image, Right: $$D_y$$ filter applied on blurred image.</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_grad_blurred.png" title="Grad mag blurred" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_edge_blurred.png" title="Binarized edge blurred" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Gradient magnitude composite, Right: Binarized edge image.</p>

The order of convolutions can also be changed, by first convolving the Gaussian blur with the finite difference filters to produce a Derivative of Gaussian filter (DoG). This works because convolution is a linear and shift-invariant operation, so the order of convolution can be rearranged. In other words, $$(G * D_x) * I = G * (D_x * I) = (G * D_x) * I$$, where $G$ is the Gaussian filter, $D_x$ is the derivative filter, and $I$ is the image. This principle allows us to combine smoothing and edge detection into one operation, improving both efficiency and results.

**Code:**
```python
# Paste your DoG filter code here
```

**Results:**
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/gaussian.jpg" title="Gaussian" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/dog_dx.jpg" title="DoG Dx" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/dog_dy.jpg" title="DoG Dy" class="img-fluid rounded z-depth-1" %}</div>
</div>

**Discussion:**

---

#### Bells & Whistles: Gradient Orientation Visualization

**Task:** Compute and visualize gradient orientations (HSV color space).

**Code & Results:**
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/orientation.jpg" title="Gradient Orientation" class="img-fluid rounded z-depth-1" %}</div>
</div>

**Discussion:**

---

### Part 2: Fun with Frequencies

#### 2.1 Image Sharpening (Unsharp Mask)

**Task:** Implement unsharp mask filter, show blurred, high-frequency, and sharpened images. Vary sharpening amount.

**Code:**
```python
# Paste your unsharp mask code here
```

**Results:**
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/taj_blur.jpg" title="Blurred" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/taj_highfreq.jpg" title="High Frequency" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/taj_sharp.jpg" title="Sharpened" class="img-fluid rounded z-depth-1" %}</div>
</div>

**Discussion:**

---

#### 2.2 Hybrid Images

**Task:** Create hybrid images, show process for one, originals and results for others.

**Code:**
```python
# Paste your hybrid image code here
```

**Results:**
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/hybrid1.jpg" title="Hybrid Example 1" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/hybrid2.jpg" title="Hybrid Example 2" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/hybrid3.jpg" title="Hybrid Example 3" class="img-fluid rounded z-depth-1" %}</div>
</div>

**Discussion:**

---

#### Bells & Whistles: Color Hybrid Exploration

**Task:** Explore color in hybrid images, justify choices.

**Results & Discussion:**

---

#### 2.3 Gaussian and Laplacian Stacks

**Task:** Implement and visualize stacks for blending.

**Code:**
```python
# Paste your stack code here
```

**Results:**
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/gaussian_stack.jpg" title="Gaussian Stack" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/laplacian_stack.jpg" title="Laplacian Stack" class="img-fluid rounded z-depth-1" %}</div>
</div>

**Discussion:**

---

#### 2.4 Multiresolution Blending (Oraple)

**Task:** Blend images with mask, show results for apple/orange and custom blends.

**Code:**
```python
# Paste your blending code here
```

**Results:**
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/oraple.jpg" title="Oraple" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/custom_blend1.jpg" title="Custom Blend 1" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/custom_blend2.jpg" title="Custom Blend 2" class="img-fluid rounded z-depth-1" %}</div>
</div>

**Discussion:**

---

### Reflection: What I Learned

_Summarize the most important thing you learned from this project._
