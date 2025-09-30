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

This project explores edge detection with finite differences and Gaussian smoothing, image sharpening with unsharp masking, hybrid images by mixing high and low frequencies, and multi-resolution blending with Gaussian and Laplacian stacks for seamless composites.

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

Below are three sample kernels I use to convolve some images. 

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
        {% include figure.liquid path="assets/img/cs180/p2/11/dx_filter_cyp.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/dy_filter_cyp.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/box_filter_cyp.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">From left to right, the original, D<sub>x</sub> applied to it, then D<sub>y</sub>, then the box filter.</p>


#### Efficiency and Accuracy of Convolution Implementations

Here is a summary of compute time to perform the 9x9 box filter convolution on the above image, compared to scipy.signal.convolve2d as a benchmark, along with a comparison of differences to the output provided by the SciPy method.

| Implementation | Time Taken (s) | Max Difference to SciPy |
|-----------------|----------------|--------------------------|
| Four loops      | 10.100        | 1.44e-15   |
| Two loops       | 0.956         | 5.55e-16    |
| SciPy           | 0.041        | 0                        |

As we can see, the maximum differences from the reference function are extremely small, which can be attributed to minor variations in floating-point precision.

---

#### 1.2 Finite Difference Operator

Finite difference operators are used to compute the derivative of an image, highlighting regions with large intensity changes (what our eyes perceive as edges). By calculating the gradient magnitude from the $$D_x$$ and $$D_y$$ filters (by simply computing the magnitude of both filtered images combined), we can produce a composite edge-detection image.

For example, let's apply the $$D_x$$ and $$D_y$$ filters to the reference image shown on the left.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_dx.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_dy.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Original, D<sub>x</sub>, D<sub>y</sub>.</p>

Now, we compute the gradient magnitude and manually 

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Justification for threshold

Comment on noise

---

#### 1.3 Derivative of Gaussian (DoG) Filter

The Derivative of Gaussian (DoG) filter is used for edge detection. It is obtained by taking the derivative of the Gaussian function, emphasizing regions of rapid intensity change.

**Task:** Construct Gaussian filters, build DoG filters, visualize and compare results.

**Code:**
```python
# Paste your DoG filter code here
```

**Results:**
<div class="row">
    <div class="col-sm">{% include fignure.liquid path="assets/img/cs180/p2/gaussian.jpg" title="Gaussian" class="img-fluid rounded z-depth-1" %}</div>
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

#### Bells & Whistles: Color Blending

**Task:** Try color blending, show and discuss results.

---

### Deliverables Checklist

- [ ] Part 1.1: Convolution implementations and comparison
- [ ] Part 1.2: Partial derivatives, gradient magnitude, binarized edge
- [ ] Part 1.3: Gaussian and DoG filters, comparison
- [ ] Bells & Whistles: Gradient orientation visualization
- [ ] Part 2.1: Unsharp mask filter, results, discussion
- [ ] Part 2.2: Hybrid images, process, results
- [ ] Bells & Whistles: Color hybrid exploration
- [ ] Part 2.3: Gaussian and Laplacian stacks, visualization
- [ ] Part 2.4: Multiresolution blending, custom blends
- [ ] Bells & Whistles: Color blending

---

### Reflection: What I Learned

_Summarize the most important thing you learned from this project._
