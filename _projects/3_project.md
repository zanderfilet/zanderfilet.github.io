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

This project explores 2D convolutions, filtering, and frequency-based image processing. The deliverables include code, results, and explanations for each section below.

---

### Part 1: Fun with Filters

#### 1.1 Convolutions from Scratch

**Task:** Implement convolution with numpy (4 loops, 2 loops, and compare with scipy). Discuss runtime and boundary handling.

**Code:**
```python
# Paste your convolution implementations here
```

**Results:**
<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/your_image.jpg" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/box_filter.jpg" title="Box Filter Result" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

**Discussion:**

---

#### 1.2 Finite Difference Operator

**Task:** Show partial derivatives in x and y, gradient magnitude, and binarized edge image.

**Code:**
```python
# Paste your finite difference and edge detection code here
```

**Results:**
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/dx.jpg" title="Dx" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/dy.jpg" title="Dy" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/gradient_mag.jpg" title="Gradient Magnitude" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/edge.jpg" title="Binarized Edge" class="img-fluid rounded z-depth-1" %}</div>
</div>

**Discussion:**

---

#### 1.3 Derivative of Gaussian (DoG) Filter

**Task:** Construct Gaussian filters, build DoG filters, visualize and compare results.

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
