---
layout: page
title: CS180 - Project 1. Colorizing Prokudin-Gorskii
description: Cyprian Zander
importance: 1
category: CS180
related_publications: false
img: assets/img/cs180/p1/cover.jpg
---

### Overview

This goal of this project is to automatically reconstruct color photographs from the filtered glass plate negatives of Sergei Mikhailovich Prokudin-Gorskii, an early 20th century photographer. Each negative contains three separate exposures of images taken through blue, green, and red filters (from top to bottom). Since I need to stitch these components together, the primary challenges of this project consists of identifying how each exposure is displaced to correctly align each color filter. Later, I extend on this algorithm with some optimizations, color correction, and cropping in postprocessing. 

---

### Input and preprocessing

<div class="text-center my-4">
  <div class="row justify-content-center">
    <div class="col-sm-6">
      {% include figure.liquid path="assets/img/cs180/p1/emir_in.jpg" title="Sample original glass plate (blue, green, red from top to bottom)" class="img-fluid rounded z-depth-1" %}
    </div>
  </div>
</div>

<div class="caption text-center mt-2">
Sample original glass plate (blue, green, red from top to bottom)
</div>

Before running our alignment algorithms, we simply divide the raw digitzed negative into thirds, since this approximately correctly divides each color filter a respective 2D filter.

<div class="row">
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/emir_01_blue_channel.jpg" title="Blue channel" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/emir_02_green_channel.jpg" title="Green channel" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/emir_03_red_channel.jpg" title="Red channel" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

---

### Part 1: Alignment Metrics

Our first approach focussed on minimizing the L2 Norm between a reference filter (green) to the other two filters. This metric simply evaluates the aggregate Euclidean distance between the reference and the target filter, which generally works well, but is highly sensitive to brightness differences between channels (i.e., matching based on the Emir's robe above is not great for L2).

Our second approach focussed on maximizing normalized cross-correlation (NCC), which is more robust, since it is invariant to linear changes in brightness and contrast. This method was generally more effective, since different different color filters naturally produce varying intensities.

##### L2 Norm

$$
E(\Delta x, \Delta y) = 
\sum_{x,y} \Big( R(x,y) - F(x+\Delta x, y+\Delta y) \Big)^2
$$

We minimize \(E(\Delta x, \Delta y)\) to find the displacement \((\Delta x, \Delta y)\) where the shifted filter \(F\) best overlaps with the reference \(R\).

##### Normalized Cross-Correlation (NCC)

$$
\text{NCC}(\Delta x, \Delta y) = 
\frac{\sum_{x,y} \big(R(x,y) - \bar{R}\big)\big(F(x+\Delta x, y+\Delta y) - \bar{F}\big)}
     {\sqrt{\sum_{x,y} \big(R(x,y) - \bar{R}\big)^2} \,
      \sqrt{\sum_{x,y} \big(F(x+\Delta x, y+\Delta y) - \bar{F}\big)^2}}
$$

We maximize \(\text{NCC}(\Delta x, \Delta y)\) to find the displacement that yields the strongest correlation between \(R\) and \(F\), regardless of brightness or contrast differences.

##### Variables

- **\(R(x,y)\):** Reference filter/channel (kept fixed, e.g. green).  
- **\(F(x+\Delta x, y+\Delta y)\):** Filter/channel being aligned (shifted version of red or blue).  
- **\((\Delta x, \Delta y)\):** Displacement vector we are solving for.  
- **\(\bar{R}, \bar{F}\):** Mean pixel intensities of \(R\) and \(F\), used for normalization in NCC.  

---

### First Examples

Here are some first outputs I achieved with these two approaches.

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/monastery_pyramid_L2_level4.jpg" title="Monastery Pyramid, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      L2 Norm alignment result
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/monastery_pyramid_NCC_level4.jpg" title="NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/tobolsk_pyramid_L2_level4.jpg" title="L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Tobolsk Pyramid, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/tobolsk_pyramid_NCC_level4.jpg" title="NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/cathedral_pyramid_L2_level4.jpg" title="L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Cathedral pyramid, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/cathedral_pyramid_NCC_level4.jpg" title="NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

---

### Part 3: Exhaustive Search (Single Scale)

[PLACEHOLDER: Search window, step size, tie-breaking, boundary handling, interior mask.]

<div class="text-center my-4">
  {% include figure.liquid path="assets/img/cs180/p1/singlescale_demo.gif" title="Single-scale alignment demo (placeholder)" class="img-fluid rounded z-depth-1 mx-auto d-block" style="max-width: 60%;" %}
</div>

<div class="caption text-center mt-2">
  [PLACEHOLDER: One-line caption describing the search visualization.]
</div>

---

### Part 4: Image Pyramid (Coarse-to-Fine)

- [PLACEHOLDER: Pyramid construction (downsample ×2).]
- [PLACEHOLDER: Recursive refine from coarsest to finest.]
- [PLACEHOLDER: Window size per level, termination criteria.]
- [PLACEHOLDER: Runtime/computation notes.]

<div class="row">
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/pyr_lvl3.jpg" title="Level 3 (coarsest)" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/pyr_lvl2.jpg" title="Level 2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/pyr_lvl1.jpg" title="Level 1 (finest)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption text-center mt-2">
  [PLACEHOLDER: Pyramid levels used for alignment.]
</div>

---

### Results on Provided Images

[PLACEHOLDER: Brief sentence about showing all images and offsets.]

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/monastery_color.jpg" title="monastery (colorized)" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/monastery_split.jpg" title="aligned channels overlay (monastery)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/cathedral_color.jpg" title="cathedral (colorized)" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/cathedral_split.jpg" title="aligned channels overlay (cathedral)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

[REPEAT BLOCKS AS NEEDED FOR: tobolsk, emir, train, harvesters, lady, icon, self_portrait, three_generations, etc.]

---

### Alignment Offsets (G→B and R→B)

[PLACEHOLDER: Replace with your actual offsets.]

| Image              | G Offset (x, y) | R Offset (x, y) | Metric | Pyramid Levels | Runtime (s) |
|-------------------:|:---------------:|:---------------:|:------:|:--------------:|:-----------:|
| monastery.jpg      | (__, __)        | (__, __)        | NCC    | 1              | __.__       |
| cathedral.jpg      | (__, __)        | (__, __)        | NCC    | 1              | __.__       |
| emir.tif           | (__, __)        | (__, __)        | EDGE   | 4              | __.__       |


---

### Additional Images From LoC Collection

<div class="row">
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/loc_1_color.jpg" title="LoC sample 1 (colorized)" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/loc_2_color.jpg" title="LoC sample 2 (colorized)" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/loc_3_color.jpg" title="LoC sample 3 (colorized)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption text-center mt-2">
  [PLACEHOLDER: One-line caption about non-handout examples.]
</div>

---

### Bells & Whistles

#### Automatic Cropping
[PLACEHOLDER: Method summary + before/after thumbnails.]

#### Automatic Contrast
[PLACEHOLDER: Method summary + before/after thumbnails.]

#### Automatic White Balance
[PLACEHOLDER: Method summary + before/after thumbnails.]

#### Better Color Mapping
[PLACEHOLDER: Brief idea + example.]

#### Edge/Gradient Features
[PLACEHOLDER: Metric swap details + result.]

#### Small Rotations/Scale
[PLACEHOLDER: Search extension + runtime note.]

#### Other Sources (e.g., Astronomy)
[PLACEHOLDER: Example + alignment/write-up note.]
