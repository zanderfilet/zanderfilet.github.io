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
Emir of Bukhara (1911), sample original glass plate (blue, green, red from top to bottom)
</div>

Before running our alignment algorithms, we simply divide the raw digitzed negative into thirds, since this approximately correctly divides each color filter into their respective color filter category.

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
<div class="caption text-center mt-3">
Blue, green, red, respectively
</div>

---

### Part 1: Alignment Metrics

Our first approach focussed on minimizing the L2 Norm between a reference filter (green) to the other two filters. This metric simply evaluates the aggregate Euclidean distance between the reference and the target filter, which generally works well, but is highly sensitive to brightness differences between channels (i.e., matching based on the Emir's robe above is not great for L2).

Our second approach focussed on maximizing normalized cross-correlation (NCC), which is more robust, since it is invariant to linear changes in brightness and contrast. This method was generally more effective, since different color filters produce varying intensities.

##### L2 Norm

$$
E(\Delta x, \Delta y) = 
\sum_{x,y} \Big( R(x,y) - F(x+\Delta x, y+\Delta y) \Big)^2
$$

We minimize $E(\Delta x, \Delta y)$ to find the displacement $(\Delta x, \Delta y)$ where the shifted filter $F$ best overlaps with the reference $R$.

##### Normalized Cross-Correlation (NCC)

$$
\text{NCC}(\Delta x, \Delta y) = 
\frac{\sum_{x,y} \big(R(x,y) - \bar{R}\big)\big(F(x+\Delta x, y+\Delta y) - \bar{F}\big)}
     {\sqrt{\sum_{x,y} \big(R(x,y) - \bar{R}\big)^2} \,
      \sqrt{\sum_{x,y} \big(F(x+\Delta x, y+\Delta y) - \bar{F}\big)^2}}
$$

We maximize $\text{NCC}(\Delta x, \Delta y)$ to find the displacement that yields the strongest correlation between $R$ and $F$, regardless of brightness or contrast differences.

##### Variables

- **$R(x,y)$:** Reference filter/channel (kept fixed, e.g. green).  
- **$F(x+\Delta x, y+\Delta y)$:** Filter/channel being aligned (shifted version of red or blue).  
- **$(\Delta x, \Delta y)$:** Displacement vector we are solving for.  
- **$\bar{R}, \bar{F}$:** Mean pixel intensities of $R$ and $F$, used for normalization in NCC.  

---

### Small Image Examples

Here are some first outputs I achieved with these two approaches.

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/monastery_pyramid_L2_level4.jpg" title="Monastery Pyramid, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Monastery, L2 Norm
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
      Tobolsk, L2 Norm
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
      Cathedral, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/cathedral_pyramid_NCC_level4.jpg" title="NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

Below are the filter displacements found by both approaches.

| Image Name | L2 Blue Displacement | L2 Green Displacement | L2 Red Displacement | NCC Blue Displacement | NCC Green Displacement | NCC Red Displacement |
|------------|---------------------|----------------------|-------------------|---------------------|----------------------|-------------------|
| monastery | (3, -2) | (0, 0) | (6, 1) | (3, -2) | (0, 0) | (6, 1) |
| tobolsk | (-3, -3) | (0, 0) | (4, 1) | (-3, -3) | (0, 0) | (4, 1) |
| cathedral | (-5, -2) | (0, 0) | (7, 1) | (-5, -2) | (0, 0) | (7, 1) |

---

### Part 3: Pyramid Search

Our next challenge is that searching for the correct alignment on large images by checking every possible shift is too slow. To speed this up, I use an image pyramid. This approach starts by finding a rough alignment on a small, downscaled version of the image, which is very fast. This rough alignment is then used to guide a more focused search on a larger, higher-resolution version. By repeating this process from coarse to fine, I can quickly and accurately align the full-resolution image, even when the initial misalignment is large.

The process begins by applying a Gaussian blur with \(\sigma = 1\) to the image. This is an anti-aliasing filter to reduce artifacts when the image is downscaled. I used a rescaling factor of 0.5, meaning each level of the pyramid is half the width and height of the one above it, and all larger images in the dataset were constructed with 4 levels. The alignment starts at the smallest scale, performing a search within a +/- 15 pixel displacement range in both x and y to find a coarse alignment. This calculated displacement is then doubled and propagated to the next higher-resolution level, where it is used to shift the image. A search with a smaller displacement range of +/- 2 pixels then refines the alignment. This process of scaling the displacement and performing a fine-tuned local search is repeated until the original, full-resolution image is reached.

##### Pyramid search time optimization

<div class="my-4">
  {% include figure.liquid path="assets/img/cs180/p1/optimization_graph.png" title="Time improvement graph" class="img-fluid rounded z-depth-1" %}
</div>

<div class="caption text-center mt-2">
  Time complexity comparison showing the speed improvement of pyramid search across levels with abovementioned hyperparameters. Level 2 onwards converges to an accurate displacement result.
</div>

---

### Full Gallery

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/church_pyramid_L2_level4.jpg" title="Church, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Church, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/church_pyramid_NCC_level4.jpg" title="Church NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/emir_pyramid_L2_level4.jpg" title="Emir, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Emir, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/emir_pyramid_NCC_level4.jpg" title="Emir NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/harvesters_pyramid_L2_level4.jpg" title="Harvesters, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Harvesters, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/harvesters_pyramid_NCC_level4.jpg" title="Harvesters NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/icon_pyramid_L2_level4.jpg" title="Icon, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Icon, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/icon_pyramid_NCC_level4.jpg" title="Icon NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/italil_pyramid_L2_level4.jpg" title="Italil, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Italil, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/italil_pyramid_NCC_level4.jpg" title="Italil NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/lastochikino_pyramid_L2_level4.jpg" title="Lastochikino, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Lastochikino, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/lastochikino_pyramid_NCC_level4.jpg" title="Lastochikino NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/lugano_pyramid_L2_level4.jpg" title="Lugano, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Lugano, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/lugano_pyramid_NCC_level4.jpg" title="Lugano NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/melons_pyramid_L2_level4.jpg" title="Melons, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Melons, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/melons_pyramid_NCC_level4.jpg" title="Melons NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/self_portrait_pyramid_L2_level4.jpg" title="Self Portrait, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Self Portrait, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/self_portrait_pyramid_NCC_level4.jpg" title="Self Portrait NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/siren_pyramid_L2_level4.jpg" title="Siren, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Siren, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/siren_pyramid_NCC_level4.jpg" title="Siren NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/three_generations_pyramid_L2_level4.jpg" title="Three Generations, L2 Norm" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      Three Generations, L2 Norm
    </div>
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/three_generations_pyramid_NCC_level4.jpg" title="Three Generations NCC" class="img-fluid rounded z-depth-1" %}
    <div class="caption text-center mt-2">
      NCC
    </div>
  </div>
</div>

Below are the filter displacements found by both approaches.

| Image Name | L2 Blue Displacement | L2 Green Displacement | L2 Red Displacement | NCC Blue Displacement | NCC Green Displacement | NCC Red Displacement |
|------------|---------------------|----------------------|-------------------|---------------------|----------------------|-------------------|
| emir | (-49, -24) | (0, 0) | (57, 17) | (-49, -24) | (0, 0) | (57, 17) |
| italil | (-38, -21) | (0, 0) | (39, 15) | (-38, -21) | (0, 0) | (39, 15) |
| church | (-25, -4) | (0, 0) | (33, -8) | (-25, -4) | (0, 0) | (33, -8) |
| three_generations | (-53, -14) | (0, 0) | (59, -3) | (-53, -14) | (0, 0) | (59, -3) |
| lugano | (-41, 16) | (0, 0) | (53, -13) | (-41, 16) | (0, 0) | (52, -13) |
| melons | (-82, -11) | (0, 0) | (96, 3) | (-82, -11) | (0, 0) | (96, 3) |
| lastochikino | (270, 270) | (0, 0) | (78, -7) | (3, 2) | (0, 0) | (78, -7) |
| icon | (-41, -17) | (0, 0) | (48, 5) | (-41, -17) | (0, 0) | (48, 5) |
| siren | (-50, 6) | (0, 0) | (47, -19) | (-49, 6) | (0, 0) | (47, -19) |
| self_portrait | (-79, -29) | (0, 0) | (98, 8) | (-79, -29) | (0, 0) | (98, 8) |
| harvesters | (-59, -17) | (0, 0) | (65, -3) | (-59, -17) | (0, 0) | (65, -3) |

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

### Bells & Whistles

##### Automatic Cropping
[PLACEHOLDER: Method summary + before/after thumbnails.]

##### Automatic Contrast
[PLACEHOLDER: Method summary + before/after thumbnails.]


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

