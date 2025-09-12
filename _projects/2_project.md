---
layout: page
title: CS180 - Project 1. Colorizing Prokudin-Gorskii
description: Cyprian Zander
importance: 1
category: CS180
related_publications: false
img: assets/img/cs180/p1/cover.jpg
---

### Background

[PLACEHOLDER: Brief historical note on Prokudin-Gorskii and RGB glass plates — 2–3 sentences max.]

---

### Overview

[PLACEHOLDER: One-paragraph summary of the assignment, goals, and constraints.]


---

### Part 1: Channel Extraction (B/G/R Slices)

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/extract_full.jpg" title="Original glass plate (BGR stacked)" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/extract_b.jpg" title="B channel (top third)" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/extract_g.jpg" title="G channel (middle third)" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p1/extract_r.jpg" title="R channel (bottom third)" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
</div>

<div class="caption text-center mt-2">
  [PLACEHOLDER: One-line caption about slicing the plate into equal thirds.]
</div>

[PLACEHOLDER: Notes on cropping borders/margins used for matching window.]

---

### Part 2: Matching Metrics

- [PLACEHOLDER: L2 / SSD definition.]
- [PLACEHOLDER: NCC definition.]
- [PLACEHOLDER: Edge/gradient-based similarity (Sobel/Canny) — bullet placeholder.]
- [PLACEHOLDER: Robustness to brightness/channel differences — bullet placeholder.]

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
