---
layout: page
title: CS180 - Project 0. Becoming Friends with Your Camera
description: Cyprian Zander
importance: 1
category: CS180
related_publications: false
---

### Part 1: Selfie — The Wrong Way vs. The Right Way

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/1.jpg" title="image1" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/2.jpg" title="image2" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/3.jpg" title="image3" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/4.jpg" title="image4" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/5.jpg" title="image5" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
</div>

<div class="caption text-center mt-2">
  Facial features become increasingly distorted as we move the camera closer to the subject and zoom out.
</div>

<p>
When you hold the camera close to the subject's face, the lens has to use a wide field of view to display the entire face...
</p>

---

### Part 2: Architectural Perspective Compression

<div class="row">
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s1.JPG" title="arch1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s2.JPG" title="arch2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s3.JPG" title="arch3" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s4.JPG" title="arch4" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s5.JPG" title="arch5" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption text-center mt-2">
  Demonstration of perspective compression across two rows of images.
</div>

<p>
When you stand far away and zoom in down a street, the lens narrows its field of view...
</p>

---

### Part 3: Bringing It All Together

<p>
When you move the camera backward while zooming in, you change the perspective but keep the subject about the same size. 
The background appears to stretch and warp around the subject, creating the unsettling “dolly zoom” effect.
</p>

<div class="text-center my-4">
  {% include figure.liquid path="assets/img/cs180/p0/p3_caught_lacking.gif" title="caught_lacking.gif" class="img-fluid rounded z-depth-1 mx-auto d-block" style="max-width: 50%;" %}
</div>

<div class="caption text-center mt-2">
  Just put the fries in the bag, bro
</div>

---

<!-- Custom CSS for thumbnails -->
<style>
.fixed-thumb img {
  height: 200px;
  object-fit: cover;
  width: 100%;
}
</style>
