---
layout: page
title: CS180 - Project 0. Becoming Friends with Your Camera
description: Cyprian Zander
importance: 1
category: CS180
related_publications: false
---

# Part 1: Selfie — The Wrong Way vs. The Right Way

<div class="row">
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/1.jpg" title="image1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/2.jpg" title="image2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/3.jpg" title="image3" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/4.jpg" title="image4" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/5.jpg" title="image5" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
  Facial features become increasingly distorted as we move the camera closer to the subject and zoom out.
</div>

When you hold the camera close to the subject's face, the lens has to use a wide field of view to display the entire face. Wide angles overemphasize perspective, with features closer to the lens (i.e., a nose) looking disproportionately large, while features farther back (ears, background) look smaller. Since selfies are typically captured from up close, perspective is exaggerated, so the subject looks unnatural.  

At a greater distance and zooming in on the subject, you’re narrowing the field of view. The relative size difference between near and far points on the face and in the background shrinks, so the proportions come out more natural. In other words, you _approach_ an orthogonal projection of the field of view.

**Fun GIF:**  
{% include figure.liquid path="assets/img/cs180/p0/p1_face.gif" title="fun gif" class="img-fluid rounded z-depth-1" %}

---

# Part 2: Architectural Perspective Compression

<div class="row">
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s1.JPG" title="arch1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s2.JPG" title="arch2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s3.JPG" title="arch3" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s4.JPG" title="arch4" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/cs180/p0/s5.JPG" title="arch5" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
  Demonstration of perspective compression in architecture.
</div>

When you stand far away and zoom in down a street, the lens narrows its field of view. This compresses perspective: objects that are actually separated by large distances along the street appear stacked closely together, making the scene look flattened. Buildings, lampposts, and people at different depths seem almost the same size.

When you move closer and use a wide field of view, the perspective exaggerates distance. Objects nearer to you appear much larger relative to those farther away, so the sense of depth is stretched. The spacing between elements down the street feels expanded, giving the scene a more natural three-dimensional look. In other words, zooming from far away visually compresses depth, while moving closer with a wide angle expands it.

---

# Part 3: Bringing It All Together

{% include figure.liquid path="assets/img/cs180/p0/p3_caught_lacking.gif" title="caught_lacking.gif" class="img-fluid rounded z-depth-1" %}

When you move the camera backward while zooming in, you change the perspective but keep the subject about the same size. The background appears to stretch and warp around the subject, creating the unsettling “dolly zoom” effect.

<div class="caption">
  Just put the fries in the bag, lil bro
</div>
