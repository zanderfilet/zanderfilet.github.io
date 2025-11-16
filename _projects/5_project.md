---
layout: page
title: Neural Radiance Fields
description: November 14, 2025
importance: 1
category: CS180
related_publications: false
img: assets/img/cs180/p4/cover.gif
---

### Overview

In this project, I built a pipeline for rendering Neural Radiance Fields from images. First, I calibrated my camera using ArUco markers, estimated poses for a 30–50 images of an object, and combined the intrinsics, poses, and undistorted images into a dataset. I then implemented a NeRF by generating camera rays, sampled points along each ray, encoded them with positional encodings, and used an MLP to predict colors and densities. Finally, I used volume rendering to synthesize new views of the object.

---

### Part 0: Calibrating Your Camera and Capturing a 3D Scan

##### 0.1: Calibrating Your Camera

I calibrated my camera using ArUco markers, capturing 37 images from different angles. For each image, I collected the marker's 4x4 corner The `calibrateCamera` function in OpenCV helped compute the camera intrinsics and distortion coefficients, which would be useful for my own object dataset curation later.

Below are some sample images.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/out_1.jpeg" title="Configuration Set Sample 1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/out_2.jpeg" title="Configuration Set Sample 2" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/out_3.jpeg" title="Configuration Set Sample 3" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

##### 0.2: Capturing a 3D Object Scan

For my target NeRF reconstruction, I captured around 35 images of a LEGO model from different angles at a consistent distance, keeping the same camera and zoom level as the calibration dataset.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/1.jpeg" title="Dataset Sample 1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/2.jpeg" title="Dataset Sample 2" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/3.jpeg" title="Dataset Sample 3" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

##### 0.3: Estimating Camera Pose

For each image containing the ArUco marker, I solved the Perspective-n-Point (PnP) to estimate the camera pose by matching the detected 2D corner positions with their known 3D coordinates. This returns a rotation vector and translation vector representing transformation between the world and the camera. With the viser library, I visualized all of the poses for each image in the dataset.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/render.jpeg" title="Poses View 1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/render2.jpeg" title="Poses View 2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

##### 0.4: Undistorting images and creating a dataset

In the final step of the data preparation process, I removed the lens distortion from each image to make sure they match the pinhole camera model that NeRF assumes. To eliminate black borders from undistortion, I computed an undistored camera matrix, cropped the images to the valid ROI, and adjusted the principal point coordinates to account for the crop offset. Finally, I split the undistorted images and their c2w poses into training, validation, and test sets.


### Part 1: Fit a Neural Field to a 2D Image

Before working with 3D Neural Radiance Fields, I first implemented a 2D neural field. For this, I built an MLP that takes normalized 2D pixel coordinates as input and outputs RGB color values. To capture high-frequency details, I applied sinusoidal positional encoding to the input coordinates, expanding them from 2D to higher dimensions: $\gamma(p) = [p, \sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)]$ where $L$ is the maximum frequency level. The network was trained using MSE loss between predicted and ground truth colors, optimized with Adam and evaluated using the Peak Signal to Noise Ratio (PSNR).

The neural field consists of a 4-layer MLP of variable width (I tested 64 and 256 channels), ReLU activations between layers, and a final Sigmoid activation. I trained for 1000 iterations with a batch size of 10,000 randomly sampled pixels per iteration, using Adam optimizer with learning rate $0.01$.

<div class="col-sm">
    {% include figure.liquid path="assets/img/cs180/p4/part1/mlp_img.jpg" title="2D Neural Field Architecture" class="img-fluid rounded z-depth-1" %}
</div>

Here you can see some examples of the neural field inference across increasing iterations.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal.jpg" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal_0.jpg" title="0 iters" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal_100.jpg" title="100 iters" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal_900.jpg" title="900 iters" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Original, 0 iterations, 100 iterations, 900 iterations.</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal_psnr_curve.jpg" title="Animal PSNR" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

Below is a grid displaying fully trained results from tweaking positional encoding frequencies and network widths.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal_freq2_width64_final.jpg" title="freq 2, width 64" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal_freq2_width256_final.jpg" title="freq 2, width 256" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">max_freq=2; left, width=64, right width=256</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal_freq10_width64_final.jpg" title="freq 10, width 64" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/animal_freq10_width256_final.jpg" title="freq 10, width 256" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">max_freq=10; left, width=64, right width=256</p>

With a low frequency encoding, the reconstruction loses high-frequency details and appears blurry. With narrow linear layers (width=64), the model struggles to capture fine details even with high-frequency encoding. The best results come from max_freq=10 with width=256, achieving a sharp reconstruction of the original image.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/nerf.jpeg" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/nerf_0.jpg" title="0 iters" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/nerf_100.jpg" title="100 iters" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part1/nerf_900.jpg" title="900 iters" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">It's NeRF or Nothing.</p>

### Part 2: Fit a Neural Radiance Field from Multi-view Images

##### 2.1: Create Rays from Cameras



##### 2.2: Sampling


##### 2.3: Putting the Dataloading All Together


##### 2.4: Neural Radiance Field


##### 2.5: Volume Rendering


##### 2.6: Training with your own data
