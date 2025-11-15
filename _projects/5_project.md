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

In this project, I built a pipeline for rendering Neural Radiance Fields from images. First, I calibrated my camera using ArUco markers, estimated poses for a 30–50 images of an object, and combined the intrinsics, poses, and undistorted images into a dataset. I then implemented a NeRF by generating camera rays, sampling points along each ray, encoding them with positional encodings, and using an MLP to predict colors and densities. Finally, I used volume rendering to synthesize new views of the object.

---

### Part 0: Calibrating Your Camera and Capturing a 3D Scan

##### 0.1: Calibrating Your Camera

I calibrated my camera using ArUco markers, capturing 37 images from different angles. For each image, I collected the marker's 4x4 corner The `calibrateCamera` function in OpenCV helped compute the camera intrinsics and distortion coefficients, which would be useful for my own object dataset curation later.

Below are some sample images.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/out_1.jpeg" title="1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/out_2.jpeg" title="1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/out_3.jpeg" title="1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Balcony set.</p>

##### 0.2: Capturing a 3D Object Scan

For my target NeRF reconstruction, I captured around 35 images of a LEGO model from different angles at a consistent distance, keeping the same camera and zoom level as calibration.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/1.jpeg" title="1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/2.jpeg" title="1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/3.jpeg" title="1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

##### 0.3: Estimating Camera Pose

For each image containing the ArUco marker, I solved the Perspective-n-Point (PnP) to estimate the camera pose by matching the detected 2D corner positions with their known 3D coordinates. This returns a rotation vector and translation vector representing transformation between the world and the camera. With the viser library, I was able to visualize all of the poses for each image in the dataset.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/render.jpeg" title="1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p4/part0/render2.jpeg" title="1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

##### 0.4: Undistorting images and creating a dataset

In the final step of the data preparation process, I removed the lens distortion from each image to make sure they match the pinhole camera model that NeRF assumes. To eliminate black borders from undistortion, I computed an undistored camera matrix, cropped the images to the valid ROI, and adjusted the principal point coordinates to account for the crop offset. Finally, I split the undistorted images and their c2w poses into training, validation, and test sets.


### Part 1: Fit a Neural Field to a 2D Image




### Part 2: Fit a Neural Radiance Field from Multi-view Images

##### 2.1: Create Rays from Cameras


##### 2.2: Sampling


##### 2.3: Putting the Dataloading All Together


##### 2.4: Neural Radiance Field


##### 2.5: Volume Rendering


##### 2.6: Training with your own data
