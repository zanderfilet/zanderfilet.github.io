---
layout: page
title: Fun With Diffusion Models
description: November 14, 2025
importance: 1
category: CS180
related_publications: false
img: assets/img/cs180/p5/cover.png
---

### Overview

Overview!! Overvieww!!!

---

### Part A: The Power of Diffusion Models

babababa

##### Part 0: Setup

Come up with some interesting text prompts and generate their embeddings.
Choose 3 of your prompts to generate images and display the caption and the output of the model. Reflect on the quality of the outputs and their relationships to the text prompts. Make sure to try at least 2 different num_inference_steps values.
Report the random seed that you're using here. You should use the same seed all subsequent parts.

##### Part 1: Sampling Loops

Overview

##### 1.1: Implementing the Forward Process

Implement the noisy_im = forward(im, t) function
Show the Campanile at noise level [250, 500, 750].

##### 1.2: Classical Denoising

For each of the 3 noisy Campanile images from the previous part, show your best Gaussian-denoised version side by side.

##### 1.3: One-Step Denoising

For the 3 noisy images from 1.2 (t = [250, 500, 750]):
Use your forward function to add noise to your Campanile.
Estimate the noise in the new noisy image, by passing it through stage_1.unet
Remove the noise from the noisy image to obtain an estimate of the original image.
Visualize the original image, the noisy image, and the estimate of the original image

##### 1.4: Iterative Denoising

Using i_start = 10:
Create strided_timesteps: a list of monotonically decreasing timesteps, starting at 990, with a stride of 30, eventually reaching 0. Also initialize the timesteps using the function stage_1.scheduler.set_timesteps(timesteps=strided_timesteps)
Complete the iterative_denoise function
Show the noisy Campanile every 5th loop of denoising (it should gradually become less noisy)
Show the final predicted clean image, using iterative denoising
Show the predicted clean image using only a single denoising step, as was done in the previous part. This should look much worse.
Show the predicted clean image using gaussian blurring, as was done in part 1.2.

##### 1.5: Diffusion Model Sampling

Show 5 sampled images.

##### 1.6: Classifier-Free Guidance (CFG)

Implement the iterative_denoise_cfg function
Show 5 images of "a high quality photo" with a CFG scale of gamma = 7. Now this prompt becomes a condition (but fairly weak) to generate conditional noise! You will use your customized prompts as stronger conditions in part 1.7 - part 1.9.

##### 1.7: Image-to-image Translation

Edits of the Campanile image, using the given prompt at noise levels [1, 3, 5, 7, 10, 20] with the conditional text prompt "a high quality photo"
Edits of 2 of your own test images, using the same procedure.

##### 1.7.1: Editing Hand-Drawn and Web Images

1 image from the web of your choice, edited using the above method for noise levels [1, 3, 5, 7, 10, 20] (and whatever additional noise levels you want)
2 hand drawn images, edited using the above method for noise levels [1, 3, 5, 7, 10, 20] (and whatever additional noise levels you want)

##### 1.7.2: Inpainting

A properly implemented inpaint function
The Campanile inpainted (feel free to use your own mask)
2 of your own images edited (come up with your own mask)
look at the results from this paper for inspiration

##### 1.7.3: Text-Conditional Image-to-image Translation

Edits of the Campanile, using the given prompt at noise levels [1, 3, 5, 7, 10, 20]
Edits of 2 of your own test images, using the same procedure

##### 1.8: Visual Anagrams

Correctly implemented visual_anagrams function
2 illusions of your choice that change appearance when you flip it upside down (feel free to take inspirations from this page).

##### 1.9: Hybrid Images

Correctly implemented make_hybrids function
2 hybrid images of your choosing (feel free to take inspirations from this page).

### Part B: Flow Matching from Scratch

##### Part 1: Training a Single-Step Denoising UNet

##### 1.1: Implementing the UNet

##### 1.2: Using the UNet to Train a Denoiser

A visualization of the noising process using 

##### 1.2.1: Training

A training loss curve plot every few iterations during the whole training process of 
.
Sample results on the test set with noise level 0.5 after the first and the 5-th epoch (staff solution takes ~3 minutes for 5 epochs on a Colab T4 GPU).

##### 1.2.2: Out-of-Distribution Testing

Sample results on the test set with out-of-distribution noise levels after the model is trained. Keep the same image and vary 

##### 1.2.3: Denoising Pure Noise

A training loss curve plot every few iterations during the whole training process that denoises pure noise.
Sample results on pure noise after the first and the 5-th epoch.
A brief description of the patterns observed in the generated outputs and explanations for why they may exist.

##### Part 2: Training a Flow Matching Model

##### 2.1: Adding Time Conditioning to UNet

##### 2.2: Training the UNet

##### 2.3: Sampling from the UNet

A training loss curve plot for the time-conditioned UNet over the whole training process.


##### 2.4: Adding Class-Conditioning to UNet

Sampling results from the time-conditioned UNet for 1, 5, and 10 epochs. The results should not be perfect, but reasonably good.
(Optional for CS180, required for CS280A) Check the Bells and Whistles if you want to make it better!

##### 2.5: Training the UNet

A training loss curve plot for the class-conditioned UNet over the whole training process.

##### 2.6: Sampling from the UNet

Sampling results from the class-conditioned UNet for 1, 5, and 10 epochs. Class-conditioning lets us converge faster, hence why we only train for 10 epochs. Generate 4 instances of each digit as shown above.
Can we get rid of the annoying learning rate scheduler? Simplicity is the best. Please try to maintain the same performance after removing the exponential learning rate scheduler. Show your visualization after training without the scheduler and provide a description of what you did to compensate for the loss of the scheduler.