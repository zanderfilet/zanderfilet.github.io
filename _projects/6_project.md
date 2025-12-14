---
layout: page
title: Fun With Diffusion Models
description: December 12, 2025
importance: 1
category: CS180
related_publications: false
img: assets/img/cs180/p5/cover.png
---

### Overview

In this project, I implemented generative models for image synthesis and denoising. In the first stage of the project, I worked with pretrained diffusion models, by using a DeepFloyd IF model with my own generated prompt embeddings. Using the pretrained model, I wrote sampling loops to produce images from noise, then extended the work to test out inpainting methods and optical illusions (like the visual anagram in this project's cover picture). Later, I built generative models from scratch by training neural networks on MNIST. One of these models is a single-step denoising UNet trained to map noisy images to clean digits, which I then extended to a flow-matching model conditioned on noisy inputs and timesteps. I implemented the UNet architecture using down- and upsampling blocks with a loss objective to predict flow between noise and data distributions.

---

### Part A: The Power of Diffusion Models

##### Part 0: Setup

To start out, I tested some prompt embeddings on the pretrained DeepFloyd model. The following text prompts were used to generate text embeddings:

```text
a high quality picture
an oil painting of a snowy mountain village
a photo of the amalfi coast
a photo of a man
a photo of a hipster barista
a photo of a dog
an oil painting of people around a campfire
an oil painting of an old man
a lithograph of waterfalls
a lithograph of a skull
a man wearing a hat
a high quality photo
a rocket ship
a pencil
long-stemmed flowers strewn on the hood of a classic Porsche
a family of four sitting in a ski lift
a green tennis court```.

The last three prompts are my own.

I used a random seed (101) and upsampled the images from 64x64 to 256x256 using DeepFloyd's pretrained stage II super-resolution model. For each of the below images, the output is the sampled output from 20, 50, and 100 steps from left to right.

###### Sampling Results

__Prompt 1: a lithograph of waterfalls__

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/1.png" title="waterfall_20" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/5.png" title="waterfall_50" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/9.png" title="waterfall_100" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

Across all of these images, the waterfalls are clearly recognizable, although adherence to the lithographic style is rather somewhat limited. Despite none of these samples reminding of true lithographies, the samples with higher step counts do seem to approach the idea of a drawing a more closely than the 20 step count image, which reminds more of an animated picture.


__Prompt 2: long-stemmed flowers strewn on the hood of a classic Porsche__

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/2.png" title="p_20" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/6.png" title="p_50" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/10.png" title="p_100" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

All of these renderings clearly display a Porsche, as indicated by the headlights and logo. However, the second iteration seemed to fail with the task of placing the flowers on the hood of the car and instead generated a flowery hedge behind the car. It seems that the 100 iteration image best adheres to the prompt, since it also incorporates the detail that the flowers are supposed to be long-stemmed. Curiously, all samples imagined a pale green vehicle, potentially hinting at some bias in the dataset regarding the green surroundings flowers are typically found in.

__Prompt 3: a family of four sitting in a ski lift__

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/3.png" title="f_20" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/7.png" title="f_50" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/11.png" title="f_100" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

These images also closely adhere to the prompt provided, although only the 100 step example gets the number of people in the ski lift correct. Also notable is that all people were generated with helmets on, but none are correspondingly wearing skis or snowboards.

__Prompt 4: a green tennis court__
<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/4.png" title="t_20" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/8.png" title="t_50" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part0/12.png" title="t_100" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

In this example, it seems that the model struggled with the combination of green coloring of a tennis court, presumably because tennis courts typically are not green, so the model had issues with correctly displaying the lines on a tennis court and the placement of the net.

##### Part 1: Sampling Loops

##### 1.1: Implementing the Forward Process

In the first stage of implementing the diffusion model, I defined the forward diffusion process. The forward diffusion process gradually transforms a clean image \(x_0\) into noise by scaling the signal and adding Gaussian noise. At timestep \(t\), the image \(x_t\) is sampled by

\[
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t)\mathbf{I}\right),
\]

which is equivalent to

\[
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon,
\quad \varepsilon \sim \mathcal{N}(0, \mathbf{I}).
\]

where \(\bar{\alpha}_t\) is the cumulative product of the noise schedule up to timestep \(t\). As \(t\) increases, \(\bar{\alpha}_t \to 0\), and \(x_t\) becomes dominated by Gaussian noise.

You can see my code for my implementation of the ```forward(im, t)````. 

Here is the Berkeley campanile at noise level [250, 500, 750].

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/a.png" title="250" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/b.png" title="500" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part2/c.png" title="750" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center"></p>

##### 1.2: Classical Denoising

To evaluate how difficult diffusion denoising is, I first applied a classical Gaussian blur to noisy images at timesteps \(t \in \{250, 500, 750\}\). Gaussian filtering smooths high-frequency noise but cannot recover lost structure, so as the noise level increases, the images become increasingly blurred without meaningful reconstruction.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/a1.png" title="c250" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/a2.png" title="c250d" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">\(t=250)</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/b1.png" title="c500" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/b2.png" title="c500d" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">\(t=500)</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/c1.png" title="c750" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/c2.png" title="c750d" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">\(t=750)</p>

##### 1.3: One-Step Denoising

For the 3 noisy images from 1.2 (t = [250, 500, 750]):
Use your forward function to add noise to your Campanile.
Estimate the noise in the new noisy image, by passing it through stage_1.unet
Remove the noise from the noisy image to obtain an estimate of the original image.
Visualize the original image, the noisy image, and the estimate of the original image

In this step, I used `stage_1.unet`, a pretrained, timestep-conditioned UNet, to estimate the Gaussian noise present in a noisy image. Given a noisy image \(x_t\), the timestep \(t\), and a text prompt embedding, the UNet predicts the noise \(\varepsilon\) that was added during the forward process, which I then appropriately scaled and removed to recover an estimate of the original image \(x_0\).

Below is the original campanile, the noisy campanile at variable $t$ noise levels, and the corresponding one-step denoised campanile using the model.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/a1.png" title="c250" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/a2.png" title="c250d" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/a3.png" title="c250c" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">\(t=250)</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/b1.png" title="c500" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/b2.png" title="c500d" class="img-fluid rounded z-depth-1" %}
    </div>
        <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/b3.png" title="c500c" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">\(t=500)</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/c1.png" title="c750" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/c2.png" title="c750d" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/c3.png" title="c750c" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">\(t=750)</p>

As we can see, the denoising UNet is much better at projecting the image onto the natural image manifold, but the sharpness worsens with more noise.

##### 1.4: Iterative Denoising

One-step denoising tries to recover a clean image \(x_0\) directly from a noisy sample \(x_t\), but diffusion models are designed to denoise gradually by repeatedly stepping from a noisier timestep to a less noisy one. Because running all \(T=1000\) steps is expensive, I used a strided schedule and denoised only at those timesteps.

At each iteration we have an image \(x_t\) at timestep \(t=\texttt{strided\_timesteps}[i]\), and we want to move to a less noisy timestep \(t'=\texttt{strided\_timesteps}[i+1]\). I first used the pretrained `stage_1.unet` (conditioned on \(t\) and the text embedding) to estimate the noise \(\hat{\varepsilon}\), then form an estimate of the clean image:

\[
\hat{x}_0 \;=\; \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\varepsilon}}{\sqrt{\bar{\alpha}_t}}.
\]

Then we update toward \(t'\) using the iterative denoising rule, which is like linearly interpolating between signal and noise:

\[
x_{t'} \;=\; \sqrt{\bar{\alpha}_{t'}}\,\hat{x}_0 \;+\; \sqrt{1-\bar{\alpha}_{t'}}\,\hat{\varepsilon} \;+\; \sigma(t,t')\,z,
\]

where \(z\sim \mathcal{N}(0,\mathbf{I})\). Repeating this update over the timesteps gradually projects the sample onto the natural image manifold, producing a much cleaner result than single-step denoising or Gaussian blur.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/690.png" title="t690" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/540.png" title="t540" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/390.png" title="t390" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/240.png" title="t240" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/90.png" title="t90" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">From left to right, iterative denoising at timesteps \(690, 540, 390, 240, 90)</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/og.png" title="og" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/iter.png" title="iter" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/1s.png" title="1s" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/gb.png" title="gb" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Original, iterative, one-step, original with Gaussian blur</p>

##### 1.5: Diffusion Model Sampling

Diffusion models can also be used as generative models by starting from pure Gaussian noise and iteratively denoising it. By setting `i_start = 0` and initializing `im_noisy` with random noise, the iterative denoising process progressively removes noise according to the learned diffusion dynamics, guided by the text prompt embedding. This procedure transforms unstructured noise into a coherent image consistent with the prompt. In this example, I used the prompt embedding for “a high quality photo,” demonstrating how diffusion models generate images from scratch.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/d1.png" title="c750" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/d2.png" title="c750d" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/d3.png" title="c750c" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/d4.png" title="c750c" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p5/part1/d5.png" title="c750c" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Five generated samples from the prompt 'a high quality photo'</p>

##### 1.6: Classifier-Free Guidance (CFG)

The unguided samples from the previous section look a bit nonsensical because, without strong conditioning, the denoising trajectory is only loosely constrained. Many different images are plausible under the model’s prior, and small errors in the predicted noise compound across iterative steps. With a weak prompt like the previous one (or effectively “null” conditioning), the model often drifts toward generic textures or unstable compositions instead of a coherent natural image.

Classifier-Free Guidance fixes this by explicitly steering the denoiser using both an unconditional and a conditional noise prediction. At each timestep I ran the UNet twice to get \(\varepsilon_u\), a noise estimate with the unconditional prompt `""` and \(\varepsilon_c\), noise estimate with the conditional prompt embedding. Then, these can be combined using the CFG rule:

\[
\varepsilon \;=\; \varepsilon_u + \gamma(\varepsilon_c - \varepsilon_u),
\]

where \(\gamma\) controls guidance strength. When \(\gamma > 1\), the update amplifies the direction that moves the sample toward satisfying the text condition, producing more coherent images at the cost of reduced diversity. This guided \(\varepsilon\) is then used in the same iterative denoising update as before.



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