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

In this project, I explored edge detection with finite differences and Gaussian smoothing, image sharpening with unsharp masking, hybrid images by mixing high and low frequencies, and multi-resolution blending with Gaussian and Laplacian stacks for seamless composites.

---

### Part 1: Fun with Filters

#### 1.1 Convolutions from Scratch

Convolutions are mathematical operations applied to images to extract features, such as edges. In this context, a kernel (or filter) slides across the image, performing element-wise multiplication and summation with the overlapping region at each position.

I implemented two convolution methods from scratch in NumPy:

- **`convolve2d_np_four`**: For each pixel, the kernel is flipped and multiplied with the corresponding image patch using four nested loops, directly computing the convolution sum.
- **`convolve2d_np_two`**: This method improves efficiency by extracting a region from the padded image and computing the convolution sum with the flipped kernel using only two loops.

```
def convolve2d_np_four(i, k):
    h, w = i.shape
    kh, kw = k.shape

    i_pad = np.pad(i, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode='constant', constant_values=0)
    
    k_flip = np.flip(k)

    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            for ki in range(kh):
                for kj in range(kw):
                    out[i, j] += i_pad[i + ki, j + kj] * k_flip[ki, kj]

    return out

def convolve2d_np_two(i, k):
    h, w = i.shape
    kh, kw = k.shape

    i_pad = np.pad(i, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode='constant', constant_values=0)
    
    k_flip = np.flip(k)

    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            roi = i_pad[i:i+kh, j:j+kw] # full pixel-wise convolution
            out[i, j] = np.sum(roi * k_flip)

    return out
```

For clarity, padding is applied to the image to ensure that the convolution operation can be performed at the edges. Adding a border of zeros around the image allows the algorithm to apply the convolution at every pixel in the image. Flipping the kernel is necessary to perform convolution, rather than cross-correlation.


##### Filters

Below are three sample kernels I used to convolve some images. 

<div class="row">
    <div class="col-sm">
        <strong>D<sub>x</sub></strong>
        <p>This finite difference filter detects vertical edges by computing the difference between horizontal pixel intensities.</p>
        $$
        D_x = \begin{bmatrix} 1 & 0 & -1 \end{bmatrix}
        $$
    </div>
    <div class="col-sm">
        <strong>D<sub>y</sub></strong>
        <p>This finite difference filter detects horizontal edges by computing the difference between vertical pixel intensities.</p>
        $$
        D_y = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}
        $$
    </div>
    <div class="col-sm">
        <strong>Box Filter</strong>
        <p>This filter replaces each pixel with the average of its neighbors (blur).</p>
        $$
        	Box = \frac{1}{81}
        \begin{bmatrix}
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
        \end{bmatrix}
        $$
    </div>
</div>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/cyprian.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/dx_filter_cyp.png" title="dx" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/dy_filter_cyp.png" title="dy" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/11/box_filter_cyp.png" title="box filter" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">From left to right, the original, $D_x$ applied to it, then $D_y$, then the box filter.</p>


##### Efficiency and Accuracy of Convolution Implementations

Here is a summary of the compute time I observed when performing the 9x9 box filter convolution on the above image, compared to scipy.signal.convolve2d as a benchmark, along with a comparison of differences to the output provided by the SciPy method.

<table class="table table-bordered text-center">
    <thead>
        <tr>
            <th>Implementation</th>
            <th>Time Taken (s)</th>
            <th>Max Difference to SciPy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Four loops</td>
            <td>10.100</td>
            <td>1.44e-15</td>
        </tr>
        <tr>
            <td>Two loops</td>
            <td>0.956</td>
            <td>5.55e-16</td>
        </tr>
        <tr>
            <td>SciPy</td>
            <td>0.041</td>
            <td>0</td>
        </tr>
    </tbody>
</table>

As can be seen, the maximum differences from the reference function are extremely small, which can be attributed to minor variations in floating-point precision.

---

#### 1.2 Finite Difference Operator

I used finite difference operators to compute the derivative of an image, highlighting regions with large intensity changes (what our eyes perceive as edges). By calculating the gradient magnitude from the $$D_x$$ and $$D_y$$ filters (by simply computing the magnitude of both filtered images combined), I produced a composite edge-detection image.

For example, I first applied the $$D_x$$ and $$D_y$$ filters to the reference image shown on the left.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_dx.png" title="dx" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_dy.png" title="dy" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Original, $D_x$, $D_y$.</p>

Next, I computed the gradient magnitude between the two convolutions, as seen in the first image below, and manually set a threshold to isolate edges from noise.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_gradient.png" title="Grad mag" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman_edges.png" title="Binarized edges" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Gradient magnitude image, binarized edge image.</p>

I selected a pixel intensity threshold of 0.3 to classify edges. My reasoning was that I preferred to preserve edge continuity, even if it meant retaining some noise in the image. This threshold struck a healthy balance, eliminating most of the noise in the grass while almost perfectly maintaining the outline of the subject.

Nevertheless, these edge detections still appear somewhat noisy, so in the next section I explore the use of Gaussian blurs to clean up the output.

---

#### 1.3 Derivative of Gaussian (DoG) Filter

To smooth our binarized edge image, I first applied a 2D Gaussian filter (outer product of a 1D Gaussian) to the original image before applying the $$D_x$$ and $$D_y$$ filters. Applying a Gaussian filter reduces high-frequency noise in the image, which helps the finite difference filters produce cleaner and more continuous edge detections. Below is each step in the procedure.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/12/cameraman.png" title="Original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_blurred.png" title="Blurred" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Original, Right: Gaussian blur applied ($kernel~size = 15$, $\sigma=2$).</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_dx_blurred.png" title="Dx blurred" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_dy_blurred.png" title="Dy blurred" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: $D_x$ filter applied on blurred image, Right: $D_y$ filter applied on blurred image.</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_grad_blurred.png" title="Grad mag blurred" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_edge_blurred.png" title="Binarized edge blurred" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Gradient magnitude composite, Right: Binarized edge image.</p>

The order of convolutions can also be changed, by first convolving the Gaussian blur with the finite difference filters to produce a Derivative of Gaussian filter (DoG). This works because convolution is a linear and shift-invariant operation, so the order of convolution can be rearranged. In other words, $$(G * D_x) * I = (G * D_x) * I$$, where $G$ is the Gaussian filter and $I$ is the image.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/derivative_gaussian_x.png" title="Grad mag blurred" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/derivative_gaussian_y.png" title="Binarized edge blurred" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: $D_x$ DoG, Right: $D_y$ DoG.</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_dx_dog.png" title="dx DoG" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_dy_dog.png" title="dy DoG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Original convolved on $D_x$ DoG, Right: Original convolved on $D_y$ DoG.</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_grad_dog.png" title="Grad mag DoG" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/13/cameraman_edge_dog.png" title="Binarized edge DoG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Gradient magnitude composite, Right: Binarized edge image.</p>

<table class="table table-bordered text-center">
    <thead>
        <tr>
            <th>Comparison</th>
            <th>Max Difference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>dx (DoG vs. Blurred)</td>
            <td>1.30e-15</td>
        </tr>
        <tr>
            <td>dy (DoG vs. Blurred)</td>
            <td>1.35e-15</td>
        </tr>
    </tbody>
</table>

As before, the maximum differences between the DoG and blurred approaches are extremely small, which is simply variation in floating-point precision, confirming that the order of operations yields identical results.

---

### Part 2: Fun with Frequencies

#### 2.1 Image Sharpening (Unsharp Mask)

Next, I implemented an image sharpening technique, by creating an unsharp mask filter. It works by first blurring the original with a Gaussian filter, which is then subtracted from the original, isolating the high-frequency components. By adding more of these high frequencies scaled by some strength paramater into the original image, the unsharp mask filter pronounces edges, producing a 'sharpening' effect. Below are some examples.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/high_freq_taj.png" title="Taj HF" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/taj.jpg" title="Taj original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/sharpened_taj.png" title="Taj sharp" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: High frequencies, Center: Original Taj Mahal Right: Sharpened Taj Mahal ($factor=1.2$).</p>

Some more examples from my personal photo library:

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/high_freq_lv_billboard.png" title="lv_billboard HF" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/lv_billboard.jpeg" title="lv_billboard original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/sharpened_lv_billboard.png" title="lv_billboard sharp" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: High frequencies, Center: Original Las Vegas sign Right: Sharpened Las Vegas sign ($factor=1.2$).</p>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/high_freq_shenoy.png" title="shenoy HF" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/shenoy.jpeg" title="shenoy original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/sharpened_shenoy.png" title="shenoy sharp" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: High frequencies, Center: Original Shenoy Right: Sharpened Shenoy ($factor=3$).</p>

Note: The last image is sharpened using a higher strength factor than the previous examples. While this increases the prominence of edges, it also amplifies high-frequency noise present in the image, producing a grainier appearance. This demonstrates that too much sharpening can enhance unwanted details and artifacts. For a final demonstration of the sharpening process, I intentionally blurred an image, then applied the unsharp mask to assess how well the original can be reconstructed. Results are below:

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/shake.jpeg" title="shake original" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/blurred_shake.png" title="shake blurred" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/21/sharpened_blurred_shake.png" title="shake resharpened" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Original, Center: Manually blurred input $(sn=8, \sigma=1)$ Shenoy Right: Sharpened blur ($factor=2.5$).</p>

---

#### 2.2 Hybrid Images

Next, I created hybrid images by combining high and low frequency components from different images. The idea comes from the fact that human visual perception is sensitive to high frequencies when viewing images up close, but primarily detects low frequencies when viewing from a distance. By blending different frequency ranges between two images, a single image can show different subjects depending on the distance it is viewed at.

The methodology works similarly to steps taken earlier. By applying a standard 2D Gaussian filter to extract low frequencies from one image, and creating a high-pass filter by subtracting the Gaussian-filtered version from the original image (equivalent to an impulse filter minus the Gaussian), we can overlay both outputs to produce the hybrid. Finding good cutoff frequencies (the $\sigma$ parameter for each Gaussian filter) requires some experimenting and really depends on the observer. For my implementation, I aligned image pairs using an alignment code (focussing on eye alignment for human subjects). Below is a rundown of the full process.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/results_high_cyp_low_flo/unaligned_high_cyp.png" title="cyprian unaligned" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/results_high_cyp_low_flo/unaligned_low_flo.png" title="flo unaligned" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Cyprian (high pass), unaligned, Right: Florentin (low pass), unaligned.</p>

Then, I manually selected the alignment reference points on each image.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/results_high_cyp_low_flo/aligned_high_cyp.png" title="cyprian aligned" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/results_high_cyp_low_flo/aligned_low_flo.png" title="flo aligned" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Cyprian, aligned, Right: Florentin, aligned.</p>

After some trial and error, I selected cutoff frequencies $\sigma_{low~pass} = 2.8,~\sigma_{high~pass} = 2.8$.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/results_high_cyp_low_flo/filtered_highpass_high_cyp.png" title="cyprian high pass" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/results_high_cyp_low_flo/filtered_lowpass_low_flo.png" title="flo low pass" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Left: Cyprian, high pass, Right: Florentin low pass.</p>

Final overlayed output:

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/results_high_cyp_low_flo/final_hybrid.png" title="cyprian flo hybrid" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<p class="text-center">Cyprentin.</p>

##### Fourier Transform Analysis

To get a better impression of how the hybrid image technique works, I analyzed the frequency content of each stage in the process. The Fourier transform visualizations show which spatial frequencies are present in each image. Low frequencies appear in the center, while high frequencies extend outward. This analysis reveals how our Gaussian and impulse filters isolate different frequency components.

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/im1_gray.png" title="cyprian gray" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/im1_fft.png"" title="cyprian fft" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/im1_highpass_sigma2.8_gray.png"" title="cyprian high pass gray" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/im1_highpass_sigma2.8_fft.png"" title="cyprian high pass fft" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/im2_gray.png" title="flo gray" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/im2_fft.png"" title="flo fft" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/im2_lowpass_sigma2.8_gray.png"" title="flo low pass gray" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/im2_lowpass_sigma2.8_fft.png"" title="flo low pass fft" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row">
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/combined_gray.png" title="combined gray" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
        {% include figure.liquid path="assets/img/cs180/p2/22/fft_FFT_Analysis/combined_fft.png"" title="combined fft" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

In the original images, the full spectrum of frequencies can be observed. When we apply the high-pass filter to Cyprian's image, the FFT shows that low frequencies (near the center) are attenuated, leaving only the high-frequency edge information that defines facial features and fine details. Conversely, the low-pass filter applied to Florentin's image preserves the central low frequencies while removing the high-frequency details. The combined hybrid image's FFT shows how the two filtered images produce complementary frequency ranges.

##### Additional Examples

Below are more examples of hybrid images using the same methodology.

<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/results_high_jj_low_khan/aligned_high_jj.png" title="JJ source" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/results_high_jj_low_khan/aligned_low_khan.png" title="Khan source" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/results_high_jj_low_khan/final_hybrid.png" title="JJ-Khan hybrid" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">JJ and Khan hybrid: High frequencies from JJ, low frequencies from Khan ($\sigma_1=8, \sigma_2=2.5, \alpha=0.9$).</p>

<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/results_nutmeg_DerekPicture/aligned_nutmeg.png" title="Cat source" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/results_nutmeg_DerekPicture/aligned_DerekPicture.png" title="Derek source" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/results_nutmeg_DerekPicture/final_hybrid.png" title="Cat-Derek hybrid" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Nutmeg and Derek hybrid: High frequencies from cat (Nutmeg), low frequencies from Derek ($\sigma_1=8, \sigma_2=1.5, \alpha=0.9$).</p>

---

#### 2.3 Gaussian and Laplacian Stacks

In the next step of this project, I worked on blending two images seamlessly by applying a smoothed mask to lower frequencies and an unsmoothed mask to the highest frequencies. The effect of this approach is that coarser features, whose changes are easily noticeable to the eye, must be gradually blended, while finer features should be more aggressively blended to avoid perceiving blur artifacts. To accomplish this, I first isolated $N$ distinct frequency ranges for each of the two images being blended. This can be achieved by repeatedly applying Gaussian blur starting from each original image to form a Gaussian stack, then subtracting each $(i+1)$-th layer from the $i$-th layer to form a Laplacian stack. The last element in the Laplacian stack is simply the residual Gaussian blur. With these frequency decompositions, we can apply increasingly smoothed masks as we move toward coarser features.

##### Naive Apprach

<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/blend_apple_orange_Vertical_Blend/apple.png" title="Cat source" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/blend_apple_orange_Vertical_Blend/orange.png" title="Derek source" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Reference images we want to blend.</p>

<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/blend_apple_orange_Vertical_Blend/apple.png" title="Cat source" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/blend_apple_orange_Vertical_Blend/orange.png" title="Derek source" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Reference images we want to blend.</p>

As we can see, simply "blending" both images by applying a uniform unsmoothed mask to both images produces an immediately identifiable stitch.

##### Gaussian Stack

Below is the Gaussian stack for both images, $N = 4$.

<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Apple_Stacks/gaussian_level_0.png" title="apple level 0" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Orange_Stacks/gaussian_level_0.png" title="orange level 0" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Level 0.</p>
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Apple_Stacks/gaussian_level_1.png" title="apple level 1" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Orange_Stacks/gaussian_level_1.png" title="orange level 1" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Level 1.</p>
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Apple_Stacks/gaussian_level_2.png" title="apple level 2" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Orange_Stacks/gaussian_level_2.png" title="orange level 2" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Level 2.</p>
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Apple_Stacks/gaussian_level_3.png" title="apple level 3" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Orange_Stacks/gaussian_level_3.png" title="orange level 3" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Level 3.</p>

##### Laplacian Stack

<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Apple_Stacks/laplacian_level_0.png" title="apple level 0" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Orange_Stacks/laplacian_level_0.png" title="orange level 0" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Level 0.</p>
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Apple_Stacks/laplacian_level_1.png" title="apple level 1" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Orange_Stacks/laplacian_level_1.png" title="orange level 1" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Level 1.</p>
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Apple_Stacks/laplacian_level_2.png" title="apple level 2" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Orange_Stacks/laplacian_level_2.png" title="orange level 2" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Level 2.</p>
<div class="row">
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Apple_Stacks/laplacian_level_3.png" title="apple level 3" class="img-fluid rounded z-depth-1" %}</div>
    <div class="col-sm">{% include figure.liquid path="assets/img/cs180/p2/22/stacks_Orange_Stacks/laplacian_level_3.png" title="orange level 3" class="img-fluid rounded z-depth-1" %}</div>
</div>
<p class="text-center">Level 3.</p>

Now, onto multiresolution blending.

---

#### 2.4 Multiresolution Blending (Oraple)

