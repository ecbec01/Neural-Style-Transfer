# Neural Style Transfer

This repository contains my implementation of the Neural Style Transfer (NST) algorithm from
[Gatys *et al.* (2015)](https://arxiv.org/abs/1508.06576).

Neural Style Transfer (NST) is a class of algorithms that take one image (or video) and modify its pixels in order to make it to look like a second picture when it comes to style, but still preserving the first image content.

In this repository, photographies are taken and provided to the NST algorithm along side with some paintings. This allows the algorithm to transform the photos into paintings.

## Main files in this repository

* **`style transfer.ipynb`**: Python notebook using applying the NTS implementation;
* **`utils.py`**: Module with helper functions used in the python notebook above;
* **`model.py`**: Module with the StyleTransfer class, with contains the NTS implementation;

## Generated images

Here are the generated images using the `style transfer.ipynb` notebook:

<img src="generated images/from_1_to_1.jpg">
<img src="generated images/from_1_to_2.jpg">
<img src="generated images/from_1_to_3.jpg">
<br></br>
<img src="generated images/from_2_to_1.jpg">
<img src="generated images/from_2_to_2.jpg">
<img src="generated images/from_2_to_3.jpg">