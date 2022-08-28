# Disentangled Image Colorization via Global Anchors

### Paper(coming soon) | [Project Page](https://menghanxia.github.io/projects/disco.html) | Online Demo(coming soon)

Conceptually, our disentangled colorization model consists of two components: (i) **anchor color representation** that predicts the global color anchors (i.e. location and probabilistic colors) as the color representation of the whole image; (ii) **anchor-guided color generation** that synthesizes the per-pixel colors deterministically by referring the specified anchor colors.

<div align="center">
	<img src="asserts/network.png" width="95%">
</div>

<br>

:blush: **This is the official PyTorch implementation of our colorization work** [DISCO](https://menghanxia.github.io/projects/disco.html).

## Dependencies and Installation


## Checkpoints
| Name |   URL  | Training Script | Description |
| :----: | :----: | :----: | :----: |
| DISCO 	 | [model](xxx) | [script](xxx) | Default colorization model used in our paper |
| DISCO-c0.2 | [model](xxx) | [script](xxx) | Colorization model with relatively mild color saturation |
| DISCO-rand | [model](xxx) | [script](xxx) | Colorization model trained with random anchor locations |
| SPixelNet-s16 | [model](xxx) | [script](xxx) | Superpixel segmentation model with primitive size of 16 |
| SPixelNet-s8 | [model](xxx) | [script](xxx) | Superpixel segmentation model with primitive size of 8 |

## Quick Inference

- **Download Pre-trained Models**:

- **Prepare Testing Data**:

- **Testing on Colorization**:


## Training

- **Prepare Training Data**:

- **Training on Colorization**:


## Acknowledgement
We borrow some codes from [SpixelFCN](https://github.com/fuy34/superpixel_fcn), [iDeepColor](https://github.com/richzhang/colorization-pytorch), and [DETR](https://github.com/facebookresearch/detr). Thanks for their awesome works.


## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@article{xia-2022-disco,
	author   = {Menghan Xia and Wenbo Hu and Tien-Tsin Wong and Jue Wang},
	title    = {Disentangled Image Colorization via Global Anchors},
	journal = {ACM Transactions on Graphics (TOG)},
	year = {2022}
}
```