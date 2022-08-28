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
| Name |   URL  | Training Script | Model Description |
| :----: | :----: | :----: | :----: |
| DISCO 	 | [model](xxx) | [script.sh](./scripts/anchorcolorprob_hint2class-enhanced-h8.sh) | default colorization model used in our paper |
| DISCO-c0.2 | [model](https://drive.google.com/file/d/1jGDOfMq4mpYe6KCc0MtuiFwdEJ7_Hcc-/view?usp=sharing) | [script.sh](./scripts/anchorcolorprob_hint2class-enhanced-h8-c0.2.sh) | colorization model with relatively mild color saturation |
| DISCO-rand | [model](https://drive.google.com/file/d/1GLLowR-0eK2U4RAHijoizEyKd5ny10OI/view?usp=sharing) | [script.sh](./scripts/anchorcolorprob_hint2class-enhanced-rand.sh) | colorization model trained with random anchor locations |
| SPixelNet-s16 | [model](https://drive.google.com/file/d/1sLIqur7Hxan8PhW0n8kd7vzNEuIXAEdI/view?usp=sharing) | [script.sh](./scripts/spixelseg_ab16-imagenet.sh) | superpixel segmentation model with primitive size of 16 |
| SPixelNet-s8 | [model](https://drive.google.com/file/d/1pZK01Si_ufyAbLiLkugA_KY5z6NFnnET/view?usp=sharing) | [script.sh](./scripts/spixelseg_ab8-imagenet.sh) | superpixel segmentation model with primitive size of 8 |


## Quick Inference

- **Download Pre-trained Models**: download a pretrained colorization model from the table above and put it a folder, e.g., `./checkpoints`.

- **Prepare Testing Data**: You can put the testing images in a folder, e.g., `./data`

- **Testing on Colorization**: As default, the input image will be resized into 256x256 and generate a colorized RGB image of that fixed resolution. Optionally, you can add `--no_resize` to colorize the image at the original resolution, and add `--diverse` to generate three different colorization results.
```
python3 inference.py --model AnchorColorProb --data [input folder] --checkpt [checkpoint path] \
--name [experiment name] --n_clusters 8		# --no_resize	--diverse	
```
The result will be saved into the created folder `[experiment name]` located in the parent directory of `[checkpoint path]`.
Note that, the colorization result may also vary a bit depending on the random seed `--seed` because the clustering based anchor location involves randomness.


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