# Segmenting salt deposits from seismic images with deeply-supervised Unet (PyTorch)

## General
I have participated in the Kaggle competition [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge) and reached the 113-th place out of 3200+ teams. This repository contains the original code using the Jupiter Notebook format (.ipynb) as well as a reorganized version(.py files).

## Data preprocessing
The original images with size 101x101 px were padded to 128x128 px, and then [integrated with position information](https://eng.uber.com/coordconv/) (transformed into 3-channel images). Random crop to the input size 128x128 px, horizontal flip, slight rotation and random linear brightness augmentation were applied.

## Model design
I used a [U-Net](https://arxiv.org/abs/1505.04597) like architecture with a ResNet34 encoder and very simple Decoder blocks. A special deep supervision structure was added to speed up training and avoid overfitting (as shown in the figure below). ![General scheme](saltdeeps.png) As a whole, the model has a very slim structure with only 22,190,693 parameters. Due to the limited computing power, I did not try deeper encoders like SE-ResNeXt50, which would possibly further boost the performance.

## Models Training
Loss function: [Lovasz hinge loss](https://arxiv.org/abs/1705.08790).

Optimizer: SGD with LR 0.01, momentum 0.9, weight_decay 0.0001.

Train stages:

1) 300 epoches, image size 128x128;
2) 300 epoches, resized to 192x192;
3) [Cosine annealing learning rate](https://openreview.net/forum?id=BJYwwY9ll) 200 epochs, 50 per cycle; max_lr = 0.01, min_lr = 0.001.

## Cross Validation
Averaged results of five folds were used in the final submission.

