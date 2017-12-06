# The Models of Abnormal Driving Detection

[简体中文](README_zh.md)

This is a description of all models. If you want to add your model, please add your model file in this folder.

## VGG-16
- option: `vgg16`
- file: `vggl6.lua`
- details:
  - 13 convolution layers
  - BN + ReLU
  - 3 full linear layers

## Wide-12
- option: `wide12`
- file: `wide12.lua`
- details:
  - 9 convolution layers
  - BN + ReLU
  - 3 full linear layers
  - more feature maps than VGG-16 in every corresponding layer

## Cardinality-25
- option: `cardinality25`
- file: `cardinality25.lua`
- details:
  - 22 convolution layers
  - BN + ReLU
  - 3 full linear layers
  - every convolution layer expect the first consist of 2 groups

## ResNet-16
- option: `resnet16`
- file: `resnet16.lua`
- details:
  - 6 blocks with every block including 2 convolution layers and shortcut using 1*1 convolution layer
  - BN + ReLU
  - 3 full linear layers

## DenseNet-88
- option: `densenet88`
- file: `densenet88.lua`
- details:
  - 4 densely block with every block including 2 convolution layers
  - 4 transition layers with reduce rate 0.5
  - BN + ReLU
  - 3 full linear layers

## DWC-88
- option: `dwc88`
- file: `dwc88.lua`
- details:
  - based on DenseNet-88
  - every convolution layers in densely block consist of 2 groups and more feature maps

## DWCR-88
- option: `dwcr88`
- file: `dwcr88.lua`
- details:
  - based on DWC-88
  - the input of every densely block is a connection from all prior blocks' output and the output of every densely block is its output adds all prior blocks' output by element

## DWC2R-29
- option: `dwc2r29`
- file: `dwc2r29.lua`
- details:
  - based on DWC-88
  - the input of every densely block is a tensor of all prior blocks' output add by element