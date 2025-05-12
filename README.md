# Scattering Angle-Resolved Bioimaging
## Project discription

## setup
Description:	Ubuntu 22.04.5 LTS

 apt list --installed 'nvidia*'
470 driver installed

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0


pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.htmlp


# Feature Extraction and Comparison

The configuration file outlines what type of features you will be comparing.

First some general information about the pipeline.
1. Images are loaded into a dataloader


NOTE:
Array sizes are 2048 by 2448. For annotations I resampeld to 2048 by 2048 for simplicity. Yolo annotations are per x,y ratio so the image size wont matter.








# Object Detection

## Overview 

The following documentation is for the bounding box object detection method.

We explored two methods
1. Semi-automated detection (using Conected Components)
2. Yolov4

To run either method you need the run_object_detection.py.

The configuration file under /config allows for specification of Yolo or Components.

Currently there has been no success with YOLOv4 (limited date, need to re run with augmentations/more samples).

This documentation focuses on Components which is the connected

## Connected Components - Semi Automated Segmentation

