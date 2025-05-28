# Scattering Angle-Resolved Bioimaging
Allison Clement

Contact: allison.clement@ox.ac.uk

## Project Discription
The Computational Imaging and Vision Lab has developed a technique for the sepa- ration of scattering light in Transmitted Light Microscopy (TLM). This method is the first to create spatial maps of the scattering property in TLM and has been used to accurately capture tissue characteristics (see link to paper [here](https://kyushu-u.elsevierpure.com/ja/publications/separation-of-transmitted-light-and-scattering-components-in-tran)).

Biological tissues are very complex. Current TLM is limited to one output image which includes all light outputs. Thus in some cases, tissue characteristics may not be fully captured. Capturing separate scattering properties will generate more detailed information about the structure of biological tissues. This may be useful for visualizing differences in characteristics between disease and healthy tissues.
Using the Scattering Angle Resolved Bioimaging (SARB) method, Mihoko collected control and disease samples. For each sample, five imaging channels were recorded. One channel with the TLM image, three channels with different scattering angles (θ1,θ2,θ3, increasing respectively), and a fifth channel containing multiple scattered light.

The goal of this project is to determine image features that can automatically differentiate between control and disease microscopy samples.

To reach this goal this repo was created. This repo is designed to load and calculate image features from the multiple scattering light images. The image features are then compared to determine statistical differences in groups.

This project is split into the following: 
1. Feature Extraction
2. Object Detection

## Setup 
<!-- If you are downloading on the computer currently set up int he lab:

Description:	Ubuntu 22.04.5 LTS

 apt list --installed 'nvidia*'
470 driver installed

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0 -->
<!-- 
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.htmlp
 -->

env_requrements.txt contains the current requirements for 

To download directly into a conda env use the following commands in ther terminal.

```
conda env create -f environment.yml
conda activate torch-env
```

Use conda list to ensure all packages have downloaded successfully.

## DATA
Data is in groups. For outline of data see Slide xx [here](). Current summary is there are 24 control samples and 19 disease (PAN) samples.
<!-- NOTE:
Array sizes are 2048 by 2448. For annotations I resampeld to 2048 by 2048 for simplicity. Yolo annotations are per x,y ratio so the image size wont matter. -->

Annotations are done for:
1. Clinical Regions of Intrest (Glomerulus, Promixal and Distal)
2. Small regions of intrest (Inner, Outer and Glomerulus edge)

See Slide xx [here]() for visuals. 

These are only done for 5 samples of the control and 5 for disease.

We are waiting for more clinical annotations.


# Feature Extraction
The configuration file outlines what type of features you will be comparing. 

General information about the pipeline.
1. Images are loaded into a dataloader, features are calculated and stored
2. Comparisons are made and defined in the configuration files.

## Features



# Object Detection
### Overview 
Ideally we would determine features over the entire image. After initial experiments, we found mimial difference in features over the entire image. After discussion with clinicians, they idenfitied localized regions of intreset (focusing on the Glomerulus or Renal Tubes). 

The following documentation is for automating the detection of regions of intrest.

We explored two methods
1. Semi-automated detection (using Conected Components)
2. Yolov4

To run either method you need the run_object_detection.py.

The configuration file under /config allows for specification of Yolo or Components.

Currently there has been no success with YOLOv4 (limited date, need to re run with augmentations/more samples).

This documentation focuses on Components which is the connected

### Connected Components - Semi Automated Segmentation


### YOLO - You Only Look Once
In the configuration files you can set the model to train on yolo. To do so you must 







