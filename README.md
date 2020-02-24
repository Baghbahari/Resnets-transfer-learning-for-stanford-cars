# Resnets transfer learning for Stanford cars classification
Stanford cars classification by resnet models

This repository is the implementation of Resnet models (resnet18, resnet50, wide_resnet50_2 and resnext50_32x4d) on Standford cars classification task by pytorch.

##### Table of Contents  
[Overview](#headers)  
[Methods](#methods)  
[Results](#results)     
<a name="headers"/>

## Overview
Implementation is on a local machine with 12GB NVIDIA Titan Xp, Intel(R) Core(TM) i9-7920X CPU @ 2.90GHz and 64GB memory.

The data set can be downloaded fron consist Kaggle with class names (https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder/download). The data path can be defined for python code. Data is normalized and augmented randomly to provide a rich dataset for trainig. All architecture is trained and evaluated with squared images (224x224).

## Methods
All resnet models are trained in a whole network fine tuning mode using pretrained models in pytorch. The output layer is added to predict 196 classes. The optimization is SGD with momentum and the learning rate decays by a factor of 0.1 every 7 epochs. The defaults learning rate is 0.001 and the number of epochs is 25. They can be changed through the python arg as well.

Resnet18 is a light model with the training completed in 26m 3s and it is suitable to monitor the performance by fine tuning the parameters. 

## Results
