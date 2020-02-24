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

Resnet18 is a light model with the training completed in 26m 3s and it is suitable to monitor the performance by fine tuning the parameters. To run the model, the following command is excuted in the clone direcoty: 

python3 transfer_learning_stanford_cars.py model_name number of epochs learning_rate images_dir

For example, you can issue the following command to retrain resnext50_32x4d:

python3 transfer_learning_stanford_cars.py resnext50_32x4d 25 0.001 ./car_data

## Results
First, we compare the best prediction per each models:

| Model  | Best test accuracy | Training time |
| ------------- | ------------- | ------------- |
| resnet18  | Content Cell  | |
| resnet50  | Content Cell  | |
| wide_resnet50_2 | | |
| resnext50_32x4d | | |
