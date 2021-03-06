# Resnets transfer learning for Stanford cars classification
Stanford cars classification by resnet models

This repository is the implementation of Resnet models (resnet18, resnet50, wide_resnet50_2 and resnext50_32x4d) on the Standford car classification task by PyTorch. The reported results on many applications indicate that Resnet is superior in performance respect to other architectures.

##### Table of Contents  
[Overview](#headers)  
[Methods](#methods)  
[Results](#results)     
<a name="headers"/>

## Overview
Implementation is on a local machine with 12GB NVIDIA Titan Xp, Intel(R) Core(TM) i9-7920X CPU @ 2.90GHz and 64GB memory.

The data set can be downloaded from Kaggle with class names (https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder/download). Data is normalized and augmented randomly to provide a rich dataset for training. All architectures are trained and evaluated with squared images (224x224).

## Methods
All resnet models are trained in a whole network fine-tuning mode using pre-trained models in PyTorch. The output layer is added to predict 196 classes. The optimization is SGD with momentum and the learning rate decays by a factor of 0.1 every 7 epochs. The default learning rate is 0.001 and the number of epochs is 25. They can be changed through the python arg as well.

Resnet18 is a light model with the training completed in 26m 3s and it is suitable to monitor the performance by fine-tuning the parameters. To run the model, the following command is executed in the clone directory: 

python3 transfer_learning_stanford_cars.py model_name number of epochs learning_rate images_dir

For example, the following command is issued to retrain resnext50_32x4d with learning rate =0.001 and 25 epochs:

python3 transfer_learning_stanford_cars.py resnext50_32x4d 25 0.001 ./car_data

## Results
We compare the best prediction per each models as well as training time:

| Model  | Best test accuracy | Training time |
| ------------- | ------------- | ------------- |
| resnet18  |  0.822161 | 26m 3s |
| resnet50  | 0.856610  | 45m 12s |
| wide_resnet50_2 | 0.878995 | 72m 15 |
| resnext50_32x4d | 0.885959 | 87m 11s |

Resnet18 has the lowest prediction accuracy and resnext50_32x4d has the best performance in prediction. However, that comes with the expense of ~3.35 times longer training time.
The training and test loss and accuracy during network training are visualized as follow:
1. Resnet18:

The network training is harder after 14 epochs. The loss profile can be seen as follow:

![Training and test loss](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/test_lossresnet18.png)

The accuracy follows the following trend over epochs:

![Training and test accuracy](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/test_accresnet18.png)

2. Resnet50

The loss profile: 

![Training and test loss](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/test_lossresnet50.png)

The accuracy profile: 

![Training and test accuracy](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/test_accresnet50.png)

3. Wide_resnet50_2

The loss profile: 

![Training and test loss](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/test_losswide_resnet50_2.png)

The accuracy profile: 

![Training and test accuracy](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/test_accwide_resnet50_2.png)


4. Resnext50_32x4d

The loss profile: 

![Training and test loss](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/test_lossresneXt50.png)

The accuracy profile: 

![Training and test accuracy](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/test_accresneXt50.png)

Now, we can compare the accuracy profile for each model:
1. Train accuracy:

![Training accuracy compare](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/comp_acc_train.png)

2. Test accuracy:

![Test accuracy compare](https://github.com/Baghbahari/Resnets-transfer-learning-for-stanford-cars/blob/master/comp_acc_test.png)


