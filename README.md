# CZ4041-Machine-Learning
This project aims to complete the [Leaf Classification](https://www.kaggle.com/c/leaf-classification) challenge in Kaggle. 

## Dependencies
* Tensorflow
* Keras
* OpenCV

## Folders
* **models:** Contains code for different models experimented.
* **tools:** Contains code for feature extraction etc.

## Feature Extraction
Besides the features provided by Kaggle dataset. Other features were extracted for experiments.
1. In *cv_feature_extraction.py*, OpenCV was used to extract extra features. Each feature was implemented in one function.
1. In *time-series_feature_extraction.ipynb*, a new kind of feature was experimented but did not produce good result in further training. 

## Models
Several model architectures were proposed and experimented.
1. In *baseline.py*, only features provided by Kaggle dataset were used. It is trained as a baseline result for further experiments.
1. In *train_manual_extracted_feature.py*, features extracted by OpenCV (*cv_featue_extraction.py*) were added for regression. 
1. In *densenet121.py*, the DenseNet architecture was applied for training the dataset.
1. In *capsule_net.py*, the CapsuleNet architecture was applied for training the dataset.
1. In *just_conv.py*, a model with simple convolutional layers followed by fully connected layers were trained and evaluated.
1. In *autoencoder* folder, a model with a convolutional autoencoder followed by fully connected layers were trained and evaluated. *ae.py* file implemented the convolutional autoencoder and *model.py* file implemented the whole model including dense layers.
