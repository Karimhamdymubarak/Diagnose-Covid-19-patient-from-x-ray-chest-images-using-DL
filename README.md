## Context
This is a repo for demonstrating how to digonse Covid-19 patients from their x-ray chest images using deep convolutional neural network.

### Genral Idea
The idea comes from we can train a machine learning model – especially deep learning model -  with large numbers of x-ray chest images for covid-19 patients and normal people and it can learn how to differntiate covid-19 patients from the normal people.

### Technical Description

 We train a Convolutional neural network or just CNN -a type of deep learning used for image recognition (more correctly "image classification" because it classifies the image into probabilistic classes) - to classify a x-ray chest image to covid-19 patient class or normal person class after extracting the features from the image.


### Data Sample
![x-ray](https://user-images.githubusercontent.com/47028466/113564043-f89ee600-9608-11eb-9a14-7793537a87b8.JPG)


### Model Architecture
![x-ray2](https://user-images.githubusercontent.com/47028466/113566631-5e8d6c80-960d-11eb-943d-2fca4b6091e2.JPG)

### Training
The training Code is training.py in which you can find the training process details

### Testing
The test code can be found as test.py

### Demo
Demo for the Model

https://drive.google.com/file/d/1ERsr9QPD4SG0fUMTU4E5vi4Q_NFniPYc/view?usp=sharing
