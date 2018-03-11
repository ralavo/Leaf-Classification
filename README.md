# Leaf-Classification
Leaf classification using Convolutional Neural Network

## Project Description

The goal of this project is to create a new, accurate model for classifying tree leaves. Instead of having to manually identify and hand-code features for an algorithm to classify, deep learning algorithms use a multi-tiered abstraction approach to automatically identify features on their own. Since publicly accessible leaf image datasets were limited, leaves were classified in this project by genus of tree rather than at the species level. 
The models created in this project are known as Convolutional Neural Networks (CNNs).  CNNs are a kind of neural network well-suited for image classification and are composed of two types of layers: convolutional and fully-connected. In the convolutional layers, a series of filters are used to extract features of an image starting with low level features like edges all the way up to more complicated features like the ears of an animal. These features are then used in one or many fully-connected layers to classify the image into one of several groups. The type of network used is shown below.

![babyroomwatchercomplete](https://cloud.githubusercontent.com/assets/5084852/25751868/78465688-3184-11e7-8753-67e0665a70b1.JPG)

In an attempt to optimize the model we manipulated some variables such as the number of layers, the number of filters, the size of filters, and the types of activations functions used in the hidden, or middle, layers of the model (e.g. ReLU). Since image data was limited, data augmentation was used to produce further image variation for model training/testing. 

## Project Resources

* Leafsnap - Dataset of 30,000 leaves images grouped in 185 species [3]
* Middle European Woods  - Dataset of 10,000 leaves grouped in 153 species [8]
* Big Red 2 - GPU access
* Anaconda – Python
* Keras - Python library for CNN creation
* Python Imaging Library (PIL) - To process images
* Tensorflow – A library for neural networks
* ResNet50 - Included with Keras
* InceptionV3 - Included with Keras

## Result

After multiple run of tweaking, we finally got 80% of accuracy from our model after running through 100 epochs. The chart below compare the training loss and the validation loss to determine if any overfitting was detected.

![babyroomwatchercomplete](https://cloud.githubusercontent.com/assets/5084852/25751868/78465688-3184-11e7-8753-67e0665a70b1.JPG)
