---
title: Assignment 16 - Convolutional Neural Networks (ConvNets, CNNs) in Code
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Identify and explain key components of a convolutional neural network (CNN)
* Implement a convolutional neural network
* 
{% endcapture %}
{% include learning_objectives.html content=content %}

# A CNN notebook
For this assignment, we have created a detailed notebook for you that give you almost all the code that you need to experiment with CNNs. Our goal here is to help you to experiment and build some intuition without spending tons of time troubleshooting code.  

However, for some of you, you might get a deeper sense of the material if you code the whole thing from scratch. There's nothing in here that you can't do, so please feel free to write your own code from scratch if it will help your learning.   

# What to submit
For your quality assessed deliverable, we may to ask you to submit answers to some of these questions, so keep that in mind as you document your work.

For people using assessment option B, you don't need to submit all of your code. We aren't giving you solutions here, so you also don't need to do the corrections. Please submit a document that answers the questions below. You will need to include some key figures (which are mostly generated for you).

# What to do and what to answer

Start by looking through the whole notebook to get the gist of what is there. Be sure to note where models are defined, where training happens, how a subset of the data is selected, and what variables you can change.

## MNIST dataset
Start with the MNIST dataset, which has grayscale images. 

Choose 3 digits to include in your model and change the code to select these.
Create a very small training set (e.g. 16 examples per class). 
Train the model called FC_only for 40 epochs.
In your write-up, show the loss over epochs plot, the test confusion matrix, and the training and test accuracy.


Research CNNs in PyTorch
Create a model called Grayscale1Convolution. The model should include 1 convolution layer and 1 max pooling layer that reduces the image size by 1/2. You will need to do some math on the sizes of each of the inputs and outputs to make this work. 
Train the model called Grayscale1Convolution for 40 epochs.
In your write-up, show your model code for Grayscale1Convolution, the loss over epochs plot, the test confusion matrix, and the training and test accuracy.

Increase the amount of data significantly and rerun both models. 
In your write-up, show the loss over epochs plot, the test confusion matrix, and the training and test accuracy.

## CIFAR10 dataset
This dataset shows 10 categories of images. While you are building your model, you may want to work with a small subset of the data. At the end, you should run it with a larger version of the data.

Create, train, and document a model with 1 convolution layer and 1 max pooling layer.

Create at least two other models that work better than this original model on your dataset.
Document your experiments by including the loss plot, confusion matrices, and relevant metrics.
If you are stuck on what to do, you might experiment with increasing the model complexity (more layers), adding dropout, changing the pooling, augmenting the data, etc.

# Transfer learning
People often use transfer learning, where we build on a pre-trained model (that was trained on a huge dataset) and then tweak it for our own purpose. This is incredibly powerful. Here's [one video](https://youtu.be/MQkVIYzpK-Y) on transfer learning, but feel free to find your own resource (and skip ahead in this video).  

The PyTorch documentation has a [nice description and example of transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). Note that you can open it in a Colab notebook at the top of the page.

You can modify our existing notebook to do transfer learning. You'll need to read through the given transfer learning example and extract relevant parts of the code. 

{% capture problem %}
Research transfer learning. Apply transfer learning to the CIFAR10 and our dessert dataset (our notebook should help you with loading these), comparing how well it works on these two datasets under a few different conditions (e.g., small number of epochs, small number of training images).

Write a short summary what you experimented with and what you learned (including key figures or pieces of information). You do not need to share your full code (and it's fine to run things and then copy an image). 

{% endcapture %}

{% include problem.html problem=problem %}

