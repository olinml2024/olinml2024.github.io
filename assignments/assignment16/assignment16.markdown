---
title: Assignment 16 - Convolutional Neural Networks (ConvNets, CNNs) in Code
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Identify and explain key components of a convolutional neural network (CNN)
* Implement convolutional neural networks, understanding the sizes of data
* Learn about transfer learning and apply it to data
{% endcapture %}
{% include learning_objectives.html content=content %}


This assignment is very open ended with the intent of creating space for you experiment and learn and then share back in class.  


# A CNN notebook
For this assignment, we have created a detailed notebook for you that give you almost all the code that you need to experiment with CNNs. Our goal here is to help you to experiment and build some intuition without spending tons of time troubleshooting code.  

However, for some of you, you might get a deeper sense of the material if you code the whole thing from scratch. There's nothing in here that you can't do, so please feel free to write your own code from scratch if it will help your learning.   

Here is the [notebook](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/Assignment_16_Convolutional_Neural_Networks_in_PyTorch.ipynb){:target="_blank"}.

# What to submit
For your quality assessed deliverable, we may to ask you to submit answers to some of these questions, so keep that in mind as you document your work.

For people using assessment option B, you don't need to submit all of your code. We aren't giving you solutions here, so you also don't need to do the corrections. Please submit a document that answers the questions below. You will need to include some key figures (which are mostly generated for you).

# What to do and what to answer

Start by looking through the whole notebook to get the gist of what is there. Be sure to note where models are defined, where training happens, how a subset of the data is selected, and what variables you can change.

## MNIST dataset
The MNIST dataset has grayscale images of digits.  


{% capture problem %}
1. Choose 3 digits to include in your model and change the code to select these.
2. Create a very small training set (e.g. 16 examples per class). 
3. Train the model called FC_only for enough epochs that the loss curve flattens (and ideally begins to overfit).
4. In your write-up, show the loss over epochs plot, the test confusion matrix, and the training and test accuracy.
{% endcapture %}
{% include problem.html problem=problem %}

{% capture problem %}
1. Research CNNs in PyTorch
2. Create a model called Grayscale1Convolution. The model should include 1 convolution layer and 1 max pooling layer that reduces the image size by 1/2. You will need to do some math on the sizes of each of the inputs and outputs to make this work. 
3. Train the model called Grayscale1Convolution for enough epochs that the loss curve flattens (and ideally begins to overfit).
4. In your write-up, show your model code for Grayscale1Convolution, the loss over epochs plot, the test confusion matrix, and the training and test accuracy.
{% endcapture %}
{% include problem.html problem=problem %}

{% capture problem %}
Experiment with at least 2 activation functions and explain how they affect your model results.
{% endcapture %}
{% include problem.html problem=problem %}

{% capture problem %}
1. Increase the amount of data significantly and rerun both models. 
2. In your write-up, show the loss over epochs plot, the test confusion matrix, and the training and test accuracy.
3. Make observations comparing to the last experiments.
{% endcapture %}
{% include problem.html problem=problem %}

## CIFAR10 dataset
This dataset shows 10 categories of images. While you are building your model, you may want to work with a small subset of the data. At the end, you should run it with a larger version of the data.

{% capture problem %}
Create, train, and document a model with 1 convolution layer and 1 max pooling layer.

{% endcapture %}
{% include problem.html problem=problem %}

{% capture problem %}
1. Create at least two other models that work better than this original model on your dataset.
2. Document your experiments by including the loss plot, confusion matrices, and relevant metrics.
If you are stuck on what to do, you might experiment with increasing the model complexity (more layers), adding dropout, changing the pooling, augmenting the data, etc.
{% endcapture %}
{% include problem.html problem=problem %}

# Transfer learning
People often use transfer learning, where we build on a pre-trained model (that was trained on a huge dataset) and then tweak it for our own purpose. This is incredibly powerful. Here's [one video](https://youtu.be/MQkVIYzpK-Y){:target="_blank"} on transfer learning, but feel free to find your own resource (and skip ahead in this video).  

The PyTorch documentation has a [nice description and example of transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html){:target="_blank"}. Note that you can open it in a Colab notebook at the top of the page.

You can modify our existing notebook to do transfer learning. You'll need to read through the given transfer learning example and extract relevant parts of the code. 

{% capture problem %}
Research transfer learning. Apply transfer learning to the CIFAR10 and our dessert dataset (our notebook should help you with loading these), comparing how well it works on these two datasets under a few different conditions (e.g., small number of epochs, small number of training images).

Write a short summary what you experimented with and what you learned (including key figures or pieces of information). You do not need to share your full code (and it's fine to run things and then copy an image). 

{% endcapture %}

{% include problem.html problem=problem %}

