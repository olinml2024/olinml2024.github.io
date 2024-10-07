---
title: Learning as Optimization - Key takeaways
toc_sticky: true 
toc_h_max: 1
layout: problemset
---


1. Big picture structural things
    * Supervised learning problem setup (X, y, model, and loss)
    * Training, validation, and testing
    * Feature engineering - you can pre-process datasets, augment datasets, and even transform data to represent non-linear information in a linear model
    * Model parameters can be turned to reduce loss. This is done using partial derivatives & gradients. We use gradient descent for this optimization.
1. Linear regression
    * The model - prediction as a weighted sum of the inputs
    * Squared loss - explain/interpret the loss function (e.g., draw a picture of what this measures)
1. Logistic regression
    * The model - linear regression and the sigmoid function to map to a probability in a binary case
    * Log loss - Explain scenarios when a model and examples would give a high or low loss. Compute the log loss of a model given model predictions (probabilities) and ground truth
1. Multilayer perceptrons / Neural networks
    * Adding layers can increase your model's ability to fit the data, can reduce the need for feature engineering, and can increase your chances of overfitting.
    * Drawings of neural networks can be mapped to a series of weighted sums and non-linear functions. These weighted sums can be represented as matrix multiplications, dot products, or summation loops. You should be able to map between drawings of a multilayer perceptron and the layers that you would construct in PyTorch.
    * In neural networks, backpropagation is the application of the chain rule for computing the partial derivatives of the model's loss with respect to the weights of the network. This can be automated using the concept of dataflow diagrams. Draw a simple data flow diagram and calculate partial derivatives.