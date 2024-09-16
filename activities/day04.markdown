---
title: Day 4
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:30am: Debrief at tables
* 10:30-10:50am: Ridge Regression
* 10:50-11:40am: Classification and train/test split
* 11:40-12:00pm: Logistic Regression Primer
{% endcapture %}

{% include agenda.html content=agenda %}

# Debrief on the last assignment (5 minutes)

Warm up your brains by refreshing on the last assignment, including the derivation of linear regression... we're going to use it in a minute.
<!--- Possibly do this without computer (unless needed)  -  have them focus on answering a few questions together and prep them for what we might ask at a gate -->

# Ridge Regression Math (20 minutes)

You'll do some more on ridge regression in the assignment, including an exploration of why it's useful. In class, we're going to go over the math of ridge regression (which will also be an exercise on your assignment).

One way to mitigate the problem of having two little data or having features that are linear combinations of each other is to modify the linear regression problem to prefer solutions that have small weights.  We do this by penalizing the sum of the squares of the weights themselves.  This is called ridge regression (or Tikhonov regularization).  Below, we show the original version of ordinary least squares along with ridge regression.


Ordinary least squares:
$$
\begin{align}
\mathbf{w^\star} &= \argmin_\mathbf{w} \sum_{i=1}^n \left ( \mathbf{w}^\top \mathbf{x_i} - y_i \right)^2  \\  
&= \argmin_\mathbf{w} \left ( \mathbf{X}\mathbf{w} - \mathbf{y} \right)^\top \left ( \mathbf{X}\mathbf{w} - \mathbf{y} \right)
\end{align}
$$

Formula for the optimal weights in linear regression:
$$
\begin{align}
\mathbf{w^\star} = \left ( \mathbf{X}^\top \mathbf{X} \right)^{-1} \mathbf{X}^\top \mathbf{y}
\end{align}
$$

Ridge regression (note that $\lambda$ is a non-negative parameter that controls how much the algorithm cares about fitting the data and how much it cares about having small weights):
$$
\begin{align}
\mathbf{w^\star} &= \argmin_\mathbf{w} \sum_{i=1}^n \left ( \mathbf{w}^\top \mathbf{x_i} - y_i \right)^2 + \lambda\sum_{i=1}^d w_i^2  \\  
&= \argmin_\mathbf{w} \left ( \mathbf{X}\mathbf{w} - \mathbf{y} \right)^\top \left ( \mathbf{X}\mathbf{w} -  \mathbf{y} \right) + \lambda \mathbf{w}^\top \mathbf{w}
\end{align}
$$

The penalty term may seem a little arbitrary, but it can be motivated on a conceptual level pretty easily.  The basic idea is that in the absence of sufficient training data to suggest otherwise, we should try to make the weights small.  Small weights have the property that changes to the input result in minor changes to our predictions, which is a good default behavior.

Time permitting, we will derive an expression to compute the optimal weights, $\mathbf{w^\star}$, to the ridge regression problem.

# Classification and Train/Test Split in scikit-learn (40 minutes)

Overfitting our model to our data can lead to diminished results when we apply our model to a new set of data. One of the ways we try to avoid overfitting is by splitting our data into a training and testing set. (In the future, we will talk about another split of the training data called cross-validation, but for now, we won't worry about that.)


[Scikit-learn](https://scikit-learn.org/stable/index.html) is a common python library for classic machine learning. 

We are going to do a guided tour of [this Colab notebook on classification](https://colab.research.google.com/drive/1cAes5ScARNwi3-naPIpl0WnCqpiS5k7x?usp=sharing)

# Logistic Regression Primer

We've met the idea of classification. Logistic regression is one algorithm for binary classification. It builds nicely on linear regression and feeds nicely into neural networks (which we will explore soon). 

{% include figure.html
        img="../assignments/assignment04/figures/linearandlogistic.png"
        alt="a schematic of a neural network is used to represent linear and logistic regression.  Circles represent nodes, which are connected to other nodes using arrows. Logistic regression looks like linear regression followed by a sigmoid function."
        caption="Graphical representation of both linear and logistic regression.  The key difference is the application of the squashing function shown in yellow. [Original Source - Towards Data Science](https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24)" %}
{% assign graphicaldataflow = figure_number %}

In your assignment, you'll be meeting loss functions for binary classification.
