---
title: Assignment 4
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* 

{% endcapture %}
{% include learning_objectives.html content=content %}



# Implement classification with a train/test split

As we (hopefully) learned in the class activity, overfitting our model to our data can lead to diminished results when we apply our model to a new set of data. One of the ways we try to avoid overfitting is by splitting our data into a training and testing set. (In the future, we will talk about another split of the training data called cross-validation, but for now, we won't worry about that.)

{% capture content %}

[//]: <> [(60 minutes)]

Work through the [Assignment 2a Companion Notebook - old to be replaced if needed, just placeholder](https://colab.research.google.com/drive/1d4EvlaSpgGB-hx78Kj2ii5ee7dczW-I5?usp=sharing) to practice implementing a simple classification algorithm called a decision tree in Python. You can place your answers directly in the Jupyter notebook so that you have them for your records.


{% endcapture %}
{% include external_resources.html content=content %}

# More Linear Regression and Ridge Regression - maybe move this up

So far, you have experimented two types of supervised learning: classification and regression. You worked through the derivation of linear regression. We are going to start this assignment by applying linear regression to our bike share dataset. You will find and plot residuals, practice splitting your data into a training and testing set, and apply a fancy twist on linear regression called ridge regression.

[estimated 120 minute notebook on bike data and maybe ridge regression](https://colab.research.google.com/drive/1p_j3y1AnX54a-Wln2J_tA50gq6iS9VEZ?usp=sharing)

## Ridge Regression Math
In the Companion Notebook, you manipulated the value of lambda ($\lambda$) to change the penalty for having large weights. One way to mitigate the problem of having two little data or having features that are linear combinations of each other is to modify the linear regression problem to prefer solutions that have small weights.  We do this by penalizing the sum of the squares of the weights themselves.  This is called ridge regression (or Tikhonov regularization).  Below, we show the original version of ordinary least squares along with ridge regression.


Ordinary least squares:
$$
\begin{align}
\mathbf{w^\star} &= \arg\min_\mathbf{w} \sum_{i=1}^n \left ( \mathbf{w}^\top \mathbf{x_i} - y_i \right)^2  \\  
&= \arg\min_\mathbf{w} \left ( \mathbf{X}\mathbf{w} - \mathbf{y} \right)^\top \left ( \mathbf{X}\mathbf{w} - \mathbf{y} \right)
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
\mathbf{w^\star} &= \arg\min_\mathbf{w} \sum_{i=1}^n \left ( \mathbf{w}^\top \mathbf{x_i} - y_i \right)^2 + \lambda\sum_{i=1}^d w_i^2  \\  
&= \arg\min_\mathbf{w} \left ( \mathbf{X}\mathbf{w} - \mathbf{y} \right)^\top \left ( \mathbf{X}\mathbf{w} -  \mathbf{y} \right) + \lambda \mathbf{w}^\top \mathbf{w}
\end{align}
$$

The penalty term may seem a little arbitrary, but it can be motivated on a conceptual level pretty easily.  The basic idea is that in the absence of sufficient training data to suggest otherwise, we should try to make the weights small.  Small weights have the property that changes to the input result in minor changes to our predictions, which is a good default behavior.



{% capture problem %}
[//]: <> [(60 minutes)]
Derive an expression to compute the optimal weights, $\mathbf{w^\star}$, to the ridge regression problem.
{% capture part_a %}
This is very, very similar to an exercise you did on the last assignment. You can click the slow solution button below this for a hint.
{% endcapture %}
{% capture part_a_hint1 %}
 If you follow the same steps as you did in ??Exercise 5??, you'll arrive at an expression that looks like this (note: $\mathbf{I}_{d \times d}$ is the $d$ by $d$ identity matrix).

$$
\mathbf{w^\star} = \arg\min_\mathbf{w} \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2\mathbf{w}^\top \mathbf{X}^\top \mathbf{y} + \mathbf{y}^\top \mathbf{y} + \lambda \mathbf{w}^\top  \mathbf{I}_{d \times d} \mathbf{w}
$$

{% endcapture %}
{% include problem_part.html label="" subpart=part_a solution=part_a_hint1 %}

{% capture part_b %}
If you want another hint, click on this solution
{% endcapture %}
{% capture part_b_hint %}
To get $\mathbf{w^\star}$, take the gradient, set it to 0 and solve for $\mathbf{w}$.
{% endcapture %}
{% include problem_part.html label=" - another hint" subpart=part_b solution=part_b_hint %}


{% capture part_c %}
Okay, now check against the full solution.
{% endcapture %}
{% capture part_c_sol %}
$$
\begin{align}
\mathbf{w^\star} &= \arg\min_\mathbf{w} \left ( \mathbf{X}\mathbf{w} - \mathbf{y} \right)^\top \left ( \mathbf{X}\mathbf{w} -  \mathbf{y} \right) + \lambda \mathbf{w}^\top \mathbf{w}  \\  
&= \arg\min_\mathbf{w} \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2\mathbf{w}^\top \mathbf{X}^\top \mathbf{y} + \mathbf{y}^\top \mathbf{y} + \lambda \mathbf{w}^\top  \mathbf{I}_{d \times d} \mathbf{w}
\end{align}
$$

$$
= \arg\min_\mathbf{w} \mathbf{w}^\top \left ( \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I_{d \times d}} \right )\mathbf{w} - 2\mathbf{w}^\top \mathbf{X}^\top \mathbf{y} + \mathbf{y}^\top \mathbf{y}  \\  
2 \left (  \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I_{d \times d}} \right ) \mathbf{w^\star} - 2 \mathbf{X}^\top \mathbf{y} &=0 \\  

\text{take the gradient and set to 0}  \\  
\mathbf{w}^\star &= \left ( \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I_{d \times d}} \right)^{-1} \mathbf{X}^\top \mathbf{y}
$$
{% endcapture %}

{% include problem_part.html label=" - Full Solution" subpart=part_c solution=part_c_sol %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}

