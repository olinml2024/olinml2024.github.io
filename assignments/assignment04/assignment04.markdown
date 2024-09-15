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

WTF, Please help me with this!?!
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


# The Classification Problem

So far in this class we've looked at supervised learning problems including a quick look at classification and a deeper look at regression. In regression, the responses $y_i$ are continuous-valued and the loss function is quadratic ($\ell(y, \hat{y}) = (y-\hat{y})^2$).  There are many times, however, where it is unnatural to frame a problem as a regression.  For instance, it may be the case that $y_i$ does not come from a continuous range but instead can only take on a few different values.  This sort of problem is known as a classification problem.  For instance, you might want to have a system that takes in an image of a person and predicts their identity.  The identity could be thought of as the output, $y_i$, and it would only make sense for $y_i$ to be one of several values (e.g., each value might represent a particular person the system was trained to recognize).  In this next section, you'll learn about a special case of the classification problem known as binary classification (where $y_i$ is either 0 or 1, e.g., a Carrie versus Sam recognizer).

In this class, we will formalize the binary classification problem and see a very useful algorithm for solving it called \emph{logistic regression}.  You will also see that the logistic regression algorithm is a very natural extension of linear regression.  Our plan for getting there is going to be pretty similar to what we did for linear regression.

* Build some mathematical foundations
* Introduce logistic regression from a top-down perspective
* Learn about logistic regression from a bottom-up perspective

In this assignment, we will focus on building the mathematical foundations.

# Formalizing the Classification Problem
Let's start by making the binary classification problem more formal.  Suppose, we are given a training set, $(\mlvec{x_1}, y_1), (\mlvec{x_2}, y_2), \ldots, (\mlvec{x_n}, y_n)$, where each $\mlvec{x_i}$ is an element of the input space (e.g., a vector) and each $y_i$ is a binary number (either 1 or 0).  In this setting we will attempt to use the training data to determine a function, $\hat{f}^\star$, that predicts the corresponding output, $y$, for any possible input, $\mathbf{x}$.  For example,  $\mathbf{x}$ could be an image and $y_i$ could be $1$ when the picture contains a puppy and $0$ otherwise.

[//]: <> [ 10 minutes] 


{% capture problem %}
{% capture part_a %}
Given this partial setup of the binary classification problem, we still need to specify the loss function, $\ell$.  Recall that $\ell$ takes as input the actual output $y$, and the predicted output $\hat{y}$.  What function could you use for $\ell$ that would result in the learning algorithm choosing a good model?  If the choice of $\ell$ depends on the application, how so?

{% endcapture %}
{% capture part_a_sol %}
An easy choice is to output a $1$ if the values don't match and a $0$ otherwise (essentially counting the number of mistakes the model makes).  Alternatively, you could have different penalties for a false positive (the model says $\hat{y} = 1$, but the actual value is $y = 0$) or false negatives (the model says $\hat{y} = 0$, but the actual value is $y = 1$). 
{% endcapture %}
{% include problem_part.html label="A" subpart=part_a solution=part_a_sol %}

{% capture part_b %}
One natural choice for $\ell$, which you may have already come up with, is to define our loss function as $\ell(y, \hat{y}) = \mathbb{I}[y \neq \hat{y}]$. The funny looking $\mathbb{I}$ is the indicator function that takes on value 1 when the condition inside is true and 0 otherwise.  Given this choice the supervised learning problem becomes:
$$
\begin{align}
\hat{f}^\star &= \argmin_{\hat{f}} \sum_{i=1}^n \mathbb{I} \left [  \hat{f}(\mlvec{x_i}) \neq y_i\right ] \enspace . \label{eq:minimizeerror}
\end{align}
$$
Convert the equation above to English to make sure you understand it.
{% endcapture %}
{% capture part_b_sol %}
The equation says that $\hat{f}^\star$ is the function that minimizes the number of mistakes it makes on the training set.
{% endcapture %}

{% include problem_part.html label="B" subpart=part_b solution=part_b_sol %}
{% endcapture %}
{% include problem_with_parts.html problem=problem %}


While the loss function given in Exercise **INSERT NUMBER FOR TWO EQUATIONS ABOVE** (minimizing mistakes on the training set) is a totally reasonable choice for the loss function, it turns out that it has a number of drawbacks.
 
* It is all or nothing.  Either we are completely right or completely wrong.
* It is not a particularly easy function to work with mathematically.  In fact, for many common classes of models, it will be difficult for the learning algorithm to find the best possible model. Note: One of the key challenges that must be met in machine learning, and modeling in general, is balancing computational considerations (e.g., how long does it take to find the best possible model) with the realism of the model (e.g., how directly does the task you pose to the learning algorithm match the problem you are solving).  Sometimes these things are in conflict and you must make tradeoffs..

It turns out that we can create a more natural loss function by thinking about predictions in terms of probabilities.

# Probability and the log loss

Imagine that instead of our model, $\hat{f}$, spitting out either 0 or 1, it outputs a confidence that the input $\mlvec{x}$ has an output $y= 1$.  In other words, rather than giving us its best guess (0 or 1), the classifier would indicate to us its degree of certainty regarding its prediction.  This notion of ``certainty'' can be formalized using the concept of a probability. % That is, the model can output a probability that the output for a particular input is 1.

We haven't formally defined probability in this class, but here are a few things to keep in mind about probabilities: 
* A probability, $p$, specifies the chance that some event occurs.  $p = 0$ means that the even will definitely not occur and $p=1$ means that it will definitely occur.
* A probability, $p$, must be between 0 and 1 ($0 \leq p \leq 1$).
* If the probability an event occurs is $p$, then the probability that the event doesn't occur is $1 - p$.

{% capture problem %}
For these questions, assume that for a given input the classifier outputs a probability that the output will be 1.

{% capture part_a %}
If a classifier has no clear idea of whether the output for a particular input is 1 or 0, what probability should the classifier output?
{% endcapture %}
{% capture part_a_sol %}
The output would be about 0.5.
{% endcapture %}
{% include problem_part.html label="A" subpart=part_a solution=part_a_sol %}

{% capture part_b %}
If a classifier is relatively certain that the output for a particular input is 1, what probability should the classifier output?
{% endcapture %}
{% capture part_b_sol %}
The output would be close to 1 (e.g., 0.99).  The degree of closeness to 1 would depend on how certain the classifier was.
{% endcapture %}

{% include problem_part.html label="B" subpart=part_b solution=part_b_sol %}

{% capture part_c %}
If a classifier is relatively certain that the output for a particular input is 0, what probability should the classifier output?
{% endcapture %}
{% capture part_c_sol %}
The output would be close to 0 (e.g., 0.01).  The degree of closeness to 0 would depend on how certain the classifier was.
{% endcapture %}

{% include problem_part.html label="C" subpart=part_c solution=part_c_sol %}
{% endcapture %}
{% include problem_with_parts.html problem=problem %}

## Log loss

If our model outputs a probability $p$ when supplied with an input $\mlvec{x}$ (i.e., $\hat{f}(\mlvec{x}) = p$), we might then ask ourselves what loss function we should choose in order to select the best possible model?  This loss function will be used to quantify how bad a prediction $p$ is given the actual output $y$ (recall that for binary classification the output is either $0$ or $1$).  To make this more intuitive, consider the task of quantifying the quality of a weatherperson's predictions.  Let's assume that on the $i$th day the weather is either sunny ($y_i = 1$) or rainy ($y_i = 0$).  Suppose that each night the weatherperson gives the probability of it being sunny the next day.  Here are two potential choices for quantifying the loss of each prediction compared to the outcome (the actual weather).

1. **0-1 loss:** we will extract from the weatherperson's prediction the most likely output (e.g., if $p = 0.75$, that would be sunny, if $p = 0.4$, that would be rainy).  If the most likely output matches the actual output we give a loss of 0, otherwise we give a loss of 1 (this is similar to Equation **~\ref{eq:minimizeerror})**.
2. **squared loss:** one downside of \emph{0-1 loss} is that it doesn't take into account the certainty expressed by the weatherperson.  The weatherperson gets the same loss if it is rainy and they predicted $p = 0.51$ or $p = 1$.  For squared loss we compute the difference between the outcome and $p$ and square it to arrive at the loss.  For example if the weatherperson predicts $p = 0.51$ and it is sunny the loss is $(1 - 0.51)^2$.  If it was rainy in this same example, the loss is $(0 - 0.51)^2$.

As an example, here are hypothetical predictions from two forecasters, the actual weather, and the resulting loss with either `0-1 loss` or `squared loss`.


| actual weather     | forecast 1 | 0-1 loss | squared loss       | forecast 2 | 0-1 loss | squared loss         |
|--------------------|------------|----------|--------------------|------------|----------|----------------------|
| sunny (y = 1)      | p = 0.2    | 1        | (1 - 0.2)^2 = 0.64 | p = 0.9    | 0        | (1 - 0.9)^2 = 0.01    |
| rainy (y = 0)      | p = 0.6    | 1        | (0 - 0.6)^2 = 0.36 | p = 0.999  | 1        | (0 - 0.999)^2 = 0.998 |
| sunny (y = 1)      | p = 0.8    | 0        | (1 - 0.8)^2 = 0.16 | p = 0.99   | 0        | (1 - 0.99)^2 = 0.0001 |
| **sum**            |            | **2**    | **1.16**            |            | **1**    | **1.01**              |




{% capture problem %}
According to the table above, which forecaster is better with regards to `0-1 loss`?  Which forecaster is better with regards to `squared loss`?

{% endcapture %}

{% capture sol %}
Forecaster 2 is better with respect to both loss functions (the losses are, on average, smaller).
{% endcapture %}
{% include problem.html problem=problem solution=sol %}


One entry in the table above is particularly interesting.  In the third row the second forecaster assigned a probability of $0.999$ to it being sunny.  It turned out to rain (boo!!!).  The forecaster was almost certain it would be sunny and it wasn't.  The 0-1 loss of course doesn't capture this at all.  The squared loss seems to assign a fairly large loss.  One might argue, though, that this loss does not fully capture how bad the prediction was (for one thing the loss can never be above 1).  This last observation motivates a third loss function that we can use to evaluate probabilistic predictions: the log loss.



{% capture content %}
[//]: <> [(30 minutes)]
[Towards Data Science](https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a) has a nice writeup that explains the concept of log loss.  Or you're welcome to search for your own resources about log loss. If you find a nice video, please post it to the Slack so others can enjoy too.
{% endcapture %}
{% include external_resources.html content=content %}





{% capture problem %}
Revisit the example from before with the two weather forecasters.  Compute the log loss for each forecaster.  Who makes better predictions according to the log loss?
{% endcapture %}

{% capture sol %}

| actual weather     | forecast 1 | log loss       | forecast 2 | log loss       |
|--------------------|------------|----------------|------------|----------------|
| sunny (y = 1)      | p = 0.2    | -ln 0.2        | p = 0.9    | -ln 0.9        |
| rainy (y = 0)      | p = 0.6    | -ln 0.4        | p = 0.999  | -ln 0.001      |
| sunny (y = 1)      | p = 0.8    | -ln 0.8        | p = 0.99   | -ln 0.99       |
| **sum**            |            | **2.75**       |            | **7.02**       |


{% endcapture %}
{% include problem.html problem=problem solution=sol %}


# Confounding variables
We discussed confounding variables in class.
{% capture problem %}
{% capture part_a %}
Please watch this video on [Simpson's Paradox](https://www.youtube.com/watch?v=sxYrzzy3cq8). Explain what happen's in Simpson's paradox.
{% endcapture %}
{% capture part_a_sol %}
See the video.
{% endcapture %}
{% include problem_part.html label="A" subpart=part_a solution=part_a_sol %}

{% capture part_b %}
Please come up with an example of Simpson's paradox that was not mentioned in the video. You don't have to know for sure that your example falls in to Simpson's paradox-- a reasonable suspicion of a confounding variable is fine. Please describe the paradox and the confounding variable that you suspect. This is a simple question, but I realize it may be challenging to answer. If you are stuck, you might consider studies related to diet or health, the pandemic, or statistics related to elections. You may look up recent studies or articles to spark your imagination. I would discourage you from Googling "Simpson's paradox examples"-- if you do, please adjust your response in the quiz accordingly.
{% endcapture %}

{% capture part_b_sol %}
 Simpson's paradox has to do with a *lurking* or *confounding* variable that isn't accounted for. 
{% endcapture %}

{% include problem_part.html label="B" subpart=part_b solution=part_b_sol %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}


# Putting stuff from potential assignment 5 (old assignment 4 here to make it easier to organize and decide what to put where.

# Learning Objectives

{% capture content %}
* Learn about the logistic regression algorithm.
* Learn about gradient descent for optimization.
* Contemplate an application of machine learning for home loans.
{% endcapture %}
{% include learning_objectives.html content=content %}

This builds on:
* Supervised learning problem framing.
* Calculating gradients.
* Log loss.

# Logistic Regression (top-down)

In the last part of the [notebook that you started in class](https://colab.research.google.com/drive/1AOUbSKhEvoSTzu_UNm-kq1SBrmmXPHVl?usp=sharing), you saw a quick implementation of logistic regression to classify if a person was looking to the left or to the right. 

In this assignment we will formalize the binary classification problem and dig the theory behind `logistic regression`.  You will also see that the logistic regression algorithm is a very natural extension of linear regression.  Our plan for getting there is going to be pretty similar to what we did for linear regression.

* Build some mathematical foundations
* Introduce logistic regression from a top-down perspective
* Learn about logistic regression from a bottom-up perspective


{% capture content %}
## Recall

In the last assignment, you were introduced to the idea of binary classification, which based on some input $\mlvec{x}$ has a corresponding output $y$ that is $y= 0$ or $y= 1$. In logistic regression, this model, $\hat{f}$, instead of spitting out either 0 or 1, outputs a confidence that the input $\mlvec{x}$ has an output $y= 1$.  In other words, rather than giving us its best guess (0 or 1), the classifier indicates to us its degree of certainty regarding its prediction as a probability.

We also explored three possible loss functions for a model that outputs a probability $p$ when supplied with an input $\mlvec{x}$ (i.e., $\hat{f}(\mlvec{x})=p$). The loss function is used to quantify how bad a prediction $p$ is given the actual output $y$ (for binary classification the output is either $0$ or $1$).

1. **0-1 loss:** This is an all-or-nothing approach. If the prediction is correct, the loss is zero; if the prediction is incorrect, the loss is 1. This does not take into account the level certainty expressed by the probability (the model gets the same loss if $y = 1$ and it predicted $p = 0.51$ or $p = 1$).
2. **squared loss:** For squared loss we compute the difference between the outcome and $p$ and square it to arrive at the loss.  For example, if $y = 1$ and the model predicts $p = 0.51$, the loss is $(1 - 0.51)^2$.  If instead $y = 0$, the loss is $(0 - 0.51)^2$.
3. **log loss:** The log loss also penalizes based on the difference between the outcome and $p$, using the formula below.
$$
\begin{align}
 logloss = -\frac{1}{N}\sum_{i=1}^n ( (y_i) \ln (p_i) + (1-y_i) \ln (1 - p_i) )\label{eq:loglosseq}
\end{align}
$$
Since $y_i$ is always 0 or 1, we will essentially switch between the two chunks of this equation based on the true value of $y_i$. As the predicted probability, $p_i$ (which is constrained between 0 an 1) gets farther from $y_i$, the log-loss value increases.

{% endcapture %}
{% include notice.html content=content %}


Now that you have refreshed on how probabilities can be used as a way of quantifying confidence in predictions, you are ready to learn about the logistic regression algorithm.

As always, we assume we are given a training set of inputs and outputs.  As in linear regression we will assume that each of our inputs is a $d$-dimensional vector $\mathbf{x_i}$ and since we are dealing with binary classification, the outputs, $y_i$, will be binary numbers (indicating whether the input belongs to class 0 or 1).  Our hypothesis functions, $\hat{f}$, output the probability that a given input has an output of 1.  What's cool is that we can borrow a lot of what we did in the last couple of assignments when we learned about linear regression.  In fact, all we're going to do in order to make sure that the output of $\hat{f}$ is between 0 and 1 is pass $\mlvec{w}^\top \mlvec{x}$ through a function that ``squashes'' its input so that it outputs a value between 0 and 1.  This idea is shown graphically in this {% include figure_reference.html fig_num=graphicaldataflow %}.

{% include figure.html
        img="figures/linearandlogistic.png"
        alt="a schematic of a neural network is used to represent linear and logistic regression.  Circles represent nodes, which are connected to other nodes using arrows. Logistic regression looks like linear regression followed by a sigmoid function."
        caption="Graphical representation of both linear and logistic regression.  The key difference is the application of the squashing function shown in yellow. [Original Source - Towards Data Science](https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24)" %}
{% assign graphicaldataflow = figure_number %}

To make this intuition concrete, we define each $\hat{f}$ as having the following form (note: this equation looks daunting. We have some tips for interpreting it below).

$$
\begin{align}
\hat{f}(\mathbf{x}) &= \mbox{probability that output, $y$, is 1} \nonumber  \\  
&=\frac{1}{1 + e^{-\mlvec{w}^\top \mathbf{x}}} \label{eq:logistichypothesis}
\end{align}
$$

Here are a few things to notice about this equation:
1. The weight vector that we saw in linear regression, $\mlvec{w}$, has made a comeback. We are using the dot product between $\mlvec{x}$ and $\mlvec{w}$ (which creates a weighted sum of the $x_i$'s), just as we did in linear regression!
2. As indicated in {% include figure_reference.html fig_num=graphicaldataflow %}, the dot product $\mlvec{w}^\top \mlvec{x}$ has been passed through a squashing function known as the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).  The graph of $\sigma(u) = \frac{1}{1+e^{-u}}$ is shown in {% include figure_reference.html fig_num=sigmoid %}.  $\sigma( \mlvec{w}^\top \mlvec{x})$ is exactly what we have in Equation~\ref{eq:logistichypothesis}. 


{% include figure.html
        img="figures/Logistic-curve.png"
        alt="a sigmoid function that is flat, curves up, and then flattens out again"
        caption="A graph of the sigmoid function $\frac{1}{1+e^{-x}}$." %}
{% assign sigmoid = figure_number %}


# Deriving the Logistic Regression Learning Rule