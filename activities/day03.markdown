---
title: Day 3
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:45am: Debrief at tables about the dessert assignment.
* 10:45-10:55am: Supervised Learning Problem SEtup
* 10:55-12:00pm: Start assignment 
{% endcapture %}

{% include agenda.html content=agenda %}

# Debrief on the last assignment

1. Introduce yourselves.
2. Present your individual findings regarding the dessert classifier (you don't have to repeat something someone else has already presented, try to just give the unique things you investigated or found).  Collectively, what are your recommendations in terms of which system might be ready for production?
3. If you were continuing to work on this project, what next steps would you take to further validate your model (e.g., having users test the model in realistic conditions, collect more test data, etc.)?
4. Note any challenges or unanswered questions regarding Colab and working with an image-based dataset.

# The Supervised Learning Problem Setup

Next, we're going to investigate the mathematical definition of supervised learning (which we touched upon last class).  This is also in the homework, but we wanted to have a chance to go over this together.

Suppose you are given a training set of data points, $(\mlvec{x_1}, y_1), (\mlvec{x}_2, y_2), \ldots, (\mlvec{x}_n, y_n)$ where each $\mlvec{x_i}$ represents an element of an input space (e.g., a d-dimensional feature vector) and each $y_i$ represents an element of an output space (e.g., a scalar target value).  In the supervised learning setting, your goal is to determine a function $\hat{f}$ that maps from the input space to the output space.  For example, if we provide an input $\mlvec{x}$ to $\hat{f}$ it would generate the predicted output $\hat{y} = \hat{f}(\mlvec{x})$.

We typically also assume that there is some loss function, $\ell$, that determines the amount of loss that a particular prediction $\hat{y_i}$ incurs due to a mismatch with the actual output $y_i$.  We can define the best possible model, $\hat{f}^\star$ as the one that minimizes these losses over the training set.  This notion can be expressed with the following equation  (note: that $\argmin$ in the equation below just means the value that minimizes the expression inside of the $\argmin$, e.g., $\argmin_{x} (x - 2)^2 = 2$, whereas $\min_{x} (x-2)^2 = 0$).

\begin{align}
\hat{f}^\star &= \argmin_{\hat{f}} \sum_{i=1}^n \ell \left ( \hat{f}(\mlvec{x_i}), y_i \right )
\end{align} 
