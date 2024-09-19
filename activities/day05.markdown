---
title: Day 5
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:30am: Debrief at tables
* 10:30-11:00am: Logistic Regression Example Problem
* 10:50-11:30am: Logistic Regression Learning Rule
* 11:30-12:00pm: Foundations of Micrograd
{% endcapture %}

{% include agenda.html content=agenda %}

# Debrief on the last assignment (5 minutes)

Warm up your brains by refreshing on the last assignment, including the derivation of linear regression... we're going to use it in a minute.

# Logistic Regression Example Problem (30 minutes)

We'll remind ourselves of the basic idea of logistic regression, and then together we'll go through a Colab notebook that shows [an example logistic regresion problem](https://colab.research.google.com/drive/1xpGvY-kg7-HOC7_To0nMZIOOHQ_Yxd89?usp=sharing).

{% include figure.html
        img="../assignments/assignment05/figures/linearandlogistic.png"
        alt="a schematic of a neural network is used to represent linear and logistic regression.  Circles represent nodes, which are connected to other nodes using arrows. Logistic regression looks like linear regression followed by a sigmoid function."
        caption="Graphical representation of both linear and logistic regression.  The key difference is the application of the squashing function shown in yellow. [Original Source - Towards Data Science](https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24)" %}
{% assign graphicaldataflow = figure_number %}

# Logistic Regression Learning Rule (40 minutes)

Let's use [assignment 5](../assignments/assignment05/assignment05) to begin to unpack some of the concepts behind choosing the best set of weights for logistic regression.  Before we start, we'll go over our high-level strategy.

# Foundations of Micrograd

