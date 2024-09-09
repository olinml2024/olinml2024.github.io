---
title: Day 2
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:45am: Debrief at tables about the last assignment.
* 10:45-10:50am: Split into two rooms, depending on desire for intro to types of ML
* 10:50-11:15am: (Room 128, else skip down) Discuss types of ML and general workflow
* 11:15-11:40am: Explore types of image transforms
* 11:40-12:00pm: Start assignment 
{% endcapture %}

{% include agenda.html content=agenda %}

# Debrief on the last assignment

1. Introduce yourselves
2. Quickly draw a confusion matrix at your table and write the equations for accuracy, precision, and recall.
3. Discuss your answers for Exercise 9 in the Colab notebook (see exercise below as a reminder).

    Exercise 9: Summarize how well the dessert classifier works for french toast and red velvet cake.
    Come to class prepared to share this at your table.
    Consider the confusion matrix, precision, and recall. How do you interpret this?
    What does it mean for life as french toast or as red velvet cake?


# Types of ML and general ML workflow

We will talk about some types of machine learning and the general machine learning workflow.

There are a few different ways to categorize machine learning problems, but most texts will reference the three main types of machine learning problems.

## Supervised Learning

<p>In supervised learning, you are given a training set of data points and corresponding desired outputs.  Let's use $\mathbf{x}_i$ to denote the $i$th training input and $y_i$ to denote the $i$th training output.  The training set is $\mlmat{X}_{train} = (\mlvec{x}_1, y_1), (\mlvec{x}_2, y_2), \ldots, (\mlvec{x}_N, y_N)$</p>

We'll talk about when this type of problem shows up and how the problem changes depending on the values that $y_i$ can take on.

<p>A supervised machine learning algorithm can take as input $\mlmat{X}_{train}$ and produce a model capable of taking in an unseen datapoint, $\mlvec{x}_{test}$, and estimating the corresponding label, $y_{test}$.  In order to evaluate the quality of these predictions, you'll want to have a set of test points, $\mlmat{X}_{test}$ to compute a relevant performance metrix (as we did in assignment 1).</p>

```mermaid!
graph TB;
    id1[X Train];
    id2[Supervised Learning Algorithm];
    id3[Predictive Model];
    id4[X Test]
    id5[Model Metrics]
    id1 --> id2;
    id2 --> id3;
    id4 --> id3;
    id3 --> id5;
```

In addition to having a test set, you may also use a validation set to help tune your machine learning model.  We'll talk a bit about how this would work.

## Unsupervised Learning

<p>In unsupervised learning, you are given a of data points (there are no corresponding outputs).  The training set is $\mlmat{X}_{train} = \mlvec{x}_1, \mlvec{x}_2 \ldots, \mlvec{x}_N$.</p>

In an unsupervised learning problem, our goal is to understand something about the structure of these training points.  For example, perhaps the data lies in some low dimensional subspace (sounding a little familiar?).  Examples of problems that fit under unsupervised learning are clustering, sequence learning (e.g., as is done in language models), and dimensionality reduction.

## Reinforcement Learning

Reinforcement learning involves an agent learning to interact with an environment in an optimal fashion.  We wont define notation for reinforcement learning as we aren't planning to cover it in this class (it could be a great final project).  Examples of reinforcement learning problems would be an agent learning to play a game (e.g., Chess), a robot learning to interact with its environment, or even determining treatment regimes in a clincal setting.  The reinforcement learning book has [a bunch of sample applications](https://rl-book.com/applications/) if you are curious.

# Exploring Image Transforms

In the next assignment, you'll be evaluating some existing models. As part of this, you may apply different transformation to the images that you'll use to evaluate the model. These same types of transforms can also be used to augment an initial dataset, often making models trained on it more robust. In class, we'll show a few transforms and how to find documentation for others.

Let's look at some simple examples in this [Google Colab notebook](https://colab.research.google.com/drive/1lBUjxz5hJleKTt_zTrFwSyER0Er3wA8-?usp=sharing).  Once we've gone over these, we'll jump to [Assignment 2](../assignments/assignment02/assignment02.markdown) and show these same concepts for the dessert classification problem.