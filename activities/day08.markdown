---
title: Day 8
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:45am: Instructor-led debrief of the homework... what just happened?!?
* 10:45-11:05am: Cross-entropy loss
* 11:05-12:00pm: From micrograd to Pytorch
{% endcapture %}
{% include agenda.html content=agenda %}

# Instructor-led Debrief

We'll debrief on what happened in the previous assignment.  The focus will be on connecting mathematical concepts to Python.  We hope that by the end of this everything is coming into focus for you (it may take a little longer to fully click).

# Cross entropy loss and softmax

Before we get to the main activity of today, we want to extend some of the concepts we built up from binary classification and extend them to the situation where we have more than 2 classes (multiclass).  Recalling binary logistic regression, we needed a way to assign a probability to the class being 1.  To do this, we passed our weighted sum of features, $s$, through the sigmoid function $\sigma(s) = \frac{1}{1+e^{-s}}$.  In the multi-class case (let's say we have $k$ classes), we assume we have a weighted sum of features for each of these k classes $s_1, s_2, \ldots, s_k$.  We now calculate the probability of each particular class using the following formula the *softmax* function.

\begin{align}
p(y = i) = \frac{e^{s_i}}{\sum_{j=1}^{k} e^{s_j}}
\end{align}

This is a bit of a handful, so let's go through it together.  One special case we can consider is when $k=2$ and we set $s_1 = 0$ (we should recover the same formula is in the binary case).  We should also verify that this gives us a valid probability distribution (all probabilities are between 0 and 1 and sum to 1)/

Now that we have a way to calculate probabilities, we need to figure out how to assign a loss to any particular prediction.  The loss function we're going to use here is called *cross entropy* and we'll use the notation $ce$ to refer to it.  Let's use the shorthand $\hat{y}_i$ to be $p(y=i)$ (as defined, for example, by the softmax formula).  We can now think of $\mlvec{\hat{y}}$ as a vector of all of these probabilties.

\begin{align}
ce(\hat{\mlvec{y}}, y) = \sum_{i=1} \log \hat{y}_i -\mathbb{I}[y = i]
\end{align}

Let's compare this formula to our formula for the log loss in the binary case and draw some parallels.  In this case $\hat{y}$ will be the probability that the class label is $1$.

\begin{align}
\ell(\hat{y}, y) = -y \log(\hat{y}) - (1-y)\log(1-\hat{y})
\end{align}

# From Micrograd to Pytorch

While it may be tempting to ride our micrograd framework for the rest of the semester, you can probably tell that there are some good reasons to move to something *a little* more powerful.  We're going to be using the `pytorch` framework for the remainder of the scaffolded work in this course (it's possible you might venture into a different framework for the final project).  Machine learning frameworks like `pytorch` provide some really important capabilities for us.

* An autograd engine
* Built-in optimizers (that do, for example, gradient descent)
* Optimized code that can efficiently handle large models (e.g., by running on a GPU or across several GPUs)
* Specific building blocks for machine learning algorithms that are used by current state of the art algorithms.
* The ability to be extended easily when the library doesn't provide the necessary functionality.

To help introduce `pytorch`, we're going to jump right into a looking at some `pytorch` code.  This is a great chance to practice reading code and looking up documentation.  Your goal should be to understand the given code as well as possible.  If there are pieces that you can't figure out, please ask us or make a note of your confusion so you can revisit it later.  You'll also get a head start on the assignment (so that is a bonus!).

The code in question is in the [assignment 8, part 2 Colab notebook](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Assignment08_part_2.ipynb). The first two code cells load a dataset of handwritten digits and visualize them.  The third code cell is where the action is, we'd like you to go over that one, read documentation, ask ChatGPT, ask an instructor, etc., so that you leave here today with a solid understanding of a training / testing loop in `pytorch`.


# More Resources on Pytorch

We're going to be introducing Pytorch functionality on an as needed basis, but if you'd like to get some more practice with the basics, we recommend checking out some of [the Pytorch tutorials](https://pytorch.org/tutorials/).  Start with the [basics of using Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).