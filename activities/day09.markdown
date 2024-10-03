---
title: Day 9
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:45am: Demystfying Pytorch
* 10:25-10:55am: Cross entropy and how to interpret the graphs from the homework
* 11:00-11:20am: Small data mini-project on classification
* 11:20-12:00pm: Choosing data for your mini-project and start working
{% endcapture %}
{% include agenda.html content=agenda %}


# Demistifying Pytorch

We'll be going through [the day 9 notebook](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Day09.ipynb).  There are three things we want you to get out of this notebook.

* Pytorch is quite similar in its basic concepts to the micrograd framework you implemented.
* We can use pytorch to compute a line of best fit.  This will allow us to visualize the optimization process more easily.
* We can use pytorch modules (e.g., `nn.Linear`) to make our lives easier.


# Cross entropy loss and softmax

In the previous assignment you generated graphs that showed the cross entropy of a model to classify handwritten digits.  These graphs looked something like this.

{% include figure.html 
  img="images/learning_curve_ce.png"
    width="100%"
    alt="A graph of training and test cross entroyp as a function of gradient descent step.  The curves begin near 2.4 and settle around 1.7"
    caption="The cross entropy on the handwritten digit classification task.  The x-axis refers to the number of gradient descent steps." %}

Right now we are going to help you interpret what these graphs mean.  The y-axis is cross entropy, which for now we can simply understand as a measure of the model's loss when its predictions are compared to the actual classes of the digits in either the training (blue line) or the test set (orange line).  The x-axis of this graph should be fairly easy to interpret.  The axis is labeled *step*, which refers to how many gradient descent steps have been taken by your optimizer in order to drive down the loss.

In order to interpret these graphs we are going to need two ingredients.  First, we need to understand how a classifier, in response to a given input, can assign a probability of that input being a member of each of $k$ possible classes (notice how this contrasts with the binary classification case where we had to assign a single probability of the input being a $1$).  Second, we need a way to assign a loss value (cross entropy in this case) given a set of predicted probabilities and the actual class label of the digit.

## Assigning probabilities when there are more than 2 classes

Recalling binary logistic regression, we needed a way to assign a probability to the class being 1.  To do this, we passed our weighted sum of features, $s$, through the sigmoid function $\sigma(s) = \frac{1}{1+e^{-s}}$.  In the multi-class case (again, where we have $k$ classes), we assume that we have computed a weighted sum of features for each of these k classes $s_1, s_2, \ldots, s_k$.  We now calculate the probability of each particular class using the following formula called the *softmax* function.

\begin{align}
p(y = i) = \frac{e^{s_i}}{\sum_{j=1}^{k} e^{s_j}}
\end{align}

Here are some exercises to help you think through this.

{% capture problem %}
* Probabilities should always be non-negative and less than or equal to 1.  Additionally, a set of probabilities that forms a probability distribution should add up to 1.  Show that both of these conditions are satisfied for the softmax function.
* Think about some limiting cases, what happens to the probability for class $i$ when $s_i$ gets really big?  What about when it becomes very negative?
* Consider the case where $k=2$ and $s_1 = 0$.  How does this relate to the sigmoid function we learned about for log loss?
{% endcapture %}
{% include problem.html problem=problem solution="" %}

## Calculating cross entropy

Now that we have a way to calculate probabilities, we need to figure out how to assign a loss to any particular prediction.  The loss function we're going to use here is called *cross entropy* and we'll use the notation $ce$ to refer to it.  Let's use the shorthand $\hat{y}_i$ to be $p(y=i)$ (as defined, for example, by the softmax formula).  We can now think of $\mlvec{\hat{y}}$ as a vector of all of these probabilties.

\begin{align}
ce(\hat{\mlvec{y}}, y) = \sum_{i=1}^{k} -\mathbb{I}[y = i] \log \hat{y}_i 
\end{align}

The following exercise will take you through some important takeaways.

{% capture problem %}
* Make sure you understand the role of the indicator function $\mathbb{I}$, what is it doing to the terms in the summation?
* The formula for log loss for binary classification is $\ell(\hat{y}, y) = -y \log(\hat{y}) - (1-y)\log(1-\hat{y})$.  Show that this formula is essentially the same as cross entropy when $k=2$.
* Imagine that at the beginning of the learning process the digit classifier assigns equal probability to each digit (0-9) regardless of what the actual class is (i.e., the model hasn't learned anything yet).  What do you think the model's cross entropy should be in this case?
{% endcapture %}
{% include problem.html problem=problem solution="" %}


# Small data mini-project on classification


# Choosing data for your mini-project and start working
Our general recommendation is to choose a dataset that has at least one other person working on that dataset. This is not a requirement, so if you have something you're passionate about, go for it! While this is a solo project, it may be helpful to have others to confer with who are also figuring out the nuances of your dataset. A lot of time in machine learning and data science go into interacting with the data before it even goes into the model. There are many canned (pre-curated data sets) out there which reduce this time, but it's still important to understand your data, as it drives your model. 

We'll do a little activity to help you find others who have overlapping dataset interest.
