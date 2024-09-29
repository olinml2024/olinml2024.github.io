---
title: Assignment 8
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Implement a multi-layer neural network
* Gain some practice writing pytorch code
* Understand how a multi-layer network can solve difficult machine learning problems. 
{% endcapture %}
{% include learning_objectives.html content=content %}

# Neural Networks Motivation

We're going to start out our journey into neural networks by revisiting the Titanic dataset that we saw a couple of classes ago.  The exercises are embedded within the [assignment 8 Colab notebook](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Assignment08.ipynb).  Go through those exercises, and then check back here as we unpack things futher.

# Neural Networks as Stacked Logistic Regression Models

Now that you've seen a neural network in action, we'll be digging into how a neural network works.  The presentation will be specific to a particular type of neural network that we used in the companion notebook known as a multilayer perceptron (MLP), but the main ideas generalize to many other types of networks.  While the name MLP might be a bit intimidating, what we'll see in just a bit is that an MLP is nothing more than some logistic regression models stacked on top of each other!

Thinking back to the Colab notebook, we observed that the features in the original dataset *age* and *sex* were not conducive to predicting whether someone would survive.  We showed that by augmenting the input features with a column called *is young male* that captured whether or not a person was young *and* male, that the algorithm could effectively learn the task.  The fundamental idea of a neural network is that the network automatically constructs useful representations of the input data *as a part of the learning process*.

Before moving on, let's show some diagrams that contrast these approaches.  First we'll show the logistic regression model that we applied in the notebook.

```mermaid!
flowchart BT
id1["age"]
id2["male"]
id3["is young male"]
id4["1"]
id5["$$p(survival) = 1/(1+e^{-s})~~~~$$"]
id6["$$s = w_1 \text{age} + w_2 \text{male} + w_3 \text{is young male} + w_4~~~$$"]
id1 --"$$w_1$$"--> id6
id2 --"$$w_2$$"--> id6
id3 --"$$w_3$$"--> id6
id4 --"$$w_4$$"--> id6
id6 --> id5
```

 A few notes here:
 * Instead of putting the weights (e.g., $w_1$) as separate boxes (which we've done when computing partial derivatives), here we are putting them on the arrows between the features and the $s$.  This is a common way of writing the architecture for neural networks, and we want you to be familiar with it.  If you prefer to think of a separate box for each weight connecting to $s$, that is fine too.
 * To be consistent with the notebook, we are using a variable that takes on 1 if the sex of the passenger is male (this is different than what we did a few classes ago when we used 1 for female).

Notice how we had to manually introduce the feature *is young male* in order for the logistic regression model to utilize it to make its prediction.  Before giving you the equivalent figure for the multi-layer perceptron, let's look at a little bit more cartoonish version of the multi-layer perceptron.  This version will leave off the math and the particular notation we are using.  Once you have a good sense of what this model is doing, we will draw a diagram to represent the multi-layer perceptron that you met in the notebook.  If you are looking for more information on how to think about MLPs, check out the 3B1B videos we linked previously.

{% capture external %}
Here are some additional resources that explain the concept of a multi-layer perceptron.  If the explanations we give below are not working for you, consider checking out some of these.  **You do not need to consult these resources if you feel like our explanations are working well for you.**
* [3B1B: What is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
* [2D fully connected network visualization](https://adamharley.com/nn_vis/mlp/2d.html)
{% endcapture %}
{% include external_resources.html content=external %}


```mermaid!
flowchart BT
id1["age"]
id2["male"]
id4["1"]
id10["$$p(survival) = 1/(1+e^{-s_3})~~~~$$"]
id6["$$s_1 = w_{1,1} \text{age} + w_{1,2} \text{male} + w_{1,3}~~~$$"]
id8["$$s_2 = w_{2,1} \text{age} + w_{2,2} \text{male} + w_{2,3}~~~$$"]
id7["$$h_1 = 1/(1+e^{-s_1})$$"]
id9["$$h_2 = 1/(1+e^{-s_2})$$"]
id11["1"]
id12["$$s_3$$"]
id1 --"$$w_{1,1}$$"--> id6
id2 --"$$w_{1,2}$$"--> id6
id4 --"$$w_{1,3}$$"--> id6
id1 --"$$w_{2,1}$$"--> id8
id2 --"$$w_{2,2}$$"--> id8
id4 --"$$w_{2,3}$$"--> id8
id6 --> id7
id8 --> id9
id7 --"$$w_{3,1}$$"--> id12
id9 --"$$w_{3,2}$$"--> id12
id11 --"$$w_{3,3}$$"--> id12
id12 --> id10
```

Oh no! Our notation has gotten more complicated.  Notice here that we are using subscripts to differentiate between our various summation nodes ($s_1$ versus $s_2$).  We are also using $w_{i,j}$ to refer to weight that corresponds to the $i$th summation node and the $j$th feature (e.g., $w_{1,2}$ tell us how much the feature *age* influences $s_2$).

Input data (in this case we just use age, male, and a bias term) are propagated via a set of connection weights to a set of hidden representations ($h_1$ and $h_2$).  These hidden representations are propagated via another set of a connection weights to the output of the network.   In the companion notebook we showed that for the Titanic dataset, the network learned two hidden representations: one that seemed to encode *is young male* and the another that encoded sex.  Of particular importance is that we did not have to manually introduce the *is young male* feature.

{% capture problem %}
Before going on, let's make sure you have a firm handle on what's being represented in the figure above.
* Just as in logistic regression, we will try to tune the weights to fit the data.  How many weights are there to tune in this network?
* While the figure looks pretty crazy, it has a lot of similarities with the logistic regression model.  Where does the logistic regression model show up in the figure?
{% endcapture %}
{% capture solution %}
* There are a total of 9 weights in the network.  There are 6 connecting the 3 input units to the 2 hidden summation units.  There are another 3 connecting the 1 output summation unit.
* There are three different logistic regression models represented.  There is one going from the inputs to the $s_1$.  There is another going from the inputs to $s_2$.  There is a third going from the hidden units ($h_1$ and $h_2$) to $s_3$.  The models are connected together such that the two lower logistic models feed into the higher-level one.
{% endcapture %}
{% include problem.html problem=problem solution=solution %}


{% capture content %}
For each of the 3 behavors described below (questions a, b, and c), determine reasonable values for the weights in this network ($w_{1,1}, w_{1,2}, w_{1,3}, w_{2,1}, w_{2,2}, w_{2,3}, w_{3,1}, w_{3,2}, w_{3,3}$) so that the MLP behaves as described. You will not need to use any training data except general knowledge that a person's reported sex is recorded as 0 or 1 and age is within a set of reasonable numbers (this question is about testing your understanding of the model itself).  Recall that the first input to the model is the passenger's age, the second is a binary variable that is 1 if the passenger is male and 0 if female, and the third is always 1.
{% capture parta %}
$h_1$ encodes whether or not the passenger is female (i.e., it should take a value close to 1 when the passenger is female and close to 0 when the passenger is male).
{% endcapture %}
{% capture partasol %}
Setting $w_{1,2} = -10$, $w_{1,1} = 0$, and $w_{1,3} = 5$ will do the trick.  If the passenger is male then $h_1 = \sigma(-10 + 5) = \sigma(-5) = 0.0067$ and if the passenger is female then $h_1 = \sigma(5) = 0.9933$
{% endcapture %}
{% include problem_part.html label="A" subpart=parta solution=partasol %}

{% capture partb %}
$h_2$ encodes whether or not the passenger is a young male (i.e., it should take a value close to 1 when the passenger is male under the age of say 5 and close to 0 otherwise).
{% endcapture %}
{% capture partbsol %}
Setting $w_{2,2} = 15$, $w_{2,1} = -1$, and $w_{2,3} = -10$ will do the trick.  If the passenger is female and one years-old $h_2 = \sigma(-10 - 1) = \sigma(-11) \approx 0$ (any older female passengers will have even lower values. and if the passenger is male and 1 years-old then $h_2 = \sigma(15 - 1 - 10) = \sigma(4) = 0.982$.  A four year old male would have $h_2 = \sigma(15 - 4 - 10) = \sigma(1) = 0.731$.  An older male (e.g., a 10 year old) would have $h_2 = \sigma(15 - 10 - 10) = \sigma(-5) = 0.0067$
{% endcapture %}
{% include problem_part.html label="B" subpart=partb solution=partbsol %}
{% capture partc %}
$p(\text{suvival})$ should be close to 1 (i.e., predict survival) when the passenger is female \emph{or} a male under the age of 5 and close to 0 otherwise.
{% endcapture %}
{% capture partcsol %}
Set $w_{3,1} = 2$ and $w_{3,2} = 2$ and $w_{3,3} = -1$.  That way if either $h_1$ or $h_2$ are close to 1, then $p(\text{suvival}) \approx \sigma(1) = 0.73$.
{% endcapture %}
{% include problem_part.html label="C" subpart=partc solution=partcsol %}

{% endcapture %}
{% include problem_with_parts.html problem=content %}

Believe it or not, computing these weights by hand was fairly common before we had algorithms for automatically tuning weights from data.  The reason for this was that early techniques for learning the weights were very inefficient and often unable to converge to good solutions.  By now you've seen how to tune these weights using gradient descent for a logistic regresion model, and given what we've learned from implementing the micrograd framework, you probably suspect that the same approach could be used here.

# Neural Networks in Pytorch

Next, we'll go back over to Colab to show how we can tune these weights automatically using pytorch.  Here is a listing of what you'll do in this notebook.
* You'll implement a multilayer perceptron to recognize handwritten digits (similar to [this previously linked example](https://adamharley.com/nn_vis/mlp/2d.html)).
* You'll see how overfitting can become an issue for more complex networks
* We'll introduce (at a very high-level) three methods for dealing with overfitting.

Okay, back to Colab for round 2.