---
title: Assignment 5
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn about the logistic regression algorithm.
* Learn about gradient descent for optimization.
* Contemplate an application of machine learning for home loans.
{% endcapture %}
{% include learning_objectives.html content=content %}

This builds on:
* [Supervised learning problem framing](/assignments/assignment03/assignment03?showAllSolutions=true#supervised-learning-problem-setup).
* Calculating gradients.
* [Log loss](/assignments/assignment04/assignment04?showAllSolutions=true#probability-and-the-log-loss)


# Logistic Regression (top-down)

<p style="color: red;">Let's modify this to be something more simple.</p>

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
2. As indicated in {% include figure_reference.html fig_num=graphicaldataflow %}, the dot product $\mlvec{w}^\top \mlvec{x}$ has been passed through a squashing function known as the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).  The graph of $\sigma(u) = \frac{1}{1+e^{-u}}$ is shown in {% include figure_reference.html fig_num=sigmoid %}.  $\sigma( \mlvec{w}^\top \mlvec{x})$ is exactly what we have in $$ \hat{f}(\mathbf{x}) =\frac{1}{1 + e^{-\mlvec{w}^\top \mathbf{x}}}$$

Equation~\ref{eq:logistichypothesis}. 
<p>
\begin{align}
\hat{f}(\mathbf{x}) &= \mbox{probability that output, $y$, is 1} \nonumber \\
&=\frac{1}{1 + e^{-\mlvec{w}^\top \mathbf{x}}}
\end{align}
</p>


{% include figure.html
        img="figures/Logistic-curve.png"
        alt="a sigmoid function that is flat, curves up, and then flattens out again"
        caption="A graph of the sigmoid function $\frac{1}{1+e^{-x}}$." %}
{% assign sigmoid = figure_number %}



# Deriving the Logistic Regression Learning Rule

Now we will formalize the logistic regression problem and derive a learning rule to solve it (i.e., compute the optimal weights). The formalization of logistic regression will combine Equation~\ref{eq:logistichypothesis} with the selection of $\ell$ to be log loss (Equation~\ref{eq:loglosseq}).  This choice of $\ell$ results in the following objective function.

<p>
\begin{align}
\mlvec{w}^\star &= \argmin_{\mlvec{w}} \sum_{i=1}^n \left ( - y_i \ln \sigma(\mlvec{w}^\top \mlvec{x_i}) - (1-y_i) \ln (1 - \sigma(\mlvec{w}^\top \mlvec{x_i}) ) \right)  \\  
&= \argmin_{\mlvec{w}} \sum_{i=1}^n \left (  - y_i \ln \left ( \frac{1}{1+e^{-\mlvec{w}^\top \mlvec{x_i}}} \right) - (1-y_i) \ln  \left (1 - \frac{1}{1+e^{-\mlvec{w}^\top \mlvec{x_i}}} \right ) \right) &\mbox{expanded out if you prefer this form} \label{eq:objective}
\end{align}
</p>

While this looks a bit intense, since $y_i$ is either 0 or 1, the multiplication of the expressions in the summation by either $y_i$ or $1-y_i$ are essentially acting like a switch---depending on the value of $y_i$ we either get one term or the other.  Our typical recipe for finding $\mlvec{w}^\star$ has been to take the gradient of the expression inside the $\argmin$, set it to $0$, and solve for $\mlvec{w}^\star$ (which will be a critical point and hopefully a minimum).  The last two steps will be a bit different for reasons that will become clear soon, but we will need to find the gradient.  We will focus on finding the gradient in the next couple of parts.

## Useful Properties of the Sigmoid Function

The equation for $\mlvec{w}^\star$ above looks really hairy! We see that in order to compute the gradient we will have to compute the gradient of $\mathbf{x}^\top \mlvec{w}$ with respect to $\mlvec{w}$ (we just wrapped our minds around this last assignment).  Additionally, we will have to take into account how the application of the sigmoid function and the log function changes this gradient.  In this section we'll learn some properties for manipulating the sigmoid function and computing its derivative.

{% capture problem %}
The sigmoid function, $\sigma$, is defined as

$$
\begin{align}
\sigma(x) &= \frac{1}{1+e^{-x}}
\end{align}
$$

{% capture part_a %}
Show that $\sigma(-x) = 1 - \sigma(x)$.
{% endcapture %}
{% capture part_a_sol %}

<p>
\begin{align}
\sigma(-x) &= \frac{1}{1+e^{x}} \\
&= \frac{e^{-x}}{e^{-x} + 1}~~\mbox{multiply by top and bottom by $e^{-x}$} \\
 \sigma(-x)  - 1&= \ \frac{e^{-x}}{e^{-x} + 1} - \frac{1 + e^{-x}}{1 + e^{-x}} ~~\mbox{subtract $-1$ on both sides} \\
 &= \frac{-1}{1+e^{-x}} \\
 &= -\sigma(x) \\
 \sigma(-x) &= 1 - \sigma(x)
\end{align}
</p>

{% endcapture %}
{% include problem_part.html label="A" subpart=part_a solution=part_a_sol %}

{% capture part_b %}
Show that the derivative of the logistic function $\frac{d}{dx} \sigma(x) = \sigma(x) (1 - \sigma(x))$
{% endcapture %}
{% capture part_b_sol %}
Two solutions for the price of 1!

Solution 1:
<p>
\begin{align}
\frac{d}{dx} \sigma(x)  &= e^{-x} \sigma(x)^2 &\mbox{apply quotient rule} \\
&= \sigma(x) \left ( \frac{e^{-x}}{1 + e^{-x}} \right) &\mbox{expand out one of the $\sigma(x)$'s}\\
&= \sigma(x) \left ( \frac{1}{e^{x} + 1} \right) & \mbox{multiply top and bottom by $e^{x}$}\\
&=  \sigma(x) (  \sigma(-x)) &\mbox{substitute for $\sigma(-x)$} \\
&=  \sigma(x) (1 -  \sigma(x) ) &\mbox{apply $\sigma(-x)=1-\sigma(x)$}
\end{align}
</p>

Solution 2:
<p>
\begin{align}
\frac{d}{dx} \sigma(x)  &=\frac{e^{-x}}{(1+e^{-x} )^2} & \mbox{apply quotient rule} \\
&= \frac{e^{-x}}{1+2e^{-x} + e^{-2x}} & \mbox{expand the bottom}\\
&= \frac{1}{e^{x}+2 + e^{-x}} & \mbox{multiply top and bottom by $e^{x}$}\\
&= \frac{1}{(1+e^{x})(1+e^{-x})} & \mbox{factor} \\
&= \sigma(x)\sigma(-x) & \mbox{decompose using definition of $\sigma(x)$}\\
&= \sigma(x)(1-\sigma(x)) &\mbox{apply $\sigma(-x)=1-\sigma(x)$}
\end{align}
</p>

{% endcapture %}

{% include problem_part.html label="B" subpart=part_b solution=part_b_sol %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}

## Chain Rule for Gradients
We now know how to take derivatives of each of the major pieces of Equation~\ref{eq:objective}.  What we need is a way to put these derivatives together.  You probably remember that in the case of single variable calculus you have just such a tool.  This tool is known as the chain rule.  The chain rule tells us how to compute the derivative of the composition of two single variable functions $f$ and $g$.  

<p>
\begin{align}
h(x)&= g(f(x))&\mbox{h(x) is the composition of $f$ with $g$} \nonumber \\
h'(x) &= g'(f(x))f'(x)&\mbox{this is the chain rule!}
\end{align}
</p>

Suppose that instead of the input being a scalar $x$, the input is now a vector, $\mlvec{w}$.  In this case $h$ takes a vector input and returns a scalar, $f$ takes a vector input and returns a scalar, and $g$ takes a scalar input and returns a scalar.

<p>
\begin{align}
h(\mlvec{w}) &= g(f(\mlvec{w}))&\mbox{h($\mlvec{w}$) is the composition of $f$ with $g$} \nonumber \\
\nabla h(\mlvec{w}) &= g'(f(\mlvec{w})) \nabla f(\mlvec{w}) & \mbox{this is the multivariable chain rule}
\end{align}
</p>


{% capture problem %}
[//]: <> [(60 minutes)]

{% capture part_a %}
Suppose $h(x) = \sin(x^2)$, compute $h'(x)$ (x is a scalar so you can apply the single-variable chain rule).
{% endcapture %}
{% capture part_a_sol %}
Applying the chain rule gives
$$
\begin{align}
h'(x) &= cos(x^2) 2x
\end{align}
$$
{% endcapture %}
{% include problem_part.html label="A" subpart=part_a solution=part_a_sol %}

{% capture part_b %}
Define $h(\mlvec{v}) = (\mlvec{c}^\top \mlvec{v})^2$.  Compute $\nabla_{\mlvec{v}} h(\mlvec{v})$ (the gradient of the function with respect to $\mlvec{v}$).
{% endcapture %}
{% capture part_b_sol %}
We can see that $h(\mlvec{v}) = g(f(\mlvec{v}))$ with $g(x) = x^2$ and $f(\mlvec{v}) = \mlvec{c}^\top \mlvec{v}$ The gradient can now easily be found by applying the chain rule.

$$
\begin{align}
\nabla h(\mlvec{v}) &= 2(\mlvec{c}^\top \mlvec{v}) \mlvec{c}
\end{align}
$$

{% endcapture %}

{% include problem_part.html label="B" subpart=part_b solution=part_b_sol %}

{% capture part_c %}

Compute the gradient of this expression, which comes from the beginning of the section on deriving the logistic regression learning rule:

$$
\begin{align}
 \sum_{i=1}^n -y_i \ln \sigma( \mlvec{w}^\top \mlvec{x_i}) - (1-y_i) \ln  \left (1 - \sigma( \mlvec{w}^\top \mlvec{x_i}) \right ) 
\end{align}
$$

You can either use the chain rule and the identities you learned about sigmoid, or expand everything out and work from that.


{% endcapture %}
{% capture part_c_sol %}

Applying the chain rule gives us

<p>
\begin{align}
 \sum_{i=1}^n -y_i \frac{\nabla \sigma( \mlvec{w}^\top \mlvec{x_i})}{\sigma( \mlvec{w}^\top \mlvec{x_i})} - (1-y_i) \frac{- \nabla \sigma( \mlvec{w}^\top \mlvec{x_i})}{1 - \sigma( \mlvec{w}^\top \mlvec{x_i})}  \enspace .
\end{align}
</p>

Applying the chain rule again gives us
<p>
\begin{align}
& \sum_{i=1}^n -y_i \frac{\sigma( \mlvec{w}^\top \mlvec{x_i})(1-\sigma( \mlvec{w}^\top \mlvec{x_i}))\nabla \mlvec{w}^\top \mlvec{x_i}}{\sigma( \mlvec{w}^\top \mlvec{x_i})} - (1-y_i) \frac{- \sigma( \mlvec{w}^\top \mlvec{x_i})(1-\sigma( \mlvec{w}^\top \mlvec{x_i}))\nabla \mlvec{w}^\top \mlvec{x_i}}{1 - \sigma( \mlvec{w}^\top \mlvec{x_i})} \nonumber \\
 &= \sum_{i=1}^n -y_i (1-\sigma( \mlvec{w}^\top \mlvec{x_i}))\mlvec{x_i} + (1-y_i)  \sigma( \mlvec{w}^\top \mlvec{x_i})) \mlvec{x_i} 
 \end{align}
 </p>
 
You could certainly stop here, but if you plug in $y=0$ and $y=1$ you'll find that the expression can be further simplified to:
 
 <p>
 \begin{align}
\sum_{i=1}^n  -(y_i - \sigma(\mlvec{w}^\top \mlvec{x_i})) \mlvec{x_i} \nonumber
 \end{align}
</p>


{% endcapture %}

{% include problem_part.html label="C" subpart=part_c solution=part_c_sol %}

{% endcapture %}
<div id="chainrule">
{% include problem_with_parts.html problem=problem %}
</div>

## Gradient Descent for Optimization

If we were to follow our derivation of linear regression we would set our expression for the gradient to 0 and solve for $\mlvec{w}$.  It turns out this equation will be difficult to solve due to the $\sigma$ function.  Instead, we can use an iterative approach where we start with some initial value for $\mlvec{w}$ (we'll call the initial value $\mlvec{w^0}$, where the superscript corresponds to the iteration number) and iteratively adjust it by moving down the gradient (the gradient represents the direction of fastest increase for our function, therefore, moving along the negative gradient is the direction where the loss is decreasing the fastest).

{% capture content %}
[//]: <> [(45 minutes)]
There are tons of great resources that explain gradient descent with both math and compelling visuals.

* Recommended: [Gradient descent, how neural networks learn - Deep learning, chapter 2, start at 5:20](https://www.youtube.com/watch?v=IHZwWFHWa-w)
* [An Introduction to Gradient Descent](https://medium.com/@viveksingh.heritage/an-introduction-to-gradient-descent-54775b55ba4f)
* [The Wikipedia page on Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
* [Ahmet Sacan's video on gradient descent](https://www.youtube.com/watch?v=fPSPdTjINi0) (this one has some extra stuff, but it's pretty clearly explained).
* There are quite a few resources out there, do you have some suggestions? (Share on Slack!)

{% endcapture %}
{% include external_resources.html content=content %}


{% capture problem %}
[//]: <> [(10 minutes)]
To test your understanding of these resources, here are a few diagnostic questions.

{% capture part_a %}
When minimizing a function with gradient descent, which direction should you step along in order to arrive at the next value for your parameters?

{% endcapture %}
{% capture part_a_sol %}
The negative gradient (since we are minimizing)

{% endcapture %}
{% include problem_part.html label="A" subpart=part_a solution=part_a_sol %}

{% capture part_b %}
What is the learning rate and what role does it serve in gradient descent?
{% endcapture %}
{% capture part_b_sol %}
The learning rate controls the size of the step that you take along the negative gradient.
{% endcapture %}

{% include problem_part.html label="B" subpart=part_b solution=part_b_sol %}

{% capture part_c %}
How do you know when an optimization performed using gradient descent has converged?
{% endcapture %}
{% capture part_c_sol %}
There are a few options.  One popular one is to check if the objective function is changing  only a minimal amount each iteration, the algorithm has converged.  You could also look at the magnitude of the gradient (which tells us the slope) to see if it is really small.
{% endcapture %}

{% include problem_part.html label="C" subpart=part_c solution=part_c_sol %}

{% capture part_d %}
True or false: provided you tune the learning rate properly, gradient descent guarantees that you will find the global minimum of a function.
{% endcapture %}
{% capture part_d_sol %}
False, the best gradient descent can do, in general, is converge to a local minimum.  If you know that the function you are optimizing has only one minimum, then this would also be the global minimum (this is the case for both linear and logistic regression).
{% endcapture %}
{% include problem_part.html label="D" subpart=part_d solution=part_d_sol %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}


If we take the logic of gradient descent and apply it to the logistic regression problem, we arrive at the following learning rule.  Given some initial weights $\mlvec{w^0}$, and a learning rate $\eta$, we can iteratively update our weights using the formula below.


We start by applying the results from our <a href="../assignment04/assignment04?showSolutions=true#chainrule">exercise on the chain rule.</a>

<p>
\begin{align}
\mlvec{w^{n+1}} &= \mlvec{w^n} - \eta \sum_{i=1}^n  -(y_i - \sigma(\mlvec{w}^\top \mlvec{x_i})) \mlvec{x_i} \\
&=  \mlvec{w^n} + \eta \sum_{i=1}^n  (y_i - \sigma(\mlvec{w}^\top \mlvec{x_i})) \mlvec{x_i}  ~~~\mbox{distribute the negative}
\end{align}
</p>

This beautiful equation turns out to be the recipe for logistic regression.

{% capture content %}

We won't be assigning a full implementation of logistic regression from scratch. In future assignments, we will spend more time applying logistic regression and gradient descent. 

If it's helpful for your learning to see a worked example with code now (to help the math make sense), you can optionally check out this [example of binary classification for admission to college](https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24), noting that some of the math notation is slightly different than ours. 

You are also welcome to implement logistic regression using gradient descent if it's helpful for your learning and/or if you already have significant experience with machine learning and want a challenge. This is completely optional, and we assume that most of you will not choose to do this. If you do decide to implement logistic regression using gradient descent, you will need to search for a good learning rate or you may consider implementing some [strategies for automatically tuning the learning rate](https://towardsdatascience.com/gradient-descent-algorithms-and-adaptive-learning-rate-adjustment-methods-79c701b086be).
{% endcapture %}
{% include notice.html content=content %}

<!-- # Machine learning for loans and mortgages

In this course, we'll be exploring machine learning from three different perspectives: the theory, the implementation, and the context, impact, and ethics.  -->


## Dataflow Diagrams



$$
\begin{align}
x &= x(t) \\
y &= y(t) \\
f &= f(x, y) \\
\end{align}
$$

```mermaid!
flowchart BT
 id1["$$f = f(x,y)~~~~$$"]
 id2["$$x = x(t)~~$$"]
 id3["$$y = y(t)~~$$"]
 id2 --> id1
 id3 --> id1
 t --> id2
 t --> id3
```

This flow chart represents how data moves from its inputs (in this case $x$ and $y$) to its outputs (in this case $f$).  If we were to take a chart like this and figure out how to evaluate a function given some inputs, you'd have to make sure you always evaluate the inputs to a block before you try to evaluate the block itself.  For instance, I wouldn't be able to evaluate the block $f = f(x,y)$ until I've evaluated the blocks $x = x(t)$ and $y=y(t)$.  To evaluate a block, you can imagine that the output of a block flows along the arrow into the downstream block, which then processes that input further until it arrives at the output.


## Data Flow Diagrams and the Chain Rule

{% capture content %}
This Harvey Mudd College calculus tutorials explain the concept of the chain rule using dataflow diagrams beautifully.  Go and read the [HMC Multivariable Chain Rule Page](https://math.hmc.edu/calculus/hmc-mathematics-calculus-online-tutorials/multivariable-calculus/multi-variable-chain-rule/)
{% endcapture %}
{% include external_resources.html content=content %}

We know from the multivariable chain rule, that we can look at the sytem in the previous section and evaluate partial derivatives within our network.  Here is what the process would yield for the non-trivial gradient of $f$ with respect to $t$.

$$
\begin{align}
x &= x(t) \\
y &= y(t) \\
f &= f(x, y) \\
\frac{\partial{f(x, y)}}{\partial t} &= \frac{\partial{x}}{\partial{t}} \frac{\partial f}{\partial x} + \frac{\partial{y}}{\partial{t}} \frac{\partial f}{\partial y}
\end{align}
$$

It turns out that we can modify the data flow diagram for computing the output of the function to instead compute this gradient automatically.  Here is what the process looks like.

```mermaid!
flowchart TB
 id1["$$\frac{\partial f}{\partial f} = 1~~~~$$"]
 id2["$$\frac{\partial f}{\partial{x}} = \frac{\partial f}{\partial{x}} \times 1 ~~$$"]
 id3["$$\frac{\partial f}{\partial{y}} = \frac{\partial f}{\partial{y}} \times 1 ~~$$"]
 id4["$$\frac{\partial f}{\partial t} = \frac{\partial f}{\partial{x}}\frac{\partial x}{\partial{t}} + \frac{\partial f}{\partial{y}}\frac{\partial y}{\partial{t}}~~~~~$$"]
 id1 --"$$\frac{\partial{f}}{\partial{x}} \times 1$$"--> id2
 id1 --"$$\frac{\partial{f}}{\partial{y}} \times 1$$"--> id3
 id2 --"$$\frac{\partial{f}}{\partial{x}} \frac{\partial{x}}{\partial{t}}$$"--> id4
 id3 --"$$\frac{\partial{f}}{\partial{y}} \frac{\partial{y}}{\partial{t}}$$"--> id4
```



```mermaid!
flowchart BT
  id1["$$z_0 = z_1 + z_2~~$$"]
  id2["$$z_1 = \cos\left(z_5\right)~~$$"]
  id3["$$z_2 = z_3 \times z_4~~$$"]
  id4["$$z_3 = x^2$$"]
  id5["$$z_4 = \sqrt{z}~~$$"]
  id6["$$z_5 = z_3 \times y~~$$"]
  id2 -- "$$\frac{\partial{z_0}}{\partial{z_1}} = 1$$" --> id1
  id3 -- "$$\frac{\partial{z_0}}{\partial{z_2}} = 1$$" --> id1
  id4 --> id3
  id6 -- "$$\frac{\partial{z_1}}{\partial{z_5}} = -\sin(z_5)~~$$" --> id2
  id5 -- "$$ $$" --> id3
  id4 --> id6
  x -- "$$\frac{\partial{z_3}}{\partial{x}} = 2x~~$$" --> id4
  y -- "$$\frac{\partial{z_5}}{\partial{y}} = z_3$$" --> id6
  z --> id5
```



{% capture problem %}

{% capture part_a %}
Draw a dataflow diagram to represent the function $f(x,y,z) = \cos(x^2 y) + x^2 \sqrt{z}$.  Compute $\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}$ using the dataflow diagram method.
{% endcapture %}
{% capture part_a_sol %}


<p>
\begin{align}
\frac{\partial f}{\partial x}&= 2x y (-sin(x^2 y)) + 2x{\sqrt z} \nonumber \\
&= -2x y sin(x^2 y) + 2x{\sqrt z} \\
\frac{\partial f}{\partial y} &= x^2 (-sin(x^2 y)) \nonumber \\
&= -x^2 sin(x^2 y) \\
\frac{\partial f}{\partial z} &= \frac{1}{2 \sqrt z} x^2
\end{align}
</p>
{% endcapture %}
{% include problem_part.html label="A" subpart=part_a solution=part_a_sol %}

{% capture part_b %}
Draw a dataflow diagram to represent the function $f(\mlvec{x}) = (\mlvec{c}^\top \mlvec{x})^2$.  Compute $\nabla_{\mlvec{x}} f$ using the dataflow diagram method.  Hint: we're generalizing what is on the HMC page a bit.  You can have vector quantities at the leaf nodes in the graph (leaf nodes are those that have no incoming arrows) and all the ideas will carry over except you will have a gradient instead of a partial derivative on the edge.  If you wanted to have a vector quantity at a non-leaf node, that would require modifying the technique on the HMC page a bit (we won't cover that in this class).
{% endcapture %}
{% capture part_b_sol %}
<p>
\begin{align}
\nabla_{\mlvec{x}} f = 2(\mlvec{c}^\top \mlvec{x}) \mlvec{c}
\end{align}
</p>
{% endcapture %}
{% include problem_part.html label="B" subpart=part_b solution=part_b_sol %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}