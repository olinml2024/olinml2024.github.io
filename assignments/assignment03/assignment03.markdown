---
title: Assignment 3
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn linear regression using a ``top-down'' approach.
{% endcapture %}
{% include learning_objectives.html content=content %}

# Supervised Learning Problem Setup

Suppose you are given a training set of data points, $(\mlvec{x_1}, y_1), (\mlvec{x}_2, y_2), \ldots, (\mlvec{x}_n, y_n)$ where each $\mlvec{x_i}$ represents an element of an input space (e.g., a d-dimensional feature vector) and each $y_i$ represents an element of an output space (e.g., a scalar target value).  In the supervised learning setting, your goal is to determine a function $\hat{f}$ that maps from the input space to the output space.  For example, if we provide an input $\mlvec{x}$ to $\hat{f}$ it would generate the predicted output $\hat{y} = \hat{f}(\mlvec{x})$.

We typically also assume that there is some loss function, $\ell$, that determines the amount of loss that a particular prediction $\hat{y_i}$ incurs due to a mismatch with the actual output $y_i$.  We can define the best possible model, $\hat{f}^\star$ as the one that minimizes these losses over the training set.  This notion can be expressed with the following equation  (note: that $\argmin$ in the equation below just means the value that minimizes the expression inside of the $\argmin$, e.g., $\argmin_{x} (x - 2)^2 = 2$, whereas $\min_{x} (x-2)^2 = 0$).

\begin{align}
\hat{f}^\star &= \argmin_{\hat{f}} \sum_{i=1}^n \ell \left ( \hat{f}(\mlvec{x_i}), y_i \right )
\end{align} 


# Linear Regression from the Top-Down

## Motivation: Why Learn About Linear Regression?
Before we jump into the *what* of linear regression, let's spend a little bit of time talking about the *why* of linear regression.  As you'll soon see, linear regression is among the simplest (perhaps *the* simplest) machine learning algorithm.  It has many limitations, which you'll also see, but also a of ton strengths.  **First, it is a great place to start when learning about machine learning** since the algorithm can be understood and implemented using a relatively small number of mathematical ideas (you'll be reviewing these ideas later in this assignment).  In terms of the algorithm itself, it has the following very nice properties.

* **Transparent:** it's pretty easy to examine the model and understand how it arrives at its predictions.
* **Computationally tractable:** models can be trained efficiently on datasets with large numbers of features and data points.
* **Easy to implement:** linear regression can be implemented using a number of different algorithms (e.g., gradient descent, closed-form solution).  Even if the algorithm is not built into your favorite numerical computation library, the algorithm can be implemented in only a couple of lines of code.


For linear regression our input data, $\mlvec{x_i}$, are d-dimensional vectors (each entry of these vectors can be thought of as a feature), our output data, $y_i$, are scalars, and our prediction functions, $\hat{f}$, are all of the form $\hat{f}(\mlvec{x}) =\mlvec{w} \cdot \mlvec{x} = \mlvec{w}^\top \mlvec{x} = \sum_{i=1}^d w_i x_i$ for some vector of weights $\mlvec{w}$ (you could think of $\hat{f}$ as also taking $\mlvec{w}$ as an input, e.g., writing $\hat{f}(\mlvec{x}, \mlvec{w}$).  Most of the time we'll leave $\mlvec{w}$ as an implicit input: writing $\hat{f}(\mlvec{x})$).

In the function, $\hat{f}$, the elements of the vector $\mlvec{w}$ represent weights that multiply various dimensions of the input.  For instance, if an element of $\mlvec{w}$ is high, that means that as the corresponding element of $\mlvec{x}$ increases, the prediction that $\hat{f}$ generates would also increase (you may want to mentally think through other cases, e.g., what would happen is the element of $\mlvec{x}$ decreases, or what would happen if the entry of $\mlvec{w}$ was large and negative).  The products of the weights and the features are then summed to arrive at an overall prediction.

Given this model, we can now define our very first machine learning algorithm: [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) (OLS)!  In the ordinary least squares algorithm, we use our training set to select the $\mlvec{w}$ that minimizes the sum of squared differences between the model's predictions and the training outputs.  Thinking back to the supervised learning problem setup, this corresponds to choosing $\ell(y, \hat{y}) = (y - \hat{y})^2$.
Therefore, the OLS algorithm will use the training data to select the optimal value of $\mlvec{w}$ (called $\mlvec{w}^\star$), which minimizes the sum of squared differences between the model's predictions and the training outputs.

\begin{align}
\mlvec{w}^\star &= \argmin_{\mlvec{w}} \sum_{i=1}^n \ell \left ( \hat{f}(\mlvec{x_i}, \mlvec{w}) , y_i \right) \\
\mlvec{w}^\star &= \argmin_{\mlvec{w}} \sum_{i=1}^n \left ( \hat{f}(\mlvec{x_i}, \mlvec{w}) - y_i \right)^2 \\
&= \argmin_{\mlvec{w}} \sum_{i=1}^n \left ( \mlvec{w}^\top \mlvec{x_i} - y_i \right)^2
\end{align}

{% capture content %}
Digesting mathematical equations like this can be daunting, but your understanding will be increased by unpacking them carefully.  Make sure you understand what was substituted and why in each of these lines.  Make sure you understand what each symbol represents.  If you are confused, ask for help (e.g., post on discord).
{% endcapture %}
{% include notice.html content=content %}

While we haven't talked at all about how to find $\mlvec{w}^\star$, that will be the focus of the next assignment, once we have $\mlvec{w}^\star$ we can predict a value for a new input point, $\mlvec{x}$, by predicting the corresponding (unknown) output, $y$, as $\hat{y} = \mlvec{w^\star} \cdot \mlvec{x}$.  In this way, we have used the training data to learn how to make predictions about unseen data, which is the hallmark of supervised machine learning!


{% capture content %}
Draw a scatter plot in 2D (the x-axis is the independent variable and the y-axis is the dependent variable).  In other words, draw five or so data points, placed wherever you like. Next, draw a potential line of best fit, a straight line that is as close to your data points.  On the plot mark the vertical differences between the data points and the line (these differences are called the residuals).  Draw a second potential line of best fit and mark the residuals.  From the point of view of ordinary least-squares, which of these lines is better (i.e. has the smallest residuals)?
{% endcapture %}


{% capture sol %}
![graph](figures/exercise3solution.pdf)
The red line (line 1) would be better since the residuals are generally smaller.  Line 2 also has several large residuals, which when squared will cause a large penalty for line 2.
{% endcapture %}

{% include problem.html solution=sol %}

# Getting a Feel for Linear Regression
In this class we'll be learning about algorithms using both a top-down and a bottom-up approach.  By bottom-up we mean applying various mathematical rules to derive a solution to a problem and only then trying to understand how to apply it and how it well it might work for various problems.  By top-down we mean starting by applying the algorithm to various problems and through these applications gaining a sense of the algorithm's properties.  We'll start our investigation of linear regression using a **top-down approach**.


## Linear Regression with One Input Variable: Line of Best Fit
If any of what we've said so far sounds familiar, it is likely because you have seen the idea of a line of best fit in some previous class.  To understand more intuitively what the OLS algorithm is doing, we want you to investigate its behavior when there is a single input variable (i.e., you are computing a line of best fit).  

\begin{externalresources} [(10 minutes)]
Use the \href{http://www.shodor.org/interactivate/activities/Regression/}{line of best fit online app} to create some datasets, guess the line of best fit, and then compare the results to the OLS solution (line of best fit).

\begin{exercise}
\bes
\item Examine the role that outliers play in determining the line of best fit.  Does OLS seem sensitive or insensitive to the presence of outliers in the data?
\begin{boxedsolution}
OLS is very sensitive to outliers.  A single outlier can change the slope of the line of best fit dramatically.  Here is an example of this phenomenon.

\begin{center}
\includegraphics[width=.6\linewidth]{figures/outlier}
\end{center}

\end{boxedsolution}

\item Were there any times when the line of best fit didn't seem to really be ``best'' (e.g., it didn't seem to capture the trends in the data)?
\begin{boxedsolution}
This could happen for many reasons.  If the dataset is pieceweise linear (e.g., composed of multiple line segments), if it has some other non-linear form (e.g., if it is quadratic), or if there are outliers.
\end{boxedsolution}

\ees
\end{exercise}

\end{externalresources}



\subsection{Linear Regression with Multiple Input Variables: Explorations in Python}
\begin{externalresources}[(60 minutes)]
Work through the \href{https://colab.research.google.com/drive/1QPsD2URWupxWjpBfKsr7AcIZ2m3VD37T?usp=sharing}{Assignment 1 Companion Notebook} to get some practice with {\tt numpy} and explore linear regression using a top-down approach.  You can place your answers directly in the Jupyter notebook so that you have them for your records.
\end{externalresources}