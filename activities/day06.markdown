---
title: Day 6
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:35am: Debrief at tables
* 10:35-10:40am: Going Over Simplification in Logistic Regression Learning Rule
* 10:40-10:50am: Preview of where we are going
* 10:50-12:00pm: Foundations of Micrograd
{% endcapture %}

{% include agenda.html content=agenda %}

# Debrief

Based on responses to the survey, we think people are still a bit fuzzy on data flow diagrams.  We recommend you come up with a function, create a dataflow diagram to represent it, and then use that dataflow diagram to compute the partial derivative of the function with respect to each of its inputs.

# Preview of where we are going

We'll go over the upcoming gate on model evaluation.  We'll also talk about the COMPAS algorithm and the readings we will be doing / the discussions we will be having.

# Buildings towards Autodifferentiation

While you may still be having a bit of difficulty applying dataflow diagrams, hopefully the process is starting to become more mechanical.  In fact, given how mechanical it is, you may be wondering if there is a way to automate the process entirely.  Of course there is, and that's where we are going to go next.  What we will be doing in assignment 7 (not this coming assignment but the next), is using the concepts of data flow diagrams to implement a system for automatically computing the gradient of any function!

Next, we'll go step-by-step through the process of going from dataflow diagrams to auto differentiation.

## Step 1: Modify our dataflow diagram to compute gradients

Let's go back to our minimal example of a multivariable function that we used to first introduce the concept of dataflow diagrams.

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

In order to compute $\frac{\partial f}{\partial t}$ we traced all possible paths from $t$ to $f$, multiplied the partial derivatives along the way, and then added up each of these paths.

\begin{align}
\frac{\partial f}{\partial t} &= \frac{\partial f}{\partial x} \frac{\partial x}{\partial t} +  \frac{\partial f}{\partial y} \frac{\partial y}{\partial t}
\end{align}

This all seems well and good, but this approach is not quite as systematic as we might like and it suffers from some computational challenges.  Consider a more complex dataflow diagram in the next problem.

{% capture problem %}
```mermaid!
flowchart BT
 id1["$$f = f(q,p)~~~~$$"]
 id2["$$x = x(t)~~$$"]
 id3["$$y = y(t)~~$$"]
 id4["$$r = r(x, y)~~$$"]
 id5["$$s = s(x, y)~~$$"]
 id6["$$q = q(r, s)~~$$"]
 id7["$$p = p(r, s)~~$$"]
 id2 --> id4
 id2 --> id5
 id3 --> id4
 id3 --> id5
 id4 --> id6
 id4 --> id7
 id5 --> id6
 id5 --> id7
 id6 --> id1
 id7 --> id1
 t --> id2
 t --> id3
```

Use the data flow diagram method to compute $\frac{\partial{f}}{\partial t}$.
{% endcapture %}
{% include problem.html problem=problem solution="" %}

You can see that things are getting a bit out of hand.  If you started adding more layers, we would be in even more trouble.  At this point things might seem a bit hopeless, but there are a few observations we can make.

* A lot of the paths from $t$ to $f$ go through the same parts of the dataflow diagram.
* The procedure above is great for computing a symbolic expression to compute $\frac{\partial{f}}{\partial t}$, however, when optimizing a function, for example using gradient descent, all we care about is computing the gradient given some specific values of our parameters.

Given these two observations we can modify our dataflow diagram to more efficiently compute the partial derivatives we need.  Let's now return to our simpler case and see how we can modify it.  Before we provide this example, let's state our assumptions and define some notation.

* Assumption: We assume that already executed our original dataflow diagram.  That is, given $t$, we have computed $x, y, f$.
* We are using the notation $grad_{v}$ to represent the evaluation of $\frac{\partial f}{\partial v}$ based on the values $t, x, y, f$ (that is, $grad_{v}$ will just be a number, not a symbolic expression).
* When we write $\frac{\partial g}{\partial v}$ we think of this as the evaluation of the partial derivative of $g$ with respect to $v$ given the values $t, x, y, f$ (again, not a symoblic expression.  it is just a number).

Given the above, let's modify our dataflow diagram to more naturally compute our gradients.

```mermaid!
flowchart TB
 id1["$$grad_f = 1 ~~~~$$"]
 id2["$$grad_x = \frac{\partial f}{\partial x} grad_f~~$$"]
 id3["$$grad_y = \frac{\partial f}{\partial y} grad_f~~$$"]
 id4["$$grad_t = \frac{\partial x}{\partial t} grad_x + \frac{\partial y}{\partial t} grad_y~~~~~~$$"]
 id1 --"$$\frac{\partial f}{\partial x} grad_f~~$$"--> id2
 id1 --"$$\frac{\partial f}{\partial y} grad_f~~$$"--> id3
 id2 --"$$\frac{\partial x}{\partial t} grad_x ~~$$"--> id4
 id3 --"$$\frac{\partial y}{\partial t} grad_y ~~$$"--> id4
```

{% capture problem %}
Make sense of the example above.  Try to understand where the expression for $grad_f$ comes from.  If you think about the concept of a recursive function, what might you call this?  Look at the other expressions for the partial derivatives.  Make sure these jibe with the rules you learned for dataflow diagrams.
{% endcapture %}
{% include problem.html problem=problem solution="" %}

Hopefully, this is making sense.  Let's do a bigger example next.

{% capture problem %}
Convert the following dataflow diagram to efficiently compute $grad_v$ for $f, x, y, r, s, q, p$.  You should wind up with something similar to the previous example.

```mermaid!
flowchart BT
 id1["$$f = f(q,p)~~~~$$"]
 id2["$$x = x(t)~~$$"]
 id3["$$y = y(t)~~$$"]
 id4["$$r = r(x, y)~~$$"]
 id5["$$s = s(x, y)~~$$"]
 id6["$$q = q(r, s)~~$$"]
 id7["$$p = p(r, s)~~$$"]
 id2 --> id4
 id2 --> id5
 id3 --> id4
 id3 --> id5
 id4 --> id6
 id4 --> id7
 id5 --> id6
 id5 --> id7
 id6 --> id1
 id7 --> id1
 t --> id2
 t --> id3
```
{% endcapture %}
{% capture solution %}

Excuse the lack of annotation of the arrows.  We can show you in class as we walk around.

```mermaid!
flowchart TB
 id1["$$grad_f = 1 ~~~~$$"]
 id2["$$grad_x = grad_r \frac{\partial r}{\partial x} + grad_s \frac{\partial s}{\partial x}~~$$"]
 id3["$$grad_y = grad_r \frac{\partial r}{\partial y} + grad_s \frac{\partial s}{\partial y}~~~~$$"]
 id4["$$grad_r = grad_q \frac{\partial q}{\partial r} + grad_p \frac{\partial p}{\partial r}~~$$"]
 id5["$$grad_s = grad_q \frac{\partial q}{\partial s} + grad_p \frac{\partial p}{\partial s}~~$$"]
 id6["$$grad_q = grad_f \frac{\partial f}{\partial q}~~$$"]
 id7["$$grad_p = grad_f \frac{\partial f}{\partial p}~~$$"]
 id4 --> id2
 id5 --> id2
 id4 --> id3
 id5 --> id3
 id6 --> id4
 id7 --> id4
 id6 --> id5
 id7 --> id5
 id1 --> id6
 id1 --> id7
 id2 --> t
 id3 --> t
```
{% endcapture %}
{% include problem.html problem=problem solution=solution %}

# Thinking Through Autodifferentiation in Python

{% capture problem %}
On assignment 7, you will be implementing autodifferentation in Python (using the exact procedure above).  We will guide you through a specific way to implement if (based on the micrograd framework by Andrej Karpathy).  If you have time though, you might think with your table about how to implement this algorithm.  You are probably not going to be able to get to the level of mapping our Python code, but you can think about the major building blocks you would need (e.g., you would need something to compute the forward pass through the dataflow diagram).
{% endcapture %}
{% include problem.html problem=problem solution="" %}