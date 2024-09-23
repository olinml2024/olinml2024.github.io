---
title: Assignment 7
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* 
{% endcapture %}
{% include learning_objectives.html content=content %}

# Logistic Regression Analysis of COMPAS data (using sklearn)

# Micrograd

It turns out that we can modify the data flow diagram for computing the output of the function to instead compute this gradient automatically.  Here is what the process looks like (let's use $grad_v$ to store the result of $\frac{\partial f}{v}$).

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


## Object-oriented Architecture

## Implementing Some Basic Building Blocks

## Topological Sorting (optional)

# Logistic Regression using Micrograd

(cross check with pytorch)

# Neural Networks top-down (this is probably assignment 8)

# Neural Networks with Micrograd

# Neural Networks with Pytorch