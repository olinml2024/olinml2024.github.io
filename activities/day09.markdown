---
title: Day 9
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:35am: Homework Debrief
* 10:35-10:55am: Cross entropy and how to interpret the graphs from the homework
* 10:45-11:00am: Demistfying Pytorch
* 11:00-11:10am: Intro to Mini Project
* 11:10-12:00pm: Mini Project Planning Time
{% endcapture %}
{% include agenda.html content=agenda %}

# Homework Debrief

Talk to folks at your tables to resolve any confusions or note common questions.

# Cross entropy and how to interpret the graphs from the homework

We're going to look back at the material we didn't cover last class on cross entropy and softmax.  The materials can be found on the [day 8 page](day08#cross-entropy-loss-and-softmax).  Let's go over it together and then we'll be in a good place to interpet the learning curves from this past assignment.

# Demistifying Pytorch

We'll be going through [the day 9 notebook](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Day09.ipynb).  There are three things we want you to get out of this notebook.

* Pytorch is quite similar in its basic concepts to the micrograd framework you implemented.
* We can use pytorch to compute a line of best fit.  This will allow us to visualize the optimization process more easily.
* We can use pytorch modules (e.g., `nn.Linear`) to make our lives easier.