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

# Cross entropy and how to interpret the graphs from the homework

We're going to look back at the material we didn't cover last class on cross entropy and softmax.  The materials can be found on the [day 8 page](day08#cross-entropy-loss-and-softmax).  Let's go over it together and then we'll be in a good place to interpet the learning curves from this past assignment.

# Small data mini-project on classification


# Choosing data for your mini-project and start working
Our general recommendation is to choose a dataset that has at least one other person working on that dataset. This is not a requirement, so if you have something you're passionate about, go for it! While this is a solo project, it may be helpful to have others to confer with who are also figuring out the nuances of your dataset. A lot of time in machine learning and data science go into interacting with the data before it even goes into the model. There are many canned (pre-curated data sets) out there which reduce this time, but it's still important to understand your data, as it drives your model. 

We'll do a little activity to help you find others who have overlapping dataset interest.
