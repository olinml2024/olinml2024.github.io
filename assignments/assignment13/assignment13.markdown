---
title: Assignment 13 - GPTs (part 2)
toc_sticky: true
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn about the role of MLPs in transformer models
* Read about issues related to collecting datasets for training large language models
{% endcapture %}
{% include learning_objectives.html content=content %}

# The Role of MLPs in Transformers

{% capture resources %}
Now, let's watch the 3B1B video [How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8).  Here are some of the key things we would like you to take away from this video.
* A transformer consists of interleaved blocks of attention and multi-layer perceptrons.
* Roughly two thirds of the parameters in a transformer are in the MLPs.
* MLPs operate on the output vectors from the attention block in parallel (i.e., the vector from two different positions do not interact with each other in the MLP).
* One way to interpret each row of the weight matrix corresponding $\mlmat{W}_{\uparrow}$ is that they each encode some sort of question (e.g., in the video's example we can think of one row as asking the question whether the input emedding corresponds to the concept "Michael Jordan").
* We can think of the role of the non-linearity in the MLP (e.g., ReLU) as deciding whether a given embedding is positive enough to consider as having answered the question in the affirmative (e.g., is this vector "Michael Jordan" versus "Michael Phelps" or "Alexis Jordan").  We can think of the neuron as being "active" if the activation exceeds the threshold and "inactive" if it does not.
* We can think of columns of the matrix $\mlmat{W}_{\downarrow}$ as referring to different concepts that we would like to add to the input embedding when forming the output from the MLP.  In the example, these concepts could relate to things like "baseketball", "Chicago Bulls", etc.
* With regards to superposition, you don't need to worry about this too much.  The main idea is that if we think of concepts in a neural network as representing vectors in the embeddings space, then we can encode a lot of facts by ensuring that each pair of these vectors is nearly perpendicular to each other.  This idea allows us to "fit" many more concepts within our embedding space.
{% endcapture %}
{% include external_resources.html content=resources %}

# Issues in Dataset Collection for LLMs

As we've seen so far in this module, in order to train GPT models we need a lot of text data for the pretraining process.  An essential question is where this data comes from and what sorts of ethical frameworks should we utilize when sourcing this data ourselves.  Here are some readings to get you thinking through some of these important issues.

{% capture resources %}
{% endcapture %}
{% include external_resources.html content=resources %}