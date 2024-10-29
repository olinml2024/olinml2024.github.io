---
title: Assignment 13 - Generative Pre-Trained Transformers (GPTs) Part 2
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* How to generalize from single-headed to multi-headed attention
* How to Interleave attention and MLPs to create a transformer
* Understand the importance of skip connections and layer normalization
* Perform an ablation experiment to understand the parts of the NanoGPT model that are relevant to text generation
* Consider issues in dataset collection and curation for training and LLM
{% endcapture %}
{% include learning_objectives.html content=content %}

# Review of What We've Done So Far
Before getting into some new stuff, let's review what we did in assignments 12 and 13.
* We learned that GPT stands for "Generative Pre-trained Transform"
* A GPT model consists of a pipeline of interleaving two major types of layers: attention and MLPs.
* The attention layers are responsbile for allowing tokens to pass information to other tokens. The degree to which a token passes information to another token depends on taking a dot prodcut between a key and query vector, which is then passed through a softmax.  The specfic value passed to the other token depends on a value vector which is computed from the input to the attenion layer multiplied by a matrix ($W_V$).
* While we haven't gotten our hands dirty with MLPs in this module, we've seen them in previous parts of the course.  The MLPs in a GPT take the output of the attention block and perform computation on them.  In the 3B1B video, we saw that one theory of what these MLPs are doing is that they are representing facts that the GPT has learned.
* We started to implement NanoGPT by starting with a simple Bigram model and then adding in a self-attention mechanism so that tokens could communicate with each other.

# Finishing Our Implementation of NanoGPT

We'll continue to work through Karpathy's video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).  Follow along with our notes and suggestions for things to try below.

{% capture external %}
* [1:11:37](https://youtu.be/kCc8FmEb1nY?t=4297): this is our starting point for this assignment.
{% endcapture %}
{% include external_resources.html content=external %}

# Ablation and NanoGPT

* Identify key pieces of the model
* Peform some code surgery
* Plot the results

# Creating a Bot to Help First-years On-board to College

Todo