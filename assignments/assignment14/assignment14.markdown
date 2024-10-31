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
* The attention layers are responsible for allowing tokens to pass information to other tokens. The degree to which a token passes information to another token depends on taking a dot product between a key and query vector, which is then passed through a softmax.  The specific value passed to the other token depends on a value vector which is computed from the input to the attention layer multiplied by a matrix ($W_V$).
* While we haven't gotten our hands dirty with MLPs in this module, we've seen them in previous parts of the course.  The MLPs in a GPT take the output of the attention block and perform computation on them.  In the 3B1B video, we saw that one theory of what these MLPs are doing is that they are representing facts that the GPT has learned.
* We started to implement NanoGPT by starting with a simple Bigram model and then adding in a self-attention mechanism so that tokens could communicate with each other.

# Finishing Our Implementation of NanoGPT

We'll continue to work through Karpathy's video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).  Follow along with our notes and suggestions for things to try below.

{% capture external %}
* [1:11:37](https://youtu.be/kCc8FmEb1nY?t=4297): this is our starting point for this assignment.
* Touch on layer norm (link to notebook)
* Touch on residual connections (discuss ablation experiment)
{% endcapture %}
{% include external_resources.html content=external %}

# Visualizing NanoGPT

https://bbycroft.net/llm
* Mapping the visualization to a class in ``model.py``.

# Ablation and NanoGPT

* Identify key pieces of the model
* Perform some code surgery
* Plot the results

Focus on having them interpret the graph and identify the line of code.

# Proposing an LLM for an Application and Context You Care About

* What would an application of large language models either at Olin or outside?  What would be valuable about it.
* What would the considerations be in developing such an application?  What issues of data management might come up? This could be privacy, legal, bias,  What value would the system provide and who would benefit.  What guardrails would you put in place?