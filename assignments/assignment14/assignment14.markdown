---
title: Assignment 14 - Generative Pre-Trained Transformers (GPTs) Part 3
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
* [1:12:08](https://youtu.be/kCc8FmEb1nY?t=4328): Karpathy is now linking the concept of attention to a more general idea of information flowing between nodes in a graph.  We don't think you need to be too concerned about this concept as we haven't learned the necessary background to think about graphs (although, you may have seen this in DSA, FOCS, or Discrete).
* [1:15:41](https://youtu.be/kCc8FmEb1nY?t=4541): self-attention is not the only type of attention (as has we've already heard about, e.g., cross attention is used in language translation tasks). This could be an interesting thing to explore in a final project if you find this concept interesting.
* [1:16:56](https://youtu.be/kCc8FmEb1nY?t=4616): we are now seeing why the normalization term $\frac{1}{\sqrt{d_k}}$ is needed.  Karpathy does a nice job showing that this term allows us to achieve the variance we want (this is called *scaled attention*).
* [1:19:18](https://youtu.be/kCc8FmEb1nY?t=4758): we can now take our self-attention code and package it up into the class ``Head``.  As we've seen before in this class, in ``pytorch`` you can create your own machine learning modules by inheriting from ``nn.Module`` (e.g., as we did with our ``MLP`` implementation).  In this part of the video, we also modify our text generation code, which you don't need to worry about.
* [1:21:59](https://youtu.be/kCc8FmEb1nY?t=4919): we'll now scale up from a single head of attention to multiple heads of attention.  Notice the use of the ``nn.ModuleList`` class, which allows multiple ``nn.Module`` objects to be grouped together into a single list.  The key idea here is that our query dimension $n_q$ and the space that our value vectors live in (also $n_q$) is now different than the number of embedding dimensions.  We concatenate the output from each attention head together to get back to the same number of dimensions as our original embedding.  Karpathy makes a reference to this idea of convolutions, which we'll learn about in the next module of this class.
* [1:24:27](https://youtu.be/kCc8FmEb1nY?t=5067): now we are going to bring in the concept of the multi-layer perceptron.  Based on the 3B1B videos, we have a conceptual idea of where these MLPs fit in and what they might do (e.g., store facts that the LLM has learned).  For the MLPs in our model, we'll follow a pretty similar implementation to what we've done previously in the course.  Initially, the MLP that Karpathy implements will look a little strange (it will be a linear layer followed by a non-linearity with no subsequent linear layer), but eventually the second linear layer will be added (matching what we did in the previous module).  Karpathy also abstracts the sequence of self-attention and an MLP into a block which can be reused / repeated.
* [1:27:59](https://youtu.be/kCc8FmEb1nY?t=5279): now we will introduce the idea of skip connections (or residual connections).  There are many reasons why this helps with the performance of the network, which the video touches upon.  The 3B1B videos give us one more way to think about this.  In those videos we talk about self-attention computing a vector that we can add onto our original embedding to modify a word's meaning in some way.  Up until now, we have actually used attention to completely overwrite the original embedding. These skip connections allow us to, instead, compute a vector that we add to our embedding to get our output.  We'll be seeing in more detail how important these connections are later in this assignment.  The concept of the projection self-attention / MLP block back into the residual pathway is confusing.  As with most matrices in neural networks, we can add this project matrix to give our network a bit more flexibility in how it integrates the results of self-attention / MLP with the original embedding.
* [1:32:56](https://youtu.be/kCc8FmEb1nY?t=5576): next we are going to meet the concept of layer norm (this is [the link to the documentation page he pulls up on layer norm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)).  The explanation given here is not particular accessible since we didn't go through the original video on batch norm that Karpathy references.  For our purposes we can unerstand that layer norm is a way of standardizing the inputs to various parts of our model.  Given a batch of data, we would like each of the input features of our data to have mean 0 and standard deviation 1.  This standardization is achieved with ``LayerNorm``, which builds on some additional bells and whistles that we don't really need to worry about.  This sort of normalization can significantly improve the performance of deep (meaning with lots of layers) neural networks.
* [1:37:57](https://youtu.be/kCc8FmEb1nY?t=5877): now we'll have some fun scaling up our network!
* [1:38:46](https://youtu.be/kCc8FmEb1nY?t=5926): we touch on the idea of dropout, which we discussed a bit in our class on preventing overfitting.
* [1:42:40](https://youtu.be/kCc8FmEb1nY?t=6160): don't worry about this part.  We are just connecting back to the "Attention is All You Need" paper with its focus on cross-attention.
* [1:46:31](https://youtu.be/kCc8FmEb1nY?t=6391): Karpathy walks through of the NanoGPT repo.  The quick summary is that some changes have been made to clean up the code and make it more efficient.
* [1:48:55](https://youtu.be/kCc8FmEb1nY?t=6535): Karpathy talks about some important steps that would happen after the pre-training step that we've learned about if you were going to train a ChatGPT-like system.  This is fascinating stuff, and it could be great fodder for a final project!
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