---
title: Assignment 12 - Generative Pre-Trained Transformers (GPTs) Part 1
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn about the concept of self-attention in neural networks and the role it plays in Generative Pre-trained Transformers (GPTs)
* Implement self-attention in Pytorch
{% endcapture %}
{% include learning_objectives.html content=content %}

# Demystifying GPT

This assignment and the next one are building towards the goal of demystifying large lanuage models (LLMs) like ChatGPT.  While we won't be able to learn everything there is to know about these models, we will be learning, in-depth, about the concept of Generative Pre-Trained Transformers (GPTs).  We hope that by seeing the GPT mechanism up close, you are able to develop a better understanding of how LLMs work and potential explore LLMs further in your final projects.  You'll also learn some useful, generalizable tricks for text processing along the way.

The roadmap for our work (over this and the next assignment) is that we are going to use two video resources.  First, we'll watch a sequence of two videos from 3B1B that will help us build a conceptual understanding of GPTs through a visual approach. The second, is a walkthrough of how to turn our conceptual understanding into an implementation of a GPT in Pytorch (we'll use NanoGPT from Andrej Karpathy for that).

# Word Embbeddings and Predicting the Next Word

{% capture externalresource %}
Let's start off by watching the 3B1B video [How large language models work, a visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M).

Here are some of the key things we would like you to take away from this video.
* That text can be tokenized in different ways (either as letters, chunks of words, or whole words)
* How predicting the next token (or word) given a piece of text can be used repeatedly to do text completion.
* That we can use the concept of embeddings to represent tokens in a high-dimensional space (make sure you understand how this connects to word embeddings)
* Why the context that surrounds a word might be important for updating its embedding vector (e.g., to disambiguate between multiple meanings of the same word).
* That the last layer of a GPT model maps from the embedding space to a real number for each possible next token (this is called the "unembedding matrix" in the video).  These numbers are called "logits."
* To take our real numbers from the previous step into a probability of the next token, we use the softmax function.
* Make a note of what materials are review from this video (based on things we've already done).
{% endcapture %}
{% include external_resources.html content=externalresource %}

# Self-attention Under the Hood

Hopefully, you found that video to connect some dots from the last assignment and set the stage nicely for where we are going next.  Our next move is going to be to watch the next chapter in the 3B1B series on deep learning.  This is where we will meet the concept of self-attention, which is going to be at the heart of our GPT model.


{% capture externalresource %}
Now, let's watch the 3B1B video [How large language models work, a visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M).

Here are some of the key things we would like you to take away from this video.
* 
{% endcapture %}
{% include external_resources.html content=externalresource %}


# Attention By Hand

For all problems our tokens consist of a-z, 0, and 1.

* Toy problem 1
Our text obeys the following rules.
1. 0 and 1 don't appear.
2. The probability of the next character being a vowel increases based on the proportion of consonants encountered in the text thus far (up to and including the current token)

* Toy problem 2
Our text obeys the following rules.
1. The text always starts with a consonant or a vowel.
2. If a consonant appears, any letter can follow with equal probability
3. If a vowel appears, the digit 1 follows as long as a consonant has appeared previously and 0 otherwise.
4. After a 0 or a 1, any letter can follow with equal probability.

Let's use self-attention to solve this problem.

[Notebook for toy problems](https://colab.research.google.com/drive/16vJqAEMOr-9U1pt67Xx-oCr8owM2hj3x?usp=sharing)
