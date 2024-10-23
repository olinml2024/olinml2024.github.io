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
* That the last layer of a GPT model maps from the embedding space to a real number for each possible next token (this is called the "unembedding matrix" in the video).  These numbers are called "logits".
* To take our real numbers from the previous step into a probability of the next token, we use the softmax function.
* Make a note of what materials are review from this video (based on things we've already done).
{% endcapture %}
{% include external_resources.html content=externalresource %}

# Self-attention Under the Hood

Hopefully, you found that video to connect some dots from the last assignment and set the stage nicely for where we are going next.  Our next move is going to be to watch the next chapter in the 3B1B series on deep learning.  This is where we will meet the concept of self-attention, which is going to be at the heart of our GPT model.


{% capture externalresource %}
Now, let's watch the 3B1B video [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc).

Here are some of the key things we would like you to take away from this video.
* That the initial embedding of a token also encodes its position (not just the token's identity)
* That it is useful for words to be able to ask questions (query) of other words.
* That queries can be specified as vectors and the answers to those queries can also be specified as vectors (called keys).
* That the degree to witch a key answers a query can be determined by taking the dot product of the key vector and the query vector and that we can compute the dot product of each query token and each query key as $QK^\top$.
* Applying a softmax to the matrix of dot products of queries and keys gives us a probability distribution of what tokens each token shoudl attend to.
* That the idea of causal attention (where we are predicting future tokens from past tokens) requires that future tokens are not allowed to send information to past tokens.  Further, to accomplish this goal, we can force entries in our query-key matrix corresponding to future tokens influencing past tokens to negative infinity (before applying softmax).  This is called "masking".
* That the token embeddings are updated by adding the value vectors from other tokens (weighted by attention).  (Note: this is presented in the video through the example of using adjectives to update the meaning of a noun.)
* Note: there is a discussion of how to cut down the number of parameters in the value map by decomposing it into the product of the value up and the value down matrices ($V_{\uparrow}$ and $V_{\downarrow}$).  While this is interesting, and we are happy to talk about it,  we don't advise getting hung up on this detail (we will not be using this architecture in the implementation to follow).  Similarly, don't worry about the note about how the $V_{\uparrow}$ matrices are all combined into one matrix called the output matrix.
* That multiple heads of attention can be used to capture multiple ways in which token embeddings can influence each other.  Note: you shouldn't have a super precise notion of what this means, but you should have a notion that multiple heads of attention might be useful.
{% endcapture %}
{% include external_resources.html content=externalresource %}

Alright, hopefully you are starting to put the pieces together.  We are going to some more steps to help thing solidify.  First, let's do some exercises to help you with your understanding of self-attention.

{% capture problem %}
Let's use a toy problem to make sure we have a handle on the mechanics of self-attention.  Instead of words, let's think of individual letters as our tokens (again, sorry for this sleight-of-hand.  We are doing this to make the problem as simple as possible to highlight the important bits of self-attention).  Let's imagine that setting in which text consists of sequences of tokens where the probability of the next token being a vowel increases based on the proportion of consonants that that appear in the last four positions of the text.  Here are some examples.

1. Input text: "cdeaeia", probability of next token being a vowel is low since there are no consonants in the last four positions of the text (keep in mind that "c" and "d" occur before these last four positions).
2. Input text: "ccrs", probability of next token being a vowel is very high since each of the last four positions are consonants.
3. Input text: "aeri", probability of next token being a vowel is about half since two of the the last four positions are consonants.

Let's use a tokenization scheme where each letter is mapped to its position in the alphabet (starting with $a \rightarrow 0$ and ending with $z \rightarrow 25$).

{% capture parta_prob %}
Explain what the features (the columns) of the input tokens (the rows) the embedding matrix $\mlmat{E}$ captures.

$$
\mlmat{E} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0  \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1  \\ 1 & 0  \\ 0 & 1  \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \end{bmatrix}
$$
{% endcapture %}
{% capture parta_sol %}
The first column of the matrix encodes whether the token is a vowel (1) or consonant (0).  The second column of the matrix encodes whether the token is a consonant (1) or a vowel (0).
{% endcapture %}
{% include problem_part.html subpart=parta_prob solution=parta_sol label="A" %}

{% capture partb_prob %}
Define a query and key matrix pair that causes all letters to attend to consonants.  As a reminder the dimensionality of $K$ and $Q$ are both $m \times n_\mbox{embd}$ where $m$ is the query dimension (you can choose this) and $n_\mbox{embd}$ is the dimensionality our embeddings (in this case 2).  You should be able to solve the problem with $m = 1$ (one row for each of the key and query matrices)
{% endcapture %}
{% capture partb_sol %}
Let's define the matrices as follows.
$$
\begin{align}
\mlmat{Q} &= \begin{bmatrix} 1 & 1 \end{bmatrix} \\ 
\mlmat{K} &= \begin{bmatrix} 0 & 5 \end{bmatrix}
\end{align}
$$

Taking it for a test spin, let's look at the different cases.

* query is vowel and key is vowel $\bigg (\mlmat{Q}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\mlmat{K} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = (1)(0) = 0$
* query is consonant and key is vowel $\bigg (\mlmat{Q}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\mlmat{K} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = (1)(0) = 0$
* query is vowel and key is consonant $\bigg (\mlmat{Q}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\mlmat{K} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = (1)(5) = 5$
* query is consonant and key is consonant $\bigg (\mlmat{Q}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\mlmat{K} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = (1)(5) = 5$

Why $5$?  This helps make the attention to consonants higher relative to attention to vowels (remember, this has to get passed through a softmax).


{% endcapture %}
{% include problem_part.html subpart=partb_prob solution=partb_sol label="B" %}

{% capture partb_sol %}
Let's define the matrices as follows.
$$
\begin{align}
\mlmat{Q} &= \begin{bmatrix} 1 & 1 \end{bmatrix} \\ 
\mlmat{K} &= \begin{bmatrix} 0 & 5 \end{bmatrix}
\end{align}
$$

Taking it for a test spin, let's look at the different cases.

* query is vowel and key is vowel $\bigg (\mlmat{Q}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\mlmat{K} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = (1)(0) = 0$
* query is consonant and key is vowel $\bigg (\mlmat{Q}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\mlmat{K} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = (1)(0) = 0$
* query is vowel and key is consonant $\bigg (\mlmat{Q}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\mlmat{K} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = (1)(5) = 5$
* query is consonant and key is consonant $\bigg (\mlmat{Q}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\mlmat{K} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = (1)(5) = 5$

Why $5$?  This helps make the attention to consonants higher relative to attention to vowels (remember, this has to get passed through a softmax).


{% endcapture %}
{% include problem_part.html subpart=partb_prob solution=partb_sol label="B" %}


{% capture partc_prob %}
Construct a short sequence of characters, $s$, consisting of some vowels and some consonants.  Compute the queries and keys for each character and use them to compute $\mlmat{Q_{s}} \mlmat{K_{s}}^\top$ (where $\mlmat{Q_{s}}$ are the query vectors for each token and $\mlmat{K_{s}}$ are the keys for each token.).  Apply masking to ensure that keys (columns) corresponding to later tokens do not influence earlier queries (rows).  Please note that the visualization of this matrix in the 3B1B video has this matrix laid out with query tokens as columns and the keys as rows (we wanted to let you know to minimize confusion).  Apply a softmax across each row to determine a weight for each token and show the resultant matrix.
{% endcapture %}

{% capture partc_sol %}
Todo
{% endcapture %}
{% include problem_part.html subpart=partc_prob solution=partc_sol label="C" %}

{% capture partd_prob %}
Define the value for each token as $\mlmat{V} r$ where $\mlmat{V}$ is the identity matrix and $r$ is the embedding of the token.  Show that if you construct the matrix $\mlmat{V_s}$ where each row is the value of a token that multiply the attention matrix from part c with $\mlmat{V_s}$ computes the output of the attention head which encodes the proportion of consonants that preceed each element of $s$.

Note: check dimensionalities match up
{% endcapture %}

{% capture partd_sol %}
Todo
{% endcapture %}
{% include problem_part.html subpart=partd_prob solution=partd_sol label="D" %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}


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

Karpathy examples
* Note that the masking looks like the transpose (upper triangular is masked rather than the lower triangular).