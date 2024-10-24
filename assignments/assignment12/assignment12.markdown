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
Let's use a toy problem to make sure we have a handle on the mechanics of self-attention.  Instead of words, let's think of individual letters as our tokens (again, sorry for this sleight-of-hand.  We are doing this to make the problem as simple as possible to highlight the important bits of self-attention.  We'll also be using a resource called NanoGPT that will implement a GPT, at first, on the character level).  Let's imagine that we want our attention head to take in a sequence of letters and compute for each token whether a consonant has occurred at any point up to and including the current token.  Here are some examples.

1. Input text: "eaeia", our attention head should output no, no, no, no, no (none of our token have the property that they are or are preceded by a consonant).
2. Input text: "ccrs", our attention head should output yes, yes, yes, yes (all tokens either are or are preceded by a consonant)
3. Input text: "aeri", our attention head should output no, no, yes, yes (starting with the third token, "r", we have at least one consonant).

We haven't quite defined how the responses "no" and "yes" will be represented as vectors, but we will get to that shortly.

Let's use a tokenization scheme where each letter is mapped to its position in the alphabet (starting with $a \rightarrow 0$ and ending with $z \rightarrow 25$).

{% capture parta_prob %}
Explain what each of the features (the rows) of the input tokens (the columns) in the embedding matrix $\mlmat{W_E}$ captures.

$$
\mlmat{W_E} = \begin{bmatrix} 1 & 0 & 0 &  0 & 1 & 0 &  0 &  0 & 1 &  0 &  0 &  0 &  0 &  0 & 1 &  0 &  0 &  0 &  0 &  0 & 1 & 0 &  0 &  0 &  0 &  0 \\ 0 &  1& 1 &  1 & 0 & 1 &  1 &  1 & 0 &  1 &  1 &  1 &  1 &  1 & 0 &  1 &  1 &  1 &  1 &  1 & 0 & 1 &  1 &  1 &  1 &  1  \end{bmatrix}
$$
{% endcapture %}
{% capture parta_sol %}
The first row of the matrix encodes whether the token is a vowel (1) or consonant (0).  The second row of the matrix encodes whether the token is a consonant (1) or a vowel (0).
{% endcapture %}
{% include problem_part.html subpart=parta_prob solution=parta_sol label="A" %}

{% capture partb_prob %}
Define a query ($\mlmat{W_q}$) and key ($\mlmat{W_k}$) matrix pair that causes all letters to attend to consonants.

$\mlmat{W_q}$ and $\mlmat{W_k}$ are both matrices with $n_{q}$ rows and $n_{e}$ columns, where $n_q$ is the query dimension (you can choose this) and $n_e$ is the dimensionality our embeddings (in this example, 2).

Hint 1: You should be able to solve the problem with $n_{q} = 1$ (that is, the key and query matrices are both 1 row and 2 columns).

Hint 2: The key equation you'll want to use is that the degree to which token $i$ attends to token $j$ can be computed from the embeddings $\mlvec{r}_i$ and $\mlvec{r}_j$ (these would be found in the appropriate column of $\mlmat{W_E}$) of tokens $i$ and $j$ respectively using the following formula.

\begin{align}
attention &= \mlmat{W_q} \mlvec{r}_i (\mlmat{W_k} \mlvec{r}_j)^\top
\end{align}

{% endcapture %}
{% capture partb_sol %}
Let's define the matrices as follows.
$$
\begin{align}
\mlmat{W_q} &= \begin{bmatrix} 1 & 1 \end{bmatrix} \\ 
\mlmat{W_k} &= \begin{bmatrix} 0 & 5 \end{bmatrix}
\end{align}
$$

Taking it for a test spin, let's look at the different cases.

* query is vowel and key is vowel $\bigg (\mlmat{W_q}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\mlmat{W_k} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = (1)(0) = 0$
* query is consonant and key is vowel $\bigg (\mlmat{W_q}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\mlmat{W_k} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg)^\top = (1)(0) = 0$
* query is vowel and key is consonant $\bigg (\mlmat{W_q}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\mlmat{W_k} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = (1)(5) = 5$
* query is consonant and key is consonant $\bigg (\mlmat{W_q}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\mlmat{W_k} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg)^\top = (1)(5) = 5$

Why $5$?  This helps make the attention to consonants higher relative to attention to vowels (remember, this has to get passed through a softmax).


{% endcapture %}
{% include problem_part.html subpart=partb_prob solution=partb_sol label="B" %}

{% capture partc_prob %}
Come up with a short sequence of characters, $s$, consisting of some vowels and some consonants (keep the length pretty small).  Compute the matrix of all queries corresponding to your sequence, $\mlmat{Q}$, where the number of rows of $\mlmat{Q}$ is equal to the number of tokens (the length of $s$) and the number of rows is equal to the query dimension.  Compute the matrix of all keys corresponding to your sequence, $\mlmat{K}$, where the number of rows of $\mlmat{K}$ is equal to the number of tokens (the length of $s$) and the number of rows is equal to the query dimension.  Compute the (pre-masking) attention of each token to each other token using the formula $\mlmat{Q} \mlmat{K}^\top$.  Apply masking to ensure that keys (columns) corresponding to later tokens do not influence earlier queries (rows).  Note: that the visualization in the 3B1B video (at [this time stamp](https://youtu.be/eMlx5fFNoYc?t=514)) has this matrix laid out with query tokens as columns and the keys as rows (we wanted to let you know to minimize confusion).  Apply a softmax across each row (as before, this is shown on columns in the 3B1B video) to determine a weight for each token and show the resultant matrix.
{% endcapture %}

{% capture partc_sol %}
Let's take our string to be $s = \mbox{abcce}$.

Step 1: Compute our embeddings by picking out appropriate columns of our matrix. $r_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $r_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $r_3 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $r_4 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, and $r_5 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

Step 2: Compute each query using the formula $\mlmat{W_q} \mlvec{r}_i$ and each key using the formula $\mlmat{W_k} \mlvec{r}_i$ and put each query as a row to form $\mlmat{Q}$ and each key as a row to form $\mlmat{K}$.

$$
\begin{align}
\mlmat{Q} &= \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \end{bmatrix} \\ 
\mlmat{K} &= \begin{bmatrix} 0 & 5 & 5 & 5 & 0 \end{bmatrix}
\end{align}
$$

Step 3: Compute the unmasked attention $\mlmat{Q} \mlmat{K}^\top$.

$$
\begin{align}
\mlmat{Q} \mlmat{K}^\top &= \begin{bmatrix} 0 & 5 & 5 & 5 & 0 \\ 0 & 5 & 5 & 5 & 0 \\ 0 & 5 & 5 & 5 & 0 \\ 0 & 5 & 5 & 5 & 0 \\ 0 & 5 & 5 & 5 & 0 \end{bmatrix}
\end{align}
$$

Step 4: Mask the matrix so that future tokens can't influence past tokens.

$$
\begin{align}
mask(\mlmat{Q} \mlmat{K}^\top) &= \begin{bmatrix} 0 & -\infty & -\infty & -\infty & -\infty \\ 0 & 5 & -\infty & -\infty & -\infty \\ 0 & 5 & 5 & -\infty & -\infty \\ 0 & 5 & 5 & 5 & -\infty \\ 0 & 5 & 5 & 5 & 0 \end{bmatrix}
\end{align}
$$

Step 5: Take softmax along the rows.

$$
\begin{align}
softmax(mask(\mlmat{Q} \mlmat{K}^\top)) &= \begin{bmatrix}    1 &  0 &  0 & 0 & 0 \\ 0.0067 &  0.9933  & 0   &      0   &    0 \\ 0.0034   & 0.4983 &   0.4983     &    0     &    0 \\   0.0022  &  0.3326  &  0.3326  &  0.3326    &     0 \\ 0.0022  &  0.3318  &  0.3318  &  0.3318  &  0.0022 \end{bmatrix}
\end{align}
$$


{% endcapture %}
{% include problem_part.html subpart=partc_prob solution=partc_sol label="C" %}

{% capture partd_prob %}
Define the value for the $i$th token as $\mlmat{W_V} \mlvec{r}_i$ where $\mlmat{W_V}$ is the identity matrix and $\mlvec{r}_i$ is the embedding of the token.  Construct the matrix $\mlmat{V}$ by computing the values of each token using the formula $\mlmat{W_V} \mlvec{r}_i$ and then transforming each value to a row of a matrix.  Show that taking your attention matrix from Part C and multiplying it on the right by $\mlmat{V}$ computes the output of the attention head which will give a vector close to $\begin{bmatrix} 1 & 0 \end{bmatrix}$ if no consonants preceded a token and $\begin{bmatrix} 0 & 1 \end{bmatrix}$ if at least one consonant preceded a token.
{% endcapture %}

{% capture partd_sol %}
The values are going to be the same as our embeddings.  We can lay them out as the rows of $\mlmat{V}$.

$$
\begin{align}
\mlmat{V} &= \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}
\end{align}
$$

We get the final outputs of our attention head by multiplying our matrix from part C by $\mlmat{V}$.

$$
\begin{align}
\begin{bmatrix}    1 &  0 &  0 & 0 & 0 \\ 0.0067 &  0.9933  & 0    &     0   &    0 \\ 0.0034   & 0.4983 &   0.4983     &    0     &    0 \\   0.0022  &  0.3326  &  0.3326  &  0.3326    &     0 \\ 0.0022  &  0.3318  &  0.3318  &  0.3318  &  0.0022 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} &= \begin{bmatrix} 1.0000     &    0 \\ 0.0067  &  0.9933 \\ 0.0034  &  0.9966 \\ 0.0022  & 0.9978 \\  0.0045  &  0.9955 \end{bmatrix}
\end{align}
$$

{% endcapture %}
{% include problem_part.html subpart=partd_prob solution=partd_sol label="D" %}

{% capture parte_prob %}
Suppose you wanted the attention head to determine the proportion of consonants that precede (rather than just whether a consonant precedes a word or not).  How would you modify $\mlmat{W_Q}$ and $\mlmat{W_K}$ to achieve this result?  You should not need to change $\mlmat{V}$.
{% endcapture %}
{% capture parte_sol %}
We could keep $\mlmat{W_Q} = \begin{bmatrix} 1 & 1 \end{bmatrix}$ the same.  We can now modify the key so that all tokens have the same key (all respond to the query) by setting $\mlmat{W_K} = \begin{bmatrix} 1 & 1 \end{bmatrix}$. Let's turn the crank.

$$
\begin{align}
\mlmat{Q} &= \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \end{bmatrix} \\ 
\mlmat{K} &= \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \end{bmatrix}
\end{align}
$$

$$
\begin{align}
\mlmat{Q} \mlmat{K}^\top &= \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}
\end{align}
$$

$$
\begin{align}
mask(\mlmat{Q} \mlmat{K}^\top) &= \begin{bmatrix} 1 & -\infty & -\infty & -\infty & -\infty \\ 1 & 1 & -\infty & -\infty & -\infty \\ 1 & 1 & 1 & -\infty & -\infty \\ 1 & 1 & 1 & 1 & -\infty \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}
\end{align}
$$

$$
\begin{align}
softmax(mask(\mlmat{Q} \mlmat{K}^\top)) &= \begin{bmatrix}    1 &  0 &  0 & 0 & 0 \\ 0.5 &  0.5  & 0   &      0   &    0 \\ 0.3333   & 0.3333 &   0.3333     &    0     &    0 \\   0.25  &  0.25  &  0.25  &  0.25    &     0 \\ 0.2  &  0.2  &  0.2  &  0.2  &  0.2 \end{bmatrix}
\end{align}
$$

Finally, combine our attention with our values (since they haven't changed from part D, let's just use those).
$$
\begin{align}
\begin{bmatrix}    1 &  0 &  0 & 0 & 0 \\ 0.5 &  0.5  & 0   &      0   &    0 \\ 0.3333   & 0.3333 &   0.3333     &    0     &    0 \\   0.25  &  0.25  &  0.25  &  0.25    &     0 \\ 0.2  &  0.2  &  0.2  &  0.2  &  0.2 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} &= \begin{bmatrix}  1.0000   &      0 \\    0.5000  &  0.5000 \\ 0.3333  &  0.6667 \\   0.2500  &  0.7500 \\   0.4000 &   0.6000 \end{bmatrix}
\end{align}
$$

{% endcapture %}
{% include problem_part.html subpart=parte_prob solution=parte_sol label="E" %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}

Next, let's see how a position embedding might help us.

{% capture problem %}
Suppose we want our attention head to take in a sequence of letters and output the vector $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ if there is a consonant at position 1 (where 1 is the first position in the sequence) and $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$ otherwise.

1. Input text: "eacia", our attention head should output $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$ (token 1 is a vowel).
2. Input text: "ccrs", our attention head should output $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (the first token is a consonant).

Let's use the same tokenization scheme as in the previous exercise. That is, each letter is mapped to its position in the alphabet (starting with $a \rightarrow 0$ and ending with $z \rightarrow 25$).

{% capture parta_prob %}
Explain what each of the features (the rows) of the input tokens (the columns) in the embedding matrix $\mlmat{W_E}$ captures.

$$
\mlmat{W_E} = \begin{bmatrix} 1 & 0 & 0 &  0 & 1 & 0 &  0 &  0 & 1 &  0 &  0 &  0 &  0 &  0 & 1 &  0 &  0 &  0 &  0 &  0 & 1 & 0 &  0 &  0 &  0 &  0 \\ 0 &  1& 1 &  1 & 0 & 1 &  1 &  1 & 0 &  1 &  1 &  1 &  1 &  1 & 0 &  1 &  1 &  1 &  1 &  1 & 0 & 1 &  1 &  1 &  1 &  1 \\ 0 &  0 & 0 &  0 & 0 & 0 &  0 &  0 & 0 &  0 &  0 &  0 &  0 &  0 & 0 &  0 &  0 &  0 &  0 &  0 & 0 & 0 &  0 &  0 &  0 &  0  \end{bmatrix}
$$

We can also specify our position embeddings for each token position (we'll stop at position $8$ since the pattern should be obvious).  Explain what the positional embedding matrix is representing.

$$
\mlmat{W_P} = \begin{bmatrix} 0 & 0 & 0 &  0 & 0 & 0 &  0 &  0 \\ 0 & 0 & 0 &  0 & 0 & 0 &  0 &  0  \\ 1 & 0 & 0 &  0 & 0 & 0 &  0 &  0  \end{bmatrix}
$$

{% endcapture %}
{% capture parta_sol %}
We have the same embedding as the previous problem but we've added a dimension that is always zero for the token embedding.  The positional embedding places a 1 in this dimension if the position is 1.
{% endcapture %}
{% include problem_part.html subpart=parta_prob solution=parta_sol label="A" %}

{% capture partb_prob %}
Define a query ($\mlmat{W_q}$) and key ($\mlmat{W_k}$) matrix pair that causes all letters to attend to only the first position in the sequence.

$\mlmat{W_q}$ and $\mlmat{W_k}$ are both matrices with $n_{q}$ rows and $n_{e}$ columns, where $n_q$ is the query dimension (you can choose this) and $n_e$ is the dimensionality our embeddings (in this example, 3).

Hint 1: You should be able to solve the problem with $n_{q} = 1$ (that is, the key and query matrices are both 1 row and 2 columns).

Hint 2: The key equation you'll want to use is that the degree to which token $i$ attends to token $j$ can be computed from the embeddings (both position and token embedding) $\mlvec{r}_i$ and $\mlvec{r}_j$ (these would be found in the appropriate columns of $\mlmat{W_E}$ and $\mlmat{W_P}$) of tokens $i$ and $j$ respectively using the following formula.

\begin{align}
attention &= \mlmat{W_q} \mlvec{r}_i (\mlmat{W_k} \mlvec{r}_j)^\top
\end{align}

{% endcapture %}
{% capture partb_sol %}
Let's define the matrices as follows.
$$
\begin{align}
\mlmat{W_q} &= \begin{bmatrix} 1 & 1 & 0 \end{bmatrix} \\ 
\mlmat{W_k} &= \begin{bmatrix} 0 & 0 & 5 \end{bmatrix}
\end{align}
$$

We leave it to you to validate that these matrices will do the job (sorry!).

{% endcapture %}
{% include problem_part.html subpart=partb_prob solution=partb_sol label="B" %}

{% capture partc_prob %}
Come up with a short sequence of characters, $s$, consisting of some vowels and some consonants (keep the length pretty small).  Compute the matrix of all queries corresponding to your sequence, $\mlmat{Q}$, where the number of rows of $\mlmat{Q}$ is equal to the number of tokens (the length of $s$) and the number of rows is equal to the query dimension.  Compute the matrix of all keys corresponding to your sequence, $\mlmat{K}$, where the number of rows of $\mlmat{K}$ is equal to the number of tokens (the length of $s$) and the number of rows is equal to the query dimension.  Compute the (pre-masking) attention of each token to each other token using the formula $\mlmat{Q} \mlmat{K}^\top$.  Apply masking to ensure that keys (columns) corresponding to later tokens do not influence earlier queries (rows).  Note: that the visualization in the 3B1B video (at [this time stamp](https://youtu.be/eMlx5fFNoYc?t=514)) has this matrix laid out with query tokens as columns and the keys as rows (we wanted to let you know to minimize confusion).  Apply a softmax across each row (as before, this is shown on columns in the 3B1B video) to determine a weight for each token and show the resultant matrix.
{% endcapture %}

{% capture partc_sol %}
Let's take our string to be $s = \mbox{cbcce}$.

Step 1: Compute our embeddings by picking out appropriate columns of our matrices (for both token and position embeddings). $r_1 = \begin{bmatrix} 0 \\ 1 \\ 1  \end{bmatrix}$, $r_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$, $r_3 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$, $r_4 = \begin{bmatrix} 0  \\ 1 \\ 0 \end{bmatrix}$, and $r_5 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$.

Step 2: Compute each query using the formula $\mlmat{W_q} \mlvec{r}_i$ and each key using the formula $\mlmat{W_k} \mlvec{r}_i$ and put each query as a row to form $\mlmat{Q}$ and each key as a row to form $\mlmat{K}$.

$$
\begin{align}
\mlmat{Q} &= \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \end{bmatrix} \\ 
\mlmat{K} &= \begin{bmatrix} 5 & 0 & 0 & 0 & 0 \end{bmatrix}
\end{align}
$$

Step 3: Compute the unmasked attention $\mlmat{Q} \mlmat{K}^\top$.

$$
\begin{align}
\mlmat{Q} \mlmat{K}^\top &= \begin{bmatrix} 5 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \end{bmatrix}
\end{align}
$$

Step 4: Mask the matrix so that future tokens can't influence past tokens.

$$
\begin{align}
mask(\mlmat{Q} \mlmat{K}^\top) &= \begin{bmatrix} 5 & -\infty & -\infty & -\infty & -\infty \\ 5 & 0 & -\infty & -\infty & -\infty \\ 5 & 0 & 0 & -\infty & -\infty \\ 5 & 0 & 0 & 0 & -\infty \\ 5 & 0 & 0 & 0 & 0 \end{bmatrix}
\end{align}
$$

Step 5: Take softmax along the rows.

$$
\begin{align}
softmax(mask(\mlmat{Q} \mlmat{K}^\top)) &= \begin{bmatrix}    1.0000     &    0     &    0      &   0     &    0 \\   0.9933 &   0.0067    &     0     &    0      &   0 \\   0.9867  &  0.0066  &  0.0066     &    0    &     0 \\    0.9802  &  0.0066  &  0.0066 &   0.0066     &    0 \\  0.9738  &  0.0066  &   0.0066 &   0.0066  &  0.0066 \end{bmatrix}
\end{align}
$$


{% endcapture %}
{% include problem_part.html subpart=partc_prob solution=partc_sol label="C" %}

{% capture partd_prob %}
Determine the of $\mlmat{W_V}$ in order to construct the matrix $\mlmat{V}$ by computing the values of each token using the formula $\mlmat{W_V} \mlvec{r}_i$ and then transforming each value to a row of a matrix. Show that taking your attention matrix from Part C and multiplying it on the right by $\mlmat{V}$ computes the output of the attention head which will give a vector close to $\begin{bmatrix} 1 & 0 \end{bmatrix}$ if the third token is a consonant and close to $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$ otherwise.
{% endcapture %}

{% capture partd_sol %}
$$
\begin{align}
\mlmat{W_V} &= \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
\end{align}
$$

This gives us $\mlmat{V}$.

$$
\begin{align}
\mlmat{V} &= \begin{bmatrix} 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}
\end{align}
$$

We get the final outputs of our attention head by multiplying our matrix from part C by $\mlmat{V}$.

$$
\begin{align}
\begin{bmatrix}    1.0000     &    0     &    0      &   0     &    0 \\   0.9933 &   0.0067    &     0     &    0      &   0 \\   0.9867  &  0.0066  &  0.0066     &    0    &     0 \\    0.9802  &  0.0066  &  0.0066 &   0.0066     &    0 \\  0.9738  &  0.0066  &   0.0066 &   0.0066  &  0.0066 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} &= \begin{bmatrix}         0   & 1.0000 \\  0  &  1.0000 \\      0  &  1.0000 \\ 0  &  1.0000 \\  0   & 0.9934 \end{bmatrix}
\end{align}
$$

{% endcapture %}
{% include problem_part.html subpart=partd_prob solution=partd_sol label="D" %}
{% endcapture %}
{% include problem_with_parts.html problem=problem %}

# Implementing Self-Attention

{% capture nanogpt %}
Hopefully the last problem got you thinking about how attention can cause tokens to attend to other tokens in a flexible manner.  While setting weights by hand can buidl intuition, we of course want to fit these to data.  Next, we're going to see how we can do that by implementing self-attention in Pytorch.  We are going to consult our old friend Karpathy (of micrograd fame) and go through his video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).  In this assignment, we're going to go from the beginning to time stamp 1:11:39.  Watching videos like this is way more valuable when you actively try things out as the video is unfolding.  To help scaffold this, below we have a sequence of time stamps in the video along with things to think about or try.

Before you start the video, you should probaby pull up the [gpt-dev.ipynb Colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) (linked in the video description).

* [10:13](https://youtu.be/kCc8FmEb1nY?t=613): make sure you understand the encoder and decoder for characters.  Try it out in the notebook on some sequences you feed in.
* [13:45](https://youtu.be/kCc8FmEb1nY?t=834): (something to ponder if you'd like, but not super critical) think through what Karpathy is doing when choosing the train / test split.  Is it a good idea to choose the first 90% of the data as train and the last 10% as test?
* [16:59](https://youtu.be/kCc8FmEb1nY?t=1019): make sure to understand the role of ``block_size`` as an upperbound on context length as well as the importance of extracting shorter contexts for training to allow the transformer to generate text starting from just a little bit of context.
* [20:59](https://youtu.be/kCc8FmEb1nY?t=1259): make sure to notice how the print outs of "inputs" and "targets" relate to each other.  Notice that targets(i,j) is what needs to be predicted given the context represented in the $i$th row of inputs up to column $j$.
* [22:44](https://youtu.be/kCc8FmEb1nY?t=1364): at this point Karpathy introduces the bigram language model.  The implementation of this Bigram model is Karpathy's way of starting with a simple model and gradually transforming it into a GPT.  This move may be a little bit unintuitive given where we are coming from, but we think it will all gel as the video goes on.  While we haven't seen the bigram model in this class, it's a pretty straightforward idea.  Imagine training a multiclass logistic regression model (linear layer followed by a softmax) that predicts the probability of the next token given the current token.  To represent these probabilities, we'll use a lookup table (implemented as a pytorch embedding) where the entry $i$, $j$ will be larger (more positive) if token $j$ often follows token $i$ and negative if token $j$ is unlikely to follow token $i$.  The entries of this lookup table will be learned from data (these would be the weights in our logistic regression model).
* [27:25](https://youtu.be/kCc8FmEb1nY?t=1645): you may want to play around (meaning running code in the notebook) with the the ``tensor.view`` function to get a sense for how Karpathy is using it to "unroll" the tensor with dimensions ``B, T, C``.
* [28:13](https://youtu.be/kCc8FmEb1nY?t=1693): notice that Karpathy is actually passing the loss as an output from the forward function.  That's a bit different to what we've been doing, but it's just a stylistic difference.  Don't get to hung up on it.
* [29:07](https://youtu.be/kCc8FmEb1nY?t=1747): Karpathy shows code for generating text (basically, continuously feeding the models predictions back into itself).  How this happens is a bit beside the point for us, so we recommend not worrying about the details of how he does this.
* [35:34](https://youtu.be/kCc8FmEb1nY?t=2134): now we are setting up our training loop.  This should look very familiar to what we've done earlier in this class.
* [40:17](https://youtu.be/kCc8FmEb1nY?t=2417): we've now transitioned to using a script.  We are estimating loss by averaging over multiple batches.  This is to avoid computing loss on the entire training set (which we've tended to do since our datasets have been relatively small).  Notice the cool decorator he uses on the ``estimate_loss`` function though (that could be handy to avoid having to using ``with torch.no_grad():``)
* [43:16](https://youtu.be/kCc8FmEb1nY?t=2596): notice that Karpathy is now switching to thinking of embedding the tokens in a space (in this case a 2-dimensional space) rather than using the embeddings as a convenient way to implement a bigram model.  This is similar to what we did when we thought about embeddings is the last assignment.  Instead of computing embeddings using ``nn.Embedding``, we're just generating them randomly to allow us to focus on the machinery of self-attention.
* [45:12](https://youtu.be/kCc8FmEb1nY?t=2712): our old friend the bag of words!  As mentioned in the video, we're only doing this simple averaging step as a brief stepping stone to the attention mechanism we learned about in the 3B1B videos.
* [47:48](https://youtu.be/kCc8FmEb1nY?t=2868): Karpathy really breaks this down nicely.  We recommend you interact with this toy example by running it yourself and making sure you understand the connection between the code and the matrix math.
* [53:35](https://youtu.be/kCc8FmEb1nY?t=3215): a quick note that if you actually run this code ``torch.allclose`` will actually give false!  Presumably some of the default values have changed in pytorch since this video was made.  Passing the keyword argument ``atol=10**-7`` along with the two matrices should give you ``True``.
* [55:35](https://youtu.be/kCc8FmEb1nY?t=3332): this should look familiar!  This is the masking we saw earlier.
* [59:10](https://youtu.be/kCc8FmEb1nY?t=3550): now we are making our bigram model look more like self-attention!  Notice how we are introducing the idea of ``n_embd`` to capture the number of embedding dimensions (this was 2 in the toy problem we did earlier).
* [1:00:17](https://youtu.be/kCc8FmEb1nY?t=3617): the version run at this point is still not doing any attention, but we have added some of the machinery necessary to implement self-attention.
* [1:00:57](https://youtu.be/kCc8FmEb1nY?t=3657): we are introducing position embedding, which was mentioned briefly in 3B1B videos since it can be important to self-attention.
* [1:05:11](https://youtu.be/kCc8FmEb1nY?t=3911): we now introduce the variable ``head_size``, which we previously referred to as the query dimension ($n_q$).  Also, not that if we didn't set ``bias=False`` we would have a constant added to the computation of our queries and keys, which we don't want.
* [1:07:08](https://youtu.be/kCc8FmEb1nY?t=4028): the way that the multiplication of two tensors works is a bit confusing for us, but hopefully we can leverage what we know about matrix multiplication.  If you want to go into this, you can check out [Understanding Broadcasting in Pytorch](https://www.geeksforgeeks.org/understanding-broadcasting-in-pytorch/).
* [1:11:37](https://youtu.be/kCc8FmEb1nY?t=4297): we made it to our stopping point for this assignment.  Look at the code to compute ``out``.  Can you connect the dots to the equation we learned about earlier for computing the output of our attention head, $softmax(mask(\mlvec{Q}\mlvec{K}^\top))\mlvec{V}$, and see how it corresponds?
{% endcapture %}
{% include external_resources.html content=nanogpt %}