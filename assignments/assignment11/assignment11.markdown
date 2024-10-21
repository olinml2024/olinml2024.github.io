---
title: Assignment 11 - Word Embeddings
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn about the concept of word embeddings and understand them as a form of unsupervised learning
* Understand the pros and cons of word embeddings versus the bag of words approach
* Examine word2vec encodings
{% endcapture %}
{% include learning_objectives.html content=content %}

# Word Embeddings

The concept of a word embedding was introduced in the [day 13 materials](../../activities/day13).  During class, we learned that a key motivation for word embeddings is to overcome a limitation that we observed with our sentiment classification algorithm from assignment 10.  Specifically, in the bag of words approach, each word is represented as an *independent* dimension in the vector that represents a particular movie review (recall that we analyzed sentiment of movie reviews in the previous assignment).

{% capture problem %}
Before getting into word embeddings in more detail, want to make sure you have a good handle on an important drawback of bag of words approaches.

Suppose, we had a training set consisting of the following movie reviews (you can assume that these are the only reviews in the training set and that we trained the model using a similar technique to what we used in assignment 10).

|--------|-------|
| Review | Label |
|---------|---------|
| The casting of the movie was impeccable | + |
| The movie was great | + |
| The movie was awful | - |
| The movie was the worst I've ever seen | - |
| The movie was an affront to the art of film-making | - |

Explain why a bag of words model trained on this data would have a difficult time evaluating the following movie reviews from a test set.

* "The movie was fantastic"
* "The cast of the movie did a superb job"

{% endcapture %}
{% capture solution %}
For the first review, "the movie was fantastic", the word "fantastic" does not appear in our training set.  Even though fantastic and great are closely related words, in the bag of words approach we treat each word as an independent dimension in our input vector.  If we want to understant fantastic as a synonym for great, we would need training data of movie reviews that contains the word fantastic.

For the second movie review, "The cast of the movie did a superb job", even though we use many similar words to what is present in the training set, the forms of the words (e.g., the chosen tenses) prevent a match with the training set.  In order to generalize to the word forms in this movie review, we would have to have the same word forms represented in the training set.
{% endcapture %}
{% include problem.html problem=problem solution=solution %}

[Word embeddings](https://en.wikipedia.org/wiki/Word_embedding) were introduced as a way to overcome the issues highlighted by the previous problem.  Instead of treating each word as an independent entity, we can learn to represent (embed) each word in a vector space that preserves key properties of the words themselves.  Let's use the symbol $r$ to represent our embedding (we'll use $r$ since it is a *representation* of the word).  We can think of $r$ as a function from words to the vector space $\mathbb{R}^d$ (don't get confused by this notation, $\mathbb{R}^d$ just means a d-dimensional vector of real numbers).

In order to learn our word embedding function $r$, we can use a form of machine learning called [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning).  As we discussed in the previous module, unsupervised learning involves learning from unlabeled data (in contrast to the supervised learning setting we've been studying for most of the term where we assume we have access to a training set consisting of input / output pairs).  We can use the concept of unsupervised learning as a way to create word embeddings.  There are quite a few ways to accomplish this goal, but two foundational approaches were proposed in the paper [Efficient Estimation of Word Representations in Vector Space
](https://arxiv.org/abs/1301.3781).  Here is the key figure from the paper.

{% include figure.html
        img="figures/word2vec.jpg"
        width="80%"
        alt="Two choices for learning word embeddings.  On the left is the continuous bag of words (CBOW) approach where the center word is predicted from the context.  On the right is the skip gram approach where the surrounding words are predicted from the center word."
        caption="Given a sequence of words, we can pose a prediction task where we try to either predict the center word based on the embeddings of the surrounding words (CBOW) or the predict the surrounding words based on the center word (skip-gram)." %}
{% assign word2vec = figure_number %}

As mentioned in the caption for {% include figure_reference.html fig_num=word2vec %}, we can use the data itself to pose a prediction task.  You might be wondering how we can call this unsupervised learning given that we are trying to predict something (either the surrounding words or the center word).  Well, the key is that the thing we are trying to predict is derived directly from the data itself (there is no need for any additional information, or label, to be added that is not in the data already).  As such, we can use this approach to learn a word embedding from a database of text (without the need for any additional labeling).

# Word2vec

As mentioned before, word2vec was introduced in the paper [Efficient Estimation of Word Representations in Vector Space
](https://arxiv.org/abs/1301.3781).  We don't think you need to read the paper (but you are certainly welcome to!), but we do want you to get a feel the word embeddings created by word2vec.  We have put together [a notebook that downloads the word embeddings and allows you to explore them a bit](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Assignment11.ipynb).

# Bias in Word Embeddings

{% capture problem %}
Depending on what experiments you tried with word2vec, you may have already seen some examples of bias.  We would like you to read the paper [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://www.researchgate.net/profile/Venkatesh-Saligrama/publication/305615978_Man_is_to_Computer_Programmer_as_Woman_is_to_Homemaker_Debiasing_Word_Embeddings/links/57a20cd508aeef8f311e0871/Man-is-to-Computer-Programmer-as-Woman-is-to-Homemaker-Debiasing-Word-Embeddings.pdf).  The paper gets quite technical in places, although many of the ideas you have seen before (PCA??!?).  We would like you to read sections 1-4 of the paper (sadly PCA only shows up in the later sections of the paper).  Please take notes on key takeaways and unanswered questions.  If you'd like to go into the latter sections of the paper (section 5 and beyond), please feel free to do so (this is not required, at all).

It's also probably worth mentioning that the literature on bias in word embeddings is quite extensive with a lot of fascinating things to explore (and we'd love to learn from you if you if you do more explorations!).

{% endcapture %}
{% include problem_with_parts.html problem=problem %}