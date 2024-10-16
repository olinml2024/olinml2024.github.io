---
title: Assignment 11 - Word Embeddings
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn about the concept of word embeddings and contrast them with bag of words
* Implement the Google word2vec model
{% endcapture %}
{% include learning_objectives.html content=content %}

# Word Embeddings

Give a basic overview of why this is important (e.g., helping to train a model using unsupervised learning, better generalization, identification of biases, etc.)

## Limitations of Bag of Words

So far we have learned how to represent text using a bag of words approach.  In the bag of words approach, each unique word was assigned a dimension in our representation of a piece of text.  As we processed each word of the text, we kept track of how many of each unique word was present by adjusting the relevant dimension (corresponding to each word) in our vector representation.  One consequence of this approach is that when we combined bag of words with machine learning, our classifier (let's use logistic regression as our example) had to learn the association between the class we were trying to predict (e.g., sentiment) and each unique word.  Let's do a quick exercise to try and understand this better.

{% capture content %}
TODO: give a training set with things like ``the movie was horrible``, ``the movie was fantastic``, and a test sample ``the movie was great``.  Have students explain why the classifier cannot predict the sentiment of the test sample despite the fact that we know that great and fantastic are similar words.  This motivates the idea that we need some way to represent words that allows a notion of semantic similarity.
{% endcapture %}
{% capture solution %}
TODO
{% endcapture %}
{% include problem.html problem=content solution=solution %}

Say something about how bag of words is an example of a sparse representation (mostly zeros in our feature vector).  Contrast this with the idea of a dense representation.

# Word2vec

TODO: describe word2vec (possibly by linking to an external source).  Students will implement some facet of it in Pytorch through Colab.

Do some sort of searching for similar words in the word2vec space.

