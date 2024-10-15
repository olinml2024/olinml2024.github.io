---
title: Assignment 10 - Bag of Words and Text Classification
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn about the field of natural language processing (NLP) and see some important problems from NLP
* Learn about bag of words methods for representing text as data
* Use a bag of words methods for text classification
{% endcapture %}
{% include learning_objectives.html content=content %}

# Text as Data

The theme of this module is text as data.  More precisely, we seek to answer the question of how we can use machine learning approaches to process text in order to solve problems (e.g., text classification or language translation).  Throughout this module, we will learn different methods to convert text to the numbers that can be operated upon using the machine learning techniques we learned about in the last module (e.g., logistic regression and MLPs).

## Key Properties

TODO
* Text is sequential
* Text is symbolic

## Important Problems in the Field of Natural Language Processing

TODO
* Machine translation
* Text completion
* Question answering
* etc.

## Beyond language: non-linguistic text

TODO: talk about other forms of text that can be modeled using machine learning (e.g., DNA sequences, proteins, etc.)

# Bag of Words

TODO: We'll talk about the basic representation and have some problems that test understanding.

## TF-IDF

TODO: show how normalizing features can make the vectors better for learning.

## Text Classification with Bag of Words

TODO: sentiment analysis is always a good one to start with.  We'll do a notebook that goes into this.

## Potential for Introducing of Unwanted Bias

TODO: something like the Amazon resume evaluator.  Maybe we can find a dataset to show how correlation between words and an outcome becomes causal when fed through an ML model.  This will probably be part of the notebook.