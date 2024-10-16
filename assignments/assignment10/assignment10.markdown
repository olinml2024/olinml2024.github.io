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

The theme of this module is text as data.  More precisely, we seek to answer the question of how we can use machine learning approaches to process text in order to solve problems (e.g., text classification or language translation).  Throughout this module, we will learn different methods to convert text to numbers that can be operated upon using the machine learning techniques we learned about in the last module (e.g., logistic regression and MLPs).

## Key Properties

Before we dive into some of the key applications of machine learning for text processing, let's take some time to think about what makes processing text different than much of the data we've looked at thus far.

### Text consists of symbols

Pieces of text are comprised of symbols.  For example, the text your reading right consists of symbols that include letters, numbers, punctuation, and other special characters.  Perhaps the most important distinction for us as machine learning practitioners is that these symbols do not necessarily have a meaningful numerical representation that we can use for learning.  As we move forward in this module, we're going to learn different methods for changing these symbols into some sort of useful numerical representation so that we can use techniques like logistic regression and MLPs for futher processing.  It's also worth mentioning that when representing text we can also choose the symbols that we use.  Some models treat each letter as an individual symbols, while others treat each word as a symbol.  Other models treat parts of words as symbols.  We'll be digging into all of this in a few assignments.

### Text has sequential structure

When we first met the supervised learning problem, we represented our input to the model as $\mlvec{x}$.  Each of the dimensions of this vector represented some characteristic of the data.  In the logistic regression model and the MLP, each dimension of $\mlvec{X}$ was treated more-or-less independently.  That is, we did not assume any specific relationship between $x_i$ and $x_j$ (we could just as easily have shuffeled the dimensions of the data and our learning approaches wouldn't have behaved any differently).  When processing text, we need to consider that pieces of text have sequential structure.  The order of the symbols (i.e., letters) matter.  Our first attempts (in this assignment) to map machine learning onto text processing will not do a great job encoding this sequential structure, but as we move through the module we will begin to use models that represent this sequential structure in important ways.

### Text has variable length

Again, thinking back to our input vector $\mlvec{x}$, it had a fixed number of dimensions.  Pieces of text consist of sequences of symbols *over variable length*.  As a concrete example, later in this document you'll learn about sentiment analysis (predicting if a piece of text is positive or negative) from text.  The individual pieces of text will contain varying numbers of symbols.  Our machine learning methods will have to be able to handle this variability, so far it's not obvious how we can make this happen (but we'll see one way by the end of this assignment).

## Important Problems in the Field of Natural Language Processing

Before we get into how to process text, it's important to ask *why* we might want to process text.  Perhaps this almost seems like a silly question given the fact that everywhere you turn these days folk are talking about processing text with large language models (LLMs).  We're going to go over a few of the specific problems that arise in a field called Natural Language Processing, but we're also going to have you do some of your own research.  First, natural language process is a field concerned with, not surprisingly, processing and making sense of natural language.  Don't let the term "natural language" confuse you, all we mean here is that we want to be able to process text that is written in natural form (i.e., how humans communicate).  In this case the world "natural" might be seen as a contrast to the notion of processing text that is constructed in some specific way as to be easily interpretable by a computer (e.g., a programming language is a good example).

* Machine translation: TODO
* Text completion: TODO
* Question answering: TODO
* Named entity recognition: TODO
* Sentence parsing: TOOD
* Sentiment analysis: TODO

{% capture content %}
Choose one of the natural language processing tasks listed above (or substitute one of your own).  Do some research to determine what are some applications of algorithms that solve the problems listed above.  The distinction here is between problems and how a solution to that problem can be used for some purpose (an application).  Some of these problems may be harder to find information on than others, so do your best.  Aim for a medium length paragraph, 5-6 sentences, for your response.  If you choose a natural language processing problem not listed above, include a brief description of the problem itself along with the applications you found.
{% endcapture %}
{% include problem_with_parts.html problem=content %}

## Beyond language: non-linguistic text

TODO: talk about other forms of text that can be modeled using machine learning (e.g., DNA sequences, proteins, etc.).

# Bag of Words

TODO: We'll talk about the basic representation and have some problems that test understanding.

## TF-IDF

TODO: show how normalizing features can make the vectors better for learning.

## Text Classification with Bag of Words

TODO: sentiment analysis is always a good one to start with.  We'll do a notebook that goes into this.

## Potential to Introduce Unwanted Bias

TODO: something like the Amazon resume evaluator.  Maybe we can find a dataset to show how correlation between words and an outcome becomes causal when fed through an ML model.  This will probably be part of the notebook.

Maybe this one? https://www.kaggle.com/code/drindeng/resume-screening-using-nlp-different-ml-algorithms/notebook