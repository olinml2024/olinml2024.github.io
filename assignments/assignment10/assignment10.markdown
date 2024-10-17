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

## Text Processing Beyond Natural Language

Many of the same techniques we will be learing about can be used to process text data other than natural language.  Examples of this sort of text data could be genomic sequences (where each symbol in the sequence consists of nucleotides A, C, T, and G), amino acid chains (where each symbol is one of the 20 amino acids present in the human body), structured text (e.g., Python code), etc.  For example, the Google's DeepMind team's [AlphaFold program for protein structure prediction just led to a Nobel prize in chemistry](https://www.nature.com/articles/d41586-024-03214-7).  [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) predicts protein structure from an Amino acid chain.  We won't be going into this sort of text processing all that much in this module (although some of the methods we will learn could be adapted fairly easily).  If you are interested in the idea of processing non-linguistic text, this might be a fruitful topic for a final project.

# Bag of Words

Next, we're going to learn about our first technique for adapting the machine learning approaches from the previous module to processing text.  In doing so, we're going to find ways of dealing with some of the unique features of text that initially might seem to make text incompatible with the techniques we've learned about.  The technique we're going to learn about is called "bag of words," and it deals with the issue of how to convert the symbols in a piece of text into a numerical representation and also how to deal with the fact that pieces of text are variable in length.  We hope you will enjoy these great external resources for learning about bag of words.

{% capture resources %}
Okay, we are now ready for our first pass adaptation of the machine learning models from the past module to text.  We'll begin by having you watch [a video from IBM called What is Bag of Words?](https://www.youtube.com/embed/pF9wCgUbRtc?si=zd1AYDQTJifqLtcZ).  Towards the end, this video gets into two, more advanced topics that we'll be digging into shortly.  The first is Tf-IDF and you'll learn about that in the notebook.  The second is the idea of word embeddigns (or word2vec), and you'll see that in the assignment after this one.  If you want one more (shorter video), we also recommend [this video from Socratica](https://www.youtube.com/embed/kLMhePA3BiY?si=MEfYE_SyhzkGBnch).
{% endcapture %}
{% include external_resources.html content=resources %}

{% capture problem %}
As a quick check of your understanding, encode the following three pieces of text using bag of words.  What would you need to do to normalize the data?  What does it mean that the bag of words is a sparse representation?  How do you see that in your solution to the exercise?

1. goodnight moon
2. goodnight cow jumping over the moon
3. and a little toy house and a young mouse
4. and goodnight mouse
{% endcapture %}
{% capture solution %}

If we examine the texts as a whole, we can identify the unique words that occur and assign each word to a dimension in our bag of words vector.  As long as we're consistent in how we do so, It doesn't matter how we assign words to vector dimensions.  Here is what the sentences would look like.

| dimension | word    | text 1 | text 2 | text 3 | text 4 |
| -------- | ------- | ------- | ----- | ----- | ---- |
| 1 | goodnight  |  1   | 1 | 0 | 1 |
| 2 | moon |   1 | 1 | 0 | 0   |
| 3 | cow    |  0 | 1 | 0 | 0   |
| 4 | jumping  | 0 | 1  |  0 | 0   |
| 5 | over    |  0 | 1 | 0 | 0   |
| 6 | the    |  0 | 1 | 0 | 0   |
| 7 | and    |  0 | 0 | 2 | 1   |
| 8 | a    |  0 | 0 | 2 | 0   |
| 9 | little    |   0 | 0 | 1 | 0  |
| 10 | toy    |  0 | 0 | 1 | 0   |
| 11 | house    |  0 | 0 | 1 | 0   |
| 12 | young    |  0 | 0 | 1 | 0   |
| 13 | mouse    |  0 | 0 | 1 | 1  |


If we were to normalize these representations, we would divide each column by the sum of the column (i.e., the total number of words in each piece of text).

The bag of words representation is sparse as most of the entries in the table are 0.  If we had a larger vocabulary the sparsity would be even more apparent (a highe proportion of entries that are 0).


{% endcapture %}

{% include problem.html problem=problem solution=solution %}

## Text Classification with Bag of Words

In the video from IBM, there were several examples used to motivate the notion of bag of words for text classification.  Let's use one of the problems mentioned, sentiment analysis, and apply it to analyzing movie reviews.  We'll be using a fairly old dataset for our analysis, but it is one that is easy to work with and big enough for us to learn some important skills about working with text.  The dataset is Stanford's [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). Here is a snippet from the README.md file that is included with the dataset.

> Large Movie Review Dataset v1.0
> 
> Overview
> 
> This dataset contains movie reviews along with their associated binary
> sentiment polarity labels. It is intended to serve as a benchmark for
>sentiment classification. This document outlines how the dataset was
> gathered, and how to use the files provided. 
> 
> Dataset
>
> The core dataset contains 50,000 reviews split evenly into 25k train
> and 25k test sets. The overall distribution of labels is balanced (25k
> pos and 25k neg). We also include an additional 50,000 unlabeled
> documents for unsupervised learning. 
>
> In the entire collection, no more than 30 reviews are allowed for any
> given movie because reviews for the same movie tend to have correlated
> ratings. Further, the train and test sets contain a disjoint set of
> movies, so no significant performance is obtained by memorizing
> movie-unique terms and their associated with observed labels.  In the
> labeled train/test sets, a negative review has a score <= 4 out of 10,
> and a positive review has a score >= 7 out of 10. Thus reviews with
> more neutral ratings are not included in the train/test sets. In the
> unsupervised set, reviews of any rating are included and there are an
> even number of reviews > 5 and <= 5.

In the [assignment 10 notebook](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Assignment10.ipynb), you'll be working with this dataset and implementing your own machine learning system for predicting the sentiment of a movie review using a bag of wordsd representation.

## Bag of Words and Machine Learning Bias


{% capture problem %}
Let's do a little spiraling back to one of the big ideas in machine learning we started the semester with.  We want to draw your attention to this specific example.

> You may have heard that [Amazon
> scrapped a secret AI recruiting tool that showed bias against women](https://www.reuters.com/article/us-amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G).
More specifically, the tool performed automatic keyword analysis of job applications to predict whether or not the applicant was worth forwarding on to a human for further evaluation. Early in the development of this system researchers discovered that the model the system had learned placed a negative weight on words such as "women's" as well as the names of some women's colleges.

Given what you just learned about the bag of words approach and what we learned about [confounding variables in assignment 4](../assignment04/assignment04#confounding-variables), how do you might Amazon's system have learned to associate negative feature weights with the gendered words or words associated with women's collleges.

{% endcapture %}
{% capture solution %}
The Amazon engineers probably didn't think to screen out particular words from their machine learning model.  Likely, they assigned a dimension to all unique words to increase the predictive power of the model.  There was a correlation between resumes not doing as well and the presence of gendered words and the names of women's colleges.  It's hard to say why this correlation might have existed without more investigation (e.g., it could have been conscious or subconscious bias on the part of the evaluations that were used to make the training set).  Given this correlation, the machine learning model associated a negative weight with these words and baked it into the model.  In this way a correlation (that having these words in your resume was correlated with being screened out) was made causal by the model (if this model were to be applied to real resumes, then people with these words would be more likely to be discriminated against).
{% endcapture %}
{% include problem.html problem=problem solution=solution %}