---
title: Assignment 12 - Generative Pre-Trained Transformers (GPTs) Part 1
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn about sequence prediction for text
* Learn about different types of tokenization
* Bigram model as a stepping stone to GPTs
* Understand and build a self-attention block
{% endcapture %}
{% include learning_objectives.html content=content %}

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
