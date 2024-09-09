title: Day 1
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:45am: Debrief at tables about the last assignment.
* 10:45-10:50am: Split into two rooms, depending on desire for intro to types of ML
* 10:50-11:15am: (Room 128, else skip down) Discuss types of ML and general workflow
* 11:15-11:40am: Explore types of image transforms
* 11:40-12:00pm: Start assignment 
{% endcapture %}

{% include agenda.html content=agenda %}

# Debrief on the last assignment

1. Introduce yourselves
2. Quickly draw a confusion matrix at your table and write the equations for accuracy, precision, and recall.
3. Discuss your answers for Exercise 9 in the Colab notebook (see exercise below as a reminder).

    Exercise 9: Summarize how well the dessert classifier works for french toast and red velvet cake.
    Come to class prepared to share this at your table.
    Consider the confusion matrix, precision, and recall. How do you interpret this?
    What does it mean for life as french toast or as red velvet cake?


# Types of ML and general ML workflow
We will talk about some types of machine learning and the general machine learning workflow.


# Explore image transforms

In the next assignment, you'll be evaluating some existing models. As part of this, you may apply different transformation to the images that you'll use to evaluate the model. These same types of transforms can also be used to augment an initial dataset, often making models trained on it more robust. In class, we'll show a few transforms and how to find documentation for others.
