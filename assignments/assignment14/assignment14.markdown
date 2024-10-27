---
title: Assignment 14 - Images as Data and Meet Convolution
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Learn!
{% endcapture %}
{% include learning_objectives.html content=content %}

# TO DO
* clean up writing
* add exercises
* add some initial fodder on images, or put in the in-class part


# Meet Convolutional Neural Networks

[1.1 hours]
Watch this lecture by Serena Yeung (Stanford) explaining convolutional neural networks. This lecture provides a little bit of history and does a nice job explaining some key terms and concepts. 

Some terms and ideas from this that you'll need for the rest of the assignment are:
* Filter size (F)
* Stride
* Padding (e.g., zero padding)
* Calculate the output size on a convolutional layer based on the terms above
* Convolution as a dot product




{% capture problem %}
{% capture part_a %}
Given an input feature map of size 32 × 32 with a single channel, a filter size of 5 × 5, a stride of 1, and no padding, calculate the dimensions of the output feature map after a single convolution operation.
{% endcapture %}
{% capture part_a_sol %}
The value after the filter (convolutional filter) goes into the spot that is the center of the filter. This means we'll lose two rows and two columns on each side (since we have no padding). This will give us an output of 28x28.
{% endcapture %}
{% include problem_part.html label="A" subpart=part_a solution=part_a_sol %}
{% capture part_b %}
Repeat the above exercise for a filter of size 4 x 4. Why would we not want this filter?
{% endcapture %}
{% capture part_b_sol %}
A 4x4 filter doesn't have a center that we can index (it's either the 2nd or 3rd item). It also changes or image size from an even to an odd number, shifting the middle of our image and losing some information in an asymmetrical way.
{% endcapture %}
{% include problem_part.html label="B" subpart=part_b solution=part_b_sol %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}

{% capture problem %}
For an RGB image of size 28x28, apply 6 different 7x7x3 filters with zero-padding of 3 and a stride of 1. What is the size of the output (give all dimensions)?
{% endcapture %}

{% capture sol %}
28x28x6. Each filter takes the image from a depth of 6 to a depth of 1, but there are 6 of them that get stacked. The padding of 3 balances out the filter size of 7, keeping the height and width at 28.
{% endcapture %}
{% include problem.html problem=problem solution=sol %}


{% capture problem %}
Given a grayscale image of size 64×64, apply a convolutional layer with the following parameters:<br/>
Filter size: 5×5 <br/>
Stride: 2 <br/>
Padding: 0 (no padding) <br/>
Calculate the size of the output feature map after applying this convolution.
{% endcapture %}

{% capture sol %}
$$ Output Dimension = \frac{Input Size - Filter  Size + (2 * Padding)}{Stride} + 1 $$
$$ \frac{64 - 5 + (2 * 0)}{2} + 1$$
The output will be 30 x 30. 
{% endcapture %}
{% include problem.html problem=problem solution=sol %}