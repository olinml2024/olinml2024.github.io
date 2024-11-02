---
title: Assignment 15 - Images as Data and Meet Convolution
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* Identify and explain key components of a convolutional neural network (CNN)
* Create and apply filters like a CNN
* Calculate the output size and values resulting from a given filter
{% endcapture %}
{% include learning_objectives.html content=content %}

# TO DO
* Make notebook without solutions.
* Decide if we will give solutions for this notebook (Sam is leaning no, since there are lots of correct solutions, and it might be more interesting to discuss the filters they created in class if we don't give them a "correct" filter, and I'd like to have something like this in the QAD).
* 


# Meet Convolutional Neural Networks
Convolutional neural networks were a major step in the world of computer vision (and image generation). In [class 17](../../activities/day17), we did some exploration of why these are cool and how they work. If you missed class, please review these materials. Now, you'll spend some more time solidifying your understanding. 


There are a huge number of resources out there. We suggest you look at two types: 
1. One that gives a high level overview and a visualization. We suggest the first one, but are providing a few other great options:
    * This [interactive visual overview of CNNs from a collaboration between Georgia Tech and Oregon State](https://poloclub.github.io/cnn-explainer/){:target="_blank"}. This one will allow you to explore each of the layers and functions. You can click on each of the parts to see more. There's a little video at the end that shows how to use the tool. 
    * This [write-up with some helpful visualizations by Ujjwal Karn](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets){:target="_blank"}.
    * [One of the earlier types of these visualizations focused on handwritten numbers](https://adamharley.com/nn_vis/){:target="_blank"}  by Adam Harley.
    * [Training on MNIST in the browser by Karpathy](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html){:target="_blank"}. This one shows the weights and the gradients.
2. This [lecture by Serena Yeung of Stanford (part of one of the most famous academic AI labs) explaining convolutional neural networks](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=6){:target="_blank"}. This lecture provides a little bit of history and does a nice job explaining some key terms and concepts, getting into the specifics and the math. This also gives you a taste of a classic academic lecture on this topic. Here's the [website from their class](https://cs231n.github.io/convolutional-networks/#conv){:target="_blank"}, which may also be a helpful but not required resource.

As always, you're welcome to find alternative resources (and share them with everyone if they are awesome)!  


For these first two exercises, we'd like you to **attempt to recall the answers based on what you learned above (without immediately looking back at these resources)**. The act of trying to recall things from your memory helps slow the forgetting process (see this [article by researcher Dr. Kathleen McDermott](https://www.annualreviews.org/content/journals/10.1146/annurev-psych-010419-051019){:target="_blank"} if you want evidence of this). Then you can check your answers with the resources and make your answers better.  

{% capture problem %}
Based on the materials above, explain the following terms/concepts:
* Convolution (conceptually and as a dot product)
* Filter size (F)
* Stride
* Padding (e.g., zero padding)
* Max pool 
* ReLu
* Flatten

{% endcapture %}

{% capture sol %}
The questions below will help you figure out if you understand some of these terms. The answers are in the suggested resources or you can look them up from other resources. We're intentionally not providing them here as we want you to practice making sense of other resources and summarizing, but feel free to use other resources or an LLM to check your understanding.
{% endcapture %}
{% include problem.html problem=problem solution=sol %}

{% capture problem %}
Describe the general architecture of a convolutional neural network for image classification. You don't need to go into a lot of detail here, we just want to draw your attention to the major things that happen and the order that they happen in. 

{% endcapture %}
{% capture sol %}
In CNNs, we start with an input image. We then apply a series of filters by sliding them across the image and getting a set of outputs that preserve the spatial information. This is typically followed by a non-linear activation function (e.g., ReLU). We often shrink the overall size by using some combination of pooling (e.g., max pool) and stride during the convolution. This can be repeated multiple times depending on the depth of the model. Finally, the output layers (that still have spatial information in their organization) are flattened (put into a vector) and then go through a series of multilayer perceptrons (or other linear layers) until a final classification layer.
{% endcapture %}
{% include problem.html problem=problem solution=sol %}

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

{% capture problem %}
Calculate the output from the following filter by hand (calculator fine).  
  
Filter:  
$$
\begin{bmatrix}
0 & -1 & 0 \\  
-1 & 4 & -1 \\  
0 & -1 & 0 \\  
\end{bmatrix}
$$

Image:  
$$
\begin{bmatrix}
10 & 0 & 10 & 0 \\  
10 & 0 & 10 & 0 \\  
10 & 10 & 10 & 10 \\  
0 & 0 & 10 & 60 \\  
\end{bmatrix}
$$

{% endcapture %}

{% capture sol %}
Because there is no padding, the output image will be smaller. You might realize that the setup for the top right and the bottom left have the same numbers, so you only have to do the math once. Notice how the relevant values in the bottom right are all 10s, so the filter outputs zero. Also notice how the big number in the bottom right corner makes no difference at all in this filtering.

$$
\begin{bmatrix}
-30 & 20 \\  
20 & 0  \\  
\end{bmatrix}
$$

Please note that ChatGPT4o got this wrong when we put it in, but ChatGPTo1-preview got it correct. 

{% endcapture %}

{% include problem.html problem=problem solution=sol %}





{% capture problem %}
In this notebook, you will create your own filters and apply them like they are part of a convolutional neural network. You will need to do a little research on filter types. 

{% endcapture %}

{% capture sol %}

[Here is a notebook with the solutions, which we might not actually provide in the real assignment, as it might be part of the QAD, so this is just for Paul to review. If we're good with this notebook, then I'll make a version w/o solutions.   ](https://colab.research.google.com/drive/1m0QRA92yrEQ0D2u2g5gh8ox8fsIb_g7k?usp=sharing){:target="_blank"}

{% endcapture %}

{% include problem.html problem=problem solution=sol %}
