---
title: Quality Assessed Deliverable - Machine Learning with Images
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

The intent of this quality assessed deliverable is to demonstrate your understanding of a few key concepts from the Images as Data module (largely focusing on convolutional neural networks).  

Most of the questions pull directly or indirectly from your prior assignments. The intent here is to minimize the workload (assuming you have been keeping up on your assignments) and to revisit concepts from the assignments now that you have had a chance to ask questions in class.

{% capture content %}
* Identify and explain key components of a convolutional neural network (CNN)
* Create and apply filters like a CNN
* Calculate the output size and values resulting from a given filter
{% endcapture %}
{% include learning_objectives.html content=content %}

Will be a Canvas quiz.

1. a. Manually calculate the output matrix resulting from applying the following filter:
$$ 
\begin{bmatrix}
0 & ~~~1 & ~~~0 \\  
1 & -4 & ~~~1 \\  
0 & ~~~1 & ~~~0 \\  
\end{bmatrix}
$$
to this image (via a convolution):
$$ 
\begin{bmatrix}
0 & 0 & 0 & 0 \\  
0 & 0 & 50 & 50 \\  
0 & 50 & 50 & 50 \\  
0 & 0 & 50 & 0 \\  
\end{bmatrix}
$$



1. b. What is the name for this kind of filter? 

2. How many weights (parameters, excluding biases) need to be learned for a convolutional layer that starts with a 10x10x3 a.k.a a 10x10 RGB color image and results in an output  with a size of 6x6x10, using no padding and a stride of 1?
    * Solution: This question requires knowledge of the effects of the filter size on output image size, knowledge that an RGB image has a depth of 3, knowledge that each filter on a color image has size fxfx3, knowledge of how stride affects output size.
        * To go from a 10x10 to a 6x6 with a stride of 1 and no padding requires a 5x5 filter (losing 2 on each side).
        * Each filter (kernel) therefore is a 5x5x3, so 75 weights to be learned.
        * The output has 10 layers (channels), so there are 10 filters to be learned, so 75x10 = 750
    * Common mistakes:
        * If 250, you may have forgotten about the RGB part.
        * If 300*360 = 108000, then you're probably doing a fully connected layer
        * if 25*7 = 175, then you may have misunderstood what happens with color images and a kernel.
        * if 270, then you assumed a filter size of 3x3 (ChatGPT 4o mini made this error)
        * 

3. Compared to the situation in the previous question (convolutional layer), how would the number of weights change if you instead had a fully connected weights between each node in the input layer and each node in the output?  
    a. The number of weights would increase significantly (fully connected weights needed are more than 10x what is needed for convolutional layer).  
    b. The number of weights would increase somewhat (fully connected weights needed are less than 10x what is needed for convolutional layer).  
    c. The number of weights would stay the same.  
    d. The number of weights would decrease somewhat (convolutional layer weights needed are less than 10x what is needed for fully connected layer).   
    e. The number of weights would decrease significantly (convolutional layer weights needed are more than 10x what is needed for fully connected layer).    

Consider the following situations where 9 3x3 filters are applied to the following images (no padding):  
A. a 24x24 grayscale image with a stride of 1.  
B. a 24x24 grayscale image with a stride of 2.  
C. a 32x32 grayscale image with a stride of 2.  

4. Rank order the number of nodes in the output from largest to smallest:  
a. A, then B, then C  
b. A, then C, then B  
c. A and B are the same, then C  
d. B and C are the same, then A  
e. B, then C, then A  
f. C, then B, then A  
g. C, then A, then B  
h. C, then A and B are the same  
i. They are all the same  

5. Ignoring the bias term, rank order each situation based on the number of weights to be learned (from largest to smallest):  
a. A, then B, then C  
b. A, then C, then B  
c. A and B are the same, then C  
d. B and C are the same, then A  
e. B, then C, then A  
f. C, then B, then A  
g. C, then A, then B  
h. C, then A and B are the same  
i. They are all the same  

6. Write code for a 5x5 filter that will result in any vertical edges showing up as positive values if the the edge is darker on the right and lighter on the left and as negative values if the edge is darker on the left and lighter on the right.  Apply appropriate zero padding to maintain image size and use a stride of 1.

You should be able to copy your padding and apply filter code from Assignment 15 directly into this and then just add the filter. 
Please note that if you were getting a swirly looking penguin in Assignment 15, this was likely due to an issue with the datatype when you created the image. To solve this, you can either start with:
image_filtered= np.zeros((image.shape[0], image.shape[1]))  
or 
image_filtered = np.zeros_like(image, dtype = np.float32)  
If you're still getting something like this, please reach out to us.


Put the values of your filter here.  


Upload the figure generated by your code here.   



7. Match the loss curve to what is happening 


(shuffle these answers in the Canvas)

8. Generally, why are convolutional layers useful for image processing? Check all that apply.  
    a. They reduce the number of weights in the model compared to a fully connected layer, reducing the number of parameters that the model needs to learn.  
    b. They increase the number of weights in the model compared to a fully connected layer, allowing to model to learn more information.  
    c. They preserve some of the spatial information of the image.  
    d. They extract image features that are relevant for the task (e.g., classifying types of images).  
    e. They extract tokens that minimize the variance.  
    f. Convolutional layers detect objects by directly classifying each pixel as an object.  
    g. Convolutional layers are useful for image processing because they treat each pixel individually without considering its neighbors.  



9. In transfer learning for image processing, why are pre-trained models often fine-tuned?   
A. To make the model smaller and more efficient  
B. To adapt the model to specific features in the new dataset  
C. To replace all layers with random weights  
D. To ignore the data used in the original training  

10. In transfer learning, what does it mean to use a model as a “feature extractor”?    
A. The model's final layers are replaced to match the new task's requirements  
B. Only the output layer is trained while other layers are frozen  
C. All layers are re-trained to fit the new dataset  
D. The model's earlier layers are used to extract general features, which are then used in a new model  

11. What is one potential drawback of using transfer learning for an image processing model?  
A. It requires significantly more data than training from scratch  
B. It can be harder to interpret and understand the model’s decisions  
C. It prevents the model from learning new features  
D. It is only effective for small-scale image processing tasks  