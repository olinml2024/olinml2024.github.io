---
title: Day 17 (Nov 4) - Goodbye Text, Hello Images and Convolution
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:50am: Closing out LLMs 
* 10:50-12:00pm: Images as data
{% endcapture %}
{% include agenda.html content=agenda %}


# Share-out your application of LLMs
In [Assignment 14](../assignments/assignment14/assignment14#proposing-an-llm-for-an-application-and-context-you-care-about), we asked you to propose an application of an LLM for a context that you care about. We'll do a share-out about our applications and possibly think about where they fall on some axes (e.g., at Olin or beyond; positive to negative; practical to whimsical). 

# Identifying Gaps in Knowledge

Let's return to the visualization from the exercise 1 of assignment 14.  With folks around you, identify aspects of the visualization that you don't understand.  That is, what are the pieces in this diagram that we did not touch on in class thus far?

## Follow-up From Class

Hi folks.  We wanted to take a little bit of time to clarify a few pieces of the walkthrough of the nano-GPT visualization.  The questions people asked were great, so hopefully this information can help.

***Point 1:*** In [this visualization](https://bbycroft.net/llm) we have 3 attention heads.  Each attention head has its own independent matrices for computing keys, queries, and values.


***Point 2:*** The value vectors for each token in each of our 3 attention heads is 16-dimensional.  These 16-dimensional vectors are added together in a weighted fashion (with the weight given by the self-attention matrix) to compute the output of each attention head.

***Point 3:*** We stack the outputs of each attention head to get back to our original 48-dimensional space (the dimensionality of the embedding space is $C=48$).

***Point 4:*** We then take the vector from the previous step and pass it through a projection matrix to translate from whatever representations were learned by each attention head to something that is appropriate to add to the input embedding (via the residual pathway).  Amanda asked a brilliant question about this in class, which was why this translation is needed since all of the attention heads have the same inputs.  This is still a hard question to answer, and Jess Brown did a nice job offering a suggestion that each of the attention heads might learn a different internal meaning of value space (the $V$ matrix), and we need a linear mapping (a matrix) in order to combine these different value spaces (across heads) in a meaningful way. After reviewing the visualization again, there is one more way to explain this.  If we look back at [this section of the 3B1B video](https://youtu.be/eMlx5fFNoYc?t=818) we see two ways to think about computing value vectors in an attention head.

We could (but don't) think of the matrix, $\mlmat{W_V}$ that maps from embeddings to value vectors a $C \times C$ matrix (where $C$ is the embedding dimension).  As Grant Sanderson, of 3B1B, points out, this approach would use many more parameters to represent the mapping from embeddings to value vectors (versus keys and queries).  To make the number of parameters similar between these three matrices (keys, queries, and values), we can instead think of two steps for computing our value vectors.  First, we use a matrix $\mlmat{V_\downarrow}$ to go from the embedding space to a lower dimensional space (in the visualization we go from $C=48$ to $16$ dimensions).  Second, we use a matrix called $\mlmat{V_\uparrow}$ to go from the 16-dimensional representation back to the $48$ dimensional representation.  As Grant explains, this change to how we compute our value vectors constrains the number of parameters versus $\mlmat{W_V}$ as a $C \times C$ (48 by 48) matrix.  Mapping this intuition onto our visualization of NanoGPT, we can think of the box labeled ``V Weights`` as playing the role of $\mlmat{V_\downarrow}$ and the box labeled ``Project Weights`` as containing the $\mlmat{V_\uparrow}$ matrices for each of the three attention heads (stacked).

# LLM Quality Assessed Deliverable and plans for the rest of the semester

We'll talk about the timeline (some details now added to the homepage).

# Images as data

Let's discuss:  
What is different about images compared to a set of variables (like in the Titanic data set)? What about compared to text data?


One common application of image data is in medical image processing. Here's a few recent papers, including one about one about clinical trials. 
* [McKinney, S.M., Sieniek, M., Godbole, V. et al. International evaluation of an AI system for breast cancer screening. Nature 577, 89–94 (2020). https://doi.org/10.1038/s41586-019-1799-6](https://www.nature.com/articles/s41586-019-1799-6)
* [Esteva, A., Chou, K., Yeung, S. et al. Deep learning-enabled medical computer vision. npj Digit. Med. 4, 5 (2021). https://doi.org/10.1038/s41746-020-00376-2](https://rdcu.be/dYXIV)
* [Abràmoff, M.D., Lavin, P.T., Birch, M. et al. Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices. npj Digital Med 1, 39 (2018). https://doi.org/10.1038/s41746-018-0040-6](https://www.nature.com/articles/s41746-018-0040-6)


# Preview of assignment and talking about convolution

We'll learn about convolutional neural networks to process images. First, we need to understand what a convolution means in this context.
[Assignment 15 - Images as Data and Convolutions](assignments/assignment15/assignment15)