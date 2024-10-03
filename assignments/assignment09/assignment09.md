---
title: Assignment 9 - Small data project on classification
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# TODO
* decide if we should do interim check-in as part of the project or as part of the regular assignments
* add details about metrics and use cases
* update hot tips doc to be less focused on images (probably)
* revise rubric


# The project

You will be working solo on this project (1.5 to 2 weeks). The syllabus calls this a mini-project, but we'll just call it a project here. 

In this project, you will contemplate a potential application for classification, you will implement and improve upon a machine learning algorithm on a data set in support of this application, and you will evaluate the models performance. 

As part of this project, you will:
* Select a "small data" dataset.
    * "Small data" here is to indicate that you don't need to choose a giant data set (too much for Colab to hold in memory). The goal is to learn about the machine learning process, building your skills and intuition. There is no requirement to get high accuracy. That said, you do need a fair number of data points (exemplars) to meaningfully train your network. This varies by the size of your network and the number of parameters. (You'll be in rough shape if you have fewer than 30 exemplars per category even if your model has a small number of parameters.)
* Document the important considerations for your application. For example: 
    * What data are available for training?
    * How well would the algorithm need to work to responsibly create value? 
    * How could the algorithm and application be tested (beyond the testing you choose to implement)? 
    * What are the implications of this application in the world; who are the stakeholders; what are the risks; how (if at all) could they be mitigated?
* Build and iterate on a classification model as a step toward this application. 
    * We don’t expect you to build the entire application, just to work toward implementing an early version of an algorithm that could be used for this application. Your dataset doesn't need to be sufficient (e.g.,  big or broad enough) to actually make the application.  
    * The iteration aspect is important. You will build and test multiple versions of your model and document the effects of these changes. You will likely manipulate a parameter and see how your results change at different values of that parameter.
    * You can use pre-built libraries like pytorch. It’s highly likely that you will use a neural network, but not required. Please talk with us if you plan on using a different type of model. 
* Visualize key aspects of your data/model. Include at least 3 meaningful visualizations in your report with clear labeling and a discussion of their meaning (though you'll likely want more than three visualizations).
    * You’ll probably find it valuable to visualize many aspects of the data and model as you work. This is helpful in sanity checking and can give good insight into how the model is working.
    * It's likely that you'll visualize: some things about your data (before even making the model); a graph that shows the loss of your model as during the training process (see figures and caption at the end of [this page](https://livebook.manning.com/concept/deep-learning/training-loss)); changes in some key metrics as you manipulate a parameter (e.g., number of hidden layers); comparison of key metrics for different versions of your model; final training and testing metrics.
* Test and evaluate your model. This will likely include accuracy and other key metrics on training and test sets. Depending on your application, you may consider testing with additional data from another source (e.g., data you collect; another internet dataset that's similar; simulated data based on theory). You will also evaluate the effectiveness of this model for your specific application and describe the limitations of your current model and data.
    * Your accuracy and other metrics don't have to be good enough for your application. The goal here is to iterate on your model, measure its effectiveness, and reflect on this in the context of an application. 
* Document your final analysis pipeline for transparency. (A simplified version that does not need to include every parameter, visualization, or tweak that you tried.) It should be well-organized and easy to follow (this takes time, so budget for it).



# Learning Objectives

{% capture content %}
This mini-project relates to all of the intended learning outcomes of this course:
* Execute the iterative machine learning workflow of model design, fitting to training data, testing, and interpretation in order to be able to successfully apply machine learning techniques in specific contexts. 
    * Yup, definitely this one.
* Successfully implement machine learning algorithms in Python (both by using only minimal external libraries and by leveraging state-of-the-art machine learning libraries).
    * You can use libraries (e.g., PyTorch)
* Consider the potential impacts of a machine learning system when deployed in a real-world context and make design decisions to mitigate potential harmful impacts. 
    * Definitely the first part of this, and you should think about the design decisions you would make.
* Understand a variety of machine learning techniques from both a mathematical and algorithmic perspective.
    * This project is not focused on the math, but in the past, this kind of project was where conceptual things clicked for several students (see Appendix).
{% endcapture %}
{% include learning_objectives.html content=content %}


# Goal-Setting and Customization
A good project as one that successfully uses a neural net on a dataset and demonstrates iteration (you try the neural net, you make improvements, you try again, and you assess the model performance thinking about a specific application). If you wish to go farther, that's great!

We've highlighted the learning objectives, and we also (informally) ask you to consider your own learning goals. Is there a specific skill you want to practice? Is there a way of learning (like, asking for help more, trying code without looking at examples first, etc) that you want to practice?  You can customize this project to support your own learning, and we are happy to help you shape the project to support your goals and challenge yourself. 


# Timeline

The project will officially launch on Thursday, October 3, 2024, and end on Thursday, October 17, 2024. While this is 2 weeks of time, we also want to note that Fall break is on October 14th and 15th (Monday and Tuesday, so no class on Oct 14), so we encourage you to work on this project like it is one week with a little time for revisions (but you should make your own choices based on your situation).

# Use of external resources (including peers, the internet, and AI)
You should lead your own project, and by "lead", we mean that you should be the active thinker and doer of the work. This is how you build your skills and intuition. You should write and understand the code and text in your project. 

We are extending our trust to you (we will not run your report through a plagiarism checker), and we expect that you will follow the [Olin Honor Code](https://www.olin.edu/student-life-student-affairs-and-resources/student-rights-responsibilities). The values of integrity and respect for others seem most directly relevant. We expect that your submitted work represents you own skills and understanding. However, we also recognize that this is a fuzzy thing to navigate. We've tried to articulate some guidelines here for equity and transparency, and we invite open discussion about what is appropriate and fair. We can navigate this fuzzy world together (like [James and his friends in the the giant peach](https://en.wikipedia.org/wiki/James_and_the_Giant_Peach)).


## Peer collaboration (students in this class or other humans, squirrels, and monkeys with typewriters)
We encourage collaboration and supporting each other to enhance everyone's learning. However, it's inappropriate to have someone else write your code for you. It is okay to talk to someone about something that you're stuck on. It's also okay for them to show you how they solved it. It's not okay to mindlessly copy several lines of code or text directly from someone else's assignment. 


## The (not actually) magical world of the internet and artificial intelligence
It's quite likely that someone has used your dataset for a machine learning project and posted that somewhere online. You can use these materials to help you when you are stuck, but you should not just copy and paste large chunks of code from the internet or AI (likely scraped from the internet). You do not need to turn off the auto-complete functions in Colab or your IDE, but please be aware that these also make mistakes.

You should not start this project by having AI generate your first draft; the thought process of thinking through each of the steps is important to build your understanding. It is great to look at examples online that use a different dataset in order to build your understanding of the process, and then to emulate that in your own code. 


## Class-owned editable resources file

We will continue to use Slack and office hours to answer questions. However, we also know that a great deal of learning happens outside of those places. [This shared, class-owned, editable document](https://docs.google.com/document/d/15zNjQp32oBqaD4CY4gAsgSnkAWUs3JbTJYNRAEr1H28/edit?usp=sharing) is intended to serve as a communal set of resources for troubleshooting as we figure things out together. Anyone can add to it, so as you run into problems (you will) and figure out solutions (sometimes on your own, sometimes with help), please add to this document to help others. Learning is not a zero-sum game.

This shared document already has two years of notes from Machine Learning students in the past (from a image-focused project), so some things may be outdated (you can delete something if it no longer works or applies to this project). We also want to note that the field of machine learning is constantly changing, as are things like libraries and toolboxes. It's likely that you'll run into something that worked 3 years ago and does not work now. Please help us find and fix these things!

# Deliverables & Assessment

## Deliverables

1. A clean report in notebook (.ipynb) format with thoughtful explanations of the considerations for your computer vision application and important highlights of your analysis. This should include in-line code and figures. This should **not** include every analysis and figure that you generated. 

###TODO: FIX THIS
Please choose three interesting things that you investigated and share your code and figures for these. You can mention other things that you investigated in the text and include the code in your repository. 

This is about quality, not quantity. 
* The submission can be a link to a notebook on Colab (please make sure you have appropriate sharing settings selected and test with someone else so we don't have to request access; let us know if you'd like our Colab account email addresses) or an uploaded .ipynb notebook.

2. A PDF of your clean notebook with all of the cells run and graphs appearing clearly. If you run into issues with this, try a different browser and experiment with the print function within Colab. If you figure something out, add it to [the shared, class-owned, editable document](https://docs.google.com/document/d/15zNjQp32oBqaD4CY4gAsgSnkAWUs3JbTJYNRAEr1H28/edit?usp=sharing).

There are rubrics in the appendices in this document. The intention of these rubrics is to create shared expectations about the project, while also giving you lots of latitude to explore and emphasize your own learning.

## Assessment

Assessment for this project will be based on the **quality of project work**, which will be assessed by the teaching team. 

During the week of October 7, we will have a project check-in. For students who chose assessment partially based on process (grading Option B), this will be an assessed check-in to create an accountability system to help you make progress on your project. Project check-ins will also be available to students who chose grading Option A. 

At the end of the project, the teaching will assess and provide feedback on your final report. We apologize if this takes us a while, as this is a large class and we're also redesigning many aspects of it given advances in machine learning.

We want to acknowledge that we are all coming to this course with varying experience with machine learning. This is great, and we’re actively trying to help you all have positive learning experiences. We intend for the course not to require prior experience with machine learning to participate, learn new things, and feasibly receive a high grade. With this in mind, we want to be explicit about how we anticipate this playing out in assessment.

With the project work and report, we will assess with a mental model of a team beginning the course with no real machine learning experience. The key aspects of this assessment are described in the rubric and project description. In practice, this means that everyone could receive an A for this project, and it will be more challenging for some people than others.


# Appendix A: Some questions to consider
[These questions built from from this webpage.](https://www.notion.so/ANN-Project-Framing-76e1b6af347f475a983487996ac9760d)

- What do you want your model to be able to do?
- How can you imagine your model (or an extension of it) being used in the real world? Feel free to get creative.
- If those ideas came true, who might they affect? In what ways?
- What measures would your model's real-world implementers need to take to ensure its effects live up to your intentions?
- What pitfalls might your model fall into, and what could you quantitatively measure to avoid those?
- Why was the dataset you used to train your model created?
- In what other ways could the same data be used? How do you feel about those possibilities?
- How was that dataset assembled? From where was the data sourced? Who or what labeled it? If there are any elements of this process you think were either particularly well done or problematic, how so?
- If your dataset contains information about people, to what degree did those people have agency over their inclusion? Do you feel that matters in this case (your application and/or for a college project)? Why or why not?
- Skimming through your dataset, does anything stand out to you about representation in its contents?
- Do you feel the potential use cases for this dataset justify it being created and published? Why or why not?

# Appendix B: Lessons from those that have come before you  

In a prior version of the class, we did a classification project using images with something called a convolutional neural network or CNN, which we'll learn about later (the images part of this course). Here we're doing a mini-project with simpler data, though you can use images if you want. At the end, we asked students to reflect on their project. We thought it might be helpful or entertaining to see [what they had to say](https://drive.google.com/file/d/1z5bXMRp2Np30TTa34uQh-GBnKykWI7rJ/view?usp=sharing).


# Appendix C: Rubrics


### Jupyter Notebook
Please submit both a PDF of your Jupyter Notebook (with the code executed) and an executable Jupyter Notebook (or link to Colab with appropriate sharing set and tested) to Canvas.

Your project will be graded on the following aspects.

**General**

1. The notebook is well-organized.
2. The notebook balances code and text well.
3. The notebook runs without errors in the Google CoLab environment.
4. The notebook is free of typos.
5. The notebook demonstrates good coding practices, including but not limited to docstrings, appropriate variable names, and comments as appropriate.
6. The code demonstrates a significant amount of effort (roughly 3 assignments' worth), made clear by careful consideration of the data, model, and implications for the model you developed.

**Motivation**

1. It is clear what data you used.
2. The application of your algorithm is clear and well-motivated.
3. The notebook explains how well the algorithm would need to work to provide value.
4. The notebook explains how the algorithm is to be evaluated (how do you know it is working well?).
5. The notebook explains the implications of this approach (including risks and stakeholders).

**Implementation**

1. The notebook explains the dataset and the features, in text and/or graphics.
2. The notebook contains a computer vision model.
3. The computer vision model is implemented correctly.
4. There is evidence that iteration has occurred in the model (this can be in the text).

**Conclusion**

1. The notebook accurately describes the effectiveness of the model in text.
2. The notebook describes the limitations of the model.
3. The code generates at least two visualizations of the data.
4. The visualizations have clear labels and connections to the text.

