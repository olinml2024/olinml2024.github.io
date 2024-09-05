---
title: Day 1
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 1-1:02pm: Everyone come hang out in MAC128.
* 1:02-1:10pm: We'll provide brief orienting remarks about the course!
* 1:10-1:15pm: Introduction to the our main activity (see below).
* 1:15-2:05pm: Mapping the Machine Learning Ecosystem
* 2:05-2:20pm: Report out
* 2:20-2:40pm: Orientation to first assignment and basic course logistics for assignment submissions.  We'll show you the Canvas page, grading options, how to find office hours, etc.
{% endcapture %}

{% include agenda.html content=agenda %}

# The Big Picture

Welcome to Machine Learning!  We're not going to spend a ton of time talking at you today (we want to get you engaging with the material as quickly as possible).  A few quick things.

## What is Machine Learning?

As a running example, let's take the idea of creating a computer program to predict various characteristics of a person (e.g., age and gender) from a picture of their face.

<p style="text-align: center;">
<img alt="A selfie of a professor that has been processed by SeeingAI.  The output of the program says that the image contains a picture of a 43-year old man looking neutral" src="images/paulseeingai.jpg"/>
</p>

One way to frame machine learning is by contrasting it with the traditional approach to writing an algorithm to solve a problem.  Here is a somewhat cartoonish version.  

```mermaid!
flowchart LR
  id1[Data]
  id2[Hand-coded Program]
  id3[Output]
  id1 --> id2
  id2 --> id3
```

This might seem like a seemingly impossible task, but it's one that the machine learning approach can be applied to quite easily.  Here is the workflow when adopting a machine learning approach.

```mermaid!
flowchart LR
  id1[Data]
  id2[Machine Learning Algorithm]
  id3[Program]
  id4[Desired Outputs]
  id1 --> id2
  id2 --> id3
  id4 --> id2
```

This picture helps us undertand the potential scope of the machine learning approach.  Is machine learning just what happens in the middle box?  What about the inputs and outputs?  We probably have a lot of questions about those.  Let's take a minute to throw out a few considerations.


## Learning Goals

Machine learning is a vast field that touches upon many disciplines.  In this class we aim to take a broad view towards the subject that covers the underlying theory, implementation, and critically evaluating how machine learning systems impact the world and its people.

* Understand a variety of machine learning techniques from both a mathematical and algorithmic perspective.
* Successfully implement machine learning algorithms in Python (both by using only minimal external libraries and by leveraging standard machine learning libraries).
* Execute the iterative machine learning workflow of model design, fitting to training data, testing, and interpretation in order to be able to successfully apply machine learning techniques in specific contexts.
* Predict the potential impacts of a machine learning system when deployed in a real-world context and make design decisions to mitigate potential harmful impacts while maximizing positive impacts.

# Mapping the machine learning ecosystem

A few years back when we were originally designing this course, we were struck by this incredible visualization of a machine learning-powered system (the Amazon Echo).  (note: click on the following link to see [the original, high-resolution, vector graphics version](https://anatomyof.ai/img/ai-anatomy-map.pdf)).

![](images/ai-anatomy-map.png)

We don't necessarily recommend diving into the nitty gritty here, but we do want to point out some of the interesting features of this map.

1. This map looks at the lifecycle of the system.  This includes development, manufacturing, usage, and disposal.
2. This map examines the diverse (along many dimensions) group of people that interacts with the Amazon Echo.
3. This map shows relationships between different organizations (e.g., transportation companies and distributors).
4. This map explores a variety of inputs and outputs to the product (e.g., data, energy, raw materials, human knowledge).

One of the hallmarks of this course will be in contextualizing machine learning systems within larger systems (e.g., economic, social, environmental) so we can better understand the likely impacts of machine learning technology and how we, as engineers, can increase positive outcomes while reducing negative ones.  We'll have dedicated class activities and readings to help make the picture clearer of how machine learning fits into various contexts, but today we want you to dive into the deep end of the pool and do some research to map out a machine learning system of your choosing.

## Step 1: Do Some Background Reading

Read the article [Machine Learning Lifecycle Explained](https://www.datacamp.com/blog/machine-learning-lifecycle-explained).  This article will give you a nice high-level view of what it takes to create a machine learning model.  The article doesn't look at all of the possible dimensions you might consider, but it does give some good jumping off points (e.g., the article doesn't talk about electricity or environmental impacts).

## Step 2: Choose a Machine Learning System to Map

You probably have a few in mind that you are interested in thinking about. For the purposes of this exercise you should probably choose examples that you are already familiar with (or that you can quickly lookup key information).

Here are some ideas off the top of our heads:
* Large language models (e.g., ChatGPT, Claude, etc.)
* Google
* Generative image models (e.g., Midjourney)
* The [SeeingAI app](https://apps.apple.com/us/app/seeing-ai/id999062298) (Microsoft's app to make various tasks more accessible for folks who are blind)
* Job applicant screening tools
* Self-driving cars
* Facial recognition software
* Fitness, health, and safety features in Apple Watches (e.g., fall detection, health monitoring, etc.)

## Step 3: Make your Map

On a whiteboard, draw a system map of the various stakeholders, stakeholder interactions, inputs into the system (e.g., energy, cost, knowledge), outputs (impacts of the system), potential pitfalls, and opportunities.  As you go, make a list of the key questions that you would like to answer to better understand how your chosen system (if you have time, you may throw these into a search engine or an LLM to see if you can get some quick ideas as to an answer).

Here are some prompts to consider to help get you thinking:
* Who is involved?  Consider data creators, research scientists, data labelers, machine learning engineers, user-experience specialists, consumers, legislators / regulators, etc.
* Who are the organizations? Of the folks involved, what organizations are they a part of (these could be governmental, commercial, or non-profit).
* Where are the interaction points?  How do individual or organizations interact with each other?  What is exchanged between them (e.g., knowledge, money, data, computing resources)?
* With respect to the benefits of the system, consider the experience of the end user (what do they get from using the system?), consider the knock-on benefits to the system creator (e.g., providing more training data), and does anyone else benefit?  With respect to potential pitfalls (or negative impacts), consider issues of environmental and financial impact along with the potential for model bias that causes differential impacts to different groups of people that interact with the system (e.g., based on some identity characteristic like race or gender).

## Step 4: Share Your Map

Make sure you take a high-resolution picture of your map and add it to [this shared Slide deck](https://docs.google.com/presentation/d/1nzF3k-ps9xWlojtOrpTjxBW6dR3eAT3gPl61rK9_QF0/edit?usp=sharing).  We'll give each team a chance to discuss an interesting feature of their map (there won't be time to present the map in its entirety, so you'll have to choose 1-2 things to share).