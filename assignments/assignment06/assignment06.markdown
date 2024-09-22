---
title: Assignment 6
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture content %}
# Everyone should submit all parts of this assignment.
This assignment contains two major components, each of which has parts that you must submit (regardless of which grading component you chose).

**Part 1: Model evaluation and basic concepts and terms in ML**
This is the first of the "Quality Assessed Assignments" (see syllabus). It will be assessed for correctness. 

#TODO: NEED A SOLUTIONS / CHATGPT / WORK ALONE POLICY EXPLICITLY STATED HERE.

**Part 2: Preparation for COMPAS recidivism discussion**
This falls under the "Context & Ethics / Discussion Prep". This will include questions about the assigned readings. The topic we are venturing into is very complex. We ask that you spend real time reading and considering these topics. It is not appropriate to just glance at this and ask ChatGPT for a summary. You are welcome to work with others to read and try to make sense of this together.

{% endcapture %}
{% include notice.html content=content %}


# Learning Objectives

{% capture content %}
* 

{% endcapture %}
{% include learning_objectives.html content=content %}



# Data flow diagrams and the chain rule revisited

It turns out that we can modify the data flow diagram for computing the output of the function to instead compute this gradient automatically.  Here is what the process looks like (let's use $grad_v$ to store the result of $\frac{\partial f}{v}$).

```mermaid!
flowchart TB
 id1["$$grad_f = 1 ~~~~$$"]
 id2["$$grad_x = \frac{\partial f}{\partial x} grad_f~~$$"]
 id3["$$grad_y = \frac{\partial f}{\partial y} grad_f~~$$"]
 id4["$$grad_t = \frac{\partial x}{\partial t} grad_x + \frac{\partial y}{\partial t} grad_y~~~~~~$$"]
 id1 --"$$\frac{\partial f}{\partial x} grad_f~~$$"--> id2
 id1 --"$$\frac{\partial f}{\partial y} grad_f~~$$"--> id3
 id2 --"$$\frac{\partial x}{\partial t} grad_x ~~$$"--> id4
 id3 --"$$\frac{\partial y}{\partial t} grad_y ~~$$"--> id4
```

# Monday class
* Go over simplification of inserting y=1 and y=0
* More on logistic regression?? Could do autograd?
* Conditional probabilities?? (don't actually need this)
* Fairness metrics??


# Potential parts of this assignment
* Disproportionate policing preamble (context & ethics)
* ProPublica COMPAS reading (context & ethics)
* Calculation and graphs of recidivism data (https://predictivemodellingearly.github.io/handout/index.html) (metric assignment)
* Analysis of another situation - how to assess (metric assignment)

# Thursday class
* Compas discussion?


# Actually start assignment here

# A refresher on key metrics and a primer on subgroup effects.

Read through [this website from Google](https://research.google.com/bigpicture/attacking-discrimination-in-ml/}) and play with the visualization. This should help refresh your memory on terms like false positive rate with some great visualizations. It also demonstrates what can happen when you have two subgroups with different rates.


# COMPAS Model: Race, Criminal Justice, and Machine Learning

{% capture content %}
We are going to think about race and criminal justice in the United States. Before we dive into this, we want to acknowledge:


* This is a complex and intricate issue that involves policy, society, technology, individual beliefs/values, and history. This topic directly (but not equally) impacts the lives of many people.
* We all have our own lenses through which we view the world.
* This topic will likely be uncomfortable to grapple with regardless of your background and identity. It may resonate differently with each of us. We (Sam/Paul) are available in person and via email, if you would like to discuss how we can best support you in class. We are planning to have some group discussion in class. One method of support could be pairing you with a partner of your choosing for this discussion. Another could be including your ideas about how class discussion can be informative and challenging without creating unnecessary pain. Please reach out to us if you have any concerns or want to discuss this more.
* In this class, we will scratch the surface of the way the US justice system works. Your instructors are not criminal justice experts, but they do care about this topic. We are also continuing to learn more.
* We believe in the importance of grappling with difficult topics. While we are in fields like engineering, math, and computing, these are not separate from the complexities of humanity.
{% endcapture %}
{% include notice.html content=content %}

## A few basics about the US criminal justice system. 

A police officer can place a person under arrest. However, an arrest does not necessarily mean that person committed a crime (both in fact and in a legal sense). Legally, someone is considered innocent until proven guilty in court. However, arrested people are often held in jail for months before trail (this is called pretrial detention). To get out of jail before trial, the arrested person can post bail. Bail is a considerable amount of money ("money bond") that is given to the court to ensure the person shows up to trial. As you might guess, bail represents a way people with money are treated differently by the system than people without money.  (Optional: For more on bail, listen to [Episode 62 of the podcast Ear Hustle](https://www.earhustlesq.com/episodes/2021/9/22/do-it-movin) (the podcast also has other stories from incarcerated people). 

Legally, a person is considered guilty if they are convicted in court. Practically, innocent people are sometimes convicted. People with a lot of money can hire many lawyers who will work many days, weeks, or years, fighting the case. People without funds for a lawyer will be assigned a public defender. Public defenders are often overwhelmed, and might have just a minute to look over the details of a case, right before trail.

[//]: <> [(20 minutes)] 

Read the [Report of The Sentencing Project to the United Nations Special Rapporteur on Contemporary Forms of Racism, Racial Discrimination, Xenophobia, and Related Intolerance](https://www.sentencingproject.org/publications/un-report-on-racial-disparities/). This article is intended to provide some background information on criminal justice and race in the US. 



We'll be spending time talking about the Correctional Offender Management Profiling for Alternative Sanctions (COMPAS) algorithm, produced by the company Northpointe, Inc. COMPAS was intended to assess the risk of recidivism. This is a well-known algorithm in machine learning communities. 

Below we will provide you a list of readings on this topic. As you read, please prepare to reflect on the following themes:
* Social justice in a non-ideal world.
* The roles of machine learning engineers and the roles of other professional roles in the application of machine learning in our society.
* The major things that you would want to consider in this type of undertaking.
* Framing things mathematically versus from a social justice standpoint.
* The level of technical debate in model choices that is brought up in this discussion (especially in the technical responses). 



First, read the [ProPublica article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing).

Next, please read the technical details of the ProPublica analysis.

We would like to be clear about when the COMPAS metric is applied. It is applied after someone is arrested, and the prediction COMPAS gives is if that person will be arrested again. A person who was arrested twice could be legally and actually innocent. It is important to be clear in our language that these are arrested people, and not convicted criminals. The term recidivism, which is generally defined as a criminal who commits a second crime, is used in the readings. ProPublica actually redefines recidivism as ``a criminal offense that resulted in a jail booking and took place after the crime for which the person was COMPAS scored.'' Here, ProPublica conflates an arrest for a crime with a conviction for that crime. Note that a jail booking is pre-trial, and different than prison, which is post-trial and conviction. This language is also used by the Northpointe rebuttal. Carrie would argue that since these are not people who have been yet convicted of crimes, they are not necessarily recidivists, and the language in the readings should be corrected.

Please read [How We Analyzed the COMPAS Recidivism Algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm).


Optional, read the [Northpointe rebuttal](https://drive.google.com/file/d/1SSbotzlsVkj4L2VhPC7R_XH0VBqOQsnT/view?usp=sharing). This is a long reading. It has a lot of jargon, and some acronyms are not defined. We strongly suggest limiting yourself to 60 minutes for this reading (perhaps read the conclusion early on). You may consider working with a classmate on this reading, so you can both discuss what you think the author is saying. We're including this whole reading here because we would like you to engage with the real-world material.


[//]: <> (30 minutes)


\item Please summarize what you see as the key parts of the ProPublica and Northpointe cases. You can use words, diagrams, concept maps, or another method that works for you.
\item Reflect on what you've just read.  We think the themes brought up above will provide good fodder for your response, but please feel free to take it in any direction.  Aim for around two paragraphs in your response.
\ees
\end{exercise}





# Quality Assessed Assignment (QAA): Model evaluation and basic concepts and terms in ML

## QAA Part 1: Develop a testing and evaluation plan

You and your team have developed a system to help mushroom foragers identify various species of mushrooms.  What is your testing and evaluation plan for ensuring the system is ready for release? Your testing plan should include specific metrics, plans for data collection, and how would use this data to validate your model.














# Acknowledgements 
Special thanks to Claire S. Lee, Jeremy Du, and Michael Guerzhoy for sharing their version of this assignment. Also thanks to Micah Reid (Olin alum), and Miranda Lao (Olin alum) for their contributions to this assignment as part of a previous ML final project. Also thanks to Carrie Nugent for some of the framing work above.





