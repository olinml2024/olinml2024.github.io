---
title: Day 6
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

{% capture agenda %}
* 10:20-10:25am: Everyone come hang out in MAC128, we'll talk about the plan and answer any logistics questions.
* 10:25-10:35am: Debrief at tables
* 10:35-10:40am: Going Over Simplification in Logistic Regression Learning Rule
* 10:40-10:50am: Preview of where we are going
* 10:50-12:00pm: Foundations of Micrograd
{% endcapture %}

{% include agenda.html content=agenda %}

# Preview of where we are going

We'll go over the upcoming gate on model evaluation.  We'll also talk about the COMPAS algorithm and the readings we will be doing / the discussions we will be having.

# Micrograd????

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