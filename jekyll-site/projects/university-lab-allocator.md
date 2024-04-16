---
layout: page
title: University Lab Allocator
---

![lab-allocator.png](/assets/images/projects/lab-allocator.png)

This is a service I worked on in 2023, which takes information about an academic year and will place students in labs within that year
 using a Genetic Algorithm to find the best solution.

It solves the problem of placing between 500 and 1000 students who choose a combination of roughly 10 subjects that have
limited lab/practical space and overlapping time slots.

Requirements gathering was complicated and involved many stakeholders. We also had to reverse engineer the behaviour
of a previous system and provide a two way interface to that system for the first year of operation.

A Genetic Algorithm (GA) is used to find the best solution to the problem of placing students in labs. The fitness
function looks at a number of factors including: keeping labs to capacity and minimising clashes.

The GA and the analysis of choices use bit arrays. This approach optimizes the use of machine resources and allows us 
to compute clashes using bitwise operations.


## The tech stack

![lab-allocator-2.png](/assets/images/projects/lab-allocator-2.png)

The system was written from scratch using Django Rest Framework, Postgres and React.

The GA and the web server are run individually on separate Google Cloud Run instances and are triggered by Google Cloud Functions
and REST calls. The GA is parallelized and uses the PyGAD library.

Interfacing with the legacy system was made possible using SQLite as an interface mechanism for dealing with MySQL dumps.

## Timetable analysis

Each lecture and practical series has its time slot converted into a timetable bit array. The way this works is that
for a two-week period each hour is represented by a bit. For instance 11-17 every Monday (both weeks) looks like this:

```
001111110000000000000000000000000000000000000000000000001111110000000000000000000000000000000000000000000000
```

So with the above you can see the two clumpings of 6 '1's, which represent the two practicals over the 2-week period.
Each 1 is 1 hour and the first set is near the left hand side, so it is a Monday morning (start of the two-week period).

Representing a timetable in this way makes it easy to spot clashes.

## Representing a student option as a GA Gene

Because there are less than 60 different practicals, you can represent the series a student could be taking as a 64-bit
integer (computationally handy for intense operations like a genetic algorithm). Each practical is assigned a number 
from 0 to 49. For instance, a student assigned to three different practicals would be represented as follows:

```
10000000000000000010000010000000000000000000000000  GeneValue for student assigned to the following:

10000000000000000000000000000000000000000000000000  GeneDescription for practical 1
00000000000000000010000000000000000000000000000000  GeneDescription for practical 2
00000000000000000000000010000000000000000000000000  GeneDescription for practical 3
```