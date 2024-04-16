---
layout: page
title: Strategy Research Organiser
---

![dalle-green-tick.png](/assets/images/projects/dalle-green-tick.png)

I was approached by an organisation, whose business ran on decades old bespoke desktop software. The client was having trouble
because the developer of this software had since left the company 5 years previously, the software only ran on old
versions of Mac OS X and had bugs that made it difficult to use, slow and prone to crashing.

My job was to make it cross-platform friendly and fix the worst of the bugs...

The software was a mix of Java desktop software and Unix scripts that only ran on Mac OS X.

I started by building test fixtures for the Unix scripts using a set of inputs and expected outputs supplied by the client.
I then wrote some Python scripts that were able to re-create the expected outputs. This allowed the client
to dispense with the Unix scripts which were unmaintainable and more complex than they needed to be.

Next was the Java Desktop app which came with some source code but when compiled didn't match the capability of the latest
version. Because the source code was essentially missing, it was necessary to decompile the class files to get this.

Having decompiled the Java source code, I went through a process of getting the source to the new Python script and 
Java code into separate GitHub projects. I added a GitHub project, so I could track issues with the client along
with a CI process. This allowed me to work methodically and fix each bug in turn.

Eventually we got to the point where the client was able to run their scripts and the Java desktop app on the latest
versions of Mac OS X and Windows and that the software was usable again.

As well as fixing the indicated problems, they were left with source code control, a project management structure,
automated regression tests and repeatable release process for the future maintenance of the software.

A review of this work can be found [here](https://www.guru.com/freelancers/westsmith/reviews).