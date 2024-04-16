---
layout: page
title: ONS Supplementary Data Service
---

![dalle-sds.png](/assets/images/projects/dalle-sds.png)

I started the development of the ONS Supplementary Data Service Office for National Statistics.
The service was part of a wider cloud-enabling strategy to make filling in business surveys easier.

As I was part of this project from a very early stage, it involved talking to stakeholders initially and then
designing an architecture using various Google cloud components for the job of indexing JSON data and JSON schemas.

The service used Cloud Firebase, FastAPI, Cloud Functions and Cloud Storage. The rationale for using these technologies
was because the main purpose was storing JSON. This meant it could either be stored in a bucket as a raw file
or stored in a semi-structured but searchable way in Firestore.

The source code is available [here](https://github.com/ONSdigital/sds).