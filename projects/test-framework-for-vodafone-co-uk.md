---
layout: page
title: Test Framework for Vodafone.co.uk
---

![vodafone-shop.png](/assets/images/projects/vodafone-shop.png)

Creating automated test pipelines and working within a team to develop a performance
testing framework. The framework runs on AWS and uses Locust (Python) and Wiremock
(Java). A lot of this work involved integrating Azure Devops pipelines (CI/CD) with various
AWS services using CloudFormation and other AWS services such as Lambda, ECS and
DynamoDB. Datadog was used to look back at historical response times and gauge endpoint
popularity. These were used as inputs for performance testing individual components.