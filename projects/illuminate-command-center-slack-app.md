---
layout: page
title: Illuminate Command Center Slack App
---

![hanzo-command-center.jpeg](/assets/images/projects/hanzo-command-center.jpeg)

This was a solo greenfield project to create a Slack App (REST API) that mirrors the
functionality of the existing web app. This was full lifecycle development, which meant starting with broad requirements,
creating a prototype and iterating over this until the client was happy.

The project uses Python, Flask,
Google Cloud Run, Cloud Build, Datastore (NoSQL) and Secrets Manager.

The biggest
challenge was integrating the Slack App with the existing authentication system, that is based
on Auth0. Day to day, the work involves making best use of the [Slack Python SDK (Bolt)](https://api.slack.com/start/building/bolt-python) and
the existing REST API (designed to work with the React-based web app).

An example challenge is that the Slack API requires data to be truncated versions of what came out of the original system.
Because of these sort of issues, I came up with novel solutions to some of the problems posed by getting two different systems to work together.

A press release for this software can be found [here](https://hanzo.co/hanzo-launches-command-center-a-slack-application-designed-for-legal-productivity/).