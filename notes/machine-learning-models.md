---
layout: page
title: Machine Learning Models
---

<img src="/assets/images/notes/models.png" alt="models">

## Introduction

In this article, we’ll demystify the term “model” and explore the evolving landscape of machine learning tools. We’ll cover the difference between classical models and modern LLMs, show you the tradeoffs between running models locally, in the cloud, or via an API, and explain when you actually *need* an LLM, and when you're better off reaching for something simpler. If you’re a software developer, system architect, or tech-curious builder wondering how to integrate ML wisely, this is for you.

### Resources or TLDR

Don't want to read this or don't have the time. Here's a list of resources I reference in this document (or you might find generally useful):

**Communities**
* [Kaggle](https://www.kaggle.com/) - The Largest AI and ML community
* [Hugging Face](https://huggingface.co/) - The platform where the machine learning community collaborates on models, datasets and applications.

**Tools**
* [Open Router Models](https://openrouter.ai/models) - Access any hosted LLM model (from GPT to DeepSeek) from this one interface!
* [Github Models](https://docs.github.com/en/github-models) - Find and experiment with AI models for free
* [Jupyter](https://jupyter.org/) - ML engineers and data scientists will use this tool to write Python code that interfaces with ML models.
* [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/) and [JAX](https://docs.jax.dev/en/latest/) - Most complex models are built on PyTorch, TensorFlow or JAX. The backbone of modern AI!
* [scikit-learn](https://scikit-learn.org/) - Python library for building most types of ML model. Less focus on neural nets (like PyTorch), more on models such as Random Forest.

**Learning**
* [W3C Schools Machine Learning course](https://www.w3schools.com/python/python_ml_getting_started.asp) - Lots of getting-started style Python examples of Machine Learning
* [Google Machine Learning Education](https://developers.google.com/machine-learning) - Highly recommended ML courses from Google 
* [StatQuest YouTube channel](https://www.youtube.com/@statquest) - The best beginner-friendly ML YouTube channel
* [Wikipedia's Machine learning Models](https://en.wikipedia.org/wiki/Machine_learning#Models) - The "official" list of model types, as penned by academics and the data-science community.
* [Machine Learning for Kids](https://machinelearningforkids.co.uk/) - We've all got to start somewhere!
* There's also free ML courses from [Harvard](https://pll.harvard.edu/course/machine-learning-and-ai-python), [IBM](https://www.coursera.org/learn/machine-learning-with-python) and [FreeCodeCamp](https://www.freecodecamp.org/learn/machine-learning-with-python/)


## What is a Model?

In AI, a "model" refers to a mathematical construct trained to make decisions or predictions based on input data. The term comes from mathematics and statistics, where a model represents a simplified abstraction of a real-world process. In machine learning, this abstraction is learned automatically by processing massive datasets.

At its core, a model:

* Takes input (e.g. text, images, audio)
* Processes it using learned parameters (weights)
* Produces output (e.g. a summary, an answer, a classification)


## What Are the Different Types of ML Models?

Before AI became known for chatbots and image generators, data scientists used a variety of machine learning (ML) models to make predictions, detect patterns, or sort information. These models were usually small, focused, and trained on structured data, like spreadsheets or databases.

Here are the major types, explained simply:

### 1. **Linear Models** – *The Straight-Line Thinker*

<img src="/assets/images/notes/linear.png" alt="Linear Model" style="width: 300px;">

Draws a line or curve through data points to spot trends and make predictions. 

See [W3C schools Python explainer on Linear Regression](https://www.w3schools.com/python/python_ml_linear_regression.asp).

* **Used for**: Forecasting sales, predicting prices
* **Strength**: Simple, fast, easy to interpret
* **Weakness**: Can’t handle complex relationships

### 2. **Decision Trees** – *The Flowchart Brain*

<img src="/assets/images/notes/decision_tree.png" alt="Decision Tree" style="width: 300px;"> 

See [W3C schools Python explainer on Decisions Tree](https://www.w3schools.com/python/python_ml_decision_tree.asp).

Asks a series of yes/no questions to make a decision.

* **Used for**: Loan approval, medical diagnoses
* **Strength**: Easy to understand and explain
* **Weakness**: Can overfit, make decisions that don’t generalize

### 3. **Random Forests** – *The Crowd of Flowcharts*

<img src="/assets/images/notes/random_forest.png" alt="Random Forest" style="width: 300px;">

Builds many decision trees and combines their answers to improve accuracy.

* **Used for**: Risk scoring, product recommendations
* **Strength**: More accurate and robust
* **Weakness**: Harder to explain decisions

### 4. **Clustering Models** – *The Natural Group Finder*

<img src="/assets/images/notes/clustering.png" alt="Clustering Models" style="width: 300px;">

Groups similar things together without knowing the labels ahead of time.

* **Used for**: Customer segments, user behavior patterns
* **Strength**: Great for discovery
* **Weakness**: Can be sensitive to noise or unclear groups

### 5. **Naive Bayes** – *The Probability Calculator*

<img src="/assets/images/notes/bayes.png" alt="Bayes" style="width: 300px;">

Makes predictions based on how likely something is, given past data.

* **Used for**: Spam filters, topic classification
* **Strength**: Very fast
* **Weakness**: Can oversimplify complex problems

### 6. **Support Vector Machines (SVMs)** – *The Border Drawer*

<img src="/assets/images/notes/svm.png" alt="SVM" style="width: 300px;">

Draws the best dividing line between different categories in your data.

* **Used for**: Image classification, face detection
* **Strength**: Precise with clean data
* **Weakness**: Not great with lots of messy or overlapping data

### 7. **Neural Networks** – *The Brain-Inspired Pattern Learner*

<img src="/assets/images/notes/neural.png" alt="SVM" style="width: 300px;">

See [Real Python article on how to build a Neural network in Python](https://realpython.com/python-ai-neural-network/)

Mathematical models inspired by biological neural networks, consisting of interconnected nodes ("neurons") organized in layers that process and transform input data.


* **Used for**: Pattern recognition, classification, prediction
* **Strength**: Can learn complex relationships
* **Weakness**: Need careful tuning, can be unstable


### 8. **Deep Learning** – *The Advanced Pattern Master*

<img src="/assets/images/notes/deep.png" alt="Deep Learning" style="width: 200px;">

Deep learning refers to neural networks with many layers ("deep" architecture). These additional layers allow the network to learn increasingly complex features from data automatically. LLM models such as GPT and Gemini fall into this category.

* **Used for**: Computer vision, language models, speech recognition, game AI
* **Strength**: Learns complex patterns automatically, state-of-the-art performance
* **Weakness**: Needs massive data/compute, complex to train, black box behavior

Common types include CNNs (for images), Transformers (for language), RNNs (for sequences), and GNNs (for network data).



### Summary:

| Model Type                         | Example Use Case           | Can it Handle Complex Data? | Needs Lots of Data? | Easy to Understand? |
| ---------------------------------- | -------------------------- | --------------------------- | ------------------- | ------------------- |
| Linear Model                       | Predicting house prices    | No                          | No                  | Yes                 |
| Decision Tree                      | Loan approval              | Some                        | No                  | Yes                 |
| Random Forest                      | Fraud detection            | Yes                         | Medium              | Kind of             |
| Clustering                         | Market segmentation        | Some                        | Medium              | Sometimes           |
| Naive Bayes                        | Spam detection             | No                          | No                  | Yes                 |
| SVM                                | Face detection             | Yes                         | Medium              | No                  |
| Neural Network                     | Voice or image recognition | Yes                         | Yes                 | No                  |
| Deep Learning (Transformers, CNNs) | Language, vision, etc.     | Yes                         | Yes (lots)          | Very hard           |


## Choosing the Right Model for the Job

### If your data is structured (tables, numbers, categories):

**Use classical ML models** like:

* Logistic Regression
* Decision Trees / Random Forests
* XGBoost or LightGBM

**Examples:**

* Predicting churn from customer data
* Scoring leads in a CRM
* Classifying transactions as fraud or not

✅ Fast
✅ Explainable
✅ Can run locally or in the browser
❌ Not great for messy or unstructured input

> 💡 *If the data fits in a spreadsheet, you probably don’t need a neural net.*


### 📄 If your input is text, and the output is a simple label:

**Use smaller NLP models** (not full LLMs):

* BERT / RoBERTa
* DistilBERT
* fastText or Scikit-learn with TF-IDF

**Examples:**

* Categorizing support tickets
* Sentiment analysis
* Spam detection

✅ Lightweight and fast
✅ More accurate than old-school methods
❌ Doesn’t generate language, just classifies

> 💡 *You don’t need ChatGPT to decide if a tweet is angry or not.*


### 🖼 If you're working with images or video:

**Use vision models** like:

* ResNet / MobileNet / EfficientNet (for image classification)
* YOLO / Detectron2 (for object detection)
* CLIP / BLIP (for image + text tasks)

**Examples:**

* Flagging inappropriate images
* Reading license plates
* Matching screenshots to UI components

✅ Purpose-built and efficient
✅ Can run on phones or edge devices
❌ Needs labeled image data to train


### 🎙 If your input is audio or speech:

**Use audio models**:

* Whisper (speech-to-text)
* wav2vec2 (speech recognition)
* TTS models (text-to-speech)

**Examples:**

* Transcribing calls
* Voice assistants
* Reading text aloud

✅ Highly accurate models are available open-source
✅ Works well offline with the right setup
❌ Audio data can be large and tricky to process


### 💬 If you need language generation, summarization, or reasoning:

**Now you’re in LLM territory**:

* GPT-4 / Claude 3 / Gemini → commercial APIs
* LLaMA / Mistral / Phi-3 → open-source options
* Use tools like **OpenRouter**, **Ollama**, or **vLLM** for access

**Examples:**

* Summarizing a legal document
* Explaining code
* Writing email drafts or documentation
* Chatbots with memory and logic

✅ Extremely powerful
✅ Very general-purpose
❌ Can be expensive
❌ May hallucinate or go off-topic
❌ Overkill for small classification tasks

> 💡 *Use LLMs for jobs that involve language reasoning, not just token matching.*


### Decision Table: What Model Should I Use?

| Task Type                   | Recommended Model Type     | Example Tool              |
| --------------------------- | -------------------------- | ------------------------- |
| Predict from tabular data   | Decision Tree, XGBoost     | Scikit-learn, LightGBM    |
| Classify short texts        | DistilBERT, fastText       | Hugging Face Transformers |
| Summarize/generate text     | LLM (GPT, Claude, Mistral) | OpenRouter, Ollama        |
| Understand images           | CNNs, CLIP                 | YOLO, ResNet, BLIP        |
| Transcribe speech           | Whisper                    | Hugging Face, OpenAI      |
| Group similar users         | K-means Clustering         | Scikit-learn              |
| Detect sentiment in reviews | RoBERTa, BERT              | Transformers              |
| Write SEO blog posts        | LLMs                       | GPT-4, Claude 3           |


### Final Advice: Use the Smallest Model That Works

You wouldn’t call a rocket scientist to fix a leaky tap, and you shouldn’t call an LLM when:

* A few `if` statements would do
* A cheap model can do it faster
* You care about speed, cost, or explainability

But sometimes LLMs are great. If:

* The task involves nuance, ambiguity, or creativity
* You need a prototype **right now**
* It’s a small task and tokens are cheap

... then go ahead and use the LLM. Just know there’s a whole toolbox behind it, and sometimes a hammer really is better than a sledgehammer.

## Acquiring models from Hugging Face

Hugging Face started hosts a wide range of **machine learning models**, especially those built with deep learning frameworks like **PyTorch**, **TensorFlow**, and **JAX**.

All the models are free or open source, but you will need to provide the compute resource to run them on. Depending on the size of the model, this may be expensive.

### Here's what Hugging Face does and does not host:

| Model Type                                              | Hosted on Hugging Face? | Notes                                                      |
| ------------------------------------------------------- | ----------------------- | ---------------------------------------------------------- |
| **Transformers (LLMs)**                                 | ✅ Yes                   | Hugging Face’s core focus (e.g. GPT-style, BERT, LLaMA)    |
| **CNNs for vision**                                     | ✅ Yes                   | Models like ResNet, YOLO, and CLIP                         |
| **Audio models**                                        | ✅ Yes                   | Whisper, wav2vec2, TTS (text-to-speech)                    |
| **Multimodal models**                                   | ✅ Yes                   | e.g. Flamingo, BLIP (image + text)                         |
| **Small/efficient LMs (SLMs)**                          | ✅ Yes                   | e.g. DistilBERT, TinyLLaMA, Phi-3                          |
| **Embeddings / vector models**                          | ✅ Yes                   | Sentence Transformers, Instructor models                   |
| **Reinforcement learning models**                       | ✅ Yes                   | RLHF-trained agents, PPO configs                           |
| **Classical ML via `sklearn`**                          | ⚠️ Limited              | A few examples exist, mostly for educational/demo purposes |
| **XGBoost / LightGBM**                                  | ⚠️ Rare                 | Not commonly hosted, but can be wrapped into pipelines     |
| **Rule-based or statistical models (e.g. Naive Bayes)** | 🚫 Not really           | Usually too simple or not practical to share as models     |


## Building a Homemade Model?

Creating your own model from scratch is not just possible, it’s educational. Here's what you need:

* **Data**: Collect or curate a training dataset
* **Framework**: Use `PyTorch`, `TensorFlow`, or `JAX`
* **Compute**: A GPU (local or cloud)
* **Training**: Run an optimization loop (SGD, Adam, etc.)
* **Evaluation**: Monitor accuracy, loss, perplexity
* **Packaging**: Export via `ONNX`, `torchscript`, or serve with `FastAPI`, `vLLM`, or `TGI`

You could build:

* A sentiment classifier
* A chatbot on your internal docs
* An image recognizer for niche applications (e.g. insects!)

## How to access and run complex Models

If you want to run or access an LLM or other complex model acquired from Hugging Face or developed locally,
there are several options. Each has tradeoffs around *power, cost,* and *privacy*.

### 1. **OpenRouter**

OpenRouter is a gateway that routes your query to various models, OpenAI’s GPT, Anthropic’s Claude, Mistral, and more, via a unified API.

* ✅ Easy integration and flexible model access
* ✅ Fast and reliable
* ⚠️ Still cloud-hosted: less control over data privacy
* ⚠️ Costs per token or per request

### 2. **Hosting Locally on Your Laptop**

Running smaller models like **Phi-3-mini** or **Gemma 2B** on a laptop is increasingly feasible.

* ✅ Full privacy: data never leaves your machine
* ✅ Free after setup
* ⚠️ Limited power: can’t run massive models
* ⚠️ Requires technical setup (e.g. Ollama, LM Studio, llama.cpp)

**Good stack for this:**

* `Ollama`, `LM Studio`, `llama.cpp`, `Python`, `Rust`, or `Go` interfaces

### 3. **GPU-Enhanced Machine (eBay Special)**

You can buy a used GPU workstation (e.g. with an NVIDIA RTX 3090 or A6000) and run even mid-sized models locally.

* ✅ Excellent balance of performance and control
* ✅ Ideal for hobbyists and researchers
* ⚠️ Expensive upfront cost, noisy, and power-hungry
* ⚠️ Requires maintenance, Linux knowledge helpful

**Great for models like:**

* Mixtral, LLaMA 3 8B, stable diffusion models

### 4. **Hourly Cloud Compute**

Platforms like RunPod, Paperspace, and LambdaLabs let you spin up a GPU machine by the hour.

* ✅ On-demand power for training or inference
* ✅ No hardware investment
* ⚠️ Pay-as-you-go can become expensive
* ⚠️ Privacy risk depending on provider/data handling

**Used for:**

* Fine-tuning models
* Serving open-source models via APIs
* Experiments with reproducibility

### 5. **Commercial APIs**

The easiest route is to use models via APIs from the big players:

* OpenAI’s GPT-4o
* Anthropic’s Claude 3
* Google’s Gemini 1.5

* ✅ Fastest time to value
* ✅ Extremely powerful models
* ⚠️ Black-box: no insight into training or operation
* ⚠️ Data may be logged (unless on enterprise tiers)
* ⚠️ Pay-per-use, costs can scale fast

### 6. **Enterprise ML Platforms**
Platforms like Snowflake ML and Amazon SageMaker that provide integrated environments for model development, deployment and management at enterprise scale.
- ✅ End-to-end ML workflow management
- ✅ Built-in security and governance
- ✅ Seamless data integration
- ⚠️ Requires enterprise licensing/subscription
- ⚠️ Platform lock-in considerations

**Good for:**
- Large-scale ML deployments
- Teams needing governance and security
- Organizations with existing data warehouse investments
- Integrated data pipelines
- Compliance and governance controls
- Collaboration features
- Production-grade reliability

The key difference from the other categories is that these platforms provide a complete ecosystem rather than just model hosting or computation resources. They're especially valuable for organizations that already use these platforms for data warehousing and analytics.

## The LLM-ification of Everything (and Why It’s a Problem)

Large Language Models are incredibly capable, they can summarize, classify, generate, reason, and even write code. Given that power, it's no surprise that **many developers are now reaching for LLMs as the default tool for every ML problem**.

But just because you *can* use an LLM doesn’t mean you *should*.

### 🚀 Why Everyone’s Using LLMs for Everything:

* **Low barrier to entry**
  You don’t need to collect data, train anything, or understand ML theory. Just write a prompt and get results.

* **One tool for many tasks**
  You can classify sentiment, summarize articles, translate languages, and chat, all from the same API.

* **Faster prototyping**
  Especially for startups and small teams, LLMs let you get a working product *today*.

* **Wide availability**
  With tools like OpenAI, Claude, Gemini, and OpenRouter, LLMs are just an API key away.

### 🧱 But Here’s the Problem: It’s Becoming a Crutch

Relying on LLMs for everything creates several long-term issues:

#### 1. **Wasteful Overhead**

You're using a billion-parameter model to do what a 5MB model (or a few if-statements) could have done:

* Classifying tweets as positive or negative? A fine-tuned BERT or even `fastText` could do it faster and cheaper.
* Matching users to product categories? A logistic regression or decision tree might outperform your LLM at scale.

#### 2. **Scaling Costs**

An LLM call might cost fractions of a cent, but multiply that by millions of users or messages and you’re bleeding money.

* Traditional models are nearly free to run once deployed.
* LLMs charge you every token, every call, every minute.

#### 3. **Latency**

Even the fastest LLMs are slower than traditional models.

* A call to a hosted LLM takes 200ms–1s+.
* A local scikit-learn model returns results in milliseconds.

#### 4. **Loss of Specialization**

LLMs are generalists. That makes them useful, but also less sharp at domain-specific tasks than smaller, fine-tuned models.

* A fine-tuned fraud detection model trained on your data will almost always beat an LLM trying to "reason" its way to a result.

#### 5. **Skills Atrophy**

When LLMs become a catch-all, developers stop learning about classical ML, statistics, feature engineering, or model evaluation. That’s dangerous in regulated, high-stakes, or performance-sensitive environments.

### 🧩 Why This Happens Anyway: Developer Psychology

* **LLMs feel like magic.** It’s easy to get hooked on the dopamine hit of seeing a prompt “just work.”
* **Machine learning feels hard.** Writing your own model or pipeline can seem intimidating, even when the task is simple.
* **APIs are safe.** You don’t have to manage GPUs, train models, or even understand the data, just call the function.

So teams default to GPT for everything from customer support classification to bug triage to basic spreadsheet analysis, often without questioning if it’s the best tool for the job.

### ✅ When It’s Fine to Use LLMs for Traditional Tasks

To be clear, this isn’t a blanket indictment. Sometimes, using an LLM **is totally appropriate**, even if it *could* be done with classical ML.

Examples:

* You’re in a rush and need something working *now*
* You don’t have labeled data
* The job is small, infrequent, or low-volume
* You want human-like flexibility (e.g. parsing vague or inconsistent text)

> **Good enough is good enough, when cost, latency, and control don’t matter.**

### 🔄 Consider a Hybrid Approach

Use LLMs for what they’re great at, *language understanding, generation, and reasoning*. Use traditional models when you want:

* Speed
* Predictable output
* Privacy
* Simplicity
* Cost-efficiency

A good architecture might look like:

1. Use LLMs at the edge, to route or clean messy data
2. Pass that to a lightweight classifier or ranking model
3. Return a response that’s fast, traceable, and explainable