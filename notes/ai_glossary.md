---
layout: page
title: Glossary of AI Terms (with a focus on LLMs)
---

What follows is my attempt to explain AI, through the medium of a glossary with a focus on LLMs (Large Language Models). These aren't wikipedia definitions, but my take on what these mean.
The purpose is to understand how LLMs can be used and the surrounding context.
I'll start with an AI hierarchy diagram and a slightly tongue-in-cheek definition of AI.

<img src="/assets/images/notes/ai_hierachy.png">

## AI

AI is a buzzword, a marketing term, a label to sell product and remain relevant in today's world (in 2024
at least). A rather cynical definition to start with but one I feel should be pointed out before digging
into a more technical definition. I say this, because I often get asked _**why every thing is "AI"**_ and
my first answer is, _**it's the new "cloud"**_. A lot of people didn't really understand what the cloud
meant and still don't. The same can be said of AI.

AI, isn't just a buzzword and there is meaning behind it. There are many places where you can look this up, so I'll just concentrate one particular AI technique and that is the [Transformer Architecture](https://towardsdatascience.com/transformers-89034557de14). This one technique has come to dominate in the last few years, mostly
due to its most successful and notorious implementation, **GPT**.

**ChatGPT**, uses **GPT**, which is a [Large Language Model(LLM)](#large-language-model-llm) 
 but is also a **Transformer Model**. This is a type of **deep learning**, which is a type of **machine learning**,
which is a type of **AI**.

So, some healthy scepticism. When someone claims, **_"my product is AI enabled"_** or **_"... uses AI"_**, I usually assume one of three things:
* They are bluffing and don't really know what AI means
* It uses an AI technique that won't have been called AI 2 years ago
* It uses an LLM such as GPT

This assumption can be wrong, but if the label **AI** was applied to something without further explanation,
it gives you a starting to point to ask what is meant by "AI", when presenting with a claim of its usage.


## Large Language Model (LLM)

LLMs are pre-trained neural nets (an artificial brain) that are expensive to run and very expensive to build. 
They contain a lot of data and are created from vast amounts of human-created text. They can be used to write poetry,
 write computer code, pretend to be a dead relative, etc. In short, the utility is endless and the result
looks disturbingly clever.

GPT4 is an example of an LLM. It is rumoured to have cost £100 million to build and is "trained" on 1.76 trillion parameters.
One example of the training data or the text that goes into GPT4 is called Common Crawl. Common Crawl is a collection of
data from 250 billion web pages, collected over the last 17 years.

Despite the huge cost to build and almost human-like capability, most LLMs can be accessed via a chatbot for free or at a low monthly fee (typically $20 or less per month).
All that endless power is democratically available to everyone and has been for the last 2 years. I feel like this is fundamentally why AI has been hyped up so much recently.

Relevant examples listed below, grouped by whether they are
proprietary or open source (hosted on [Hugging Face](#hugging-face)). Each model is generally a family of models, and in each instance I am referring to the latest or biggest unless specified otherwise

All the models listed below are examples of **foundation models**. Foundation models are generally fined-tuned (turned in into a **fine-tuned-model**) before they can be used for applications, like chatting or language translation. Without fine-tuning, models can be less useful and potentially dangerous.

### Proprietary

| Model  | Description                                                                                                        | Size (in parameters)                                    |
|--------|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| GPT    | Models from "Open" AI (yes, the quotes are deliberate). Forms the backed to ChatGPT and Copilot (tools that I use) | 1.76 trillion (GPT4)                                    |
| Gemini | Models from Google that power the Gemini assistant. Competes with GPT                                              | Unknown but rumoured to be 1.56 trillion (Gemini Ultra) |
| Claude | From Anthropic. Reputation for being safe and ethical.                                                             | Unknown but rumoured to be 2 trillion (Claude 3 Opus)   |

All three of these models are comparable and have various strengths and weaknesses. Different sources will claim that one of these are better than the other, however Claude appears
to beat the others in IQ tests and is often regarded as generally outperforming GPT and Gemini.

### Open Source

These models are generally smaller than proprietary models, so may be less capable but are still good
enough for many applications. Because they are smaller they don't require as big an investment to build
and therefore can be given away for free and may be easier to include in a commercial product

See [Hugging Face Models](https://huggingface.co/models) for a canonical listing.

| Model   | Description                                                                                                 | Size (in parameters)        |
|---------|-------------------------------------------------------------------------------------------------------------|-----------------------------|
| BLOOM   | From Hugging Face                                                                                           | 176 billion                 |
| Llama   | Models from Meta                                                                                            | 70 billion                  |
| Mistral | Models from Mistral                                                                                         | 141 billion (Mixtral 8x22B) |
| Gemma   | Smaller models from Google built in a similar way to Gemini. Notable for bwing small enough to run at home. | 7 Billion                   |


## Hugging Face

Hugging Face is a website, a bit like GitHub that hosts LLM models, tools to build LLM models and 
tools to make use of LLM models. Everything is free and community maintained.

## LLM apps

**ChatGPT** (_again with the ChatGPT_) is an example of an LLM app. It hosts a user interface to allow a human
to interact with a fine-tuned LLM model by allowing the human to ask questions of the LLM and responding with
relevant answers.

You can also interact with LLMs via an API. This allows you to build whatever you like on top of the LLM and
potentially build an **AI agent**. This is an application that performs intelligent tasks on your behalf such as looking up train times or the weather for today.

### Langchain

Langchain is a toolkit for creating LLM apps and AI agents. It allows you to chain together various components
such as an LLM with an API. This way you could ask the LLM "when is the next train to Winchester from London".
Langchain would get the LLM to turn this question into an API request. The request would be sent to the relevant
API endpoint (rail enquires API). The API would respond with some XML or JSON, which would be fed back into 
the LLM. The LLM would then turn this into a conversational output such as "the next train is at 10:00 on platform 9 (and three quarters)".

### Chroma DB

If you've ever noticed how slow ChatGPT takes to respond to a question, you may be thinking that it will
be a bottleneck for an app. That is where Chroma DB comes in. It stores something called "Vector Embeddings".

These are somewhat like a cache of LLM responses. So if I asked, "What are the price of rucksacks?" I could store the response in Chroma DB. Then if someone asked in the future or a similar sounding question like "What are the cost of rucksacks?", then it would respond more quickly and thus make it viable
for a performant app.