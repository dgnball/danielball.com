---
layout: page
title: Integrating AI into Your Project
---

This is a practical guide for developers who want to integrate a large language model into an application. Before
reaching for an API key, it helps to understand what's actually happening under the hood. A lot of the common mistakes
come from treating LLMs like a black box when the underlying mechanics are actually quite straightforward.

<!-- prettier-ignore -->
- TOC
{:toc}

## TLDR and Resources

Don’t want to read this or don’t have the time. Here’s a list of resources you might find generally useful or
interesting...

**Docs**

- [OpenAI's guide to function calling](https://developers.openai.com/api/docs/guides/function-calling)

## Glossary

- **Autoregressive generation** — The process by which a model generates output one token at a time, with each new token
  conditioned on all previous tokens. This is why output is slower and more expensive than input processing.

- **Context window** — The maximum number of tokens a model can process in a single request. Includes the system prompt,
  message history, tool definitions and the model's own output. Measured in tokens.

- **Fine-tuning** — Additional training on a specific dataset after the initial pre-training phase. Used to specialise a
  general-purpose model for a particular task or domain without training from scratch.

- **Forward pass** — A single run of data through all the layers of a neural network from input to output. For
  generation, each output token requires its own forward pass. Input tokens are processed in one parallel forward pass.

- **Hallucination** — When a model generates content that is plausible-sounding but factually incorrect or entirely made
  up. A consequence of the model predicting likely tokens rather than retrieving verified facts.

- **Inference** — Running a trained model to generate a response. Distinct from training (where the model learns from
  data). Every API call is an inference request.

- **Input tokens** — The tokens in the request you send to the model. This includes the system prompt, conversation
  history and any tool definitions. Cheaper to process than output tokens because they are handled in one parallel pass.

- **Multimodal** — A model that can process more than one type of input. Most frontier models are now multimodal,
  accepting text alongside images, audio or documents.

- **Output tokens** — The tokens the model generates in its response. More expensive per token than input tokens because
  each one requires its own forward pass through the model.

- **RAG (Retrieval-Augmented Generation)** — A pattern where relevant documents or data are retrieved from an external
  source and inserted into the context before the API call. Avoids stuffing the full corpus into the context window and
  keeps responses grounded in specific sources.

- **System prompt** — Instructions passed to the model before the conversation begins. Sets the model's role, tone,
  constraints and any background context. Not visible to the end user but shapes every response.

- **Temperature** — A parameter that controls how predictable the model's output is. A temperature of 0 makes the model
  always pick the most probable next token. Higher values introduce more randomness. Useful for creative tasks but
  should be lowered when you need consistent or structured output.

- **Token** — The basic unit of text that an LLM processes. Not equivalent to a word. One token is roughly three to four
  characters or 0.75 words in English. Pricing and context limits are both measured in tokens.

- **Tokenisation** — The process of converting text into a sequence of integer tokens before it goes into the model. The
  same process in reverse (detokenisation) converts the model's output back into readable text.

- **Tool** — A function definition passed to the model as part of the request. When the model determines that calling
  the function would help it respond, it returns a structured tool call instead of plain text. The calling application
  executes the function and feeds the result back as a new message. See also [AI Agents](./ai_agents).

- **Tool call** — The structured response a model returns when it decides to invoke a tool rather than generate text
  directly. Contains the tool name and the arguments the model wants to pass to it.

## Anatomy of an LLM API call

Most LLM APIs share a common structure regardless of which provider you use. OpenAI, Anthropic and Google all have
slightly different SDKs but the underlying request looks similar. There are three main inputs.

**System prompt**

This is where you set the model's role, tone and constraints. It runs before the conversation starts and shapes every
response. It might be one sentence or several paragraphs. A chatbot might have a system prompt like "You are a helpful
customer support agent for Acme Corp. You only answer questions about our products." An agentic tool might have
thousands of tokens in its system prompt covering instructions, tool descriptions and project context.

**Messages**

This is the conversation history. Each message has a role (either `user` or `assistant`) and content. The model sees the
entire history on every turn. There is no memory between turns other than what's in this list. If you want the model to
remember something, it must be in the messages.

**Tools**

Tools are optional function definitions you pass to the model. When the model decides a tool is appropriate, it returns
a structured call rather than a plain text response. Your code executes the function and feeds the result back as a new
message. This is the mechanism behind agentic behaviour. See [AI Agents](./ai_agents) for more on this pattern.

A minimal API call in Python looks like this:

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.content[0].text)
```

## Multimodal inputs

Modern frontier models are not limited to text. Most now accept a mixture of input types in the same request.

**Images** can be passed alongside text. This is useful for tasks like analysing screenshots, reading diagrams or
describing photos. You include the image as a base64-encoded string or a URL alongside your text content.

**Audio** is handled differently depending on the provider. Some accept audio files directly. Others expect you to
transcribe first (using something like [Whisper](https://openai.com/index/whisper/)) and then pass the text to the LLM.

**Video** is still experimental at the frontier. Google's Gemini models have the most mature video support, accepting
clips directly in the request.

**Documents** such as PDFs are supported by some APIs. Others expect you to extract the text first and include it in the
message content.

The key mental model is that all of these inputs eventually become tokens. Images are converted to a representation the
model was trained on (vision transformers handle this). Audio is transcribed or embedded. Everything becomes part of the
same flat token sequence the model processes.

Here is an example of a multimodal call that sends an image URL alongside a text question:

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
                    },
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                }
            ],
        }
    ],
)

print(response.content[0].text)
```

The `content` field becomes a list instead of a plain string. Each item in the list is a block with a `type`. You can
mix as many image and text blocks as you need in a single message. For images stored locally, use `"type": "base64"` and
include the raw base64 data and media type instead of a URL.

## Tokens: the unit of everything

Tokens are how LLMs represent text. They are not words. A rough rule of thumb is that one token is approximately three
to four characters, or about 0.75 words in English. The word "integration" is one token. The word "integrations" might
be two.

Tokenisation matters for three reasons.

**Cost** - you are billed per token. More tokens in a request means a more expensive request.

**Context limits** - the model can only process a finite number of tokens at once. This is the context window.

**Behaviour** - models can behave differently near the edge of their context window. Long conversations can degrade in
quality as older context is crowded out.

You can experiment with how text tokenises using tools like [Tiktokenizer](https://tiktokenizer.vercel.app/) for OpenAI
models or the tokeniser tools on [Hugging Face](https://huggingface.co/).

## The context window

The context window is the total number of tokens the model can see at once. It includes the system prompt, the full
message history, any tool definitions and the model's own responses.

Modern frontier models have large windows. As of early 2026, Claude's context window is 200,000 tokens. Gemini 1.5 Pro
went to one million. For most applications this is more than enough, but it still has real implications for cost and
architecture.

**Cost** - the entire context is re-sent to the model on every turn of a conversation. A chat session with 50,000 tokens
of history costs more per turn than one with 5,000 tokens.

**Retrieval over stuffing** - for applications that need access to a large corpus of documents, you generally do not
want to cram everything into the context. Instead, retrieve only the relevant chunks at query time. This is the basis of
RAG (Retrieval-Augmented Generation).

**Long context is not free** - some providers charge more per token for very long contexts. Check the pricing page of
the provider you are using.

## What actually goes into the model

It is worth tracing the full path from a user typing a message to the moment the model generates a response. There are
more layers than people usually realise.

**1. The application layer**

Your application receives the user's message. It might do some pre-processing such as sanitising input, looking up user
preferences, fetching relevant documents from a database or applying rate limiting.

**2. Context assembly**

Before calling the API, your code assembles the full context. This includes the system prompt (which might be loaded
from a file or database), the conversation history (loaded from wherever you store it) and the new user message.

**3. The API call**

Your code calls the LLM provider's API with the assembled context. This goes over the network to the provider's
infrastructure.

**4. Tokenisation**

The provider tokenises everything. All those messages, the system prompt, tool definitions, the whole lot becomes a flat
sequence of integer tokens. From this point on, the model has no concept of "messages" or "users". It just sees a stream
of tokens.

**5. The forward pass**

The tokenised sequence goes through the model. The model produces a probability distribution over every possible next
token. The highest-probability token (or a sampled one, depending on the temperature setting) is selected and appended
to the sequence. This repeats until the model produces a stop token or hits the max tokens limit.

**6. Detokenisation and return**

The output tokens are converted back to text and returned to your application. Your application stores the assistant's
response in the conversation history and renders it to the user.

## Why output tokens cost more

This is one of the less obvious aspects of LLM pricing and it trips people up when estimating costs.

When you send a request, all the input tokens are processed in one go. The model runs a single forward pass over the
entire input sequence in parallel, which is relatively efficient.

Output tokens are different. Generation is autoregressive, which means the model generates one token at a time and each
new token depends on all the tokens that came before it. To generate the second token, the model must run another
forward pass that includes the first token. To generate the third, it must run another pass that includes the first two.
Each token requires its own forward pass through all the model's layers.

This is why generating a 1,000-token response is significantly more expensive and slower than processing a 1,000-token
input. Most providers charge roughly two to four times more per output token than per input token.

The practical implication is that you should be mindful of how much output you ask for. If you only need a yes/no
answer, say so. If you need a structured JSON response, tell the model to keep it concise. Verbose model outputs cost
more and take longer.

## Hosting choices

Where you run the model has significant implications for cost, privacy, latency and capability. This builds on the
hosting comparison in [Machine Learning Models](./machine-learning-models#how-to-access-and-run-complex-models), but
with a focus on the practical decisions you face when integrating into a real application.

### Commercial APIs

The simplest option. You call an endpoint, you get a response. The big three are OpenAI, Anthropic and Google. These
give you access to the most capable frontier models with no infrastructure to manage.

Pros:

- Highest capability models
- No infrastructure to manage
- Scales automatically
- Fast time to value

Cons:

- Your data is sent to a third party
- Costs scale with usage and can become expensive at volume
- You are dependent on their uptime and pricing decisions
- Data may be used for training unless you are on an enterprise tier (check the terms)

**Privacy note** - for most commercial tiers, assume your data could be logged. If you are handling sensitive or
regulated data (healthcare, legal, financial) you need to either use an enterprise tier with a data processing agreement
or choose a different hosting option.

### Open-weight models via a gateway

Services like [OpenRouter](https://openrouter.ai/) and [Together AI](https://www.together.ai/) host open-weight models
(LLaMA, Mistral, Qwen and others) and expose them through a standard API. This gives you access to capable models at
lower cost than the frontier providers.

Pros:

- Cheaper per token than frontier APIs
- Access to a wide range of models from one integration
- Some providers offer GDPR-compliant infrastructure

Cons:

- Still cloud-hosted, still requires trust in the provider
- Quality ceiling below frontier models for complex tasks

### Self-hosted on your own hardware

[Ollama](https://ollama.com/) and [LM Studio](https://lmstudio.ai/) make it straightforward to run open-weight models
locally. Small to mid-sized models (7B to 14B parameters) run reasonably well on a modern laptop with 16GB of RAM.
Larger models need a GPU.

Pros:

- Complete data privacy, nothing leaves your machine
- No per-token costs after the initial setup
- Useful for development and testing

Cons:

- Performance is limited by your hardware
- Smaller models are noticeably less capable than frontier models for complex reasoning
- You are responsible for updates and maintenance

### GPU cloud compute

Platforms like [RunPod](https://www.runpod.io/) and [Lambda Labs](https://lambdalabs.com/) let you rent a GPU server by
the hour. You can run large open-weight models, serve them via a local API and connect your application to them.

Pros:

- Run large models with full control over the infrastructure
- Better privacy than commercial APIs (especially with a reputable provider)
- Pay only for what you use

Cons:

- More infrastructure work to set up and maintain
- Costs can add up for sustained workloads

### Enterprise platforms

If you are already on AWS, GCP or Azure, each has its own managed LLM offering. AWS Bedrock, Google Vertex AI and Azure
OpenAI Service all let you run frontier and open-weight models inside your existing cloud environment with
enterprise-grade security and compliance controls.

Pros:

- Data stays within your existing cloud boundary
- Compliance frameworks already in place
- Integrates with your existing infrastructure and IAM

Cons:

- Can be more expensive than calling the provider's API directly
- More configuration to set up

### Choosing

The decision comes down to three things: how sensitive your data is, how much volume you expect and how much
infrastructure complexity you want to take on. For most applications starting out, a commercial API is the right choice.
As volume grows or data sensitivity increases, it is worth revisiting.

| Scenario                             | Recommended approach    |
| ------------------------------------ | ----------------------- |
| Prototype or low-volume app          | Commercial API          |
| Sensitive data, no cloud             | Self-hosted (Ollama)    |
| Sensitive data, cloud infrastructure | Enterprise platform     |
| Cost-sensitive at scale              | Open-weight via gateway |
| Research or large open-weight models | GPU cloud compute       |
