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
interesting.

**Provider docs**

- [Build with Claude](https://www.anthropic.com/learn/build-with-claude) — Anthropic’s starting point for developers
- [Anthropic prompt engineering guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
  — Anthropic’s official prompting reference
- [OpenAI API reference](https://platform.openai.com/docs/api-reference/introduction) — OpenAI’s API docs
- [OpenAI’s guide to function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI Cookbook](https://cookbook.openai.com/) — practical recipes and worked examples from OpenAI
- [Gemini API docs](https://ai.google.dev/gemini-api/docs) — Google’s developer documentation for Gemini

**Playgrounds and studios**

- [Anthropic Console](https://console.anthropic.com/) — iterate on prompts and export code for Claude
- [OpenAI Playground](https://platform.openai.com/playground) — interactive prompt testing for GPT models
- [Google AI Studio](https://aistudio.google.com/) — browser-based environment for Gemini
- [OpenRouter](https://openrouter.ai/) — single API to access and compare models from many providers
- [Tiktokenizer](https://tiktokenizer.vercel.app/) — see how text tokenises before it hits the model

**Learning**

- [Anthropic interactive prompt engineering tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) —
  hands-on Jupyter notebooks for prompting Claude
- [AI Python for Beginners](https://www.deeplearning.ai/short-courses/ai-python-for-beginners/) — DeepLearning.AI short
  course
- [Real Python AI tutorials](https://realpython.com/tutorials/ai/) — practical AI tutorials for Python developers

**Evaluation and observability**

- [LangSmith](https://www.langchain.com/langsmith) — tracing, evaluation and prompt versioning from LangChain
- [Langfuse](https://langfuse.com/) — open-source observability and evaluation platform
- [Promptfoo](https://www.promptfoo.dev/) — open-source CLI for prompt testing and red teaming
- [Braintrust](https://www.braintrust.dev/) — eval-driven development platform with CI integration

**Visual builders**

- [Flowise](https://flowiseai.com/) — open-source drag-and-drop canvas for LLM pipelines
- [Dify](https://dify.ai/) — open-source RAG application builder
- [LangFlow](https://www.langflow.org/) — visual canvas for agents and retrieval pipelines
- [n8n](https://n8n.io/) — automation tool with strong AI workflow support and self-hosting option

**Self-hosting**

- [Ollama](https://ollama.com/) — run open-weight models locally with minimal setup
- [LM Studio](https://lmstudio.ai/) — desktop app for running and chatting with local models
- [Hugging Face](https://huggingface.co/) — model hub, datasets and hosted inference

**Community**

- [r/RAG](https://www.reddit.com/r/Rag/) — subreddit for retrieval-augmented generation discussion
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — self-hosting, open-weight models and hardware

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

## Prompt engineering

Prompt engineering is the practice of structuring your inputs to get better outputs from a model. It is the first thing
to reach for before considering RAG or fine-tuning. Frontier models are capable of surprising things if you ask clearly,
and a lot of apparently bad model behaviour turns out to be bad prompting.

### The system prompt is your primary lever

The system prompt runs before anything else and shapes every response in the conversation. Most of the heavy lifting
belongs here. A well-designed system prompt covers the model's role, the task it is performing, any constraints, and the
format you want back.

Vague system prompts produce vague behaviour. Compare these two:

- "You are a helpful assistant."
- "You are a support agent for a B2B SaaS product. Answer only questions about the product. If a question is outside
  scope, apologise and redirect to the sales team. Always respond in plain English. Keep answers under 150 words."

The second leaves much less room for the model to improvise.

### Be explicit about output format

Models will produce whatever format seems most natural unless you tell them otherwise. If you need a specific structure,
describe it precisely. Better still, use your provider's structured output feature, which constrains the response to a
JSON schema and validates it before it reaches your application.

If you cannot use structured outputs, show the model an example of the exact format you want. The pattern "Respond in
this format: ..." followed by a concrete example is one of the most reliable techniques in prompting.

### Few-shot examples

Few-shot prompting gives the model examples of input/output pairs to imitate. It is particularly effective for tasks
where the desired behaviour is easier to demonstrate than to describe, such as classification, extraction or a specific
writing style.

One well-chosen example often beats several paragraphs of description. The examples do not need to cover every case.
They just need to show the model the pattern.

```
Classify the following customer message as either "billing", "technical" or "other".

Message: "I was charged twice last month."
Category: billing

Message: "The dashboard won't load on my laptop."
Category: technical

Message: "{{ customer_message }}"
Category:
```

### Chain-of-thought

By default, models jump to an answer. For complex tasks this means they sometimes get it wrong in ways that would have
been caught by reasoning through the steps. Asking the model to think through the problem first before giving a final
answer reliably improves accuracy on tasks involving logic, maths or multi-step reasoning.

The simplest version is adding "Think through this step by step before giving your answer." A more structured version
gives the model a specific sequence of steps to follow.

For tasks where you want the reasoning hidden from the user, you can instruct the model to produce its reasoning in a
separate field or XML tag and only surface the conclusion.

### XML tags for structure

When a prompt contains multiple distinct sections — instructions, retrieved context, conversation history, the current
question — mixing them in plain prose makes it harder for the model to separate them. Wrapping sections in XML tags
gives clear boundaries.

```
<instructions>
You are a legal research assistant. Summarise the key points of the provided document.
</instructions>

<document>
{{ document_text }}
</document>

Produce a structured summary with headings.
```

This technique is particularly recommended by Anthropic for Claude but it works well across providers. It is especially
useful as prompts grow longer and more complex.

### Temperature and other parameters

Temperature controls how deterministic the output is. At zero the model picks the highest-probability token every time,
producing consistent and predictable output. Higher values introduce randomness, which is useful for creative tasks but
harmful for tasks needing reliable structured output or factual accuracy.

For classification, extraction, code generation or anything you need to parse programmatically, set temperature to zero
or close to it. For creative writing or brainstorming, raise it.

`max_tokens` sets a ceiling on response length. Setting it too low will cause the model to truncate mid-sentence. Set it
comfortably above your expected output length.

### Iterating on prompts

Treat prompts like code. Write them down, version them and test them against a set of representative inputs. One of the
most common mistakes is tuning a prompt on a single example until it works and then being surprised when it fails on
others.

Build a small evaluation set of 20 to 50 inputs with expected outputs. When you change a prompt, run it against the set.
Most providers have a playground for interactive iteration; most production systems should have something more
systematic.

### When prompting is not enough

Prompting has real limits. It cannot give the model knowledge it does not have. It cannot reliably teach it new facts.
It cannot fundamentally change its capabilities. If you are hitting these limits, the next options are RAG for external
knowledge and fine-tuning for deep behavioural change.

## How the API got its shape

The interface you are looking at today has a specific history, and understanding it helps explain some of its quirks.

### The original Completions API

When OpenAI opened GPT-3 to developers in 2020, the API was much simpler. You sent a prompt — a block of text — and the
model completed it. There were no roles, no message lists. You might write "Translate the following English text to
French: Hello, how are you?" and the model would produce the continuation. It was essentially a very sophisticated
autocomplete.

This worked, but it put a lot of burden on the developer to engineer prompts carefully. Role-play prompting ("You are a
helpful assistant. User: ... Assistant: ...") was a common hack to get conversational behaviour out of a completion
model.

### Chat Completions and the chatbot era

In March 2023, OpenAI introduced the Chat Completions API alongside GPT-3.5. Instead of a freeform prompt, you now sent
a structured list of messages with roles. This was a direct response to the success of ChatGPT, which had launched a few
months earlier. The new format formalised what developers had already been doing manually.

The `system`, `user` and `assistant` roles are the fingerprints of that era. They are chatbot concepts. Even when you
are using an LLM to process a CSV, analyse code or extract structured data, the API still asks you to frame your request
as a user message and receive an assistant response. The abstraction leaks through.

The Chat Completions format took over quickly. Within months it accounted for the vast majority of OpenAI's API usage,
and the older Completions endpoints were deprecated in early 2024.

### Industry convergence

Because OpenAI was first and had the largest developer mindshare, everyone else followed their lead. Anthropic, Google,
Mistral and most others adopted similar conventions. Messages arrays, role fields, system prompts — these became
industry idioms even though no standards body defined them.

The similarity goes deep enough that gateway services like [OpenRouter](https://openrouter.ai/) can proxy requests
across multiple providers using a single OpenAI-compatible interface. You can often swap one provider's SDK for another
with minimal code changes.

Where providers diverged is mostly at the edges. Anthropic puts `system` as a top-level field in the request rather than
a message with a `system` role, which is slightly cleaner. Google's Gemini uses `contents` instead of `messages` and
calls the model turn `model` rather than `assistant`. These are cosmetic differences compared to the underlying shared
shape.

### The Responses API

In 2025, OpenAI introduced a new interface called the
[Responses API](https://platform.openai.com/docs/guides/responses-vs-chat-completions). It keeps the same conceptual
model but adds server-side state. Rather than sending the full conversation history on every turn, you pass a
`previous_response_id` and the provider reconstructs the context. It also ships built-in tools like web search and a
code interpreter, blurring the line between API and agent platform.

Chat Completions is not going away — OpenAI has committed to supporting it indefinitely — but the Responses API is the
recommended path for new projects. Whether the rest of the industry follows is an open question.

For most applications you build today, Chat Completions is what you will use. The chatbot flavour is just something to
be aware of, not a limitation in practice.

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

## Retrieval-Augmented Generation

RAG is a pattern for giving an LLM access to external knowledge without baking that knowledge into its training data.
Instead of hoping the model already knows the answer, you retrieve relevant information at query time and include it in
the context. The model then generates a response grounded in what you retrieved.

Three problems drive adoption of RAG.

The first is stale knowledge. LLMs have a training cutoff. If your product documentation changed last week, the model
does not know about it.

The second is scale. Your entire document corpus almost certainly cannot fit in a context window. Even at 200,000
tokens, there is a practical limit. Stuffing everything in is also expensive.

The third is hallucination. Models fabricate plausible-sounding facts when they do not know something. RAG gives the
model actual source material to work from, which significantly reduces this tendency and gives you sources to cite.

### The basic pipeline

A RAG system has two phases. The ingestion phase runs offline. The retrieval phase runs at query time.

**Ingestion**

1. Load your documents (PDFs, HTML pages, database records, whatever)
2. Split them into chunks — smaller pieces of text that each represent a meaningful unit of information
3. Convert each chunk into an embedding — a vector of numbers that encodes its meaning
4. Store the chunks and their embeddings in a vector database

**Retrieval**

1. Take the user's query and convert it into an embedding using the same model
2. Search the vector database for chunks whose embeddings are closest to the query
3. Inject the top results into the context alongside the user's question
4. The model generates a response with that retrieved material available

This is the naive version. It works reasonably well but has enough failure modes that production systems add several
layers on top.

### Embeddings

An embedding model converts text into a dense vector of floating-point numbers (commonly 768 to 3072 dimensions
depending on the model). Semantically similar passages produce similar vectors, which makes it possible to search by
meaning rather than by keyword.

Embedding models are separate from generation models. You use one model to embed your corpus and the same model to embed
queries at retrieval time. Mixing different embedding models or even different versions of the same model produces
incompatible vectors.

Popular options include OpenAI's `text-embedding-3-large`, Cohere's Embed, and open-weight options like
`bge-large-en-v1.5` from BAAI. Domain-specific models can outperform general-purpose ones by 12 to 30 percent on
industry-specific retrieval tasks, so it is worth evaluating on your own data before committing to one.

### Chunking

How you split documents into chunks matters more than most people expect. The chunk is the unit of retrieval. Too large
and it dilutes the signal and wastes context budget. Too small and it loses the surrounding context that makes it
interpretable.

**Fixed-size chunking** splits text at a fixed token count (say, 512 tokens) with some overlap between adjacent chunks.
Simple and predictable, but it can split a sentence in the middle.

**Recursive character splitting** tries to split on natural boundaries first (paragraphs, then sentences, then words)
before falling back to fixed size. This is a better default than pure fixed-size splitting and is what most frameworks
use out of the box.

**Semantic chunking** uses embeddings to detect where the topic shifts and places boundaries there. It produces more
coherent chunks at the cost of more computation during ingestion.

**Late chunking** embeds the full document first, so every token's representation reflects the whole document's context,
and only then extracts chunk-level embeddings by averaging. This is a newer technique that improves retrieval quality
for chunks that would lose meaning if extracted in isolation.

For most projects, recursive splitting with a chunk size around 512 tokens and a 10 to 15 percent overlap is a
reasonable starting point.

### Retrieval

Pure vector search (also called dense retrieval) is good at semantic matching. Ask "how do I cancel my subscription" and
it retrieves text about account cancellation even if the word "cancel" does not appear. Where it struggles is exact
matching. Product codes, version numbers, names and error codes often retrieve poorly with embeddings alone.

BM25 is a classic keyword-based ranking algorithm. It works like a traditional search engine, scoring results by term
frequency and rarity across the corpus. It is predictable, transparent and excellent at exact matches.

In production, combining both outperforms either technique alone by 10 to 30 percent across most benchmarks. You run
both searches and merge the ranked results using Reciprocal Rank Fusion (RRF), a simple algorithm that combines ranked
lists without needing to tune score weights. Hybrid search is the safe default for production systems.

**Reranking** is a second pass that runs after the initial fetch. A reranker model takes the query and each retrieved
chunk and scores them together, producing a more precise relevance ranking. The standard approach is to retrieve a broad
set (say, 50 results with hybrid search) and rerank down to the top 5 or 10 to pass to the model. The key insight is
that a reranker only helps once retrieval recall is already solid. If the right document is not in your top 50 results,
a reranker will not recover it.

### Making retrieval smarter

The query the user types is often not the ideal retrieval query. Several techniques address this.

**Query rewriting** uses the LLM to rephrase the question before retrieval. A follow-up like "what about the pricing?"
needs expanding to "what is the pricing for Product X?" to retrieve meaningfully. This can be a simple prompt asking the
model to produce a standalone search query from the conversation context.

**HyDE (Hypothetical Document Embeddings)** generates a hypothetical answer to the query and embeds that instead of the
question itself. The idea is that an answer embedding is closer in vector space to a real answer than a question
embedding is. It improves recall for vague or poorly phrased queries at the cost of an extra LLM call.

**Query decomposition** breaks a complex question into simpler sub-queries that are each retrieved separately. Useful
for questions that span multiple topics or require synthesising across several sources.

### GraphRAG

Standard RAG retrieves individual chunks. It handles pinpoint lookups well but struggles with questions that require
synthesising across many documents, such as "what are the recurring themes in our customer complaints?"

GraphRAG, developed by Microsoft Research and released as open source in 2024, builds a knowledge graph over your corpus
during ingestion. It extracts entities and relationships using an LLM and organises documents around those nodes. At
query time it can traverse the graph to answer theme-level questions with full traceability.

The cost is significant. Ingestion requires three to five times more LLM calls than standard RAG, entity resolution
accuracy ranges from 60 to 85 percent depending on the domain, and updating the corpus can trigger expensive
recomputation. GraphRAG is not the right default. It is the right choice when your use case is about understanding and
synthesising a corpus rather than retrieving specific facts from it.

### Agentic RAG

In basic RAG, retrieval happens once at query time. Agentic RAG puts retrieval under the control of an agent that can
plan multiple retrieval steps, choose between different retrieval tools, evaluate whether it has enough information and
retrieve again if not.

This is more powerful for complex multi-step questions but also harder to build, evaluate and debug. See
[AI Agents](./ai_agents) for more on the underlying patterns.

### RAG vs fine-tuning

These are often compared but they solve different problems.

Fine-tuning changes the model's weights. It teaches the model new skills or behaviour by training it further on specific
data. It does not reliably inject specific facts. Models fine-tuned on a dataset of facts will still hallucinate
details. Fine-tuning is the right choice when you want to change how the model writes or what it is capable of.

RAG injects facts into the context at query time without changing the model at all. The model's knowledge is only as
current as your last ingestion run. RAG is the right choice when you need the model to know specific facts, especially
facts that change.

Using both together is also valid. Fine-tune for domain-specific style or terminology and use RAG to keep the model
current.

### Common pitfalls

**Retrieving the wrong chunks** is the most common failure mode. If retrieval is poor, the model either hallucinates
because it has no relevant context or gives a wrong answer based on a tangentially related chunk. Evaluate retrieval
quality separately from answer quality.

**Chunk boundaries that destroy meaning** are easy to introduce with naive splitting. If a chunk starts mid-sentence or
contains an unresolvable reference ("it was discontinued in 2023" — what was?), retrieval may surface it but the model
cannot use it.

**Embedding drift** happens when you update your embedding model. The new model produces different vectors, so
embeddings generated by the old model and the new model are not comparable. You need to re-embed the entire corpus after
any model change.

**Ignoring latency** is a common oversight at the prototype stage. A basic vector search might add 20ms. A hybrid search
with reranking and query rewriting can add several hundred milliseconds. Measure it before going to production.

**Too many chunks in context** can hurt as much as too few. Providing 20 loosely relevant chunks means the model has to
work through noise. Precision matters more than recall by the time you are assembling the final context.

## Fine-tuning

Fine-tuning is continued training on a pre-trained model using your own data. It adjusts the model's weights rather than
injecting information into the context. This is what distinguishes it from RAG and from prompt engineering, both of
which leave the model itself unchanged.

It is also the most expensive and complex option. The standard advice is to exhaust prompt engineering and RAG before
reaching for fine-tuning, because those two cover the majority of practical use cases at a fraction of the cost and
complexity.

### What fine-tuning can and cannot do

Fine-tuning is good at changing how a model behaves. You can teach it to write in a specific style, follow a particular
output format, adopt a domain-specific tone or decline certain types of requests. You can specialise a general-purpose
model to behave like an expert in a narrow domain.

Fine-tuning is not a reliable way to inject facts. A model fine-tuned on a dataset of facts will still hallucinate. The
training process does not store facts the way a database does. If you need the model to answer accurately from specific
documents, use RAG.

### Types of fine-tuning

**Supervised Fine-Tuning (SFT)** is the foundation. You provide pairs of inputs and ideal outputs and train the model to
reproduce that behaviour. This is how base models are turned into instruction-following assistants, and it is the
starting point for most fine-tuning projects. Quality matters far more than quantity. Around 1,000 well-crafted examples
will outperform 100,000 noisy ones.

**LoRA (Low-Rank Adaptation)** is a parameter-efficient technique that avoids updating all of the model's weights.
Instead, it trains small adapter matrices that sit alongside the original weights. The result is a much smaller set of
trainable parameters, which means lower memory requirements, lower compute cost and faster training. QLoRA adds 4-bit
quantisation on top, reducing memory further. For most practitioners, LoRA or QLoRA is the practical default rather than
full fine-tuning.

**Full fine-tuning** updates every parameter in the model. It is more powerful but requires significantly more compute
and carries a higher risk of catastrophic forgetting, where the model loses capabilities it had before training. Most
application-level fine-tuning does not need it.

**RLHF (Reinforcement Learning from Human Feedback)** is the technique that turned base models into ChatGPT-style
assistants. It involves training a separate reward model on human preference data and then using reinforcement learning
to steer the main model towards higher-reward outputs. It is complex to implement, requires managing several model
copies during training and is prone to instability. It is generally not something you implement yourself.

**DPO (Direct Preference Optimisation)** achieves similar alignment goals to RLHF with significantly less complexity.
Instead of a reward model and RL training loop, DPO trains directly on preference pairs — examples where a human has
indicated which of two responses is better. Research from Stanford and others shows it achieves comparable or better
results than PPO-based RLHF at roughly half the compute. DPO is the approach most teams use when they need to align
model behaviour with human preferences.

**RFT (Reinforcement Fine-Tuning)** is a newer approach suited to tasks with verifiable outcomes, such as coding or
maths. The model is rewarded when its outputs are provably correct, which drives improvement on reasoning-heavy tasks.
OpenAI introduced this in late 2024 and it underpins much of the improvement in their o-series reasoning models.

### Data requirements

The data is usually the hardest part. The format is typically JSONL, where each line contains a prompt/completion pair
or a conversation thread. Providers like OpenAI, Anthropic and Google all have specific schemas for their fine-tuning
endpoints.

A few hundred to a few thousand high-quality examples is a reasonable starting point for SFT. The examples need to be
representative of what you actually want the model to do. If your examples are inconsistent or contain errors, the model
will learn the noise.

For DPO you need preference pairs — the same prompt with two responses, one preferred and one not. Collecting this data
is slower but the results are more targeted at behavioural alignment.

### Managed fine-tuning

All three major providers offer managed fine-tuning APIs where you upload your data, kick off a training job and receive
a fine-tuned model identifier to use in your application. This removes the infrastructure burden entirely. OpenAI's
fine-tuning API supports GPT-4o and GPT-4o mini. Anthropic offers fine-tuning for Claude. Google offers it via Vertex
AI.

Managed fine-tuning is the right default for most application developers. Rolling your own training infrastructure is
only worth it if you have very specific requirements around hardware, data privacy or model architecture.

### Common pitfalls

**Fine-tuning to compensate for bad prompts** is a common waste of effort. If you have not first tried a well-crafted
system prompt with examples, you are skipping a cheaper and faster fix.

**Catastrophic forgetting** happens when fine-tuning on a narrow dataset degrades the model's general capabilities. The
model gets better at your specific task but worse at everything else. Using LoRA reduces this risk because the original
weights are preserved.

**Overfitting** on a small dataset produces a model that handles training examples well but fails on anything slightly
different. A small, high-quality dataset is better than a large inconsistent one, but you still need enough variety to
cover realistic inputs.

**Expecting facts to stick** is a fundamental misunderstanding of what fine-tuning does. If you need accurate recall of
specific information, use RAG.

**Underestimating maintenance** is easy to overlook. A fine-tuned model is another artefact to version, evaluate and
retrain when the underlying base model is updated.

## Development environments and evaluation

The tooling landscape for AI development has matured quickly. There are now three distinct categories of environment
depending on where you are in the development cycle: provider studios for interactive experimentation, visual builders
for assembling pipelines without code and evaluation platforms for systematic testing.

### Provider studios

Every major provider ships a browser-based studio where you can iterate on prompts, try different models and inspect
what the API is actually doing. These are the fastest way to experiment before writing any application code.

**[Anthropic Console](https://console.anthropic.com/)** (Workbench) is the starting point for working with Claude. You
can write system prompts, adjust parameters, compare model versions side by side and export working prompts directly as
code. It also exposes token usage and cost estimates per request, which is useful for early sizing.

**[OpenAI Playground](https://platform.openai.com/playground)** is the most mature of the provider studios, reflecting
OpenAI's head start in the market. It supports chat, assistants and fine-tuning workflows in one interface and has the
largest ecosystem of community examples to borrow from.

**[Google AI Studio](https://aistudio.google.com/)** is Google's browser-based environment for Gemini models. It stands
out for its multimodal capabilities, support for very long context windows and a fast path to deploying via Vertex AI
for production workloads. Grounding with Google Search is available as an option.

**[AWS Bedrock](https://aws.amazon.com/bedrock/)** gives access to models from multiple providers (Anthropic, Meta,
Mistral and others) within AWS infrastructure. If your application already runs on AWS, it is often the most
straightforward way to use Claude or Llama without introducing a new vendor relationship.

**[Azure AI Foundry](https://ai.azure.com/)** (formerly Azure AI Studio) is Microsoft's unified environment for
building, testing and deploying AI applications. It includes prompt flow, model catalogue, evaluation tooling and direct
access to OpenAI models inside the Azure boundary.

**[OpenRouter](https://openrouter.ai/)** is not a studio in the traditional sense but deserves a mention. It provides a
single API that routes to over a hundred models across providers. For comparing how different models handle the same
prompt, it removes a lot of setup friction.

### No-code and visual builders

If you want to wire together a retrieval pipeline, an agent or a multi-step workflow without writing code, visual
builders are the fastest route. They are also useful for prototyping architectures before committing to a code
implementation.

**[Flowise](https://flowiseai.com/)** is an open-source canvas built on LangChain and LlamaIndex. You drag components
onto a canvas — a vector store, an embedding model, a language model, a retriever — connect them and test in the same
screen. It ships with three builders covering simple assistants, single-agent chatflows and multi-agent orchestration.

**[Dify](https://dify.ai/)** is an open-source platform that blends a visual workflow builder with a
backend-as-a-service approach. It is aimed at teams that want to get a RAG-backed application running quickly, with
built-in support for knowledge bases, prompt management and model switching.

**[LangFlow](https://www.langflow.org/)** is a visual canvas for agents and RAG pipelines with a live chat pane so you
can test while you build. It integrates with all major LLMs and vector databases and has recently added support for
building and deploying MCP servers.

**[n8n](https://n8n.io/)** started as a general-purpose automation tool and has grown into a capable AI workflow
builder. Its strength is integration breadth. It connects AI steps to hundreds of external services, supports branching
and error paths and has a self-hosted option for teams where data residency matters.

**[Azure PromptFlow](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/overview-what-is-prompt-flow)**
is Microsoft's visual environment for building and evaluating LLM workflows. It is tightly integrated with Azure AI
Foundry and fits naturally into teams already using Azure DevOps for CI/CD.

**[Vertex AI Agent Builder](https://cloud.google.com/products/agent-builder)** is Google's no-code option for building
grounded agents backed by Google Search or your own documents. It is aimed at enterprise teams on GCP rather than
developers prototyping new things.

### Evaluation platforms

Testing an LLM application is different from testing conventional software. Outputs are non-deterministic. The same
input can produce slightly different responses on different runs. Whether a response is "correct" is often a matter of
judgement rather than an exact string comparison. This requires a different approach to quality assurance.

**[LangSmith](https://www.langchain.com/langsmith)** is LangChain's observability and evaluation platform. It captures
traces of every LLM call in your application, which means you can replay real production inputs in your test suite. Its
playground and prompt hub handle versioning, and its evaluation runner lets you run datasets against prompts and score
the results.

**[Langfuse](https://langfuse.com/)** is an open-source alternative to LangSmith with strong self-hosting support. It
covers tracing, monitoring, dataset management and evaluation in one place. Because it is MIT-licensed and API-first, it
integrates cleanly with custom tooling and is a good choice when data control is a concern.

**[Braintrust](https://www.braintrust.dev/)** is a closed-source platform built specifically around the eval-driven
development loop. Production traces become test cases with one click. Eval results surface in pull requests via CI
integration. It is designed for teams where product managers and engineers need to iterate on prompts together.

**[Promptfoo](https://www.promptfoo.dev/)** is an open-source CLI and library for systematic prompt testing. You define
test cases in YAML or JSON config files, which can be committed to git and run in CI like any other test suite. Its
standout feature is red teaming — it can probe your prompts for vulnerabilities, test for prompt injection, check for
PII leaks and identify guardrail failures.

**[DeepEval](https://github.com/confident-ai/deepeval)** is an open-source framework built around LLM unit testing. The
API is modelled after pytest and ships with metrics for answer relevancy, hallucination, faithfulness (does the response
match retrieved context?), and task completion. You write test cases in Python and run them as part of your normal test
suite.

### How to evaluate LLM outputs

The core challenge is defining what "correct" means for a given task.

**Exact match** works only for tasks with a single correct answer. Classification labels, boolean decisions and
structured JSON outputs can be tested this way. If your model should return `{"sentiment": "positive"}` and it does, the
test passes.

**Semantic similarity** compares the meaning of the output to an expected answer rather than the exact wording. An
embedding model scores how close the two are. This works for summarisation or question answering where there are
multiple acceptable phrasings of a correct answer.

**LLM-as-judge** uses a capable model (typically GPT-4o or Claude) to score responses against criteria you define. You
write a rubric — accuracy, tone, completeness, brevity — and the judge model rates each response. This handles
subjective quality dimensions that rule-based metrics cannot capture. It is slower and costs more tokens, but for tasks
where human judgement is the real ground truth it is the most practical automated proxy.

**Human review** remains the ground truth for anything where nuance matters. Build a feedback mechanism into your
application so you can capture when users mark a response as unhelpful or flag it as wrong. Those signals feed directly
back into your evaluation dataset.

### Building a test set

A useful test set covers representative inputs, edge cases and known failure modes. Start building it from day one, not
after you have a problem.

The typical workflow looks like this. Pick 30 to 100 diverse inputs that represent real usage. Write down what a good
response looks like for each — the expected output does not need to be a single correct answer but it should be specific
enough to evaluate against. Run your current prompt and model against the set and record the scores. When you change a
prompt, a model or a retrieval strategy, run the set again and compare.

The goal is to catch regressions. A prompt change that improves one case often degrades another. Without a test set you
will not notice until a user complains.

For teams with a CI pipeline, tools like Promptfoo and Braintrust can run evaluations automatically on every pull
request, blocking merges when quality scores drop below a threshold.

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

## Language and architecture choices

Integrating an LLM into an application is not a machine learning problem. It is an API integration problem, and the most
important factor in choosing a language is SDK availability rather than raw language capability.

### Python

Python is the default for AI work and the safest choice if you do not have a strong reason to pick something else. Every
provider ships a first-party Python SDK. Every orchestration framework — LangChain, LlamaIndex, LangGraph, AutoGen — is
Python-first. Every evaluation tool, every vector database client and the vast majority of example code is Python.

If your application involves a data pipeline, RAG ingestion, fine-tuning, embeddings or anything involving data
processing alongside LLM calls, Python is the most productive choice by a significant margin. The ecosystem depth means
you rarely hit a wall.

The main weakness is that Python is not a natural fit for frontend work or for applications where startup time and
concurrency matter at the edges.

### TypeScript and JavaScript

TypeScript is the strongest alternative to Python for AI application development, and for full-stack web applications it
is often the better choice. All three major providers ship first-party TypeScript SDKs that are well-maintained and kept
in step with their Python equivalents.

The main advantage is that you can write the backend (Node.js or a framework like NestJS), the frontend (React, Vue,
Svelte) and infrastructure as code in the same language. This matters in practice. It reduces context switching,
simplifies the monorepo and means a single developer can work across the whole stack without language boundaries slowing
them down.

TypeScript also handles streaming well. LLM responses are inherently streamed, and TypeScript's async iterator model
maps naturally to reading a streaming API response and forwarding it to the client.

The gap with Python is in the broader ML ecosystem. If your project starts needing data pipeline work, fine-tuning or
custom retrieval logic, you will eventually find Python libraries that have no TypeScript equivalent.

### Go, Java and C#

All three are viable options if you already have a codebase in one of them.

**Go** has official SDKs from Anthropic and an OpenAI-compatible community SDK. Its concurrency model is well suited to
building API proxies and services that sit between your application and LLM providers. For teams building backend
services rather than data pipelines, Go is a solid choice.

**Java** has mature support via the Spring AI framework, which provides a unified abstraction over multiple LLM
providers. It is the right choice for enterprise codebases already on the Spring stack.

**C#** is the natural choice on the Microsoft and Azure stack. The Azure OpenAI SDK is first-party and well integrated
with the rest of the .NET ecosystem.

### Rust

Rust has thin AI library support compared to the options above. There are community SDKs but no first-party support from
any major provider as of early 2026. For applications where performance is critical and you are comfortable building
more from scratch, it is possible, but it is not the pragmatic default for most projects.

### Backend or frontend

Do not call LLM APIs directly from the browser. The reason is simple: you cannot call an LLM API from the frontend
without exposing your API key in client-side code, where any user can read it. An exposed API key means anyone can make
requests billed to your account.

The correct pattern is to put a thin backend between your frontend and the LLM provider. The frontend sends the user's
message to your backend. Your backend authenticates the request, assembles the full context, calls the LLM API and
returns the response. Your API key never leaves the server.

This backend can be a dedicated service, a serverless function or simply a route in your existing web application. The
important thing is that the LLM call happens there, not in the browser.

**Streaming** complicates this slightly but does not change the principle. Users expect to see tokens appear as they are
generated rather than waiting for the full response. To achieve this, your backend proxies the streaming response from
the LLM provider and forwards it to the frontend using Server-Sent Events or a WebSocket connection. All major LLM SDKs
support streaming, and it is straightforward to proxy.

The one exception worth noting is Anthropic's and OpenAI's client-side SDKs, which are designed for use in environments
like React Native or Electron where you control the runtime. These are not for browser applications where the source is
publicly accessible.

### A practical starting point

For a new web application with an AI feature, the simplest stack that covers most cases is a TypeScript backend
(Node.js) with a TypeScript or React frontend. This gives you a single language across the stack, first-party SDK
support, clean streaming support and a fast path from prototype to production.

If the project involves significant data work, RAG ingestion, model evaluation or anything ML-adjacent, Python is the
better backend language. In that case, a common split is Python for the AI backend service and TypeScript for the
frontend, with a REST or WebSocket interface between them.

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
