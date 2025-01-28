---
layout: page
title: AI Agents
---

What follows is my attempt to explain what is meant by the term "AI agent", list out some examples and resources and separate the buzzword from the facts.

For simplicity, this document only refers to LLM-powered agents using technology and concepts from 2023 onwards. The real history and facts 
about AI agents goes back decades and incorporates a lot of different underlying technologies and techniques (I had to draw the line somewhere).

## Basic definition of an AI agent

An AI agent is an autonomous program designed to perform specific tasks by understanding its environment, reasoning about it, and taking actions to achieve a goal. AI agents often have roles to define their purpose and responsibilities in a multi-agent system or as part of a larger application.

At the core of any AI Agent is a Large Language Model. This enables the agent to converse in natural language and transform between unstructured and structured text.

Here are some examples of what an AI agent or Agentic-based system will do:

* A personal assistant bot that checks the weather or books a meeting
* A web-crawling bot gathering information for market research
* A financial bot deciding on stock investments based on historical data

## Other terms in the context of AI agents

* **Role** - Defines the identity or perspective the AI adopts in the interaction.
* **Instructions** - Guidelines or constraints provided to the AI to shape its behavior or responses.
* **Message** - Content exchanged between the AI and the user or system during the interaction
* **Agentic** - Means a system can operate independently, making decisions based on data analysis and predefined goals
* **Model** - The underlying LLM used by the agent
* **No Code** - No Code agent systems use drag and drop plus text boxes for configuration instead of code (usually Python). [Relevance AI](https://relevanceai.com/agents) would be an example of such a system
* **RAG** - Retrieval-Augmented Generation. A technique for building LLM-powered applications.
* **ReAct** - Reasoning and Action
* **Multi-agent** - A system using multiple specizilised agents to collaborate on a desired outcome such as summarising today's news into an email
* **Task-Oriented Agent** - Executes specific tasks based on given inputs, like fetching data, summarizing content, or scheduling events
* **Collaborative Agent** - Works with other agents or users to solve problems or complete objectives.
* **Exploratory Agent** - Collects and analyzes data to identify trends or new insights.
* **Decision-Making Agent** - Uses reasoning or ML models to make informed decisions.
* **Interactive Agent** - Communicates with users or systems, often using natural language.

## What is the difference between a chatbot and an agent?

Where chatbots largely follow rules-based dialogues and are limited to answering predefined questions, 
AI agents can reason and ground answers in relevant knowledge and content. Chatbots, unlike agents, 
need extensive training on hundreds of conversations to be able to understand natural-language requests, 
making agents significantly quicker and easier to implement and launch.

AI agents can understand context, learn from interactions, and adapt their behavior to achieve specific goals.
Unlike simpler systems, AI agents can handle ambiguity, make autonomous decisions, and execute multi-step plans to 
solve complex problems, making them suitable for more challenging and open-ended tasks.

## No Code vs Code

**No-code AI systems** are ideal for users without programming expertise, offering intuitive drag-and-drop interfaces
and pre-built components to create AI solutions quickly. They reduce development time and are great for prototyping or 
simple applications like chatbots or data analysis. However, they can be limited in customization, flexibility, and
scalability for complex or unique use cases. In contrast, **coded AI systems** provide full control, customization, 
and the ability to optimize performance, making them suitable for advanced or highly specific requirements. 
While they require programming skills and more development time, coded solutions are more adaptable and capable 
of handling complex tasks effectively.

## Python-configurable agent systems

List of agent systems that can be configured with Python (other language interfaces may be available). I have also linked to the documentation and open source
example code showing how it can be used.

| System name                                                   | Official docs                                                                         | Code samples                                                                                |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [Phidata](https://www.phidata.com/)                           | [Phidata introduction](https://docs.phidata.com/introduction)                         | [Phidata samples ](https://github.com/phidatahq/phidata/tree/main/cookbook/examples/agents) |
| [OpenAI Swarm](https://github.com/openai/swarm)               | [OpenAI Swarm docs](https://github.com/openai/swarm?tab=readme-ov-file#documentation) | [OpenAI Swarm samples ](https://github.com/openai/swarm/tree/main/examples)                 |
| [LangGraph](https://langchain-ai.github.io/langgraph/)        | [LangGraph docs](https://langchain-ai.github.io/langgraph/tutorials/introduction/)    | [LangGraph samples ](https://github.com/langchain-ai/langgraph/tree/main/examples)          |
| [Microsoft Autogen](https://microsoft.github.io/autogen/0.2/) | [Autogen docs](https://microsoft.github.io/autogen/stable/)                           | [Autogen samples](https://github.com/microsoft/autogen/tree/main/python/samples)            |
| [CrewAI](https://www.crewai.com/)                             | [Autogen docs](https://docs.crewai.com/introduction)                                  | [Autogen Code samples ](https://github.com/crewAIInc/crewAI-examples)                       |
| [Vertex AI](https://cloud.google.com/vertex-ai?hl=en)         | [Vertex AI Code docs ](https://cloud.google.com/vertex-ai/docs)                       | [Vertex AI Code samples ](https://github.com/GoogleCloudPlatform/vertex-ai-samples)         |
| [Langflow](https://www.langflow.org/)                         | [Langflow docs](https://docs.langflow.org)                                            | [Langflow samples ](https://github.com/langflow-ai/langflow_examples)                       |

## No Code agent systems

List of No-Code agent systems. These don't require coding skills but may be more expensive to run, less flexible and in
some cases slower to execute. However, you may end up a good-enough solution with a lot less effort, especially for 
individual or small scale deployment. Some of these systems are built specifically for an eco-system such as Agentforce
which is designed to work with Salesforce.

* [make.com AI Automation](https://www.make.com/en/ai-automation)
* [Zapier agents](https://zapier.com/agents)
* [Voiceflow](https://www.voiceflow.com/)
* [Vapi](https://vapi.ai/)
* [Relevance AI](https://relevanceai.com/agents) 
* [n8n](https://n8n.io/)
* [Copilot agents](https://support.microsoft.com/en-gb/topic/introducing-copilot-agents-943e563d-602d-40fa-bdd1-dbc83f582466)
* [Agentforce](https://www.salesforce.com/uk/agentforce/)

## Do you need a framework to build AI agents?

I would not recommend building an AI agent framework or system from scratch... 

If you're building a simple AI agent with limited functionality, you can write everything from scratch using libraries
like Python’s asyncio for concurrency, requests for APIs, or ML libraries like TensorFlow or PyTorch. [This article](https://www.analyticsvidhya.com/blog/2024/07/build-ai-agents-from-scratch/)
describes how you can do this.

My feeling is that, other than for an academic exercise, there are so many well-thought-out frameworks that you will most 
likely end up re-inventing the wheel. If you're going to do AI agents, use a system designed for that purpose

## AI Agent approach vs traditional coding approach

For a basic use case, traditional procedural Python code is often the better choice due to its simplicity, speed of 
implementation, and ease of debugging. Tasks like processing data files, automating repetitive actions, or performing
straightforward calculations are efficiently handled with well-defined, step-by-step code. It’s cost-effective and ideal 
for problems with clear rules and predictable inputs.

An AI agent system, while powerful, is unnecessary for most basic tasks. It’s better suited for complex, dynamic scenarios
requiring adaptability, such as natural language processing or pattern recognition. AI introduces added complexity, 
requiring more time, resources, and expertise. For simple use cases, the overhead of an AI system outweighs its benefits.

## Local hosted vs cloud hosted

For implementing an AI agent system, locally hosted solutions are ideal when data privacy, control, and low latency are 
priorities. Hosting locally gives you full control over the environment, allowing customization and operation without 
reliance on external networks. This approach is cost-effective for small-scale setups with minimal infrastructure needs,
and it avoids ongoing subscription fees. However, it may require higher upfront hardware investment, and scaling up for 
large workloads can be challenging.

In contrast, cloud-hosted solutions excel in scalability, ease of setup, and access to powerful resources like GPUs or 
TPUs. They’re ideal for dynamic or large-scale applications where rapid scaling or collaboration is required. 
Cloud providers handle maintenance, updates, and high availability, saving time and effort. However, recurring 
costs can add up, and reliance on internet connectivity and third-party providers can introduce security concerns 
and latency, depending on the use case.

## Free vs paid solutions

**Free solutions** provide an accessible starting point for AI development, especially for small-scale projects or
experimentation. Many platforms offer free tiers that include basic features and limited resources, such as compute
hours, API calls, or smaller models. Open-source frameworks and large language models (LLMs) like GPT-Neo or Falcon 
can also be deployed on personal hardware at no cost, making them an attractive option for developers with technical
expertise. Free solutions are ideal for learning, prototyping, or low-demand applications but often come with
restrictions like limited scalability, performance caps, or reduced support.

**Paid solutions** offer enhanced capabilities, including access to larger models, higher computational power,
and advanced features like fine-tuning or analytics dashboards. They are suited for production-level systems requiring
reliability, scalability, and support. Paid plans also often include managed services, reducing the burden of 
infrastructure setup and maintenance. While they incur ongoing costs, the efficiency, ease of use, and ability to
handle complex workloads often justify the investment for businesses or mission-critical applications.

## Resources

This is a list of articles, YouTube channels and other useful resources related to AI agents

* [Multiagent AI frameworks](https://getstream.io/blog/multiagent-ai-frameworks/)
* [Demystifying AI Agents](https://www.mongodb.com/resources/basics/artificial-intelligence/ai-agents)
* [Ben AI YouTube channel](https://www.youtube.com/@BenAI92)
* [react agent model](https://klu.ai/glossary/react-agent-model)
* [exploring microsoft copilot and autogen](https://www.topview.ai/blog/detail/microsoft-ai-tour-boston-exploring-microsoft-copilot-and-autogen-with-experts)
* [rag comparative analysis](https://www.digitalocean.com/community/conceptual-articles/rag-ai-agents-agentic-rag-comparative-analysis)
* [comparing rag systems and ai agents](https://aisutra.com/comparing-rag-systems-and-ai-agents-2ea9082c80d6)
* [what is agentic rag](https://weaviate.io/blog/what-is-agentic-rag)
* [awesome agents](https://github.com/kyrolabs/awesome-agents)