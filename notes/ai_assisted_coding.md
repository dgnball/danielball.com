---
layout: page
title: AI Assisted Coding
---

This is my attempt to make sense of the AI-assisted coding landscape. I wrote this to educate myself and as a reference
for anyone trying to navigate this space without the hype.

<!-- prettier-ignore -->
- TOC
{:toc}

## AI Coding Glossary

- **Agentic AI** — An AI system that doesn't just respond to a single prompt but takes sequences of actions
  autonomously: running commands, reading files, making decisions and iterating toward a goal with minimal human
  intervention between steps.

- **Cargo-Cult Programming** — Writing code by copying patterns or snippets without understanding why they work, hoping
  the result will behave correctly by association. Named after the anthropological phenomenon of mimicking the form of
  something without grasping its function. Stack Overflow made this easy; AI tools risk amplifying it further.

- **Context Window** — The maximum amount of text (code, instructions, conversation history) an AI model can "see" at
  once when generating a response. Larger context windows allow tools like Claude Code to reason across more files
  simultaneously. Measured in tokens.

- **Code Completion** — A feature in IDEs that suggests the next token, method or block of code as you type, ranging
  from simple symbol lookup (early IDEs) to ML-ranked suggestions (PyCharm) to full-line and multi-line generation
  (GitHub Copilot).

- **Copilot** — [GitHub Copilot](https://github.com/features/copilot), a widely adopted AI code completion tool built on
  OpenAI's Codex model. Sits inside your editor and suggests whole lines or functions in real time as you type, trained
  primarily on public GitHub repositories.

- **Embedding** — A numerical representation of code or text as a vector in high-dimensional space, used by AI systems
  to measure semantic similarity. Allows a model to understand that `fetch_user` and `get_user` are likely related
  concepts even though the strings are different.

- **Fine-tuning** — Taking a general-purpose language model and training it further on a specific dataset — a particular
  codebase, language or domain — so it becomes more accurate and idiomatic for that context.

- **Hallucination** — When an AI model confidently generates something that is plausible-looking but factually wrong: a
  method that doesn't exist, an API with the wrong signature, a library that was never published. A significant risk in
  AI-assisted coding that makes verification essential.

- **Inference** — The process of running a trained AI model to generate output. Distinct from training. When you ask
  Claude Code to refactor a function, you're triggering inference against a model that was trained earlier.

- **LLM (Large Language Model)** — The class of AI model underpinning modern coding assistants. Trained on vast text and
  code corpora to predict likely next tokens, LLMs develop emergent abilities to reason, explain, translate between
  languages and generate syntactically and semantically coherent code.

- **MCP (Model Context Protocol)** — An open standard for connecting AI agents to external data sources and tools. With
  MCP, a coding assistant isn't limited to your local files — it can be connected to Jira, Slack, Google Drive, Figma,
  or any custom internal tooling that exposes an MCP server. Increasingly supported across Claude Code, Cursor, Windsurf
  and other agentic tools.

- **Prompt Engineering** — The practice of carefully crafting inputs to an AI model to get better outputs. In coding
  contexts this might mean specifying the language, describing constraints, providing examples of desired style or
  including relevant error messages to steer the model toward a useful response.

- **RAG (Retrieval-Augmented Generation)** — A technique where an AI system retrieves relevant documents or code
  snippets from an external source (like your codebase or documentation) and injects them into the prompt before
  generating a response, allowing the model to reason over information it wasn't trained on.

- **Static Analysis** — Examining code without executing it, to infer types, detect errors or suggest completions. The
  basis of pre-AI tools like [Rope](https://github.com/python-rope/rope) and
  [Jedi](https://jedi.readthedocs.io/en/latest/). Fast and deterministic, but limited by what can be known without
  running the program.

- **Token** — The basic unit an LLM processes: roughly a word, part of a word or a symbol. Models have token limits for
  both input and output. Code tends to be token-dense because of punctuation, indentation and repeated keywords.

- **Type Inference** — Automatically deducing the type of a variable or expression without explicit annotations.
  Critical to good Python autocompletion since Python doesn't require type declarations, though PEP 484 type hints have
  made this significantly more tractable.

- **Vibe Coding** — A term coined by Andrej Karpathy in early 2025 describing a style of development where the
  programmer describes intent in natural language and largely accepts whatever the AI generates, iterating through
  prompts rather than writing code directly. Prioritises speed and outcome over deep understanding of the
  implementation. Powerful for prototyping; risky for production systems where the developer may not fully understand
  what they've shipped.

- **Zero-shot / Few-shot Prompting** — Zero-shot means asking an AI to perform a task with no examples, relying purely
  on its training. Few-shot means providing one or more examples in the prompt to demonstrate the desired pattern or
  output format before asking it to continue. Few-shot reliably improves output quality for structured or stylistically
  specific tasks.

## The Evolution of Code Assistance

**Paper Manuals & Reference Books** The original developer companion. You'd thumb through language references, API docs,
or O'Reilly books to find the right method signature or understand a concept. Entirely passive — the knowledge sat on
the shelf until you went looking. Slow, but surprisingly thorough if you had the right book. Many developers kept
dog-eared copies of K&R C or the Python Cookbook within arm's reach.

**Online IDE Documentation & Language References** As the web matured, documentation moved online —
[docs.python.org](https://docs.python.org), [MSDN](https://learn.microsoft.com/en-us/),
[Javadoc](https://docs.oracle.com/en/java/). IDEs integrated this so you could hover over a symbol and get inline docs
pulled from docstrings or type stubs. Still reactive, but dramatically faster. The shift to hyperlinking meant you could
follow a rabbit warren of related APIs in minutes rather than flicking through an index.

**Copying from Stack Overflow, GitHub & Open Source** Arguably the most impactful "tool" of the mid-2000s to 2010s
developer toolkit — and it wasn't just [Stack Overflow](https://stackoverflow.com). Developers were simultaneously
mining [GitHub](https://github.com), reading through open source repositories on [SourceForge](https://sourceforge.net)
and later [GitLab](https://gitlab.com) and lifting patterns from permissively licensed projects. The cognitive model was
matching natural language intent to human-curated solutions or real-world production code. Stack Overflow gave you
explained snippets; GitHub gave you full working implementations you could study in context. It worked remarkably well
for common problems but broke down for anything domain-specific, requiring you to adapt boilerplate to your situation —
often the fiddly part. It also introduced a generation of developers to cargo-cult programming: paste first, understand
later. Crucially, this vast public corpus — millions of answered questions, billions of lines of openly licensed code —
became the training substrate that made AI coding tools possible. Without Stack Overflow's decade of annotated
problem-solution pairs, and without the open source community's collective output on GitHub, there would be nothing
substantial enough for models to learn the shape of good code from.

**Early Code Completion (2000s IDEs)** Tools like early [Eclipse](https://eclipseide.org),
[Visual Studio](https://visualstudio.microsoft.com) and [IntelliJ](https://www.jetbrains.com/idea/) began offering basic
autocomplete — typically triggered by typing a dot after an object. The engine was largely syntax-aware symbol lookup:
parse the file, infer the type, offer members from that class. It was often brittle — break the type chain and
suggestions vanished. Useful for reducing typos in method names but rarely capable of suggesting whole patterns or
intent.

**Python Autocompletion Before AI — Jedi & Rope** Python's dynamic typing made static autocompletion genuinely hard.
Early tools like [Rope](https://github.com/python-rope/rope) and later [Jedi](https://jedi.readthedocs.io/en/latest/)
tackled this with deep static analysis: they'd trace variable assignments, follow imports, infer types through call
chains and build an in-memory symbol index. Jedi in particular became the backbone of completion in editors like
[Vim](https://www.vim.org), [Emacs](https://www.gnu.org/software/emacs/), [Atom](https://github.com/atom/atom) and
[VS Code](https://code.visualstudio.com). It handled decorators, `*args/**kwargs` and even some basic type inference
without annotations. Its big limitation was dynamic runtime behaviour — if a type was only knowable at execution time,
Jedi had to guess or give up.

**PyCharm's Autocompletion — Basic AI & Deep Analysis** [PyCharm](https://www.jetbrains.com/pycharm/) pushed Python
completion significantly further by combining Jedi-style static analysis with its own proprietary type inference engine
and, more recently, a locally-run ML model (introduced around 2019–2020 as
"[ML-assisted completion](https://plugins.jetbrains.com/plugin/14823-full-line-code-completion)"). This model was
trained on large Python corpora to rank completion candidates by likelihood given surrounding context — so it wouldn't
just offer every valid method, it would surface the one you probably meant. It also integrates full type checking via
its own type system, understanding PEP 484 annotations deeply. The result felt almost anticipatory compared to pure
static tools, though it still operated at the token/symbol level rather than understanding intent.

**Copying and Pasting into ChatGPT (Pre-Agentic)** When [ChatGPT](https://chatgpt.com) launched publicly in late 2022 it
changed developer workflows overnight, even in its most primitive form. The pattern was manual but transformative: copy
a function or error message, paste it into the chat window, describe what you wanted and get back working code with an
explanation. For the first time you could have a conversation about your code — ask follow-up questions, request a
refactor or say "that didn't work, here's the error." The model had no access to your wider codebase, no ability to run
code and no memory between sessions, so you were constantly ferrying context back and forth by hand. Despite the
friction it was a dramatic leap over Stack Overflow for anything non-trivial, because the response was tailored to your
exact situation rather than someone else's similar-but-different problem. Developers quickly learned the art of prompt
crafting — how much context to include, how to describe the desired output, when to paste the full traceback versus a
summary.

**Claude Code** [Claude Code](https://claude.ai/code) operates at a fundamentally different level. Rather than
completing tokens or ranking symbols, it understands intent across an entire codebase. You describe what you want in
plain language, and it can generate, refactor, debug and explain across multiple files simultaneously. It holds context
about your architecture, naming conventions and patterns — not just the open file. It reasons about _why_ code works,
not just _what_ comes next. Used via the CLI, it can run commands, read outputs and iterate — behaving less like a tool
and more like a junior engineer pair-programming alongside you.

**The Near Future — Agents with Persistent Context** The next step is likely persistent, project-aware agents that
maintain a living model of your codebase between sessions — knowing not just the current state but _why_ decisions were
made, tracking tech debt and proactively surfacing problems before you hit them. Think less "answer my question" and
more "autonomous collaborator" that files its own PRs, writes tests as code changes and flags when a new feature
conflicts with an architectural decision made six months ago.

**The Speculative Horizon — Self-Improving Systems** Further out, the boundary between tool and team blurs. Models
trained continuously on a specific codebase — learning your idioms, your team's preferences, even your personal style —
could make suggestions indistinguishable from a senior colleague's review. The frontier question isn't capability, it's
trust and verifiability: how do you confidently delegate to a system you can't fully inspect? The tools that win will
likely be the ones that make their reasoning transparent enough that engineers stay in control of the craft, even as the
mechanical labour of coding largely disappears.

## How to Use an AI Coding Assistant

There's a temptation, when you first get access to an AI coding assistant, to treat it like a very fast search engine.
You type a vague question, get something back and move on. That approach works for simple, isolated tasks, but it misses
the point entirely. Getting real value from these tools requires a shift in how you think about the interaction — less
like querying a database, and more like briefing a capable but context-blind colleague who has just walked into your
project for the first time.

### Starting with the Right Mental Model

AI coding assistants operate within a **context window** — a finite amount of text they can "see" at any one time.
Everything relevant to your task needs to fit within that window: your question, the code you're working on, any files
you've shared, prior conversation and the assistant's own responses. When the context fills up, older information falls
away. The assistant doesn't remember last session's decisions. It doesn't know your project's conventions, your team's
opinions or the architectural choices made six months ago — unless you tell it.

This is the most important thing to understand about working effectively with these tools. **Context is everything, and
context costs tokens.**

### Customising Your Assistant

Most AI coding tools offer several layers of customisation, and taking the time to set these up properly pays dividends
on every subsequent interaction.

**System-level or project-level instructions** are the most powerful lever you have. In Claude Code, for instance, you
can create a `CLAUDE.md` file in the root of your project. This file is read at the start of every session and can
contain anything you'd want a new developer to know before touching your codebase: the tech stack, coding conventions,
which directories to leave alone, how tests are structured, which commands to run to start the dev server. Think of it
as your project's onboarding document for the AI. The difference between a session that starts with this context and one
without is striking — you stop spending tokens re-explaining the basics every time.

Beyond project-level files, most tools allow you to configure **permission levels** (whether the assistant can run shell
commands, edit files autonomously or only make suggestions), **model selection** (trading off speed against quality
depending on the task) and **tool access** (whether the assistant can browse the web, read connected services like Jira
or Google Drive via integrations such as MCP servers).

**Custom commands and slash commands** are another underused feature. Claude Code lets you define custom `/commands`
that encapsulate common workflows — running your test suite, generating a summary of recent changes or triggering a code
review prompt against your own standards. These are tiny upfront investments that save disproportionate amounts of time
over a project's life.

### The Token Economy: Where People Go Wrong

Tokens are the currency of AI interactions. You're charged for them financially (if you're on a usage-based plan) and
constrained by them architecturally. Wasting tokens on unnecessary content makes the tool slower, more expensive and
sometimes less accurate as useful information gets crowded out. Here are the most common mistakes:

**Pasting entire files when only a function is relevant.** If you're asking about a bug in a 40-line function, you don't
need to include the entire 800-line module. Be surgical. Use `@`-mentions in Claude Code or similar context-attachment
features to include only what's needed.

**Repeating context that's already been established.** Once you've explained your stack and architecture, you shouldn't
keep re-stating it in every message. This is exactly what project-level markdown files solve — the context is loaded
once, automatically.

**Ignoring conversation history bloat.** Long, meandering sessions accumulate tokens fast. Every message in the
conversation history is re-sent to the model on each turn. If you've spent thirty messages debugging a side issue,
consider starting a fresh session once that's resolved rather than carrying all that history forward into an unrelated
task.

**Asking for broad rewrites when targeted edits would do.** "Refactor this entire service" consumes far more tokens than
"Extract the database logic from this controller into a separate repository class." The more specific your request, the
more efficient and accurate the response.

**Not using Plan Mode or equivalent.** Many tools offer a mode where the assistant describes what it intends to do
before doing it. This is invaluable. It lets you catch misunderstandings before the assistant has burned tokens (and
potentially made changes) on the wrong approach.

### Planning as a First-Class Activity

One of the biggest mindset shifts in working well with AI coding assistants is learning to plan _before_ you prompt.
This feels counterintuitive — wasn't the whole point to move faster? — but unplanned sessions are where the real time
gets lost.

Before starting a significant task, it's worth spending a few minutes writing down in plain language what you're trying
to achieve, what constraints apply and what a successful outcome looks like. You can put this directly into your prompt,
or better yet, maintain a `TASKS.md` or `PLANNING.md` file in your project that evolves as the work progresses.

This approach has several compounding benefits. It forces you to think clearly about scope, which surfaces ambiguities
before the assistant has a chance to make confident wrong assumptions. It gives the assistant a clear success criterion
to work toward. And it creates a written record you can refer back to if a session goes sideways.

Some developers go further, maintaining a suite of markdown files that together form a kind of living project memory: an
`ARCHITECTURE.md` covering high-level design decisions, a `CONVENTIONS.md` for style and naming rules, a `DECISIONS.md`
logging why certain approaches were chosen and a `PROGRESS.md` tracking what's done and what's next. These files make
every new session dramatically more productive because the assistant has genuine context to work with rather than having
to infer everything from the code alone.

The payoff is that you stop treating every session as a blank slate. The markdown files persist what the context window
cannot.

### Asking Well

Clear, specific prompts produce better results than vague ones. This isn't a character flaw in the AI — it's a
reflection of how language models work. They generate responses based on the probability distribution of what plausibly
follows your input. Vague inputs lead to generic outputs.

Useful habits include: being explicit about what you already know (so the assistant doesn't explain it), specifying the
format you want for the response, breaking complex tasks into discrete steps and using the `@filename` pattern to attach
precise context rather than pasting content into the message body. If a response isn't quite right, it's almost always
more efficient to refine the prompt than to argue with the output.

## How AI Coding Assistants Work

Not all AI coding assistants are the same. The category label "AI coding assistant" currently covers everything from a
light autocomplete plugin that suggests the next line of your function, to a fully autonomous agent that can clone a
repository, understand its architecture, write new features, run the tests and open a pull request — all without you
touching the keyboard. Understanding the differences matters, because the right tool depends entirely on what you're
trying to do.

### The Spectrum of Assistance

It helps to think of AI coding assistants as sitting on a spectrum from **ambient/reactive** at one end to **agentic**
at the other.

At the reactive end, tools like **GitHub Copilot** (in its traditional form) sit close to your cursor. They observe what
you're typing, infer your intent from the surrounding code and comments, and suggest completions inline. The interaction
model is passive: you write, it suggests, you accept or ignore. Copilot is deeply integrated into the IDE and operates
almost entirely on the immediate local context — the file you're editing, perhaps a few open tabs, your recent
keystrokes. It does not browse the web, run your code or orchestrate multi-step tasks. Its intelligence is applied in a
single forward pass: here is the context, what comes next?

At the agentic end, tools like **Claude Code** and **Lovable** take a fundamentally different approach. Rather than
predicting the next token of your code, they reason about goals, decompose tasks into steps, invoke tools, observe the
results and decide what to do next. This loop — reason, act, observe, repeat — is what makes them agents rather than
autocomplete engines.

### How Agentic Tools Work

The engine underneath an agentic coding assistant is a large language model (an LLM) equipped with a set of tools it can
call. When you give Claude Code a task, it doesn't just generate text — it generates _decisions about what actions to
take_. These might include reading a file, running a shell command, searching the codebase for a pattern, fetching
documentation from the web or editing a file. Each action produces a result that is fed back into the model's context,
informing the next decision.

Claude Code reads your codebase, edits files, runs commands and integrates with your development tools. This description
sounds simple, but the underlying mechanism is a continuous loop: the model maintains a running view of the task,
decides on the next tool call, executes it, processes the result and iterates until the task is complete or it needs
your input.

The key insight is that the model's intelligence is not applied once, upfront — it's applied repeatedly, with fresh
information each time, informed by real results from the environment. This is what allows agentic tools to handle
genuinely complex, multi-file tasks that a reactive autocomplete tool simply couldn't approach. Project-specific context
can be injected into this loop at initialisation — Claude Code, for instance, reads a `CLAUDE.md` file from your project
root at the start of every session, shaping every subsequent decision the agent makes.

### Lovable: Generation, Not Assistance

Lovable occupies an interesting and distinct position on this spectrum. Where Claude Code is designed to work _within_
your existing codebase — editing, debugging, extending — Lovable is primarily a **generative** tool aimed at building
applications from scratch through conversation. You describe what you want, and it produces a full-stack application:
component structure, styling, data model and deployment configuration.

The difference in philosophy is significant. Lovable is optimised for the early, greenfield phase of a project, where
you're moving quickly from idea to working prototype and the codebase doesn't yet have accumulated conventions or
complexity. Claude Code is optimised for the sustained development phase — the much longer period where an existing
codebase needs to be understood, extended and maintained. Both are powered by LLMs with tool access, but they are
designed for different moments in a project's life and make different trade-offs.

### OpenCode: Open Source Flexibility

**OpenCode** ([opencode.ai](https://opencode.ai)) takes a different philosophical stance again. It is an open source AI
coding agent that runs in your terminal, IDE or desktop, and is explicitly designed to be model-agnostic. Rather than
locking you into a single LLM provider, OpenCode connects to any model from any provider — Claude, GPT, Gemini and more
— with support for over 75 LLM providers and even locally running models. GitHub Copilot users can log in and use their
existing subscription directly.

This matters for developers who want the agent architecture — the reasoning loop, the tool access, the file editing —
without being tethered to a single vendor's pricing or model quality decisions. OpenCode's approach reflects a broader
trend in the tooling ecosystem: separating the _agent infrastructure_ (how the loop works, how tools are invoked, how
results are processed) from the _model_ (which LLM does the reasoning). The two concerns are increasingly being treated
as independent, and with over 100,000 GitHub stars and 2.5 million monthly developers, OpenCode is clearly meeting a
real demand for that flexibility.

### Cursor: The AI-Native IDE

**Cursor** ([cursor.com](https://www.cursor.com)) takes yet another approach: rather than operating in the terminal or
as an extension, it replaces the editor itself. Built as a fork of VS Code, Cursor embeds AI deeply into the editing
experience — not as an add-on, but as a first-class participant in how you navigate, write and refactor code.

Its most distinctive feature is **Composer**, which handles multi-file edits autonomously: you describe a goal, and it
plans and applies changes across your codebase, showing you diffs before committing them. Tab completion goes beyond
single lines, predicting entire functions based on surrounding context and your recent edits. The net effect is an
editor that feels less like a tool you use and more like one that anticipates you.

Cursor's commercial success — valued at around $29B — reflects how many developers prefer this deeply integrated model
over a terminal agent. It occupies the same AI-native space as Claude Code and Windsurf but bets that most developers
will ultimately want their AI inside their editor, not alongside it.

### The Role of the Context Window

Every LLM-based tool, regardless of where it sits on the spectrum, is fundamentally constrained by its context window —
the maximum amount of text the model can process in a single pass. For agentic tools, managing this window intelligently
is one of the core engineering challenges.

A large repository can contain millions of tokens worth of code. The model can't read all of it at once. Agentic tools
therefore need strategies for deciding _what to include_ in the context at any moment. Common approaches include
semantic search (finding the most relevant files by embedding similarity), language server protocol (LSP) integration to
understand code structure and relationships and explicit user-directed context attachment (the `@filename` pattern).
OpenCode automatically loads the right LSPs for the LLM, which is one example of how tools are increasingly automating
this context management on your behalf.

The context window also explains why session management matters. Longer sessions accumulate more history, which consumes
window space that could otherwise be used for relevant code. Well-designed tools handle this through summarisation,
selective history pruning and the ability to resume sessions with their key decisions intact without replaying every
token.

### Model Context Protocol

One architectural development worth understanding is the **Model Context Protocol (MCP)**, an open standard for
connecting AI agents to external data sources. With MCP, a coding assistant isn't limited to your local files — it can
be connected to your Jira board, your Slack workspace, your Google Drive, your Figma designs or any custom internal
tooling that exposes an MCP server.

This is significant because it shifts the agent's effective context from "what's in this repository" to "what's in your
entire development environment." The agent becomes a genuine participant in your workflow rather than an isolated tool
that only knows about code files.

### Putting It Together

The fundamental difference between a tool like GitHub Copilot and a tool like Claude Code or OpenCode is the degree to
which reasoning is externalised from a single forward pass into a multi-step loop with real-world tool use. Copilot
applies intelligence once, inline, in a fraction of a second. An agentic tool applies intelligence repeatedly, across
minutes or longer, with each step informed by the results of the last.

Both are useful. The right choice depends on the task. For fast, flow-state coding where you want suggestions without
interruption, a reactive autocomplete tool is often the better fit. For complex, multi-file work where you need
something that can genuinely understand a problem and work toward a solution, an agentic tool is more appropriate. And
for the earliest phase of a project — when you're still figuring out what you're building — a generative tool like
Lovable may be the right starting point before you hand the codebase over to something with deeper editing and reasoning
capabilities.

What's clear is that the category is moving rapidly in one direction: more agency, more tool access, more integration
with the broader development environment and increasingly sophisticated strategies for managing the context that makes
all of it possible.

## The Landscape

### 🛠️ AI Coding Tools

| Tool                                                      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Lovable](https://lovable.dev)                            | Uses Anthropic models. Browser-based vibe coding platform that turns natural language prompts into full-stack React/Supabase apps. No local setup required — ideal for rapid prototyping and non-developers. Free tier available; paid plans from around **$25/mo**.                                                                                                                                                                                                                     |
| [Claude Code](https://www.anthropic.com/claude-code)      | Uses Anthropic models. Agentic coding tool that lives in your terminal. Reads, edits and reasons across entire codebases. Excels at large-scale refactoring and complex multi-file tasks with a 200K token context window. Pay-per-use via the Anthropic API — costs vary by usage, with a flat **$100/mo** option available.                                                                                                                                                            |
| [GitHub Copilot](https://github.com/features/copilot)     | Supports multiple models — GPT-4o, Claude and Gemini are all switchable. The original AI coding assistant, deeply integrated into VS Code, JetBrains and the wider GitHub ecosystem. Covers inline completions, chat, agent mode and PR summaries. Free tier available; paid plans from around **$10/mo**.                                                                                                                                                                               |
| [Cursor](https://www.cursor.com)                          | Uses a mix of GPT-4o, Claude and custom Cursor models. AI-native code editor (VS Code fork) purpose-built for AI-first development. Composer mode handles multi-file edits autonomously; Tab completion predicts entire functions. One of the most popular tools in the space, valued at ~$29B. Free trial; Pro around **$20/mo**.                                                                                                                                                       |
| [Windsurf](https://windsurf.com)                          | Uses multiple models including Claude and GPT-4o — switchable. IDE built on VS Code with a unique "Cascade" agentic system that tracks your real-time actions and responds contextually. Acquired by OpenAI in May 2025. Strong value at its price point. Free tier; paid plans from around **$15/mo**. Codeium ([codeium.com](https://codeium.com)) is the underlying platform — it also offers a free standalone completion plugin for all major IDEs as a GitHub Copilot alternative. |
| [Devin](https://devin.ai)                                 | Built on OpenAI models with proprietary RL fine-tuning. The original "autonomous AI software engineer" — operates independently via Slack or a VSCode-style interface, spawning its own environment to plan, code, test and open PRs. Famously launched at $500/mo before dropping prices with Devin 2.0. Entry plans are affordable but serious usage gets expensive quickly.                                                                                                           |
| [Poolside](https://poolside.ai)                           | Uses proprietary in-house models. Enterprise-only platform deploying custom-trained models within your own infrastructure (on-prem or VPC). Targets Global 2000 and defence customers. Backed by Nvidia at a **$12B valuation**. Enterprise pricing only — contact for a quote.                                                                                                                                                                                                          |
| [Amazon Q Developer](https://aws.amazon.com/q/developer/) | Uses Amazon's proprietary models. AWS-native AI coding assistant for building, debugging and transforming code. Deep integration with AWS services and IDEs. Includes a `/transform` feature for upgrading Java/.NET/COBOL codebases. Free tier available; paid plans from around **$19/mo**.                                                                                                                                                                                            |
| [Aider](https://aider.chat)                               | Model-agnostic — bring your own API key (Claude, GPT-4o, Gemini or local models via Ollama). The most popular open-source terminal coding agent. Git-native by design — it stages changes and writes commit messages automatically. Excellent for multi-file refactors across an existing repo. Terminal-first, but terminal-native developers consistently rate it their most productive tool. **Free**; you pay for API calls.                                                         |
| [Continue.dev](https://continue.dev)                      | Model-agnostic — bring your own API key (Claude, GPT-4o, Gemini or run locally via Ollama). Open-source VS Code and JetBrains extension for building fully custom AI coding assistants. Highly configurable; 20K+ GitHub stars. The go-to choice for teams in regulated industries who need 100% on-premise AI. **Free**; you pay for API calls.                                                                                                                                         |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Uses Google's Gemini models. Open-source terminal coding agent that uses a ReAct loop to plan and execute multi-step tasks directly from your shell. Integrates with MCP servers and Google Search for real-time context. Made waves with its extremely generous free tier — **free** via a Google account, with pay-as-you-go or team plans available.                                                                                                                                  |
| [OpenAI Codex CLI](https://github.com/openai/codex)       | Uses OpenAI's Codex models. Open-source terminal coding agent built in Rust. Reads, edits and runs code in your local directory with configurable approval modes. Supports multimodal input, web search and MCP. **Free to install**; usage is bundled with a ChatGPT subscription or pay-as-you-go via API.                                                                                                                                                                             |
| [Cline](https://cline.bot)                                | Model-agnostic — bring your own API key (Claude, GPT-4o, Gemini or local models via Ollama). Open-source VS Code extension that acts as a fully agentic coding assistant. Shows diffs inline before applying them and requires explicit approval before running terminal commands — agentic power with human-in-the-loop control. **Free**; you pay for API calls.                                                                                                                       |
| [OpenCode](https://opencode.ai)                           | Model-agnostic — connects to any of 75+ providers including Claude, GPT, Gemini and local models; or log in with an existing Copilot or ChatGPT subscription. Open-source terminal, IDE and desktop coding agent with 100K+ GitHub stars. LSP-aware, multi-session and privacy-first. **Free** to use; a paid tier offers access to curated hosted models.                                                                                                                               |

### 🧠 Code-Specific Models

These are the underlying models purpose-built or heavily optimised for code generation. Most are accessible via API and
can power your own tooling.

| Model                                                       | Country of origin | Description                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Claude Sonnet 4](https://docs.anthropic.com)               | USA               | Top SWE-bench scores (>72% Verified). 200K context window. Best-in-class for agentic coding, refactoring and multi-file reasoning. Powers Claude Code, Cursor, Windsurf and Lovable. Input: **$3** / Output: **$15** per 1M tokens.                                                           |
| [Claude Opus 4](https://docs.anthropic.com)                 | USA               | Anthropic's most capable model. Superior at complex reasoning, large-scale refactoring and extended agentic workflows where quality takes priority over cost. Same 200K context window as Sonnet. Best reserved for the most demanding tasks. Input: **$15** / Output: **$75** per 1M tokens. |
| [GPT-4.1 / o3](https://platform.openai.com)                 | USA               | o3 scores 69.1% on SWE-bench Verified. GPT-4.1 is strong on multi-language, multi-module enterprise tasks. Powers GitHub Copilot and Devin. GPT-4.1 Input: **$2** / Output: **$8**; o3 Input: **$10** / Output: **$40** per 1M tokens.                                                        |
| [Gemini 2.5 Pro](https://ai.google.dev)                     | USA               | 63.8% on SWE-bench Verified; massive **1M token context window** ideal for large codebases. Strong at multimodal coding (e.g. code from diagrams). Input: **$1.25** (≤200K) / Output: **$10** per 1M tokens.                                                                                  |
| [Codestral 2501](https://mistral.ai/news/codestral)         | France            | Specialist code model supporting **80+ languages** and a 256K context window. 2× faster than its predecessor. Scores 86.6% on HumanEval Python and 95.3% on Fill-in-the-Middle tasks. Optimised for low-latency IDE completions. Input: **$0.30** / Output: **$0.90** per 1M tokens.          |
| [DeepSeek Coder V2 / V3](https://platform.deepseek.com)     | China             | 236B parameter MoE (21B active). Competitive with frontier closed models on coding benchmarks. Extremely aggressive pricing makes it a go-to for cost-conscious teams. Open weights available. Input: **$0.14** / Output: **$0.28** (V3) per 1M tokens.                                       |
| [Qwen2.5-Coder 32B](https://huggingface.co/Qwen)            | China             | Open-weight coding specialist, 32B params. Rivals GPT-4o on HumanEval (91.0%). Strong multilingual coding support across 40+ languages. Can be self-hosted or accessed via API. Via Together/Fireworks: ~**$0.90** / **$0.90** per 1M tokens; free to self-host.                              |
| [Qwen3-Coder 480B](https://huggingface.co/Qwen)             | China             | Massive MoE model. 69.6% on SWE-bench Verified — the highest open-source score to date. Supports 256K+ context and agentic workflows. A genuine open-source challenger to Claude and GPT. Via API: pricing varies by provider; free to self-host.                                             |
| [Devstral / Codestral (Mistral family)](https://mistral.ai) | France            | Devstral Medium is an "agentic coding" model designed for multi-step code planning and automated review. Part of Mistral's growing code-focused family. Open-source version available (Apache 2.0). Devstral via API: **$0.50** / **$1.50** per 1M tokens; open weights free.                 |

_Prices correct as of February 2026. The AI tooling market moves fast — always check vendor pages for the latest._

### Chinese models: a separate conversation

DeepSeek and Qwen (from Alibaba) deserve their own section because they've changed the economics of the space in a way
that can't be ignored.

DeepSeek V3 and R1 arrived in late 2024 and early 2025 and matched or beat frontier Western models on key coding
benchmarks — at a fraction of the reported training cost. Qwen Coder is competitive at its size class. Both are open
weight, meaning you can download and run them yourself.

**The case for using them:**

- API pricing is significantly cheaper than OpenAI or Anthropic
- Open weight models can be run locally with no API costs at all
- Performance on coding tasks is genuinely competitive

**The case for caution:**

- If you use the hosted API versions, your code is processed on Chinese infrastructure. For anything sensitive —
  commercial IP, government work, client code — this is a meaningful concern
- "Open weight" is not the same as open source. The weights are available but the training data and full methodology
  aren't always disclosed
- The models haven't had the same level of independent security auditing as Western counterparts

The practical split: DeepSeek and Qwen running locally via Ollama is a reasonable choice for personal projects and
experimentation. The hosted API versions require a more deliberate decision about what code you're comfortable sending
where.

| Model          | Provider         | Open Weight | Data Concerns               | Cost Tier |
| -------------- | ---------------- | ----------- | --------------------------- | --------- |
| Claude Sonnet  | Anthropic        | No          | Low                         | High      |
| GPT-4o         | OpenAI           | No          | Low-Medium                  | High      |
| Gemini 2.5 Pro | Google           | No          | Low-Medium                  | Medium    |
| DeepSeek V3    | DeepSeek (China) | Yes         | High (hosted) / Low (local) | Low       |
| Qwen Coder     | Alibaba (China)  | Yes         | High (hosted) / Low (local) | Low       |

## Building in a Safety Net

Vibe coding introduces a specific risk that's easy to underestimate: you can end up with working code that you don't
fully understand and that contains security flaws you wouldn't recognise even if you read it.

Human code review breaks down in this situation. If the reviewer doesn't understand the code, they can't meaningfully
assess it. This is where automated tooling becomes not just useful but arguably essential.

### Static analysis tools

It's worth distinguishing between linters and security analysers — they're not the same thing.

**Ruff** is a fast Python linter and formatter. It catches style issues, unused imports, obvious bugs. It's not a
security tool.

**Bandit** analyses Python code for common security issues — hardcoded credentials, use of unsafe functions, SQL
injection patterns. This is what you want running on generated code.

**Semgrep** is more powerful and language-agnostic. It uses rules to detect patterns across a codebase and has a large
library of security-focused rules. For JavaScript, ESLint with the `eslint-plugin-security` plugin covers similar
ground.

All of these can be run as pre-commit hooks or integrated into a CI pipeline via GitHub Actions, so they run
automatically on every code change without requiring manual intervention.

### The critic agent pattern

A more sophisticated approach is to use a second AI agent whose sole job is to review the output of the first. This is
sometimes called a critic agent or adversarial agent — a pattern from multi-agent architecture (see my
[AI Agents article](/notes/ai_agents)).

In practice this means setting up a second model session with a system prompt focused entirely on security review: look
for injection vulnerabilities, check input validation, flag any use of deprecated or unsafe functions, identify missing
error handling. It reviews the generated code before it goes anywhere near a pipeline.

The limitation worth acknowledging: an LLM reviewer can have the same blind spots as an LLM coder. It's not a substitute
for understanding your code, but it catches a meaningful category of issues that static analysis misses — particularly
logical security flaws rather than pattern-based ones.

### A practical pipeline

A reasonable minimal setup for vibe-coded projects:

1. Generate code with your tool of choice
1. Run Bandit or Semgrep automatically via pre-commit hook
1. Run a critic agent review for anything going to production
1. Human sign-off before merge

GitHub Actions is the natural glue for steps 2 and 3 in a team setting.

## Saving Money Without Sacrificing Quality

Token costs add up quickly when you're using agentic tools, because every tool call and every file read consumes
context. A few principles that help.

**Match the model to the task.** Frontier models are expensive. For straightforward tasks — generating a migration,
writing a test, reformatting code — a smaller, cheaper model (Claude Haiku, GPT-4o mini, Gemini Flash) will do the job.
Save the heavy models for complex reasoning tasks.

**Understand what drives cost.** You pay for input tokens (everything sent to the model, including context) and output
tokens (everything the model generates). Long context windows are expensive on the input side. Verbose responses are
expensive on the output side. Prompts that require lots of back-and-forth are expensive overall.

**Write better prompts upfront.** A precise prompt that gets the right result in one shot is significantly cheaper than
a vague prompt that requires five rounds of correction. Time spent on the prompt pays for itself in token savings.

**Use local models for exploration.** When you're experimenting — trying an approach, exploring an API, sketching out a
design — run a local model via Ollama. DeepSeek or Qwen locally costs nothing per token. Switch to a frontier model when
you need the best result.

**Subscription vs. pay-as-you-go.** For heavy daily use, a subscription (Claude Pro, ChatGPT Plus, Cursor) usually wins
on cost. For occasional or project-based use, pay-as-you-go API access gives you more control and visibility over what
you're spending.

## When It Works and When It Doesn't

**Where vibe coding genuinely shines:**

- Prototypes and MVPs where speed matters more than perfection
- Solo developers and non-technical founders building internal tools
- Boilerplate — CRUD operations, API wrappers, data transformations
- Learning — seeing working code helps you understand patterns faster than reading documentation

**Where it struggles:**

- Large, complex codebases where the model can't hold enough context to reason about the whole system
- Security-sensitive code where you need to understand exactly what's happening
- Anything requiring deep domain knowledge the model doesn't have
- Production systems where maintainability and testability matter long-term

The "it works but I don't know why" problem compounds over time. A prototype built with vibe coding that becomes a
production system accumulates decisions nobody understands. When something breaks, debugging becomes archaeology.

A simple heuristic: use these tools freely for anything throwaway or exploratory. Apply more discipline — better
prompts, more review, more understanding — as the code gets closer to production.

## Resources

- [Awsome list-style list of AI coding tools](https://github.com/sourcegraph/awesome-code-ai)
- [Andrej Karpathy — vibe coding post](https://x.com/karpathy/status/1886192184808149383?lang=en)
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code)
- [DeepSeek technical report](https://arxiv.org/abs/2501.12948)
- [Bandit — Python security linter](https://bandit.readthedocs.io/)
- [Semgrep documentation](https://semgrep.dev/docs/)
- [Ollama — run models locally](https://ollama.com/)
- [SWE-bench — coding benchmark](https://www.swebench.com/)
- [LiveCodeBench](https://livecodebench.github.io/)
