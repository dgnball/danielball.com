---
layout: page
title: AI Assisted Coding
---

This is my attempt to make sense of the AI-assisted coding landscape. I wrote this to educate myself and as a reference
for anyone trying to navigate this space without the hype. If you don't want to read this or don't value my opinion,
feel free to skip to read these highly recommended resources instead:

- https://github.com/sourcegraph/awesome-code-ai

## AI Coding Glossary

- **Agentic AI** — An AI system that doesn't just respond to a single prompt but takes sequences of actions
  autonomously: running commands, reading files, making decisions, and iterating toward a goal with minimal human
  intervention between steps.

- **Cargo-Cult Programming** — Writing code by copying patterns or snippets without understanding why they work, hoping
  the result will behave correctly by association. Named after the anthropological phenomenon of mimicking the form of
  something without grasping its function. Stack Overflow made this easy; AI tools risk amplifying it further.

- **Context Window** — The maximum amount of text (code, instructions, conversation history) an AI model can "see" at
  once when generating a response. Larger context windows allow tools like Claude Code to reason across more files
  simultaneously. Measured in tokens.

- **Code Completion** — A feature in IDEs that suggests the next token, method, or block of code as you type, ranging
  from simple symbol lookup (early IDEs) to ML-ranked suggestions (PyCharm) to full-line and multi-line generation
  (GitHub Copilot).

- **Copilot** — [GitHub Copilot](https://github.com/features/copilot), a widely adopted AI code completion tool built on
  OpenAI's Codex model. Sits inside your editor and suggests whole lines or functions in real time as you type, trained
  primarily on public GitHub repositories.

- **Embedding** — A numerical representation of code or text as a vector in high-dimensional space, used by AI systems
  to measure semantic similarity. Allows a model to understand that `fetch_user` and `get_user` are likely related
  concepts even though the strings are different.

- **Fine-tuning** — Taking a general-purpose language model and training it further on a specific dataset — a particular
  codebase, language, or domain — so it becomes more accurate and idiomatic for that context.

- **Hallucination** — When an AI model confidently generates something that is plausible-looking but factually wrong: a
  method that doesn't exist, an API with the wrong signature, a library that was never published. A significant risk in
  AI-assisted coding that makes verification essential.

- **Inference** — The process of running a trained AI model to generate output. Distinct from training. When you ask
  Claude Code to refactor a function, you're triggering inference against a model that was trained earlier.

- **LLM (Large Language Model)** — The class of AI model underpinning modern coding assistants. Trained on vast text and
  code corpora to predict likely next tokens, LLMs develop emergent abilities to reason, explain, translate between
  languages, and generate syntactically and semantically coherent code.

- **Prompt Engineering** — The practice of carefully crafting inputs to an AI model to get better outputs. In coding
  contexts this might mean specifying the language, describing constraints, providing examples of desired style, or
  including relevant error messages to steer the model toward a useful response.

- **RAG (Retrieval-Augmented Generation)** — A technique where an AI system retrieves relevant documents or code
  snippets from an external source (like your codebase or documentation) and injects them into the prompt before
  generating a response, allowing the model to reason over information it wasn't trained on.

- **Static Analysis** — Examining code without executing it, to infer types, detect errors, or suggest completions. The
  basis of pre-AI tools like [Rope](https://github.com/python-rope/rope) and
  [Jedi](https://jedi.readthedocs.io/en/latest/). Fast and deterministic, but limited by what can be known without
  running the program.

- **Token** — The basic unit an LLM processes: roughly a word, part of a word, or a symbol. Models have token limits for
  both input and output. Code tends to be token-dense because of punctuation, indentation, and repeated keywords.

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
and later [GitLab](https://gitlab.com), and lifting patterns from permissively licensed projects. The cognitive model
was matching natural language intent to human-curated solutions or real-world production code. Stack Overflow gave you
explained snippets; GitHub gave you full working implementations you could study in context. It worked remarkably well
for common problems but broke down for anything domain-specific, requiring you to adapt boilerplate to your situation —
often the fiddly part. It also introduced a generation of developers to cargo-cult programming: paste first, understand
later. Crucially, this vast public corpus — millions of answered questions, billions of lines of openly licensed code —
became the training substrate that made AI coding tools possible. Without Stack Overflow's decade of annotated
problem-solution pairs, and without the open source community's collective output on GitHub, there would be nothing
substantial enough for models to learn the shape of good code from.

**Early Code Completion (2000s IDEs)** Tools like early [Eclipse](https://eclipseide.org),
[Visual Studio](https://visualstudio.microsoft.com), and [IntelliJ](https://www.jetbrains.com/idea/) began offering
basic autocomplete — typically triggered by typing a dot after an object. The engine was largely syntax-aware symbol
lookup: parse the file, infer the type, offer members from that class. It was often brittle — break the type chain and
suggestions vanished. Useful for reducing typos in method names but rarely capable of suggesting whole patterns or
intent.

**Python Autocompletion Before AI — Jedi & Rope** Python's dynamic typing made static autocompletion genuinely hard.
Early tools like [Rope](https://github.com/python-rope/rope) and later [Jedi](https://jedi.readthedocs.io/en/latest/)
tackled this with deep static analysis: they'd trace variable assignments, follow imports, infer types through call
chains, and build an in-memory symbol index. Jedi in particular became the backbone of completion in editors like
[Vim](https://www.vim.org), [Emacs](https://www.gnu.org/software/emacs/), [Atom](https://github.com/atom/atom), and
[VS Code](https://code.visualstudio.com). It handled decorators, `*args/**kwargs`, and even some basic type inference
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
a function or error message, paste it into the chat window, describe what you wanted, and get back working code with an
explanation. For the first time you could have a conversation about your code — ask follow-up questions, request a
refactor, or say "that didn't work, here's the error." The model had no access to your wider codebase, no ability to run
code, and no memory between sessions, so you were constantly ferrying context back and forth by hand. Despite the
friction it was a dramatic leap over Stack Overflow for anything non-trivial, because the response was tailored to your
exact situation rather than someone else's similar-but-different problem. Developers quickly learned the art of prompt
crafting — how much context to include, how to describe the desired output, when to paste the full traceback versus a
summary.

**Claude Code** [Claude Code](https://claude.ai/code) operates at a fundamentally different level. Rather than
completing tokens or ranking symbols, it understands intent across an entire codebase. You describe what you want in
plain language, and it can generate, refactor, debug, and explain across multiple files simultaneously. It holds context
about your architecture, naming conventions, and patterns — not just the open file. It reasons about *why* code works,
not just *what* comes next. Used via the CLI, it can run commands, read outputs, and iterate — behaving less like a tool
and more like a junior engineer pair-programming alongside you.

**The Near Future — Agents with Persistent Context** The next step is likely persistent, project-aware agents that
maintain a living model of your codebase between sessions — knowing not just the current state but *why* decisions were
made, tracking tech debt, and proactively surfacing problems before you hit them. Think less "answer my question" and
more "autonomous collaborator" that files its own PRs, writes tests as code changes, and flags when a new feature
conflicts with an architectural decision made six months ago.

**The Speculative Horizon — Self-Improving Systems** Further out, the boundary between tool and team blurs. Models
trained continuously on a specific codebase — learning your idioms, your team's preferences, even your personal style —
could make suggestions indistinguishable from a senior colleague's review. The frontier question isn't capability, it's
trust and verifiability: how do you confidently delegate to a system you can't fully inspect? The tools that win will
likely be the ones that make their reasoning transparent enough that engineers stay in control of the craft, even as the
mechanical labour of coding largely disappears.

## How These Tools Actually Work

There are two broad categories worth distinguishing.

**Chat-based assistants** (Claude.ai, ChatGPT) let you describe a problem and receive code in return. You copy it into
your project yourself. The AI has no visibility into your codebase and no ability to run or test anything. It's powerful
but requires you to do the integration work.

**Agentic coding tools** (Claude Code, Cursor, Lovable, Bolt) go further. These tools operate in a loop: they receive a
task, use tools to gather information, take actions, observe the results and repeat until the task is done. They can
read files, search codebases, run shell commands, install dependencies and make edits across multiple files
autonomously.

The key concept underpinning all of this is the **context window** — the amount of text an LLM can hold in memory at
once. Larger context windows mean the model can reason about more of your codebase at once, which generally produces
better results. But context costs tokens and tokens cost money.

### How Claude Code decides what to read

A common misconception is that agentic tools like Claude Code send your entire codebase to the LLM on every turn. They
don't — that would be prohibitively expensive and often unnecessary.

Instead, Claude Code builds context incrementally through a series of tool calls. It might start by listing the
directory structure to understand the project layout, then read a `package.json` or `pyproject.toml` to understand
dependencies, then follow imports to find the files most relevant to the task at hand. It uses grep-style search to
locate specific functions or patterns. Each of these is a discrete tool call that returns results which the model uses
to decide what to look at next.

What actually gets sent to the LLM is the accumulated results of those tool calls — the files and snippets the model has
decided are relevant — plus the conversation history. Files that weren't explored don't get sent.

One practical implication of this: if you create a `CLAUDE.md` file in your project root, Claude Code will always read
it at the start of a session. This is your opportunity to give the model persistent context about the project —
architecture decisions, conventions, things to avoid — without it having to rediscover them every time.

This matters for three reasons: cost (you're only paying for context that's actually needed), privacy (not everything in
your repo goes to the API) and quality (a focused context tends to produce better output than a noisy one).

## The Landscape

### 🛠️ AI Coding Tools

| Tool                                                      | Model(s) Used                                                                                                                    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                            | Price                                                                                                                                       |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| [Lovable](https://lovable.dev)                            | Claude Sonnet (Anthropic)                                                                                                        | Browser-based vibe coding platform that turns natural language prompts into full-stack React/Supabase apps. No local setup required — ideal for rapid prototyping and non-developers.                                                                                                                                                                                                                                                                  | Free (5 credits/day); Pro from **$25/mo** (100 credits); Business from **$50/mo**; Enterprise custom                                        |
| [Claude Code](https://www.anthropic.com/claude-code)      | Claude Sonnet 4 / Opus 4 (Anthropic)                                                                                             | Agentic coding tool that lives in your terminal. Reads, edits, and reasons across entire codebases. Excels at large-scale refactoring and complex multi-file tasks with a 200K token context window.                                                                                                                                                                                                                                                   | Pay-per-use via **Anthropic API** (included in Claude Max plan at **$100/mo**)                                                              |
| [GitHub Copilot](https://github.com/features/copilot)     | GPT-4o, Claude 3.5/3.7, Gemini (switchable)                                                                                      | The original AI coding assistant, deeply integrated into VS Code, JetBrains, and the wider GitHub ecosystem. Covers inline completions, chat, agent mode, and PR summaries.                                                                                                                                                                                                                                                                            | Free (50 requests/mo); Pro **$10/mo**; Pro+ **$39/mo**; Enterprise **$19/user/mo**                                                          |
| [Cursor](https://www.cursor.com)                          | GPT-4o, Claude Sonnet, custom Cursor models                                                                                      | AI-native code editor (VS Code fork) purpose-built for AI-first development. Composer mode handles multi-file edits autonomously; Tab completion predicts entire functions. One of the most popular tools in the space, valued at ~$29B.                                                                                                                                                                                                               | Free (2-week trial); Pro **$20/mo**; Business **$40/user/mo**                                                                               |
| [Windsurf](https://windsurf.com)                          | Claude Sonnet, GPT-4o, own SWE model (switchable)                                                                                | IDE built on VS Code with a unique "Cascade" agentic system that tracks your real-time actions and responds contextually. Acquired by OpenAI in May 2025. Strong value at its price point.                                                                                                                                                                                                                                                             | Free tier; Pro **$15/mo**; Teams **$35/user/mo**; Enterprise custom                                                                         |
| [Devin](https://devin.ai)                                 | Proprietary (built on OpenAI GPT-4o + RL)                                                                                        | The original "autonomous AI software engineer" — operates independently via Slack or a VSCode-style interface, spawning its own environment to plan, code, test, and open PRs. Famously launched at **$500/mo** before dropping prices with Devin 2.0.                                                                                                                                                                                                 | Core **$20/mo** (9 ACUs, pay-as-you-go); Teams **$500/mo** (250 ACUs); Enterprise custom                                                    |
| [Poolside](https://poolside.ai)                           | Malibu (complex tasks) + Point (fast completions) — proprietary                                                                  | Enterprise-only platform deploying custom-trained models within your own infrastructure (on-prem or VPC). Targets Global 2000 and defence customers. Backed by Nvidia at a **$12B valuation**. No public pricing.                                                                                                                                                                                                                                      | Enterprise only — **contact for pricing**                                                                                                   |
| [Amazon Q Developer](https://aws.amazon.com/q/developer/) | Amazon proprietary + CodeWhisperer                                                                                               | AWS-native AI coding assistant for building, debugging, and transforming code. Deep integration with AWS services and IDEs. Includes a `/transform` feature for upgrading Java/.NET/COBOL codebases.                                                                                                                                                                                                                                                   | Free tier; Individual **$19/mo**; Pro (via AWS) **$19/user/mo**                                                                             |
| [Windsurf (Codeium)](https://codeium.com)                 | See Windsurf above                                                                                                               | Codeium also offers a standalone code completion plugin for all major IDEs as a free GitHub Copilot alternative, powering Windsurf under the hood.                                                                                                                                                                                                                                                                                                     | Free forever (individual); Team plans available                                                                                             |
| [Aider](https://aider.chat)                               | Any (Claude, GPT-4o, Gemini, local via Ollama — BYOK)                                                                            | The most popular open-source terminal coding agent. Git-native by design — it stages changes and writes commit messages automatically. Excellent for multi-file refactors across an existing repo. Terminal-first, so not for everyone, but terminal-native developers consistently rate it their most productive tool. Apache 2.0 licence.                                                                                                            | **Free** (you pay only for your chosen model's API usage)                                                                                   |
| [Continue.dev](https://continue.dev)                      | Any (Claude, GPT-4o, Gemini, local via Ollama — BYOK)                                                                            | Open-source VS Code and JetBrains extension that lets you build fully custom AI coding assistants inside your IDE. Highly configurable via "blocks" — connect your own docs, rules, and models. 20K+ GitHub stars. Used in production by teams at Siemens and Morningstar. The go-to choice for teams in regulated industries who need 100% on-premise AI. Apache 2.0 licence.                                                                         | **Free** (you pay only for your chosen model's API usage; enterprise support available)                                                     |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Gemini 2.5 Pro / Gemini 3 Flash (auto-routed)                                                                                    | Google's open-source terminal coding agent (Apache 2.0). Uses a ReAct loop to plan and execute multi-step tasks — fixing bugs, writing features, improving test coverage — directly from your shell. Integrates with MCP servers and Google Search for real-time context. Gemini 3 Flash scores 78% on SWE-bench Verified for agentic coding. Made waves with its extremely generous free tier, widely seen as a direct shot at Claude Code's pricing. | **Free** (via Google account, generous daily quota); pay-as-you-go via Gemini API key; Code Assist Standard/Enterprise from **$19/user/mo** |
| [OpenAI Codex CLI](https://github.com/openai/codex)       | GPT-5 Codex / codex-mini (default)                                                                                               | OpenAI's open-source terminal coding agent (MIT licence), built in Rust. Reads, edits, and runs code in your local directory with configurable approval modes. Supports multimodal input (screenshots, wireframes), web search, MCP, and parallel cloud tasks via Codex Web. The CLI is free to install; usage is bundled with your ChatGPT subscription — no separate Codex fee.                                                                      | **Free to install**; usage included in ChatGPT Plus (**$20/mo**), Pro (**$200/mo**), Business (**$30/user/mo**); or pay-as-you-go via API   |
| [Cline](https://cline.bot)                                | Any (Claude, GPT-4o, Gemini, local via Ollama — BYOK)                                                                            | Open-source VS Code extension (MIT licence) that acts as a fully agentic coding assistant inside your editor. Shows diffs inline before applying them and requires explicit approval before running terminal commands — giving you agentic power with human-in-the-loop control. Model-agnostic and fully auditable. One of the most popular open-source coding agents in 2025/26 alongside Aider.                                                     | **Free** (you pay only for your chosen model's API usage)                                                                                   |
| [OpenCode](https://opencode.ai)                           | Any — 75+ providers via Models.dev, incl. Claude, GPT, Gemini, local models; GitHub Copilot and ChatGPT Plus/Pro login supported | Open-source terminal, IDE, and desktop coding agent with 100K+ GitHub stars and 2.5M monthly users. LSP-aware, multi-session (run parallel agents on the same project), and privacy-first — no code or context is stored. Works across terminal, VS Code extension, and a beta desktop app. Paid **Zen** tier provides access to pre-benchmarked, optimised models for coding agents.                                                                  | **Free** (BYOK or use existing Copilot/ChatGPT sub); **Zen** paid tier for curated hosted models                                            |

### 🧠 Code-Specific Models

These are the underlying models purpose-built or heavily optimised for code generation. Most are accessible via API and
can power your own tooling.

| Model                                                       | Made By         | Highlights                                                                                                                                                                                                                       | API Pricing (per 1M tokens)                                                 | Link                                                   |
| ----------------------------------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------ |
| [Claude Sonnet 4](https://www.anthropic.com/api)            | Anthropic       | Top SWE-bench scores (>72% Verified). 200K context window. Best-in-class for agentic coding, refactoring, and multi-file reasoning. Powers Claude Code, Cursor, Windsurf, and Lovable.                                           | Input: **$3** / Output: **$15**                                             | [docs.anthropic.com](https://docs.anthropic.com)       |
| [GPT-4.1 / o3](https://platform.openai.com)                 | OpenAI          | o3 scores 69.1% on SWE-bench Verified. GPT-4.1 is strong on multi-language, multi-module enterprise tasks. Powers GitHub Copilot and Devin.                                                                                      | GPT-4.1 Input: **$2** / Output: **$8**; o3 Input: **$10** / Output: **$40** | [platform.openai.com](https://platform.openai.com)     |
| [Gemini 2.5 Pro](https://ai.google.dev)                     | Google DeepMind | 63.8% on SWE-bench Verified; massive **1M token context window** ideal for large codebases. Strong at multimodal coding (e.g. code from diagrams).                                                                               | Input: **$1.25** (≤200K) / Output: **$10**                                  | [ai.google.dev](https://ai.google.dev)                 |
| [Codestral 2501](https://mistral.ai/news/codestral)         | Mistral AI      | Specialist code model supporting **80+ languages** and a 256K context window. 2× faster than its predecessor. Scores 86.6% on HumanEval Python and 95.3% on Fill-in-the-Middle tasks. Optimised for low-latency IDE completions. | Input: **$0.30** / Output: **$0.90**                                        | [mistral.ai](https://mistral.ai)                       |
| [DeepSeek Coder V2 / V3](https://platform.deepseek.com)     | DeepSeek        | 236B parameter MoE (21B active). Competitive with frontier closed models on coding benchmarks. Extremely aggressive pricing makes it a go-to for cost-conscious teams. Open weights available.                                   | Input: **$0.14** / Output: **$0.28** (V3)                                   | [platform.deepseek.com](https://platform.deepseek.com) |
| [Qwen2.5-Coder 32B](https://huggingface.co/Qwen)            | Alibaba Cloud   | Open-weight coding specialist, 32B params. Rivals GPT-4o on HumanEval (91.0%). Strong multilingual coding support across 40+ languages. Can be self-hosted or accessed via API.                                                  | Via Together/Fireworks: ~**$0.90** / **$0.90**; free to self-host           | [huggingface.co/Qwen](https://huggingface.co/Qwen)     |
| [Qwen3-Coder 480B](https://huggingface.co/Qwen)             | Alibaba Cloud   | Massive MoE model. 69.6% on SWE-bench Verified — the highest open-source score to date. Supports 256K+ context and agentic workflows. A genuine open-source challenger to Claude and GPT.                                        | Via API: pricing varies by provider; free to self-host                      | [huggingface.co/Qwen](https://huggingface.co/Qwen)     |
| [Devstral / Codestral (Mistral family)](https://mistral.ai) | Mistral AI      | Devstral Medium is an "agentic coding" model designed for multi-step code planning and automated review. Part of Mistral's growing code-focused family. Open-source version available (Apache 2.0).                              | Devstral via API: **$0.50** / **$1.50**; open weights free                  | [mistral.ai](https://mistral.ai)                       |

*Prices correct as of February 2026. The AI tooling market moves fast — always check vendor pages for the latest.*

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

- [Andrej Karpathy — vibe coding post](https://x.com/karpathy/status/1886192184808149383?lang=en)
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code)
- [DeepSeek technical report](https://arxiv.org/abs/2501.12948)
- [Bandit — Python security linter](https://bandit.readthedocs.io/)
- [Semgrep documentation](https://semgrep.dev/docs/)
- [Ollama — run models locally](https://ollama.com/)
- [SWE-bench — coding benchmark](https://www.swebench.com/)
- [LiveCodeBench](https://livecodebench.github.io/)
