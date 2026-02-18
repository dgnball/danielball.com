---
layout: page
title: AI Assisted Coding
---

What follows is my attempt to make sense of the AI-assisted coding landscape — the tools, the models, the risks, and the
trade-offs. I wrote this to educate myself and as a reference for anyone trying to navigate this space without the hype.

## What is Vibe Coding?

Vibe coding is a term coined by Andrej Karpathy in February 2025. The idea is simple: instead of writing code yourself,
you describe what you want in plain language and let an AI generate it. You're coding by intent rather than by syntax.
You might not read every line of what gets produced. You trust the output because it works, not because you understand
it.

This is distinct from earlier AI coding tools like GitHub Copilot, which offered autocomplete and suggestions within
your editor — you were still writing the code, just with assistance. Vibe coding goes further. You describe a feature or
a whole application and the AI builds it.

The result is a spectrum. At one end, experienced developers use these tools to move faster, skipping boilerplate and
delegating the tedious parts. At the other end, people with no coding background at all are building working software —
side projects, internal tools, MVPs — without ever learning to program in the traditional sense.

## How These Tools Actually Work

There are two broad categories worth distinguishing.

**Chat-based assistants** (Claude.ai, ChatGPT) let you describe a problem and receive code in return. You copy it into
your project yourself. The AI has no visibility into your codebase and no ability to run or test anything. It's powerful
but requires you to do the integration work.

**Agentic coding tools** (Claude Code, Cursor, Lovable, Bolt) go further. These tools operate in a loop: they receive a
task, use tools to gather information, take actions, observe the results, and repeat until the task is done. They can
read files, search codebases, run shell commands, install dependencies, and make edits across multiple files
autonomously.

The key concept underpinning all of this is the **context window** — the amount of text an LLM can hold in memory at
once. Larger context windows mean the model can reason about more of your codebase at once, which generally produces
better results. But context costs tokens, and tokens cost money.

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
your repo goes to the API), and quality (a focused context tends to produce better output than a noisy one).

## The Landscape of Models

### General purpose and coding-capable

These are the frontier models from Western labs. They're capable across a wide range of tasks, regularly updated, and
well-documented.

| Model          | Provider  | Notes                                             |
| -------------- | --------- | ------------------------------------------------- |
| Claude Sonnet  | Anthropic | Strong reasoning, good at multi-step coding tasks |
| GPT-4o         | OpenAI    | Solid all-rounder, wide ecosystem support         |
| Gemini 2.5 Pro | Google    | Strong benchmark performance on coding tasks      |

### Agentic coding environments

These are tools built on top of models that add the agentic layer — file access, terminal access, project awareness.

| Tool        | Built on           | Notes                                  |
| ----------- | ------------------ | -------------------------------------- |
| Claude Code | Claude             | Terminal-based, strong autonomy        |
| Cursor      | OpenAI / Anthropic | Editor-native, popular with developers |
| Lovable     | GPT-4o             | Web-based, aimed at non-developers     |
| Bolt        | Various            | Web-based, fast prototyping            |
| Replit      | Various            | Browser IDE with AI integration        |

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
fully understand, and that contains security flaws you wouldn't recognise even if you read it.

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
