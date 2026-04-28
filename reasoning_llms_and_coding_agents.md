# Reasoning LLMs and Coding Agents

### DS542 — Deep Learning for Data Science

Based on two essays by Sebastian Raschka:
- [*Understanding Reasoning LLMs* (Feb 2025)](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)
- [*Components of a Coding Agent* (Apr 2026)](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent)

.footnote[.red.bold[*] Created with Claude]

---

## Today's Roadmap

**Part 1 — Reasoning LLMs**

 1. What is a "reasoning model"?
 2. When should we (and shouldn't we) use one?
 3. The DeepSeek-R1 family as a case study
 4. Four ways to build/improve reasoning models
 5. Doing it on a budget: Sky-T1, TinyZero, Journey Learning

**Part 2 — Coding Agents**

6. LLM vs. reasoning model vs. agent
7. Six components of a coding harness
8. Why a good harness can matter more than the model

**Wrap-up** — takeaways, discussion

---

# Part 1: Reasoning LLMs

---

## Stage 4 of the LLM Lifecycle

![LLM development stages 1-3 plus specialization](https://substackcdn.com/image/fetch/$s_!QwUc!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd6ebc5c9-461f-4d3a-889b-b8ea4e14e5ba_1600x830.png)

1. **Pre-training** — next-token prediction on web-scale text
2. **Supervised fine-tuning (SFT)** — instruction following
3. **Preference tuning** — RLHF / DPO / etc.
4. **Specialization** — RAG, code assistants, **reasoning models**, …

Reasoning models are one of the most important specializations to emerge in 2024–2025. Specialization *adds* capability — it does not replace general-purpose LLMs.

---

## Defining "Reasoning"

No universally agreed-upon definition, but a working one:

> **Reasoning** = answering questions that require complex, multi-step generation with intermediate steps.

| Query | Reasoning needed? |
|---|---|
| "What is the capital of France?" | No — factual lookup |
| "A train at 60 mph travels 3 hours. How far?" | Yes — relate distance, speed, time |
| "Prove that √2 is irrational." | Yes — multi-step proof |

--

Daniel Kahneman, _Thinking, Fast and Slow_.

* _System 1 (fast)_ thinking and _System 2 (slow)_ thinking.

---

## Where Does "Reasoning" Show Up?

<img src="https://substackcdn.com/image/fetch/$s_!8oZo!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff2987079-25f4-45fb-a020-1ac936ed16cb_1424x820.png" alt="Regular LLM vs reasoning model output" style="max-width: 60%; display: block; margin: 0.4em auto;" />

Intermediate steps can appear in two places:

1. **In the visible response** — the model writes out its work
2. **In hidden iterations** — e.g. OpenAI's `o1` runs multiple internal passes; only the final answer is shown to the user

Most modern reasoning models do **both**. Non-reasoning LLMs can *also* produce intermediate steps when prompted — the difference is *default behavior* and *training*.

---

## When Is a Reasoning Model the Right Tool?

**Good fit**
- Math proofs, competition problems
- Logic puzzles, riddles
- Complex / multi-file coding tasks
- Multi-step planning under constraints

**Poor fit**
- Summarization
- Translation
- Fact retrieval / basic Q&A
- Short-form creative writing

Reasoning models are **more expensive**, **more verbose**, and can **overthink** simple prompts.

> **Rule of thumb:** use the right tool for the task.

---

## Strengths & Weaknesses

![Strengths and weaknesses of reasoning models](https://substackcdn.com/image/fetch/$s_!lnf2!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46dbe029-ab7d-4278-8dfe-7bc4af79a103_1352x524.png)

| **Strengths** | **Weaknesses** |
|---|---|
| Excellent on complex, multi-step problems | Higher per-token cost |
| Self-verification / error correction | Longer responses, higher latency |
| Better math & coding benchmarks | Can "overthink" easy prompts |
| Produces inspectable reasoning traces | Inference costs scale with thought length |

This trade-off is why providers offer *both* a regular and a "thinking" variant.

---

## Case Study: The DeepSeek-R1 Family

DeepSeek released **three distinct models**, each teaching a different lesson:

| Model | Lesson |
|---|---|
| **DeepSeek-R1-Zero** | Reasoning can emerge from **pure RL**, no SFT |
| **DeepSeek-R1** | **SFT + RL** produces the strongest model |
| **DeepSeek-R1-Distill** | Smaller models catch up via **SFT distillation** |

All three are built on top of the [DeepSeek-V3 671B](https://arxiv.org/abs/2412.19437) base model. The
[technical report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) (or [arXiv](https://arxiv.org/abs/2501.12948)) is freely available and serves as a blueprint for the field.

---

## The DeepSeek Training Pipeline (Bird's-Eye View)

![DeepSeek R1 training pipeline for all three models](https://substackcdn.com/image/fetch/$s_!z-dr!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdb19df56-c5bf-4a0c-aafb-4629a39b13f5_1542x1166.png)

All three models start from the same DeepSeek-V3 671B base. The recipes diverge — and each teaches a different lesson. We'll walk through each branch in turn.

---

# The 4 Ways to Build a Reasoning Model

1. **Inference-time scaling** — no retraining
2. **Pure RL** — no SFT before RL
3. **SFT + RL** — the flagship recipe
4. **Pure SFT / distillation** — small, efficient models

These are *complementary*, not mutually exclusive.

---

## Approach 1 — Inference-Time Scaling

**Idea:** spend more *compute at inference* to get better answers. No weight updates. Analogy: humans give better answers when given more time.

Three common flavors:
- **Chain-of-thought (CoT) prompting** — "think step by step"
- **Majority voting / self-consistency** — sample N answers, pick most common
- **Search-based decoding** — beam search, MCTS, process reward models

![Zero-shot chain-of-thought prompting example](https://substackcdn.com/image/fetch/$s_!VFAa!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F523eee5e-afb6-4019-a11b-e0a291d2c286_1600x419.png)

Adding *"Let's think step by step"* shifts the model into a different response distribution — often turning wrong answers into right ones. (Kojima et al., 2022)

???
MCTS: Monte Carlo Tree Search

---

## Search-Based Inference Scaling

![Different search-based methods with process reward model](https://substackcdn.com/image/fetch/$s_!YGJO!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5cb10e5a-738b-4c9e-ba65-5850d4793706_1600x919.png)

A **process reward model (PRM)** scores partial reasoning chains. Common variants:
- **Best-of-N** — sample N answers, pick the highest-scored
- **Beam search with PRM** — keep top-k partial chains at each step
- **MCTS** — explore branching trees of reasoning, using stats from previous simulations to balance exploration and exploitation

See [Snell et al., *Scaling LLM Test-Time Compute Optimally* (2024)](https://arxiv.org/abs/2408.03314).

---

## A Surprising Finding from DeepSeek

The DeepSeek-R1 report puts PRM-based and MCTS-based approaches under **"unsuccessful attempts"**.

Their finding:
- Explicit search didn't help them much
- But R1 *naturally* produces longer reasoning chains than V3
- This is an **implicit** form of inference-time scaling

**Caveat:** explicit inference-time scaling often lives at the *application layer* (e.g. a wrapper around the model), not inside the weights. It is widely suspected that OpenAI's `o1`/`o3` use some form of it — which would help explain why they are pricey per token.

---

## Approach 2 — Pure Reinforcement Learning

<img src="https://substackcdn.com/image/fetch/$s_!_9Z-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5bb6ecc-7e46-45fe-abff-1eb02e6b0e3a_1556x1162.png" alt="DeepSeek-R1-Zero development process" width="400" />

The headline finding of DeepSeek-R1:

> **Reasoning can emerge as a learned behavior from RL alone, with no SFT warm-start.**

Recipe for **R1-Zero**: start from DeepSeek-V3 base, skip SFT entirely, apply RL with *rule-based* rewards. This is unusual — standard RLHF has an SFT stage first.

---

## The Reward Design for R1-Zero

Two rule-based rewards — no human preference model needed:

**Accuracy reward**
- Code → compile and run against tests (LeetCode-style)
- Math → deterministic verification of final answer

**Format reward**
- An LLM judge checks that responses wrap their reasoning in `<think> … </think>` tags
- Enforces a structural convention, not a style preference

> **Key point:** when the task has a *verifiable* answer, rule-based rewards sidestep the reward-modeling bottleneck of RLHF.

---

## The "Aha!" Moment

![The emergent "Aha!" moment during R1-Zero training](https://substackcdn.com/image/fetch/$s_!Prn2!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F30f8e37b-ba60-49d2-a95e-9c06b2033ee4_1600x1019.png)

During R1-Zero training, the model spontaneously began to **pause, reconsider, and rewrite** its approach mid-answer. Nobody trained it to do that — the behavior emerged from reward alone.

R1-Zero isn't the strongest reasoning model — but it is the **proof of concept** that reasoning can be *grown*, not just *taught*.

---

## Approach 3 — SFT + RL (the Flagship Recipe)

<img src="https://substackcdn.com/image/fetch/$s_!19pK!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdf7f99f0-d154-49e5-b60a-4d148e0a61be_1548x1154.png" alt="DeepSeek-R1 development process" width="400" />

This is how you get **DeepSeek-R1** — and probably `o1` and friends. Four stages:

1. Use R1-Zero to generate "cold-start" SFT data → instruction-tune V3
2. RL stage (accuracy + format + **language-consistency** reward)
3. Collect new SFT data: 600K CoT examples + 200K general examples
4. Final RL stage with *both* verifiable rewards and human-preference rewards

---

## Why Add a Language-Consistency Reward?

A quirky failure mode of reasoning-trained LLMs:

> **Language mixing** — the model starts answering in English, then drifts into Chinese mid-chain-of-thought, then back.

Why? Because the model's best reasoning path in latent space may not respect language boundaries.

Fix: a reward term that penalizes mixing languages within a single response.

Minor-looking detail, major UX improvement.

---

## R1 vs. R1-Zero: Does the Extra Work Pay Off?

![Benchmark comparison of OpenAI o1 and DeepSeek R1](https://substackcdn.com/image/fetch/$s_!22Cm!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff7f73f16-db4e-4047-89b0-823f16cefb33_1556x490.png)

SFT + RL consistently beats pure RL on challenging benchmarks. The RL-only model (R1-Zero) is scientifically more interesting; the SFT+RL model (R1) is practically more useful.

R1 is also broadly competitive with `o1` — and substantially cheaper at inference time.

---

## Approach 4 — Pure SFT (Distillation)

<img src="https://substackcdn.com/image/fetch/$s_!xUjE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7db7c46b-fe67-49f4-9f65-b0e7b7e5ac08_1444x1174.png" alt="DeepSeek-R1-Distill development process" width="300" />

DeepSeek also shipped **R1-Distill** models: Qwen and Llama bases, sizes 1.5B–70B, fine-tuned on outputs from R1.

> ⚠️ This is **not classical knowledge distillation** (no logit matching). It's instruction fine-tuning on R1-generated answers.

Two reasons: **efficiency** (7B/14B run on a laptop) and **research signal** (how far does pure SFT get you?).

---

## Distillation Results (R1-Distill)

![Benchmark comparison of distilled versus non-distilled models](https://substackcdn.com/image/fetch/$s_!XwZe!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Febc749fb-6a79-483f-bcda-b219f284bc09_1168x604.png)

The distilled models are surprisingly strong:
- **R1-Distill-Qwen-32B** ≈ matches `o1-mini` on many reasoning benchmarks
- **R1-Distill-Llama-70B** approaches full R1 on some tasks
- Far cheaper to serve than the 671B R1

Plausible interpretation: `o1-mini` may itself be a distilled version of `o1`. Tradeoff: distillation needs a *stronger teacher model to already exist*.

---

## A Revealing Experiment: Pure RL on a Small Model

![Benchmark comparison of distillation vs RL on a smaller 32B model](https://substackcdn.com/image/fetch/$s_!5_5L!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F05514c9f-eb04-496b-bd98-bb4710c65b14_1448x408.png)

DeepSeek also tried applying R1-Zero's pure-RL recipe to **Qwen-32B** directly. Result: worse than R1-Distill-Qwen-32B.

Interpretation:
- **Pure RL** is effective when the base model is already large/strong enough
- **Pure SFT on high-quality reasoning data** is more effective at smaller scales

Small base model → distill. Strong base model → RL.

---

## Summary: When to Use Which Approach

| Approach | Strength | Weakness |
|---|---|---|
| **1. Inference-time scaling** | No retraining, immediate gains | Ongoing per-query cost |
| **2. Pure RL** | Research insights, clean signal | Needs strong base model |
| **3. SFT + RL** | Best overall quality | Most expensive pipeline |
| **4. Distillation (SFT)** | Cheap, efficient, deployable | Ceilinged by teacher quality |

Practice: most production systems combine **SFT + RL (training)** with **inference-time scaling (serving)**.

---

# Reasoning Models on a Budget

You do not need millions of dollars to play with these ideas.

---

## Sky-T1 — $450 to Train an o1-Preview-Level Model

![Sky-T1 training cost and performance](https://substackcdn.com/image/fetch/$s_!Y8HI!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8865a313-2326-4f07-a6dc-72cc94cb2ebe_1364x570.png)

From NovaSky (UC Berkeley), Jan 2025:
- **Base model:** Qwen2.5-32B
- **Data curation via rejection sampling:** generate many candidate traces from a strong teacher, keep only the ones whose final answer verifies (math / code tests), SFT on the survivors.
- **Technique:** SFT only, no RL
- **SFT dataset:** just **17K examples** (generated from a stronger model)
- **Total training cost:** **~$450** — less than an AI conference registration
- **Result:** matches `o1-preview` on several reasoning benchmarks

Moral: high-quality, well-curated data > lots of data.

---

## TinyZero — Pure RL for $30

<img src="https://substackcdn.com/image/fetch/$s_!Ykdn!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6111f4b4-cfb9-494c-8390-ec251702914b_1600x955.png" alt="TinyZero self-verification capability" style="width: 50%; margin: 0.4em auto;" />

From Berkeley, Jan 2025:
- **Base model:** Qwen2.5-3B
- **Technique:** pure RL (R1-Zero-style) on the *Countdown* number puzzle
- **Total training cost:** **<$30**
- **Observation:** even this small model developed **self-verification** behavior

Shows the R1-Zero finding replicates at very small scale, on a narrow domain. Not a general reasoner — but a powerful proof of concept.

---

## Journey Learning — A Twist on SFT

![Journey learning vs. shortcut learning](https://substackcdn.com/image/fetch/$s_!TxCO!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7a0bfcd0-6d93-4c91-a0d6-28178839b7cf_1492x724.png)

From [*O1 Replication Journey — Part 1* (Oct 2024)](https://arxiv.org/abs/2410.18982):

**Shortcut learning (standard SFT):** train only on *correct* paths — model learns "what a good answer looks like."

**Journey learning:** train on *both correct and incorrect* paths with reflection — "I tried X, that didn't work because Y, so I tried Z."

Resembles the self-correction that emerges in RL — but achievable with pure SFT.

---

## Prompting Reasoning Models — Quick Tips

Reasoning models behave differently from regular LLMs. From the R1 paper and community findings:

1. **Zero-shot beats few-shot** — few-shot examples often *hurt* reasoning models
2. **Describe the problem directly** — skip elaborate prompting patterns
3. **Keep language consistent** — mixing languages in the prompt causes mixing in the output
4. **Let it think** — don't ask for "short" answers on hard problems

> Your elaborate prompt-engineering tricks for GPT-3.5 may actively *hurt* `o1` and R1.

---

## Part 1 Takeaways

- "Reasoning LLM" ≈ LLM specialized for multi-step, verifiable tasks
- Four complementary techniques: **inference-time scaling, pure RL, SFT+RL, distillation**
- **SFT + RL** is the flagship recipe (DeepSeek-R1, probably o1)
- **Reasoning can emerge** from pure RL with rule-based rewards (R1-Zero's headline finding)
- **Distillation** is the most cost-effective path for smaller models
- Budget-friendly variants exist: **Sky-T1 ($450)**, **TinyZero (<$30)**

Now: what happens when we put one of these models inside an **agent**?

---

# Part 2: Coding Agents

---

## Why Talk About Agents?

<img src="https://substackcdn.com/image/fetch/$s_!zcE_!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc90147bc-4574-4b52-914f-8bda96620063_3467x2270.png" alt="Claude Code CLI, Codex CLI, and Mini Coding Agent side by side" style="max-width: 480px; display: block; margin: 1em auto;" />

In 2024–2025, a lot of practical LLM progress was **not** about better models. It was about **better scaffolding around the models**: tool use, context management, memory, control loops.

This is why **Claude Code** or **Codex CLI** can feel dramatically more capable than the *same model* in a plain chat interface.

> The model is the engine. The harness is the car.

---

## LLM vs. Reasoning Model vs. Agent

![Relationship between LLM, reasoning LLM, and agent harness](https://substackcdn.com/image/fetch/$s_!if1o!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F09a4d839-d572-4eab-a2ee-47f644a746e5_3501x885.png)

| Concept | What it is |
|---|---|
| **LLM** | A next-token model |
| **Reasoning model** | An LLM optimized to produce intermediate reasoning and verify itself |
| **Agent** | A control loop that uses a model **+ tools + memory + environment feedback** |
| **Agent harness** | Software scaffold around the agent (prompts, tools, state, control flow) |
| **Coding harness** | Task-specific harness for software engineering |

Claude Code and Codex CLI are **coding harnesses**.

---

## The Agent Loop

An agent repeatedly runs a loop inside an environment:

```
        ┌─── observe ──► gather info from environment
        │
        ▼
      inspect ──► analyze the info
        │
        ▼
      choose ──► pick the next action
        │
        ▼
        act ──► execute it (tool call, edit, command)
        │
        └────► back to observe
```

Stop when the goal is met (or budget/limit is reached).

---

## A Coding Harness Has Three Layers

![Three layers of a coding harness: model, agent loop, runtime supports](https://substackcdn.com/image/fetch/$s_!l4hd!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F76f2c37e-1996-4f30-96cd-e2e555169873_2227x992.png)

1. **Model family** — the LLM / reasoning model (the engine)
2. **Agent loop** — observe → inspect → choose → act
3. **Runtime supports** — repo context, tools, caching, memory, sandboxing

> **Same model + better harness = dramatically different UX.**

---

## The Claim: Harness > Model (Often)

Today's frontier LLMs from different vendors have roughly comparable raw capability.

Raschka's claim:

> *If you drop a strong open-weight model (say GLM-5) into the same harness as Codex, it will likely perform on par with GPT-5.4 in Codex or Claude Opus 4.6 in Claude Code.*

The harness is frequently the distinguishing factor.

(Small caveat: there is usually some *harness-specific post-training*, e.g. OpenAI maintains `gpt-5.3` and `gpt-5.3-codex` variants.)

---

## Coding Is Not Just "Writing Code"

Why coding needs a harness more than most tasks:

- **Repo navigation** — thousands of files, which ones matter?
- **Function lookup** — where is `encode_batch` defined?
- **Diff application** — edit existing code, don't rewrite files
- **Test execution** — run tests, read output, fix failures
- **Error inspection** — parse tracebacks, map to source
- **Iterative feedback** — try, observe, revise

Next-token prediction alone doesn't solve these. A harness does.

---

# The Six Components of a Coding Harness

![Main harness features overview](https://substackcdn.com/image/fetch/$s_!iPcp!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F82c0f343-afec-4a9f-b8fe-b60f7ad5db5f_3396x971.png)

1. **Live repo context**
2. **Prompt shape and cache reuse**
3. **Structured tools, validation, and permissions**
4. **Context reduction and output management**
5. **Transcripts, memory, and resumption**
6. **Delegation with bounded subagents**

We'll walk through each.

---

## Component 1 — Live Repo Context

<img src="https://substackcdn.com/image/fetch/$s_!mPz4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2e3a1e6a-4eb5-4b60-b52f-0a4ee880dbca_2085x1428.png" alt="Workspace summary combined with user request" width="300" />

When the user says *"fix the tests"*, the model cannot answer in a vacuum. The harness must know:

- Git repo? Branch? Uncommitted changes?
- Is there an `AGENTS.md` / `CLAUDE.md` / `README`?
- Repo layout? What build/test commands are defined?

--

**Without it:** *"I'll run `pytest`"* — wrong, this project uses `make test`.
**With it:** *"`AGENTS.md` says tests run via `make test`. Running now…"*

This is why files like `AGENTS.md` and `CLAUDE.md` have become a de facto ecosystem convention.

---

## Component 2 — Prompt Shape and Cache Reuse

Naive approach: on every turn, concatenate everything and send it to the model.

Problem: **most of that prompt doesn't change between turns.**

- System instructions — stable
- Tool descriptions — stable
- Workspace summary — mostly stable
- Short-term memory — changes
- Recent transcript — changes
- Latest user turn — changes every time

---

## Split Into "Stable Prefix" + "Dynamic Tail"

![Stable prompt prefix and changing session state](https://substackcdn.com/image/fetch/$s_!keF3!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F92d9467c-1333-40f3-8d5d-c8bc0ffebb11_2698x1001.png)

The **stable prefix** contains system instructions, tool descriptions, and workspace summary — all mostly unchanging across turns.

The **dynamic section** contains short-term memory, recent transcript, and the new user request.

Smart runtimes **cache the Key-Value weights for the prefix** → massive latency and cost wins on long coding sessions. Anthropic and OpenAI both expose prompt caching APIs for exactly this reason.

---

## What is Prompt Caching?

**Problem:** every API call re-processes the full prompt through all attention layers — expensive for long system prompts, tool schemas, and growing agent transcripts.

--

**Idea:** the provider caches the **KV (key/value) tensors** computed for a prompt *prefix* on their servers. Identical prefixes on later calls skip the prefill and load cached state.

--
- **Prefix-only, exact-token match** — one character change invalidates the rest
- **TTL-based eviction** (Anthropic: 5 min or 1 hr; OpenAI: ~5–10 min idle)
- **Pricing:** writes cost *more* (~1.25–2x), reads cost *much less* (~0.1–0.5x)

---

## Prompt Caching in Practice

**Providers:**
- **Anthropic** — explicit `cache_control` breakpoints (up to 4)
- **OpenAI** — automatic for prompts ≥1024 tokens
- **Google Gemini** — explicit `CachedContent` handle

**Why it's critical for coding agents:**
- Agents replay a *growing* transcript on every tool-call iteration
- Large, stable system prompts + tool schemas + file context
- Without caching: a 20-step agent task reprocesses the full prompt 20×
- With caching: each turn only prefills the new tool result → **5–10× lower latency, ~10× lower cost**

**Design rule:** put **stable content first** (system prompt → tools → docs → history → new user turn). Never interpolate timestamps or request IDs near the top.

---

## Component 3 — Structured Tools

A plain LLM can *suggest* shell commands in prose:

> *"You could run `pytest tests/test_auth.py -v`"*

A coding agent *actually runs them* — and uses the output. But not via free-form generation. It uses **structured tool calls**:

- `list_files(path)`
- `read_file(path, start, end)`
- `search(pattern, path)`
- `run_shell(cmd)`
- `write_file(path, contents)`
- `apply_diff(path, diff)`

---

## The Tool-Use Flow

![Tool-use flow: model emits action, harness validates, executes, feeds back](https://substackcdn.com/image/fetch/$s_!yL-u!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7aff251f-0ce2-44f1-9792-5449c24d5600_2172x927.png)

1. Model emits a **structured action**
2. Harness **validates**: known tool? valid args? path inside workspace? needs approval?
3. (If needed) prompt user for **approval**
4. **Execute**, capture output
5. Feed **clipped result** back into the loop

---

## Why Validation & Permissions Matter

<img src="https://substackcdn.com/image/fetch/$s_!nD22!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6ff4770e-c3a5-442f-a657-ef9bec6b9862_2493x2031.png" alt="Tool call approval request in the Mini Coding Agent" style="max-width: 35%; height: auto; display: block; margin: 0.8em auto;">

The harness restricts the model's freedom — on purpose.

- **Rejects malformed actions**
- **Enforces path sandboxing** — only files inside the repo
- **Approval gates dangerous actions** — rm, force push, network calls
- **Limits blast radius** — no `sudo`, no writes outside workspace

Counter-intuitively, **giving the model less freedom makes it more useful and more trustworthy**.

---

## Component 4 — Minimizing Context Bloat

Coding agents are context-hogs. Every step can produce:

- A file read (hundreds of lines)
- A tool output (stack traces, logs)
- A search result (many hits)
- A diff

Left unchecked, the context fills up in a few turns. Even with million-token context windows, performance degrades on long contexts (and cost scales with tokens).

---

## Two Compaction Strategies

![Context compaction: clipping and transcript reduction](https://substackcdn.com/image/fetch/$s_!ksfT!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1d61d701-1f2b-4010-9a1f-2cefc7265bde_2495x1213.png)

**Clipping** — truncate long outputs (file reads, test logs, search results); show head/tail, elide middle. Prevents any one thing from dominating the budget.

**Transcript reduction** — turn full session history into a smaller summary.
- **Recency-weighted**: keep recent events richer, compress older ones more
- **Deduplicate repeat file reads** — don't show the same file 5 times

---

## Context Quality > "Model Quality" (Often)

Raschka's aside:

> *"A lot of apparent 'model quality' is really context quality."*

If you've ever watched Claude Code or Cursor repeatedly re-read the same file, or fail to recall something from 20 turns ago — that's a context-management problem, not usually a model problem.

Good context engineering is the "boring" infrastructure work that separates great coding agents from mediocre ones.

---

## Component 5 — Structured Session Memory

<img src="https://substackcdn.com/image/fetch/$s_!xWhc!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe58efdfb-9c19-42e8-a016-90b41b51ef15_2438x1346.png" alt="Full transcript and working memory as two JSON files" style="max-width: 40%; height: auto;">

Two layers of state, with different jobs:

| Layer | Format | Role |
|---|---|---|
| **Full transcript** | JSONL, append-only | Durable record. Resumable if agent crashes. |
| **Working memory** | JSON, structured | Distilled state: current task, key files, recent notes. Updated and compacted, not just appended. |

Compact transcript (Component 4) ≠ working memory:
- **Compact transcript** → for prompt reconstruction
- **Working memory** → for task continuity

???
JSONL: JSON Lines

JSONL is a format where each line is a valid JSON object. It is similar to JSON, but each object is on a separate line. This is useful for storing large datasets or logs.

Example:

```jsonl
{"name": "John", "age": 30}
{"name": "Jane", "age": 25}
```

---

## What Goes Into Working Memory?

Examples of structured working-memory fields:

```json
{
  "current_task": "fix failing auth tests",
  "key_files": ["src/auth.py", "tests/test_auth.py"],
  "recent_notes": [
    "test_login_timeout fails due to missing mock",
    "auth.py line 142 uses deprecated API"
  ],
  "open_questions": ["should we keep backward compat?"]
}
```

Small, curated, editable. Not a dump of everything that happened.

---

## Why Separate Memory from Transcript?

- **Transcripts** are honest but noisy and linear
- **Memory** is curated and queryable
- **Transcript** = "what happened"
- **Memory** = "what matters right now"

Close your laptop, reopen tomorrow → the agent loads both. The transcript grounds it; the working memory reminds it what it was doing.

---

## Component 6 — Delegation via Subagents

![Subagent inherits context but runs inside tighter boundaries](https://substackcdn.com/image/fetch/$s_!Ygjt!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F81b2ae42-ea42-40bf-b2ba-fdfcf9b87912_2438x990.png)

Sometimes the main agent needs a side answer:
- "Which file defines `encode_batch`?"
- "What does this config say?"
- "Why is this test failing?"

Options:
- **Inline**: do it in the main loop → pollutes context, may lose focus
- **Delegate**: spawn a bounded **subagent** → parallel, isolated work

Claude Code has supported subagents for a while; Codex added them more recently.

---

## Binding the Subagent

> *"The tricky design problem is not just how to spawn a subagent, but how to bind one."*

A subagent needs **enough context to be useful** — but must be **constrained** or it'll duplicate work, spawn its own subagents, edit files it shouldn't, or run forever.

**Typical constraints:**
- **Read-only by default** (sometimes — Codex is more permissive)
- **Scoped context**: inherits relevant parts of parent's state
- **Recursion depth limit**
- **Tool whitelist** — smaller than parent's
- **Timeout / step budget**
- **Cannot spawn its own subagents** (usually)

Return a focused answer to the parent, then disappear.

---

## The Six Components at a Glance

![Six main features of a coding harness summary](https://substackcdn.com/image/fetch/$s_!ml47!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4fe9e9f0-b04f-4c3e-be1f-fbd74b41c4aa_3396x971.png)

| # | Component | Why it matters |
|---|---|---|
| 1 | **Live repo context** | Grounds ambiguous instructions in concrete facts |
| 2 | **Prompt shape + caching** | Saves cost & latency on repeated state |
| 3 | **Structured tools + permissions** | Deterministic, safe, auditable actions |
| 4 | **Context reduction** | Keeps sessions long without drowning in tokens |
| 5 | **Transcripts + memory** | Durability + task continuity |
| 6 | **Bounded subagents** | Parallel side-tasks without losing focus |

Together, they turn "an LLM that writes code" into "a system that ships PRs."

---

## Coding Harness ≠ General Agent Harness

Not every agent is a coding agent.

**Coding harness (Claude Code, Codex)** — optimized for:
- one developer, one repo at a time
- tight feedback with files, tools, tests
- minutes-to-hours sessions

**General agent platform (e.g. OpenClaw)** — optimized for:
- many long-lived agents
- across chats, channels, and workspaces
- coding is *one* workload among many

Both use similar building blocks. The emphasis differs.

---

# Putting It All Together

---

## How Part 1 and Part 2 Connect

Part 1: how do we **build** a model that can reason well?
Part 2: how do we **deploy** that model in a system that actually ships useful work?

Neither is sufficient alone:

- A great reasoning model in a bad harness → feels dumb
- A great harness around a weak model → feels stuck

The state of the art in 2026: **strong reasoning model** × **strong coding harness** = what we see in Claude Code, Codex, Cursor.

---

## Key Takeaways

1. **Reasoning is a specialization**, not a replacement. Use it when tasks need multi-step verification.

2. **SFT + RL** is the flagship recipe; **distillation** is the budget path; **inference-time scaling** is the serving-side boost.

3. **Rule-based rewards** (accuracy + format) can replace human preference models when answers are verifiable.

4. **A harness can matter as much as the model.** Coding is partly about next-token prediction, but mostly about managing context, tools, and state.

5. **Context quality is often mistaken for model quality.** Clipping, deduplication, caching, and working memory do a lot of the heavy lifting.

6. **Budget-friendly research is possible:** Sky-T1 ($450), TinyZero (<$30), and the Mini Coding Agent are all doable at lab scale.

---

## Further Reading

**Reasoning models**
- [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)
- DeepSeek-R1 technical report — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Snell et al., *Scaling LLM Test-Time Compute Optimally* — [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- *O1 Replication Journey — Part 1* — [arXiv:2410.18982](https://arxiv.org/abs/2410.18982)
- Sky-T1 blog — [novasky-ai.github.io/posts/sky-t1/](https://novasky-ai.github.io/posts/sky-t1/)
- TinyZero repo — [github.com/Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero)

**Coding agents**
- Raschka, *[Components of a Coding Agent](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent)* (Apr 2026)
- Raschka's *Mini Coding Agent* — [github.com/rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent)

<!--
**Readings:**
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903)
- [Large Language Models are Zero-Shot Reasoners](https://proceedings.neurips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf)
- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [OpenAI Harmony Response Format](https://cookbook.openai.com/articles/openai-harmony)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](https://machinelearning.apple.com/research/illusion-of-thinking)
- [DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning](https://github.com/deepseek-ai/DeepSeek-Math-V2/blob/main/DeepSeekMath_V2.pdf)
- [AlphaGeometry: An Olympiad-level AI system for geometry](https://deepmind.google/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/)
- [How Does A Blind Model See The Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth)
- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)

**Extra Readings:**
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)
- [Language Models, World Models, and Human Model-Building](https://lingo.csail.mit.edu/blog/world_models/)
- [LLMs and World Models](https://aiguide.substack.com/p/llms-and-world-models-part-1)
-->

---

## Discussion Questions

1. Where would you expect a reasoning model to *fail* in ways a standard LLM wouldn't?

2. The R1-Zero "Aha!" moment — is this genuine emergence, or an artifact of the reward shape? How would you design an experiment to tell?

3. If harness > model for many tasks, what does that mean for *benchmarking*? Are current LLM leaderboards measuring the right thing?

4. For your own research/work: which of the six harness components would be hardest to reuse in a non-coding domain (e.g., a data-analysis agent)?

5. Where do you expect the next bottleneck to be — model capability, harness design, or something else entirely?

---

## Thank You

**Questions?**

*Slides and references available on the course page.*
