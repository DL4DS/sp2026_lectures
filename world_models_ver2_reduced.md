# World Models — Understanding the World or Predicting the Future?

## DS 542 / DL4DS — Spring 2026
*Boston University*

> Under construction

A short tour, structured around Ding et al., *"Understanding World or Predicting Future? A Comprehensive Survey of World Models"* (ACM Computing Surveys, 2025).

.footnote[.red.bold[*] Created with Claude]

---

## Hook: understanding *or* predicting?

DeepMind's **Genie 3** generates interactive, navigable 3D environments at 24 frames per second from a single text prompt — staying consistent for several minutes.

![Genie 3 demo still](https://lh3.googleusercontent.com/BirHhzrtzHQtXvNG0dXcj9QFA7vGUNWQYbeQiMiY7RL8a7-xGrLDGybf9YZZ1nML7lRVZh4KYARz0PFO8xBAfe7FAatH6xVlRnakHtnESHlpzEBTEA=w2880-h1620-n-nu-rw-lo)

Does the network *understand* how the world works, or merely *predict* what its next frame should look like? The survey we'll follow argues these are two different research programs.

---

## Lecture roadmap

![Survey outline](https://raw.githubusercontent.com/tsinghua-fib-lab/World-Model/main/asset/outline.png)

1. **Background & the dual taxonomy** — what is a world model?
2. **Understanding the world** — internal/implicit world models
3. **Predicting the world** — generative/video world models
4. **Applications** — game, embodied, urban, societal
5. **Open problems** — and the JEPA alternative

*Source: Ding et al., [arXiv:2411.14499](https://arxiv.org/abs/2411.14499).*

---

## Callbacks to other things we've covered

| Component | Lectures it draws on |
|:---|:---|
| Encoding observations | VAE, ViT, CNN |
| Sequence dynamics | RNN, Transformer |
| High-fidelity decoding | Diffusion, GAN |
| Joint-embedding objectives | GNN, contrastive self-supervised |
| Action conditioning | (new today: Reinforcement Learning) |

World models are a *recombination* of techniques you already know.

---

# Part 1 — Background & the Dual Taxonomy

---

## What is a world model?

A **world model** is a learned function that predicts how the world evolves:

$$\hat{s}_{t+1} = f_\theta(s_t, a_t)$$

Read this as: *given the current state $s_t$ and action $a_t$, predict the next state.* The network's parameters are $\theta$.

The deep disagreement in the field is about **what "state" means**:
- raw pixels of a video frame?
- a learned latent vector?
- a symbolic description in language?

Each choice leads to a different kind of world model.

---

## Why learn a world model?

- **Sample efficiency** — train policies in imagination, not in the real world
--

- **Planning** — roll out counterfactual futures before acting
--

- **Self-supervised learning** — prediction as the pretraining objective
--

- **Simulation** — synthetic data, games, robotics, autonomous driving

---

## Where world models live: model-based RL

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/1280px-Reinforcement_learning_diagram.svg.png" alt="RL agent-environment loop" style="max-width: 45%; height: auto;">

In **reinforcement learning**, an agent observes a state, takes an action, gets a reward, and lands in a new state. *The "transition function" — what the next state will be — **is** the world model.*

---

## Model-free vs. model-based RL

| | Model-free RL | Model-based RL |
|---|---|---|
| Learns | Policy directly | Policy *plus* a world model |
| Examples | DQN, PPO | Dreamer, MuZero |
| Sample cost | Millions of steps | Strong — train in imagination |
| Risk | Stable | World-model errors compound |

---

## The survey's dual taxonomy

![Survey structure](https://raw.githubusercontent.com/tsinghua-fib-lab/World-Model/main/asset/structure.png)

Ding et al. split the field into **two research programs**:

- **Understanding** — the world model is an *internal representation* that supports decisions and encodes knowledge. (Latents, not pixels.)
- **Predicting** — the world model is a *generative simulator* that rolls future video forward. (Pixels, not just latents.)

Same toolbox (transformers, diffusion), very different *objectives*.

---

## Five paradigms across the deep-learning era

![Roadmap of world model paradigms](https://raw.githubusercontent.com/tsinghua-fib-lab/World-Model/main/asset/roadmap.png)

The survey organises the field into five intersecting paradigms:

- **Model-based RL** — Dreamer, MuZero
- **Self-supervised learning** — JEPA, V-JEPA, DINO-WM
- **Large language models** — language as compressed world state
- **Video generation** — Sora, Genie, Cosmos
- **Interactive 3D environments** — Habitat, CARLA

Each will reappear later in the lecture.

---

# Part 2 — Understanding the World

*Implicit/representation-based world models. Survey §3.*

---

## Two ways to *use* a world model

> **[FIGURE PLACEHOLDER — Survey Fig. 2: Two schemes for using a world model in decision-making]**

Once you have a learned world model, there are two ways to use it:

1. **Train a policy in imagination** — let the agent dream rollouts inside the model, and learn from the dream.  *(Dreamer family.)*
2. **Plan at decision time** — search over imagined futures whenever the agent must act.  *(MuZero, model-predictive control.)*

Same world model, very different control strategies.

---

## Ha & Schmidhuber (2018) — the V / M / C blueprint

![V, M, C architecture](https://worldmodels.github.io/assets/world_model_overview.svg)

The paper that started the modern field. Three components:

- **V — Vision.** A small network that compresses each frame into a short latent vector.
- **M — Memory.** A recurrent network that, given the current latent and action, predicts the *next* latent.
- **C — Controller.** A tiny linear policy that maps the latent state to an action.

The trick: train each part separately on its own objective, then deploy them as a pipeline.

*Source: Ha & Schmidhuber 2018, [worldmodels.github.io](https://worldmodels.github.io).*

---

## The kicker: train entirely in the dream

![Training inside the dream](https://worldmodels.github.io/assets/world_model_schematic.svg)

1. Collect a few random rollouts in the real environment.
2. Train V (the encoder) on frames; train M (the predictor) on latent sequences.
3. **Train the controller entirely inside M's dreams** — no real rollouts.
4. Deploy the controller in the real environment — *it works.*

This established the central claim of the field: **you can transfer a policy from imagination to reality.**

---

## Scaling up: PlaNet → the Dreamer family

The 2018 blueprint had three weaknesses at scale:

- Pixel reconstruction wastes capacity on textures
- Long-horizon dreams drift
- Continuous control needs gradients, not just evolutionary search

The **Dreamer** line (Hafner et al., 2019–2025) fixes all three with a smarter latent: **part deterministic** (reliable memory) and **part stochastic** (room for uncertainty about the future). The agent then uses standard actor-critic learning entirely *inside imagined latent rollouts*.

![DreamerV3 world model and behavior loops](https://danijar.com/asset/dreamerv3/header.gif)

*Source: Hafner et al., DreamerV3 / Nature 2025 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104).*

---

## Validation: Minecraft Diamond

DreamerV3 was the **first algorithm to collect a diamond in Minecraft** — no human demonstrations, no curriculum, just raw pixels and ~30M environment steps.

**Dreamer 4** (Sept 2025) does it from *offline data alone* — over 20,000 keyboard/mouse actions planned entirely in imagination, with no further interaction.

> A good world model + planning in imagination = far-sighted strategic behaviour.

---

## A different lineage: MuZero

MuZero (DeepMind 2020) is a counterpoint: it **never reconstructs pixels**. The latent is trained only to predict what planning needs — reward, value, policy.

It beats AlphaZero at Go, Chess, and Shogi *without being given the rules*.

> Maybe pixel reconstruction is the *wrong* objective entirely. — This idea returns at the end of the lecture as **JEPA**.

---

## Do LLMs already contain world models?

> **[FIGURE PLACEHOLDER — Survey Fig. 3: World knowledge taxonomy in LLMs]**

The survey asks: even *without* explicit dynamics training, do large language models internally encode world knowledge? Three categories:

- **Global physical world** — geography, scale, time
- **Local physical world** — affordances, intuitive physics
- **Human society** — norms, theory of mind, social roles

A canonical probe is **Othello-GPT** (Li et al. 2023): train a transformer on game-move sequences only. Researchers find an internal *board representation* — emerging without any spatial supervision.

LeCun's counter: LLMs hallucinate physics, and word-level prediction does not enforce world consistency. **Open question.**

---

# Part 3 — Predicting the World

*Generative/video world models. Survey §4.*

---

## The 2024 regime shift

Pre-2023 world models were **small, task-specific, trained on game data**.

After 2023, the same recipe that worked for language models — **internet-scale pretraining + transformers** — gets applied to video.

A new question emerges:

> *If you train a video generator at sufficient scale, does it become a world model?*

A working video world model would need: object permanence, 3D consistency, action conditioning, and long-horizon coherence. Most current systems excel at one or two and fail at the rest.

---

## Sora — "video generation as world simulation"

OpenAI, Feb 2024. A **diffusion transformer** denoises *spacetime patches* of video.

![Sora spacetime patches](https://images.ctfassets.net/kftzwdyauwt9/1d2955dd-9d05-4f33-13073dc9301d/8dc0bae8cb98054d083ab3cc3ade6859/figure-patches.png?w=3840&q=90&fm=webp)

A pre-trained encoder compresses each clip; the result is cut into small space-and-time patches; a transformer denoises them step by step.

*Callbacks to your vision-transformer and diffusion lectures.*

*Source: [OpenAI Sora technical report](https://openai.com/index/video-generation-models-as-world-simulators/).*

---

## Sora's emergent properties — and its failures

**Emergent**:
- Object permanence across cuts
- Primitive 3D consistency
- Multi-character interactions

**Documented failure modes**:
- Glasses fall *through* tables
- Hands with seven fingers
- Wolves spontaneously appear
- No conservation of mass or causal structure

> *Photorealism is not understanding.*

---

## Genie & Genie 3 — the *interactive* turn

![Genie 3 navigable scene](https://lh3.googleusercontent.com/BirHhzrtzHQtXvNG0dXcj9QFA7vGUNWQYbeQiMiY7RL8a7-xGrLDGybf9YZZ1nML7lRVZh4KYARz0PFO8xBAfe7FAatH6xVlRnakHtnESHlpzEBTEA=w2880-h1620-n-nu-rw-lo)

DeepMind. **Genie** (Feb 2024): trained on 200,000 hours of unlabelled gameplay video. With *no action labels*, it discovers a small action vocabulary by watching transitions, and learns to predict next frames given those latent actions.

**Genie 3** (Aug 2025): 24 fps, 720p, several minutes of consistent dynamics from a text prompt. Mid-session text commands ("start a thunderstorm") modify the running simulation.

Subtle question: is Genie a true world model, or just a controllable video generator?

---

## Cosmos — physical-AI foundation model

**NVIDIA**, Jan 2025 ([arXiv:2501.03575](https://arxiv.org/abs/2501.03575)). Explicitly branded a **"World Foundation Model"** for physical AI.

- Trained on 20M hours of *real-world* video (vs. Genie's gameplay)
- Both diffusion and autoregressive variants
- Built-in evaluation for physics plausibility (conservation, collisions)
- Open weights, downstream fine-tuning recipes — the *Hugging Face of world models*

Targeted at robotics, autonomous driving, and simulated training data.

---

## Comparing the big three — and the central tension

| | **Sora** | **Genie 3** | **Cosmos** |
|---|---|---|---|
| Input | Text | Text + interactive | Text / image / video |
| Output | Fixed-length video | Real-time playable world | Conditioned video |
| Training data | Mixed video | Gameplay video | Real-world video |
| Primary goal | Cinematic | Interactive simulation | Physical realism |

But all three share the same problems:

- **Errors compound** in long autoregressive rollouts
- **Pixel-level loss does not enforce physics**
- **Unclear utility for planning** — can an agent actually use these for decisions?

---

## From video to embodied environments

> **[FIGURE PLACEHOLDER — Survey Fig. 4: classification of interactive embodied environments]**

The survey distinguishes video models that you *watch* from environments that you *act in*.

| Family | Examples | Role |
|---|---|---|
| Indoor (homes, offices) | Habitat, ProcTHOR, Holodeck | Manipulation, navigation |
| Outdoor (driving) | CARLA, MetaDrive | AV simulators |
| **Learned simulators** | GAIA-1, Cosmos, DriveDreamer | Replace hand-built simulators with neural ones |

The shift in 2024–2025: from rigid hand-built sims to *learned* simulators, trading exactness for diversity.

---

# Part 4 — Applications

*Survey §5 — where world models meet the real world.*

---

## Game intelligence

**World model as game-maker:** Genie / Genie 3 turn text or image prompts into *playable* games. The action set is *learned from gameplay video*, not specified in advance.

**World model as game-player:** Dreamer 4 plans entirely offline in Minecraft and collects diamonds via long-horizon imagined rollouts.

Why games matter: a closed environment with rich actions — clean for evaluation; partial physics; long-horizon planning matters; a stepping-stone to harder applications.

---

## Embodied intelligence — robotics

The implicit-world-model lineage applied to robots:

- **DINO-WM** (Zhou et al. 2024) — a world model built on **frozen DINO features** (a self-supervised image representation); enables zero-shot planning.
- **V-JEPA 2** (Meta, June 2025) — pre-trained on >1 million hours of internet video, then post-trained on just 62 hours of unlabelled robot video. Demonstrates **zero-shot robot planning from image goals**.
- **Cosmos** — used as a *generative simulator* to fine-tune robotic policies in synthetic data.

The premise: the latent need not reconstruct pixels — it only needs to *support action*.

> **[FIGURE PLACEHOLDER — Survey Fig. 6: robotic world-model trajectory]**

---

## Urban intelligence — autonomous driving

> **[FIGURE PLACEHOLDER — Survey Fig. 5: autonomous-driving WM framework]**

Two distinct uses of world models in autonomous driving:

- **As internal representation** — predict latent occupancy or bird's-eye-view scenes (OccWorld, UniAD).
- **As driving simulator** — neural surrogates for hand-built sims like CARLA. Examples: GAIA-1 (Wayve, 9B parameters), DriveDreamer, Drive-WM, Vista.

Cosmos directly targets this market. Sim-to-real transfer remains the bottleneck.

---

## Societal intelligence

> **[FIGURE PLACEHOLDER — Survey Fig. 7: social simulacra]**

A *world model* of a **social** environment, not a physical one.

- **Generative Agents / Smallville** (Park et al. 2023) — 25 LLM-driven agents in a simulated town; emergent dating, gossip, party planning.
- **EconAgent**, **AgentGroupChat**, **GovSim** — economic, social, and governance simulations populated by language-model agents.

What these need that physical world models don't:
- **Theory of mind** — agents must model each others' beliefs
- **Value alignment** — simulated populations should match target distributions
- **Privacy & ethics** — modelling real people is fraught

---

# Part 5 — Open Problems & the JEPA Counterpoint

*Survey §6, plus the alternative LeCun proposes.*

---

## LeCun's critique of pixel prediction

> *Generative pixel-prediction is fundamentally wasteful.*

Most pixels carry no decision-relevant information. Predicting them spends capacity on irrelevant texture detail.

LeCun's alternative: predict **representations** of the future, not pixels. This is the **Joint-Embedding Predictive Architecture**, or **JEPA** — *joint-embedding* meaning that an encoder is trained on both past and future, and the model predicts the future's embedding.

Reference: LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf).

---

## JEPA — predict in representation space

![JEPA vs generative architectures](https://scontent-lga3-3.xx.fbcdn.net/v/t39.2365-6/347632356_201757625702076_1813962196800732436_n.png?_nc_cat=102&ccb=1-7&_nc_sid=e280be&_nc_ohc=_uLGhys0OkwQ7kNvwFGAGyJ&_nc_oc=AdpurO8E2VRqlIMy27p1Slx1qxUuzNfjJ7qWG5XGenndwS-z7OzAhawxYfgZ6KUEb3w&_nc_zt=14&_nc_ht=scontent-lga3-3.xx&_nc_gid=ld4pnAXoy4I7zYthbFs1Hw&_nc_ss=7b289&oh=00_Af1_C0lJATWz_WMlQOw61OJmyU8nbOx5Mm31aymruIRw2Q&oe=6A079DA0)

- **Generative** (left): decode future *pixels* from past pixels.
- **JEPA** (right): predict the future's *embedding* from the past's embedding.

Loss is computed in **representation space**, not pixel space. The risk: the encoder can collapse to a constant — so JEPA needs a regulariser to keep representations informative.

JEPA aligns with MuZero (no reconstruction) and the contrastive self-supervised methods you saw earlier.

---

## Open problem 1 — physical understanding

Next-frame prediction loss does **not** measure physical understanding. New benchmarks shift toward causal metrics:

- **IntPhys 2** (Bordes et al. 2025) — intuitive physics: object permanence, gravity, solidity
- **Long-horizon consistency** — does the world stay coherent over 60 seconds?

Persistent failure modes in 2025: errors compound over long rollouts; conservation of mass and identity is routinely violated; action conditioning is brittle.

---

## Open problems 2–5

**Sim-to-real transfer** — knowledge from a generative simulator doesn't yet reliably transfer to real robots.

**Efficiency** — autoregressive video generation is sequential and expensive. 20-million-hour training corpora are not sustainable.

**Social dimension** — value alignment and theory-of-mind in multi-agent simulations are still under-evaluated.

**Ethics & safety** — privacy of training video; deepfakes; provenance/watermarking lags generation capability; compute concentrates the field in a few labs.

---

# Synthesis & Wrap-Up

---

## The V/M/C blueprint, evolved

| Module | 2018 | 2025 — Understanding | 2025 — Predicting |
|---|---|---|---|
| **V** (perception) | Small VAE | DINO / JEPA encoder | Vision-transformer patches |
| **M** (dynamics) | Recurrent net | Latent transformer | Diffusion or autoregressive transformer |
| **C** (control) | Tiny linear policy | Actor-critic in imagination | Text prompts, world events |

Every prior lecture in the course (CNN, transformer, ViT, diffusion, GAN, VAE, GNN, self-supervised) plays a role in this table.

---

## The 2025–2026 inflection

- **Dreamer 4** solves Minecraft Diamond from offline data (Sep 2025)
- **Genie 3** real-time interactive worlds (Aug 2025)
- **V-JEPA 2** demonstrates zero-shot robot planning (Jun 2025)
- **Cosmos 2.5** — open world foundation models, 2M+ downloads (Oct 2025)
- **World Labs** launches **Marble** — commercial 3D world generation (Nov 2025)
- **LeCun leaves Meta** to start AMI Labs, raising €500M for JEPA-style world models (late 2025)

> The field is forking — generative video on one branch, JEPA on the other — but both share the survey's two-camp DNA.

---

## Three questions to take away

1. Is the path to general AI through scaling **language models**, scaling **generative world models**, or scaling **JEPA-style implicit world models** — or fusing them?

2. Can a model trained only on *pixels* learn *causal* physics, or do we need **action-conditioned** training data at scale?

3. What is the right **evaluation metric** for a "good" world model — and who decides?

---

## Reading list

**The survey (the spine of this lecture)**:
- Ding et al., *Understanding World or Predicting Future? A Comprehensive Survey of World Models*, ACM CSUR 2025 — [arXiv:2411.14499](https://arxiv.org/abs/2411.14499)
- Curated paper list & figures: [github.com/tsinghua-fib-lab/World-Model](https://github.com/tsinghua-fib-lab/World-Model)

**Primary sources**:
- Ha & Schmidhuber, *World Models* — [worldmodels.github.io](https://worldmodels.github.io)
- DreamerV3 — Hafner et al., *Nature* 2025; [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Dreamer 4 — Hafner et al., [arXiv:2509.24527](https://arxiv.org/abs/2509.24527)
- Sora — OpenAI technical report (Feb 2024)
- Genie — Bruce et al., [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
- Cosmos — NVIDIA, [arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
- V-JEPA 2 — Assran et al., [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf)

*Questions?*
