# World Models — Understanding the World or Predicting the Future?

## DS 542 / DL4DS — Spring 2026
*Boston University*

> Under construction

A survey-driven tour, structured around Ding et al., *"Understanding World or Predicting Future? A Comprehensive Survey of World Models"* (ACM CSUR 2025).

.footnote[.red.bold[*] Created with Claude]

---

## Hook: Understanding *or* predicting?

DeepMind's **Genie 3** generates interactive, navigable 3D environments at **24 fps** from a single text prompt — with consistency for several minutes.

![Genie 3 demo still](https://lh3.googleusercontent.com/BirHhzrtzHQtXvNG0dXcj9QFA7vGUNWQYbeQiMiY7RL8a7-xGrLDGybf9YZZ1nML7lRVZh4KYARz0PFO8xBAfe7FAatH6xVlRnakHtnESHlpzEBTEA=w2880-h1620-n-nu-rw-lo)

To do this, does the network *understand* the world's mechanisms, or merely *predict* its pixels?
*The survey we'll follow argues these are two different research programs.*

---

## Lecture roadmap

![Survey outline](https://raw.githubusercontent.com/tsinghua-fib-lab/World-Model/main/asset/outline.png)

1. **Background & the dual taxonomy** — what is a world model?
2. **Implicit representation** *(Survey §3)* — world models that *understand*
3. **Future prediction** *(Survey §4)* — world models that *generate*
4. **Applications** *(Survey §5)* — game, embodied, urban, societal
5. **Open problems** *(Survey §6)* — and the JEPA counterpoint
6. **Synthesis**

*Source: Ding et al., [arXiv:2411.14499](https://arxiv.org/abs/2411.14499).*

---

## Callbacks to other things we've covered

| Component | Lectures it draws on |
|:---|:---|
| Encoder of observations | VAE, ViT, CNN |
| Sequence dynamics | RNN, Transformer |
| High-fidelity decoding | Diffusion, GAN |
| Joint-embedding objectives | GNN, contrastive SSL |
| Action conditioning | (new today: RL) |

World models are a *recombination* of techniques you already know.

---

# Part 1 — Background & the Dual Taxonomy

---

## Definition

A **world model** is a learned function predicting how the world (or its representation) evolves:

$$\hat{s}_{t+1} = f_\theta(s_t, a_t)$$

Often probabilistic:

$$p_\theta(s_{t+1} \mid s_t, a_t)$$

The choice of *what counts as $s$*, *what counts as $a$*, and *what we use the prediction for* is the central source of disagreement in the field.

---

## Why learn a world model?

- **Sample efficiency** — train policies in imagination, not in the real world
--

- **Planning** — roll out counterfactual futures before acting
--

- **Self-supervised representation learning** — prediction as the pretraining objective
--

- **Simulation** — synthetic data, games, robotics, autonomous driving

---

## The agent–environment loop

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/1280px-Reinforcement_learning_diagram.svg.png" alt="RL agent-environment loop" style="max-width: 50%; height: auto;">

At each step $t$: agent observes $s_t$, takes action $a_t$, receives reward $r_{t+1}$, and lands in $s_{t+1}$.

Goal: learn a **policy** $\pi(a \mid s)$ that maximizes long-term reward.
*Source: Wikipedia / Sutton & Barto.*

---

## The MDP formalism

A Markov Decision Process: $(S, A, P, R, \gamma)$

- **Transition function** $P(s_{t+1} \mid s_t, a_t)$ — *this is exactly the world model*
- **Reward** $R(s, a)$
- **Discount** $\gamma \in [0, 1)$

Return: $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$

Value function: $V^\pi(s) = \mathbb{E}_\pi [G_t \mid s_t = s]$

In standard ("model-free") RL, we **never learn $P$** — we just learn from samples.

---

## Model-free vs. model-based RL

| | **Model-free** | **Model-based** |
|---|---|---|
| Learns | Policy / value directly | Policy *and* a model $\hat P$ of the world |
| Examples | DQN, PPO, SAC | World Models, Dreamer, MuZero |
| Sample efficiency | Poor (millions of steps) | Strong |
| Real-world cost | Often prohibitive | Can train *in imagination* |
| Risk | Stable | Errors in $\hat P$ compound |

> *If you have a good world model, RL becomes much cheaper.*
> The rest of this lecture is **how do we learn one** — and what kind?

---

## The survey's dual taxonomy

![Survey structure](https://raw.githubusercontent.com/tsinghua-fib-lab/World-Model/main/asset/structure.png)

Ding et al. organise the field into **two distinct research programs**:

- **Understanding** *(§3)* — the world model is an *internal/implicit* representation that supports decision-making and encodes knowledge.
- **Predicting** *(§4)* — the world model is a *generative simulator* that rolls future states forward in observation space.

The same technique (e.g. a transformer) can serve either program — but the *objective* and *use* differ.

*Source: Ding et al. 2025, Figure 1.*

---

## Five paradigms across the deep-learning era

![Roadmap of world model paradigms](https://raw.githubusercontent.com/tsinghua-fib-lab/World-Model/main/asset/roadmap.png)

The survey identifies five intersecting paradigms — each will reappear later in the lecture:

- **Model-based RL** — Dreamer, MuZero, TD-MPC
- **Self-supervised learning** — JEPA, V-JEPA, DINO-WM
- **LLM / MLLM** — language as a compressed world state
- **Video generation** — Sora, Genie, Cosmos
- **Interactive 3D environments** — Habitat, ProcTHOR, CARLA

*Source: Ding et al. 2025, "Roadmap" figure.*

---

# Part 2 — Implicit Representation of the External World

*Survey §3 — world models that **understand**.*

---

## §3.1 — Two schemes for using a world model in decision-making

> **[FIGURE PLACEHOLDER — Survey Fig. 2: two schemes for using a world model in decision-making]**

The survey identifies two distinct ways to *use* a learned world model:

1. **Train policy in imagination** — backprop through learned dynamics. *Examples:* Dreamer family.
2. **Plan at test time over learned dynamics** — search/MCTS in latent space. *Examples:* MuZero, TD-MPC.

Same world model, two very different control strategies.

---

## Ha & Schmidhuber (2018) — the V/M/C blueprint

![V, M, C architecture](https://worldmodels.github.io/assets/world_model_overview.svg)

The paper that started the modern field. Three components:

- **V** (Vision): VAE — compresses each frame into latent $z_t$
- **M** (Memory): MDN-RNN — predicts $p(z_{t+1} \mid z_t, a_t, h_t)$
- **C** (Controller): tiny linear policy — outputs $a_t$ from $[z_t, h_t]$

*Source: Ha & Schmidhuber 2018, [worldmodels.github.io](https://worldmodels.github.io).*

---

## V / M / C internals (compressed)

**V — Convolutional VAE on $64{\times}64{\times}3$ frames** → 32-dim latent $z_t$. Smooth, sampleable space (so we can *dream*).

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))$$

**M — Mixture-density RNN.** Predicts a multimodal next-latent — because the future is multimodal:

$$p(z_{t+1} \mid z_t, a_t, h_t) = \sum_{k=1}^{K} \pi_k(h_t)\, \mathcal{N}\!\left(z_{t+1};\, \mu_k(h_t),\, \sigma_k^2(h_t)\right)$$

**C — Tiny linear controller** (~1000 parameters), $a_t = W_c [z_t, h_t] + b_c$, trained with **CMA-ES** (no gradients). Optional sampling temperature $\tau$ controls dream uncertainty — training in a *harder* dream than reality produces a more *robust* policy.

---

## The kicker: train entirely in the dream

![Training inside the dream](https://worldmodels.github.io/assets/world_model_schematic.svg)

1. Collect random rollouts in the real environment.
2. Train V on frames; train M on the latent sequence.
3. **Train C entirely inside M's dreams** (no real rollouts).
4. Deploy C in the real environment — it works.

*Established the V+M+C decomposition, prediction-as-pretraining, and imagination-to-reality transfer.*

---

## Scaling up: PlaNet → Dreamer

The 2018 blueprint had three weaknesses at scale:

- **Pixel reconstruction is wasteful** — VAE spends capacity on textures
- **Long-horizon credit assignment fails** in MDN-RNN dreams
- **Continuous control** (robotics) needs gradients, not just evolution

**PlaNet** (Hafner 2019) introduced the latent-space planner; the **Dreamer** line (V1–V4, 2019–2025) added actor-critic in imagination and engineered it for scale.

---

## RSSM — Recurrent State-Space Model

Latent state has **two parts**:

- $h_t$ — **deterministic** GRU hidden state (reliable memory)
- $z_t$ — **stochastic** sample (uncertainty)

$$
\begin{aligned}
h_t &= f_\phi(h_{t-1},\, z_{t-1},\, a_{t-1}) && \text{recurrent backbone} \\[4pt]
z_t &\sim q_\phi(z_t \mid h_t,\, o_t) && \text{posterior — uses observation} \\[4pt]
\hat z_t &\sim p_\phi(z_t \mid h_t) && \text{prior — used for dreaming}
\end{aligned}
$$

Plus heads for: reconstructed observation $\hat o_t$, reward $\hat r_t$, episode continuation $\hat c_t$.

The prior $p_\phi$ is trained to match the posterior $q_\phi$ (KL term) — so we can *imagine* without observations.

---

## DreamerV1 → V3 — actor-critic in imagined latent space

![DreamerV3 world model and behavior loops](https://danijar.com/asset/dreamerv3/header.gif)

(a) Train the world model from real experience.
(b) Imagine latent rollouts; train **actor** and **critic** entirely inside them.

**V3 hardening tricks**: symlog predictions, KL balancing + free bits, **categorical latents** (32×32 with straight-through gradients), adaptive gradient clipping. **A single fixed hyperparameter set** solves continuous control, Atari, BSuite, Crafter, **and Minecraft from scratch**.

*Source: Hafner et al., DreamerV3 / Nature 2025 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104).*

---

## The Minecraft Diamond milestone

DreamerV3 was the **first algorithm to collect a diamond in Minecraft** — no human data, no curriculum, ~30M environment steps from raw pixels.

**Dreamer 4** (Sept 2025) does it from *offline data alone*: a sequence of >20,000 keyboard/mouse actions planned entirely in imagination.

This validates the central claim of *world model + planning-in-imagination = far-sighted strategic behavior.*

---

## MuZero — implicit world models

A different lineage (Schrittwieser et al., DeepMind 2020).

- **Never reconstructs pixels.**
- Predicts only what planning needs: **reward**, **value**, **policy**.
- Beats AlphaZero on Go, Chess, Shogi *without being given the rules*.

| | **Generative (Dreamer)** | **Implicit (MuZero)** |
|---|---|---|
| Reconstructs observations? | Yes | No |
| Latent trained for | Reconstruction + reward | Reward + value + policy |
| Pros | Interpretable; debuggable | Compact; no wasted capacity |
| Cons | Wastes capacity on textures | Latent is opaque |

> Maybe pixel reconstruction is the *wrong* objective entirely. — This idea returns as **JEPA** in Part 5.

---

## §3.1.2 — Language-backbone world models

Once LLMs were strong, a new question: *can the LLM itself act as the world model?*

- **RAP** (Hao et al. 2023) — reasoning via planning: the LLM generates the next state and reward inside an MCTS loop.
- **DynaLang** (Lin et al. 2024) — multi-modal world model conditioned on natural-language descriptions of dynamics.
- **LLM-MCTS / RAFA** — the LLM serves as both transition model and value heuristic.

The intuition: **language is already a highly compressed world state**. Whether that compression is faithful enough to plan over remains an open question (Part 5).

> **[FIGURE PLACEHOLDER — adapted from Survey Table 1: language-backbone WMs]**

---

## §3.2 — World knowledge learned by models

> **[FIGURE PLACEHOLDER — Survey Fig. 3: LLM world-knowledge taxonomy (global / local / societal)]**

Even *without* explicit dynamics training, LLMs may **internally encode** world knowledge.

The survey decomposes it along three axes:

- **Global physical world** — geography, time, scale; "What city is north of Paris?"
- **Local physical world** — affordances, intuitive physics, spatial reasoning
- **Human society** — norms, theory-of-mind, social roles

A canonical probe: **Othello-GPT** (Li et al. 2023) — train a transformer on move sequences; an internal *board representation* emerges with no spatial supervision.

---

## The big debate: do LLMs already have world models?

**Pro** (autoregressive prediction is sufficient):
- **Othello-GPT** — emergent internal board state from move tokens alone
- Sora exhibits primitive object permanence
- Mechanistic interpretability finds world-model-like structures in LLMs

**Con** (LeCun's view):
- LLMs hallucinate physics
- Token-level loss doesn't enforce world consistency
- True world models need representation-space prediction + planning

This is an *open* question. Two reasonable views — we'll return to it in Part 5.

---

# Part 3 — Future Prediction of the Physical World

*Survey §4 — world models that **generate**.*

---

## §4.1 — The 2024 regime shift to video

Pre-2023 world models were **small, task-specific, trained on game data** (Dreamer on Atari; PlaNet on DeepMind Control).

Post-2023, the same recipe that worked for LLMs — **internet-scale pretraining + transformers** — gets applied to video.

A new question emerges: *if you train a generative video model at sufficient scale, does it become a world model?*

---

## What a video world model must satisfy

The survey identifies a **capability rubric** for video WMs (§4.1.2):

- **Object permanence** — entities persist through occlusion / cuts
- **3D consistency** — geometry survives camera moves
- **Action conditioning** — the model accepts and responds to control inputs
- **Long-horizon coherence** — minutes, not seconds
- **Multimodality** — text, image, audio, depth, action streams

Most current video models excel at one or two and fail at the rest.

> **[FIGURE PLACEHOLDER — adapted from Survey Table 3: video WM capability checklist]**

---

## Sora — "video generation models as world simulators"

OpenAI, Feb 2024. Architecture: a **diffusion transformer** operating on **spacetime patches** of video latents.

![Sora spacetime patches](https://images.ctfassets.net/kftzwdyauwt9/1d2955dd-9d05-4f33-13073dc9301d/8dc0bae8cb98054d083ab3cc3ade6859/figure-patches.png?w=3840&q=90&fm=webp)

A pretrained video VAE compresses each clip; the result is cut into *spacetime patches*; a DiT denoises them.

*Callback to ViT (patches) and diffusion (denoising) lectures.*

*Source: [OpenAI Sora technical report](https://openai.com/index/video-generation-models-as-world-simulators/).*

---

## Sora's emergent properties — and failures

**Emergent**:
- Object permanence across cuts
- Primitive 3D consistency
- Multi-character interactions

**Failure modes** documented in the technical report and follow-up analyses:
- Glasses fall *through* tables
- Hands with seven fingers
- Wolves spontaneously appearing
- No conservation of mass or causality

> *Photorealism is not understanding.*

---

## Genie — the *interactive* turn

Bruce et al., DeepMind, Feb 2024 ([arXiv:2402.15391](https://arxiv.org/abs/2402.15391)). **11B parameters**, trained on **200,000 hours** of unlabeled internet gameplay video.

Three components:

- **Video tokenizer** — ST-Transformer (spatiotemporal) VQ-VAE
- **Latent action model** — infers a *learned* discrete action vocabulary (8 actions, embedding size 32) from observed transitions, **with no action labels**
- **Dynamics model** — autoregressive ST-Transformer with MaskGIT, predicts next frame token given past tokens + latent action

**Subtlety**: the latent actions are *not interpretable* — "action 1" might mean diagonal-up, not "jump." Is Genie a true *world model* or just a *controllable video generator*? Debated.

---

## Genie 3 — real-time playable worlds

![Genie 3 navigable scene](https://lh3.googleusercontent.com/BirHhzrtzHQtXvNG0dXcj9QFA7vGUNWQYbeQiMiY7RL8a7-xGrLDGybf9YZZ1nML7lRVZh4KYARz0PFO8xBAfe7FAatH6xVlRnakHtnESHlpzEBTEA=w2880-h1620-n-nu-rw-lo)

DeepMind, Aug 2025. **24 fps, 720p, several minutes** of consistent dynamics from a text prompt. No hard-coded physics.

Key new mechanism: **promptable world events** — mid-session text commands ("start a thunderstorm") modify the running simulation.

*Source: [DeepMind Genie 3 blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/).*

---

## Cosmos — physical AI foundation model

NVIDIA, Jan 2025 ([arXiv:2501.03575](https://arxiv.org/abs/2501.03575)). Explicitly a **"World Foundation Model"** for physical AI.

- Trained on **20M hours of real-world video** (vs. Genie's gameplay)
- Both **diffusion** and **autoregressive** variants
- Built-in evaluation for **physical alignment** — conservation, plausibility
- Aimed at robotics, autonomous driving, simulation

Strategy: the **Hugging Face of world models** — open weights, downstream fine-tuning recipes.

---

## Architectural comparison & central tension

| | **Sora** | **Genie 3** | **Cosmos** |
|---|---|---|---|
| Input | Text prompt | Text + interactive | Text / image / video |
| Output | Fixed-length video | Real-time playable world | Video, conditioned |
| Backbone | DiT (diffusion + transformer) | Autoregressive latent diffusion | Both DiT and AR |
| Training data | Mixed video | Gameplay video | Real-world video |
| Primary goal | Cinematic generation | Interactive simulation | Physical realism |

But all three share the same problems:

- **Compounding error** in long autoregressive rollouts
- **No causal grounding** — pixel-level loss does not enforce physics
- **Unclear utility for planning** — can an agent actually use these for decision-making?

---

## §4.2 — From video to embodied environments

> **[FIGURE PLACEHOLDER — Survey Fig. 4: classification of interactive embodied environments]**

Survey §4.2 distinguishes video models that you *watch* from environments that you *act in*.

Three sub-axes:

- **Indoor** — homes, offices (navigation, manipulation)
- **Outdoor** — streets, drives (autonomous navigation)
- **Dynamic** — weather, traffic, multi-agent dynamics layered on top

Watching ≠ acting: an embodied environment must accept arbitrary actions and remain physically consistent.

---

## Indoor embodied environments

| Environment | Year | Distinctive feature |
|---|---|---|
| **AI2-THOR / RoboTHOR** | 2017 | Photorealistic apartments, manipulation |
| **Habitat 2.0 / 3.0** | 2021–23 | Fast simulation; humanoids and rearrangement |
| **ProcTHOR** | 2022 | **Procedural** generation — 10K+ houses |
| **Holodeck** | 2024 | LLM-generated scenes from text |
| **RoboGen** | 2024 | Auto-generated robotic tasks + assets |

Trade-offs: photorealism vs. physics fidelity vs. scale. Procedural generation now dominates.

---

## Outdoor & dynamic embodied environments

- **CARLA** (2017) — open-source autonomous-driving simulator with weather, traffic, pedestrians.
- **MetaDrive** (2021) — procedural driving scenarios for ML benchmarks.
- **GAIA-1** / **DriveDreamer** / **OccWorld** — *neural* driving simulators (Part 4).

The shift: from *hand-built simulators* (rigid physics, fixed assets) to *learned simulators* (latent dynamics, generative content). Each trades exactness for diversity.

---

# Part 4 — Applications

*Survey §5 — where world models meet the real world.*

---

## §5.1 — Game Intelligence

**Game generation**: Genie 1/2/3 turn text or image prompts into *playable* games. The action space is *learned from gameplay video*, not specified.

**Game agents**: Dreamer 4 plans entirely offline in Minecraft and collects diamonds via long-horizon imagination.

Why games matter for the survey:
- A **closed environment** with rich action spaces — clean evaluation.
- A **bridge**: rules/physics partial; aesthetics matter; long-horizon planning matters.
- The **substrate** for many later applications (sim-to-real, social agents).

---

## §5.2 — Embodied Intelligence (a): implicit reps for control

The implicit / JEPA lineage applied to robots:

- **DINO-WM** ([Zhou et al. 2024](https://arxiv.org/abs/2411.04983)) — world model on **frozen DINOv2 features**; zero-shot planning.
- **V-JEPA** (Bardes et al. 2024) — masked-feature prediction on internet video; generic visual world model.
- **GR-2**, **SWIM**, **DayDreamer** — earlier robotic Dreamer variants.

The premise: the latent need not reconstruct pixels — it only needs to *support action*.

---

## §5.2 — Embodied Intelligence (b): V-JEPA 2

Two-stage training (Meta, June 2025):

1. **Pretrain** on >1 million hours of internet video — *action-free* masking objective
2. **Post-train** an action-conditioned variant on just **62 hours** of unlabeled robot video → **V-JEPA 2-AC**

Demonstrates **zero-shot robotic planning from image goals**.

77.3% top-1 on Something-Something v2; 39.7 R@5 on Epic-Kitchens-100.

> Connects internet-scale pretraining (LLM lineage) with planning (Dreamer lineage).

> **[FIGURE PLACEHOLDER — Survey Fig. 6: robotic WM trajectory]**

---

## §5.2 — Embodied Intelligence (c): sim-to-real & generative simulators

The complementary direction: instead of using a WM to *plan*, use it to *generate training data*.

- **Cosmos** (NVIDIA) — open-weights generative simulator targeted at robotics fine-tuning.
- **NVIDIA Isaac Lab** + **MimicGen** — synthetic demonstration generation.
- **Sim-to-real** still hinges on *domain randomisation*, but the randomiser is now learned.

The key open question: do agents trained in a generative WM behave differently than those trained in a hand-built one? *Survey §6 says: not yet known.*

---

## §5.3 — Urban Intelligence (autonomous driving)

> **[FIGURE PLACEHOLDER — Survey Fig. 5: AV/urban WM application framework]**

The survey distinguishes two AV uses of world models:

**(a) Implicit representations for AV perception/planning** — latent occupancy or BEV prediction:
- **OccWorld**, **OccSora** — voxel-level 3D occupancy world models
- **UniAD**, **MILE** — joint perception–prediction–planning latents

**(b) Full driving simulators** — neural surrogates for CARLA-style sims:
- **GAIA-1** (Wayve) — 9B-param video WM conditioned on actions and text
- **DriveDreamer**, **Drive-WM**, **Vista** — diffusion-based controllable drives

Cosmos targets this market — link back to slide 33.

---

## §5.4 — Societal Intelligence (a)

A *world model* for a **social** environment, not a physical one.

- **Generative Agents / Smallville** ([Park et al. 2023](https://arxiv.org/abs/2304.03442)) — 25 LLM-driven agents in a simulated town; emergent dating, gossip, party planning.
- **AgentGroupChat** — social dynamics in multi-agent dialogue.
- **EconAgent** — macroeconomic simulation with LLM consumers and firms.
- **GovSim** — governance and norm-formation experiments.

> **[FIGURE PLACEHOLDER — Survey Fig. 7: social simulacra]**

---

## §5.4 — Societal Intelligence (b)

What *societal* world models need that physical ones don't:

- **Theory of mind** — agents must model each other's beliefs.
- **Value alignment** — simulated populations should reflect target distributions.
- **Controllability** — for policy experiments, interventions must be precise.
- **Privacy & ethics** — modelling real people is fraught.

Survey claim: this is the *youngest* sub-field and the most under-evaluated.

---

## Each application surfaces a different open problem

| Domain | Dominant unsolved problem |
|---|---|
| **Game intelligence** | Long-horizon coherence; learned-action interpretability |
| **Embodied intelligence** | Sim-to-real transfer; physics consistency |
| **Urban intelligence** | Causal physics; safety-critical evaluation |
| **Societal intelligence** | Value alignment; ethics; benchmark scarcity |

These map directly onto the open-problem categories of survey §6 — Part 5.

---

# Part 5 — Open Problems & the JEPA Counterpoint

*Survey §6, plus the alternative LeCun proposes.*

---

## LeCun's critique ([2022 AMI position paper](https://openreview.net/forum?id=BZ5a1r-kVsf))

> *Generative pixel-prediction is fundamentally wasteful.*

Most pixels carry no decision-relevant information. Predicting them spends capacity on irrelevant texture detail.

LeCun proposes an alternative: predict **representations** of the future, not pixels.

This becomes the **Joint-Embedding Predictive Architecture (JEPA)**.

---

## JEPA architecture

![JEPA vs generative architectures](https://scontent-lga3-3.xx.fbcdn.net/v/t39.2365-6/347632356_201757625702076_1813962196800732436_n.png?_nc_cat=102&ccb=1-7&_nc_sid=e280be&_nc_ohc=_uLGhys0OkwQ7kNvwFGAGyJ&_nc_oc=AdpurO8E2VRqlIMy27p1Slx1qxUuzNfjJ7qWG5XGenndwS-z7OzAhawxYfgZ6KUEb3w&_nc_zt=14&_nc_ht=scontent-lga3-3.xx&_nc_gid=ld4pnAXoy4I7zYthbFs1Hw&_nc_ss=7b289&oh=00_Af1_C0lJATWz_WMlQOw61OJmyU8nbOx5Mm31aymruIRw2Q&oe=6A079DA0)

- **Generative** (left, Masked AE): decode pixel $y$ from $x$
- **JEPA** (right): predict the *embedding* $s_y$ from $s_x$, conditioned on action / latent $z$

Loss is computed in **representation space**, not pixel space.

*Source: [Assran et al., I-JEPA (CVPR 2023)](https://arxiv.org/abs/2301.08243). Note: the Facebook CDN URL above is a transient signed link — replace with a stable mirror if it breaks.*

---

## JEPA loss & non-collapse

$$\mathcal{L}_{\text{JEPA}} = \big\| \, \text{Predictor}\big(\text{Enc}_\theta(x),\, a\big) \;-\; \text{sg}\!\left[\text{Enc}_{\bar\theta}(y)\right] \,\big\|^2$$

Plus a **non-collapse mechanism** (EMA target encoder, VICReg regularization, or similar).

| | **MAE** | **Contrastive** | **JEPA** |
|---|---|---|---|
| Predicts | Pixels | Distance to negatives | Representations |
| Needs negatives? | No | Yes | No |
| Collapse risk | Low | Low | **High** (needs reg) |

---

## Where JEPA fits in the lecture map

JEPA is in the same family as:

- **Contrastive image SSL** (SimCLR, MoCo, DINO)
- **GNN representation learning** (callback to your GNN lecture)
- **MuZero** (implicit prediction, no reconstruction)

The "implicit world model" lineage running parallel to the "generative" one — and in 2025–26, the two camps are increasingly competing for attention.

---

## Open problem 1 — physical rules & counterfactual simulation

Next-frame prediction loss does **not** measure physical understanding. New benchmarks shift toward causally grounded metrics:

- **IntPhys 2** (Bordes et al., 2025) — intuitive physics: object permanence, gravity, solidity
- **CausalProbe** — causal ordering of events
- **Long-horizon consistency** — does the world stay coherent over 60s?

Even with photorealistic outputs, persistent failure modes remain: **compounding error** in long rollouts, **conservation violations** (mass, energy, identity), **action-conditioning brittleness**, lack of **causal vs. correlational** structure.

---

## Open problems 2–4 — social, sim-to-real, efficiency

**Social dimension** *(§6.2)*. Subjective evaluation at scale; behavioural/cognitive theory needed; alignment of simulated populations.

**Sim-to-real** *(§6.3)*. Multi-modal, multi-task, 3D capability gaps; knowledge transfer from generative WMs to real robots remains unproven.

**Efficiency** *(§6.4)*. Autoregressive video generation is sequential; 20M-hour training corpora are unsustainable. Real-time deployment (Genie 3 at 24 fps) was an engineering tour-de-force, not yet a generic capability.

---

## Open problem 5 — ethics, safety, authenticity

*Survey §6.5.*

- **Privacy** — training on internet video captures real people without consent.
- **Misuse** — high-fidelity simulators enable deepfakes and synthetic decision-making at scale.
- **Watermarking & detection** — provenance tooling lags generation capability.
- **Evaluation gaming** — "looks plausible to humans" is not the same as "is causally correct."

Compute and licensing burden of video pretraining at scale further concentrates the field in a few large labs.

---

# Synthesis & Wrap-Up

---

## The V/M/C blueprint, evolved

| Module | 2018 | 2025 — Understanding | 2025 — Predicting |
|---|---|---|---|
| **V** (perception) | ConvVAE | DINO / JEPA encoder | VQ-VAE → ViT spacetime patches |
| **M** (dynamics) | MDN-RNN | RSSM / latent transformer | DiT / autoregressive ST-Transformer |
| **C** (control) | Linear + CMA-ES | Actor-critic in imagination; MCTS | Text prompts; promptable events |

Every prior lecture in the course (FCN, CNN, VAE, GAN, transformer, ViT, diffusion, GNN) plays a role somewhere in this table.

---

## The 2025–2026 inflection

- **Dreamer 4** solves Minecraft Diamond from offline data (Sep 2025)
- **Genie 3** real-time interactive worlds (Aug 2025)
- **V-JEPA 2** demonstrates zero-shot robotic planning (Jun 2025)
- **Cosmos 2.5** — open World Foundation Models, 2M+ downloads (Oct 2025)
- **World Labs** launches **Marble** — commercial 3D world generation (Nov 2025)
- **LeCun leaves Meta** to start **AMI Labs**, raising €500M to build JEPA-style world models (late 2025)

> The field is forking into a generative-video lineage and a JEPA lineage — but both branches share the survey's two-camp DNA.

---

## Three open questions for you

1. Is the path to general AI through **scaling LLMs**, **scaling generative world models**, or **scaling JEPA-style implicit world models** — or fusing them?

2. Can a model trained only on *pixels* learn *causal* physics, or do we need **action-conditioned data** at scale?

3. What is the right **evaluation metric** for a "good" world model — and who decides?

---

## Reading list

**Survey (the spine of this lecture)**:
- Ding et al., *Understanding World or Predicting Future? A Comprehensive Survey of World Models*, ACM CSUR 2025 — [arXiv:2411.14499](https://arxiv.org/abs/2411.14499)
- Curated paper list & figures: [github.com/tsinghua-fib-lab/World-Model](https://github.com/tsinghua-fib-lab/World-Model)

**Primary sources**:
- Ha & Schmidhuber, *World Models* — [worldmodels.github.io](https://worldmodels.github.io)
- DreamerV3 — Hafner et al., *Nature* 2025; [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Dreamer 4 — Hafner et al., [arXiv:2509.24527](https://arxiv.org/abs/2509.24527)
- Sora — OpenAI technical report (Feb 2024)
- Genie — Bruce et al., [arXiv:2402.15391](https://arxiv.org/abs/2402.15391); DeepMind blogs for Genie 2 / 3
- Cosmos — NVIDIA, [arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
- V-JEPA 2 — Assran et al., [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- DINO-WM — Zhou et al., [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- Generative Agents — Park et al., [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)

*Questions?*
