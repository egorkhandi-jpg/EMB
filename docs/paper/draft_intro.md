## 1 Introduction

Language models are increasingly proposed as components of systems that monitor and control physical
processes — as planners for embodied agents [Ahn 2022; Huang 2022], as policies initialized from
pretrained weights [Li 2022; Reid 2022], and, in industrial settings, as diagnostic layers that
interpret sensor streams and explain anomalies to human operators [Dave 2024; Li 2026; Liu 2025]. In
safety-critical infrastructure such as power generation, the value of such a component depends less
on how fluent its explanations are than on whether what it says reflects what it has actually
registered. A diagnostic layer whose verbal report diverges from its internal state is a liability:
operators act on the report, not on the state.

This question is not hypothetical for us. The author operates gas-engine power generation and develops
KONTUR, an observation-only monitoring system for a generator fleet that includes a machine-learning
diagnostic layer producing technician-facing explanations. In that system no automated decision touches
the plant; every action is taken by an operator. Deciding whether a language model can be trusted in
such a role — and what must be read from it, its output or its internal state — requires knowing how
much of what a model represents about a physical situation actually reaches its output.

We study this question in a deliberately minimal setting, designed to separate mechanisms rather than
to be realistic. A simulated agent in a two-dimensional foraging environment experiences discrete body
events — which kind of item it ate and whether its energy rose or fell — and these events are presented
to a frozen pretrained language model as a short token sequence. A latent variable of the episode,
which item kind is currently food, is never stated in the context; it can only be inferred from the
conjunction of events. This lets us ask three precise questions: what pretraining contributes to
processing such event streams, whether the latent variable is represented in the model's hidden states,
and whether the model can report it when asked directly.

Across three model families, we find a consistent dissociation between representation and report. The
latent variable is linearly decodable from hidden states almost perfectly, while the model's own answer
recovers it only weakly; the meaning of a word is fully present at that word's position but barely
reaches the position where an episode-level readout would use it; and when word meaning conflicts with
in-context experience, the experience leaves a measurable but small trace in the answer and essentially
does not change it. Pretraining contributes sequence-processing machinery that works equally well on
arbitrary symbols, rather than the meaning of the words. Probing studies on factual knowledge have
reported related gaps between what models encode and what they express [Orgad 2024; Gekhman 2025;
Burns 2022; Azaria 2023]. Unlike prior studies of hidden factual knowledge, our target variable is
not stored in pretrained parameters: it is constructed online from the agent's current experience.

Our contributions are:
1. a controlled setting in which a latent situation variable must be inferred from in-context body
   events, with a random-weight control of the same architecture for every comparison;
2. measurements of the gap between representation and report — for a latent episode variable, for word
   meaning and for meaning against experience — replicated across Pythia, Qwen2.5 and OLMo-2, which we
   interpret as a three-layer picture of representation, routing and expression;
3. a preregistered methodology with timestamped commits and a registry of all attempts, including a
   documented set of confounds that misled or would have misled our conclusions during the project — survivorship-induced
   frequency cues, shortcuts through the agent's own closed-loop behavior, information loss in symbolic
   interfaces, and readout artifacts — together with the controls that caught them.

We make no claim that these findings transfer directly to deployed monitoring systems, which use larger
models, richer inputs and often trained output channels. The practical implication we draw is narrower:
in a role where a language model's output is taken as evidence about a physical system, the
correspondence between its output and its internal state should be measured rather than assumed.
