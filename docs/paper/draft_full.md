# Knowing More Than It Can Say — full draft (focus 1)

<!-- Собрано build_draft.py 2026-10-07 из docs/PAPER_OUTLINE.md и docs/paper/draft_*.md. Комментарии-источники сохранены; в итоговом тексте будут удалены. -->

## Abstract

Title options:
- Knowing More Than It Can Say: Frozen Language Models in Minimal Bodies
- Represented but Not Reported: Probing Frozen Language Models as Embodied Readers

Frozen pretrained language models are increasingly used as components of embodied agents, yet it
is unclear what pretraining contributes in such settings and how much of what the model internally
represents is available through its output. We study this in a minimal two-dimensional foraging
environment where an agent's body events are presented to a frozen language model as a short
sequence of discrete tokens, and compare each pretrained model against the same architecture with
random weights and matched embedding norms. Across three model families (Pythia, Qwen2.5, OLMo-2;
1–1.5B parameters, with Pythia up to 6.9B for some analyses), we find that: (i) where pretraining helps, the advantage comes from sequence-processing machinery rather than word
meaning — arbitrary event words work as well as or better than meaningful ones, and in one family
(Qwen2.5) the advantage appears only with arbitrary words; (ii) a latent episode variable that is never
stated explicitly in the context is linearly decodable from hidden states with AUC 0.93–0.99, but
the model's own answer to a direct question recovers it only at AUC 0.61–0.64, and in Pythia,
in-context examples from other episodes do not close this gap; (iii) the meaning of a word (food
vs. harmful) is fully present at that word's position but reaches the final position, where an episode-level readout
would use it, at most weakly (0.97–1.00 vs. 0.51–0.62); and (iv) when word meaning
conflicts with in-context experience, a statistically reliable trace of the experience reaches the
answer but is substantially weaker than the meaning effect (4–28% of it) and leaves the answers
essentially unchanged. All primary analyses were preregistered with timestamped commits, use exact
permutation or bootstrap tests with Holm correction, and include controls for confounds that misled or
would have misled our conclusions during the project, such as survivorship-induced frequency cues. Taken together, the
results suggest that frozen language models in this regime represent substantially more about
their embodied situation than they can express, and that the bottleneck lies in routing and
expression rather than in representation.

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
does not change it. Where pretraining helps, it contributes
sequence-processing machinery that works on arbitrary symbols at least as well as on meaningful ones,
rather than the meaning of the words; in Qwen2.5 the advantage appears only with arbitrary symbols. Probing studies on factual knowledge have
reported related gaps between what models encode and what they express [Orgad 2024; Gekhman 2025;
Burns 2022; Azaria 2023]. Unlike prior studies of hidden factual knowledge, our target variable is
not stored in pretrained parameters: it is constructed online from the agent's current experience.

Our contributions are:
1. a controlled setting in which a latent situation variable must be inferred from in-context body
   events, with a random-weight control of the same architecture for every comparison;
2. measurements of the gap between representation and report — for a latent episode variable, for word
   meaning and for meaning against experience — replicated across Pythia, Qwen2.5 and OLMo-2, which we
   interpret as a three-layer picture of representation, routing and expression;
3. a preregistered methodology with timestamped commits and a registry of all attempts, including a documented set of confounds that misled or would
   have misled our conclusions during the project — among them survivorship-induced frequency cues and
   shortcuts through the agent's own closed-loop behavior — together with the controls that caught them
   (Section 4).

We make no claim that these findings transfer directly to deployed monitoring systems, which use larger
models, richer inputs and often trained output channels. The practical implication we draw is narrower:
in a role where a language model's output is taken as evidence about a physical system, the
correspondence between its output and its internal state should be measured rather than assumed.

## 2 Setup

### 2.1 Environment

All analyses use a minimal two-dimensional foraging world, *flip-2* <!-- env_flip.py, ATTEMPTS п. 4 -->. An episode lasts 100 steps <!-- env_flip.py T --> in the unit square, with two items of each of two kinds, A and B <!-- env_flip.py N_PER -->. In every episode one kind is food (+0.35 energy) and the other is poison (−0.20) <!-- env_flip.py FOOD_GAIN; ENVS flip-2 -->. Which kind is food — the *episode sign* — is drawn by a coin at the start, flips once at a random step between 30 and 69 <!-- env_flip.py REV_LO, REV_HI -->, and is never observed directly. The body reports only events: which kind, if any, was eaten on this step, and whether energy went up, down, or stayed the same. Energy drifts down without food, so the body has its own time. For the analyses reported here, trajectories are generated by a *blind* policy, a reflex that steers to the nearest item of either kind. The model therefore reads experience that it did not shape, and closed-loop shortcuts through behavior are excluded <!-- A13; RESULTS_2026-10-01_flip.md v3-replay -->.

### 2.2 Interfaces

**Event words.** Each step is written as two tokens: the kind eaten (" A", " B", or " none") and the outcome (" up", " down", or " same"). The window is preceded by a start token and covers the last 32 steps <!-- A3; A28 -->. The episode sign is not stated anywhere in the window. It can only be inferred by binding a kind to the outcomes that follow it. In an earlier interface, a fixed random projection of continuous sensor features into embedding space, pretrained models did not outperform random ones (Section 3.1). For the arbitrary-word control (Section 3.2), the six event words are replaced by unrelated single-token words (e.g. " cat", " table", " stone" / " river", " chair", " door") <!-- A4; replicate.py ARB -->.

**Questions.** To read the model's *answer*, a question is appended to the window (" Food:", " Safe to eat:", " Good kind:"), and the answer is the logit difference between " A" and " B" at the last position <!-- A22 -->.

**Kind words with meaning.** To test meaning (Section 3.4), the two kind symbols are replaced by words whose meaning is food (e.g. apple, bread) or harmful (e.g. mold, glass). These come from pools of 30 single-token words each, and words that name edibility directly are excluded <!-- A24, meaning_words.json -->.

**Synthetic windows.** To separate meaning from experience without any frequency cue (Section 3.5), we construct windows of short sentences ("I ate w. Energy went up." / "… went down." / "Nothing happened.").
- Each window contains n ∈ {2, 4, 6} events, with exactly n/2 per kind. The kinds of the first and last events, and the reference kind used for scoring, are balanced within every cell <!-- A27 -->.
- A frequency reference, a last-mentioned reference and a first-mentioned reference all score exactly 0.5 on these windows. A Bayesian reader of the kind–outcome binding scores 1.0 <!-- A27, проверка конструкции -->.
- The control keeps words, positions and order, and permutes outcomes across events so that the binding is uninformative <!-- A27 -->.
- Ten sets of 648 windows are used <!-- A27 -->.

### 2.3 Models and controls

- **Models:** Pythia 160M–6.9B, Qwen2.5-1.5B and OLMo-2-1B, all frozen and run in fp32 with TF32 disabled <!-- CLAUDE.md, правило 10; A28 -->.
- **Random-weight control.** Every pretrained model is compared with the same architecture with random weights; its input embeddings are rescaled to the mean row norm of the pretrained embeddings <!-- A3; A28 -->. The random network is reseeded for every set, except in the meaning probe (Section 3.4), which uses one random network per model <!-- A24 -->.
- **Tokenizer transfer.** For the cross-family replication, every protocol was transferred without other changes <!-- A28 -->:
  - the start token is each model's end-of-document token, "<|endoftext|>". It is the BOS token in Pythia and OLMo-2; Qwen2.5 has no BOS by default <!-- A28 -->;
  - every word is a single token with a leading space in all three tokenizers, so no substitutions were needed <!-- A28 -->;
  - all inputs were verified by decoding;
  - Pythia-1B run through the transferred code reproduced the original numbers (e.g. 0.983 / 0.526 / 1.000 for the meaning probe) <!-- A28, проверка переноса -->.

### 2.4 Measures

**Probes.** Probes are L2-regularized logistic regressions on the final-layer hidden state. Features are standardized on the training part of each fold, and the regularization strength is chosen by inner validation <!-- probe_fast.py; meaning_probe.py -->. Folds are grouped by episode when the probe is tested on new episodes with the same symbols (Sections 3.1–3.3). When the probe must generalize to new words, folds are grouped by word, so test words are never seen in training (Section 3.4) <!-- A24 -->. The sign probe reports balanced accuracy <!-- A3, A4 -->; the other probes report AUC.

**Answers.** Answers are logit differences between the two answer tokens after a question, scored by AUC against the true sign <!-- A22 -->, or as mean logit shifts (Section 3.5) <!-- A27 -->. Reading logits rather than sampled text follows the argument that prompted judgments can understate what a model encodes [Hu 2023].

### 2.5 Statistics and preregistration

- **Main test.** Exact paired sign-flip permutation tests over 10 sets (1,024 permutations), one-sided where the direction was preregistered <!-- A22, A27, A28 --> and two-sided in the earliest analyses <!-- A3, A4 -->.
- **Meaning probe.** It used a one-sided bootstrap (10,000 resamples) <!-- A24 -->.
- **Intervals.** Where a probe generalizes to new words, intervals are cluster bootstraps over words, not over episodes <!-- уточнение 2026-10-07, RESULTS_2026-10-05_meaning.md -->. Other intervals are 95% t-intervals over sets.
- **Multiplicity.** Holm correction within each preregistered family.
- **Preregistration.** Every primary analysis was preregistered in a commit made before any data were computed, with all outcomes and their interpretation written down in advance. Every attempt, including abandoned environment variants, is listed in a registry <!-- ATTEMPTS.md -->. Analyses added after seeing data are labeled post hoc.

## 3 Results

### 3.1 The interface decides whether pretraining helps

**Continuous projector.** Pretrained Pythia was no better than random weights when body states entered through the continuous projector: at 160M it was significantly worse (−0.053, Holm p = 0.012) <!-- A2 -->.

**Start token and event words.** Adding a start token removed most of the gap: at 160M the pretrained–random difference went from −0.054 to −0.008, an improvement of +0.047 (Holm p = 0.004) <!-- A3, b − a -->. Writing events as words gave pretrained models their first advantage: relative to the projector interface, the pretrained–random gap improved by +0.191 (Holm p = 0.004), giving sign-probe balanced accuracy 0.950 vs. 0.813 at 160M <!-- A3, c − a -->, and, descriptively, 0.951 vs. 0.856 at 410M <!-- A3; 410M — описательно -->.

**Replication.** The advantage replicates on OLMo-2-1B (0.963 vs. 0.938, Holm p = 0.006) <!-- A28 R1 --> and, descriptively, on Pythia-1B (0.976 vs. 0.906) <!-- A28 R1 -->. It does **not** replicate on Qwen2.5-1.5B with these event words: 0.906 vs. 0.921, with the pretrained model lower in 9 of 10 sets <!-- A28 R1 -->. With arbitrary event words, pretrained Qwen2.5 is again clearly ahead (0.983 vs. 0.920) <!-- A28 R1, R2 --> (Fig. F2). We did not investigate why the letter-like symbols disadvantage Qwen2.5. The claim that event words let pretraining help therefore holds with a caveat about model family and symbol set.

### 3.2 Machinery, not meaning

**Pythia-160M.** If the advantage came from the meaning of the event words, unrelated words should remove it. They do not. At 160M, arbitrary words were *better* than meaningful ones by 0.016 (Holm p = 0.004), and pretrained models on arbitrary words beat random weights by 0.156 (Holm p = 0.004) <!-- A4 -->. Replacing the six event embeddings of the pretrained model with random vectors of the same norm still gave, descriptively, 0.984 <!-- A4, e1 -->. An apparent preference for one arbitrary word set was later traced to token luck, so we state the result as "not worse" rather than "better" <!-- v3-replay -->.

**Replication.** Across the three families, arbitrary words are non-inferior to meaningful ones and above random weights (Holm p = 0.006 in Qwen2.5 and OLMo-2 for the preregistered test requiring both; Pythia-1B descriptively) <!-- A28 R2 -->. The differences against meaningful words are +0.016 for Pythia-1B, +0.078 for Qwen2.5 and +0.024 for OLMo-2 <!-- A28 R2 --> (Fig. F2).

What pretraining contributes here is sequence-processing machinery over discrete symbols, not the meaning of the symbols. This parallels findings that language-model initialization helps decision making whether or not inputs are natural language [Li 2022], and that random transformers already have some algorithmic capabilities [Zhong 2024].

### 3.3 Represented but not reported

**Representation.** The episode sign is never stated in the window, yet a linear probe at the question position decodes it with AUC 0.993, 0.932 and 0.975 (Pythia-1B, Qwen2.5, OLMo-2; first question) <!-- A28 R3 -->.

**Answer.** Asked directly, the same models answer far worse: AUC 0.607–0.612, 0.623–0.637 and 0.615–0.627 across the three questions <!-- A28 R3 -->. This repeats the original Pythia result (zero-shot AUC 0.61 against a probe at 0.98–0.99) <!-- A22 -->.

**Preregistered criterion.** The answer's excess over chance is at most half of the probe's, (AUC_answer − 0.5) ≤ 0.5 · (AUC_probe − 0.5). It holds for all three questions in all three families <!-- A28 R3 --> (Fig. F1). The probe is trained on the hidden state at the same position from which the answer is read; it is thus a trained linear bridge from that state. The gap between the probe and the answer therefore points to the mapping from this state to the answer tokens, rather than to an absence of the information at this position. <!-- report_body.run_lm: зонд и ответ — одно скрытое состояние на последней позиции -->

**Two observations qualify this picture.**
- Random-weight networks also support high probes (0.892–0.937), while their answers stay at chance (0.50–0.53) <!-- A28 R3 -->. Descriptively, pretraining adds more to the answer channel (+0.09 to +0.13 AUC) than to the probe (+0.04 to +0.06) <!-- A28 R3; PAPER_OUTLINE сверка -->. We did not test this difference.
- In Pythia only, in-context examples did not close the gap. Examples from the same episode were simply copied: few-shot AUC 0.974 matched a "repeat the last example's answer" control at 0.975 <!-- A22 -->. Examples from other episodes gave chance (AUC 0.505 at 160M, 0.469 at 1B), not above random weights or the copy control (Holm p ≥ 0.6) <!-- A23 -->. A23 was run on Pythia only.

The pattern — internal states that encode more than the model's output reveals — has been reported for factual knowledge and truthfulness [Orgad 2024; Gekhman 2025; Azaria 2023]; here it concerns a variable that exists only in the agent's recent bodily experience.

### 3.4 Meaning stays local

**Meaning in language.** The models know which words denote food. A probe on the word's hidden state in a neutral sentence ("The apple."), tested on held-out words, separates food from harmful words with AUC 1.000, 1.000 and 0.999 for Pythia 1B, 2.8B and 6.9B <!-- A24 -->, and 1.000 for Qwen2.5 and OLMo-2 <!-- A28 R4 -->. Direct words (edible vs. poison) serve as a positive control for the method (1.000 / 0.967 / 0.967; 0.900; 0.967) <!-- A24; A28 R4 -->.

**Meaning in the event window.** When the same words appear as kind symbols, the probe — tested on words never seen in training — reads their meaning at the kind word's own position: 0.983–0.999 for Pythia <!-- A24 -->, 0.972 for Qwen2.5 and 0.996 for OLMo-2 <!-- A28 R4 -->. At the final position, where an episode-level readout would look, it reaches at most 0.53–0.62: 0.526 / 0.586 / 0.620 for Pythia 1B / 2.8B / 6.9B <!-- A24 -->, 0.595 for Qwen2.5 and 0.514 for OLMo-2 <!-- A28 R4 --> (Fig. F3).

**Intervals.** Because the probe generalizes to new words, intervals are cluster bootstraps over the ~56 test words. For Pythia-6.9B at the final position, this gives 0.620 [0.51, 0.73] against 0.447 [0.33, 0.56] for random weights <!-- уточнение 2026-10-07, f3_check.txt -->. We therefore do not claim that 6.9B carries more meaning to the final position than a random network. We claim only that no model reaches the preregistered threshold of 0.8 <!-- A24; уточнение 2026-10-07 -->.

**Random-weight values below 0.5.** These (e.g. 0.315 for Pythia-2.8B) are not an artifact of the word-grouped split. A simulation of the same procedure on random word vectors gives a mean of 0.495, with independent random networks spread over 0.36–0.63 <!-- f3_check.txt -->.

The meaning of a word is thus present where the word is, but is not routed to the position that summarizes the episode. This is consistent with work on how models bind attributes to entities in context [Feng 2023].

### 3.5 Meaning against experience: a weak trace that does not change answers

**Why synthetic windows (post hoc, Pythia).** A first test used simulated bodies (Pythia 1B–6.9B) <!-- A26 -->.
- Kind words were truthful, lying (a food word naming the poisonous kind) or neutral.
- The preregistered shift toward experience was formally positive in all six cells. But it persisted when kind and outcome slots were shuffled <!-- A26 -->.
- Post hoc, we traced it to survivorship: queries came only from living bodies, so the food kind had been eaten more often (frequency reference 0.62 instead of the assumed 0.5) <!-- A26 -->. The answer followed the more frequently mentioned word, at +1.1 to +1.9 logits per mention <!-- A26, анализ после расчёта -->.
- With equal mentions, the shift vanished in 5 of 6 cells <!-- A26, анализ после расчёта -->.

This frequency channel is what the synthetic windows remove.

**Synthetic windows.** We compare windows where the kind words lie, against the same windows with outcomes permuted (Fig. F4).
- In Pythia-6.9B, experience moves the answer toward the true food by +0.020 logits relative to permuted outcomes, more than it does in random weights (Holm p = 0.002 for the preregistered test requiring both comparisons). This is small against a meaning effect of −0.51 <!-- A27, H2 -->.
- The shift replicates in Qwen2.5 and OLMo-2 (Holm p = 0.006; Pythia-1B descriptively), with random weights near zero <!-- A28 R5 -->: +0.022 (Pythia-1B), +0.098 (Qwen2.5), +0.044 (OLMo-2) logits.
- Relative to the meaning effect, the shift is 3.8% (Pythia-6.9B), 7.7% (Pythia-1B), 6.3% (OLMo-2) and 27.6% (Qwen2.5) <!-- PAPER_OUTLINE раздел 5 -->.
- Descriptively, it does not grow from 1B to 6.9B in Pythia <!-- A27 -->.
- The answers themselves barely change. The share of conflict windows answered toward experience is 0.450 vs. 0.444 with and without the binding for Pythia-1B, 0.464 vs. 0.453 for Qwen2.5, 0.392 vs. 0.372 for OLMo-2 <!-- A28 R5 -->, and 0.36 vs. 0.36 for Pythia-6.9B <!-- A27 -->.
- The preregistered companion test — that experience alone, with neutral kind words, is answered above random weights — was not met: AUC 0.509 exceeded the permuted control (p = 0.001) but not random weights (0.489 ± 0.034, p = 0.105) <!-- A27 H1 -->.

A hint from the first test (Pythia-6.9B with sentences, +0.43 logits under equal mentions) did not survive the clean construction (+0.02) <!-- A27; A26 -->.

We conclude that the binding of a word to its outcome in context leaves a statistically reliable but small trace in the answer, and that when meaning and experience conflict, meaning decides.

## 4 Pitfalls and controls

<!-- Черновик 2026-10-07. Источники после чисел будут удалены в итоговом тексте. -->

Several results and design choices during the project misled or would have misled our conclusions. For each we give what happened, how it misled or would have, and the control that caught it.

**Readout artifact in the original interface.** In the first interface, body states entered through a continuous projector, and the readout used the raw final hidden state. With that readout the pretrained Pythia lagged its random-weight twin, suggesting that pretraining hurts. Its hidden states were nearly collinear (cosine 0.986 between states) <!-- FINAL_REPORT п. 6; п. 1 -->, consistent with a few very large activations dominating the representation [Sun 2024]. With a fixed standardization of hidden states the difference disappeared: fitness (fraction of the episode survived plus half the final energy) 1.455 vs. 1.453 (3 seeds) and 1.453 vs. 1.457 in a ten-seed repeat (p = 0.20) <!-- п. 1, п. 2 -->. Standardized readouts have been the default since.

**Shortcuts through closed-loop behavior.** When a readout learns while the agent acts, the agent's own behavior becomes part of its input. In one series, a random-weight model appeared to read the episode sign better than the pretrained one, and a raw event window solved the task without a memory model <!-- A11; A13 -->. Out-of-loop evaluation on trajectories of a blind policy removed both effects.
- The raw window fell from 0.759 to 0.570: it had used a "lost, so switch" rule, not a memory of the sign <!-- A13 -->.
- The random-weight head fell from 0.670 to 0.513 <!-- A13 -->.

All analyses in this paper use trajectories the model did not shape.

**A frequency cue created by survivorship.** In the first test of meaning against experience (Pythia 1B–6.9B), queries came only from living bodies <!-- A26 -->. Bodies that ate more poison were more often dead, so among the living the food kind had been eaten more often. A frequency reference that the preregistration assumed would score 0.5 by construction scored 0.62 <!-- A26 -->. The preregistered shift toward experience was formally positive in all six cells but persisted when kind and outcome slots were shuffled. Post hoc, the answer followed the more frequently mentioned word, at +1.1 to +1.9 logits per mention; with equal mentions the shift vanished in 5 of 6 cells <!-- A26, анализ после расчёта -->. We now check every "0.5 by construction" reference on the actual inputs before preregistering. The synthetic windows of Section 3.5 pass this check exactly <!-- A27 -->.

**Information lost by a symbolic interface.** In a separate line of the project, input was quantized into event symbols. Before running any language model we computed what a simple rule could extract from them: 0.62, against 0.985 from the continuous sensors <!-- Г3; A25 -->. The model then scored 0.48–0.52 <!-- A25 -->. We recorded this null result with that caveat, not as a statement about language models.

**A condition against random weights that silently required a sign.** In the final meaning-versus-experience test, the preregistered condition required the pretrained answer on conflict windows to exceed that of random weights. By symmetry, random weights answer near zero there, so the condition demanded that experience *outweigh* meaning, not merely shift the answer. We noticed this before computing any data. In a separate commit, the condition was replaced by a comparison of shifts relative to permuted outcomes; the original condition is reported descriptively <!-- A27, дополнение до расчёта -->.

**Intervals over observations instead of units of generalization.** The meaning probe of Section 3.4 is tested on unseen words, but its intervals and test resampled episodes <!-- A24 -->. With about 56 distinct test words, this understates uncertainty: in a simulation of the same procedure, independent random networks spread over AUC 0.36–0.63 <!-- уточнение 2026-10-07, f3_check.txt -->. Cluster bootstraps over words widened the intervals roughly 1.5–2×. A preregistered claim that Pythia-6.9B carries more meaning to the final position than random weights can no longer be supported; the threshold verdict (below 0.8 for every model) is unchanged <!-- уточнение 2026-10-07 -->. Where a probe generalizes to new units, intervals and tests are now computed over those units.

## 5 Discussion

**A three-layer interpretation.** We read the results as pointing to three layers with very different
capacities, while noting that this is an interpretation of the measurements rather than a measured
mechanism. *Representation* is rich: the episode sign is linearly decodable almost perfectly, and so is
the meaning of a word. *Routing* is limited: word meaning is present at the word's own position but
barely reaches the position that summarizes the episode. *Expression* is narrow: the answer channel
recovers the latent sign only weakly, and when word meaning conflicts with in-context experience, the
answer follows the word. On this reading, the bottleneck of a frozen model reading an embodied situation
lies not in what it registers but in what reaches its output. We use representation, routing and
expression as operational descriptions of the measurements — linear decodability from a specified
hidden state, the change in decodability between positions, and the effect on the model's answer
distribution — not as claims about separately localized mechanisms.

High probe accuracy alone does not show that a model uses the information [Hewitt 2019; Belinkov 2021;
Elazar 2020], and our random-weight controls make this point concrete: random networks also support
probes at 0.89–0.94 while answering at chance. Descriptively, what distinguishes pretrained models in our data is
mainly the answer channel. We therefore treat the answer as the measure of observable use at the tested output channel, and the probe as an
upper bound on what could be used — in line with the observation that prompted judgments can
understate what a model encodes [Hu 2023], and with reports that models encode more about facts than
they express [Orgad 2024; Gekhman 2025].

**Why expression might lag.** One account, consistent with the data but not tested here, is that the
answer channel is shaped by the statistics of pretraining text: it maps contexts to words through
associations learned before the episode began. Word meaning is such an association, and it dominated
the answers in every model we tested (Section 3.5); in Pythia, a post hoc analysis suggests that mention
frequency acted the same way (Section 3.5). The episode sign is not such an association: it is a binding
of a kind to its outcomes that exists only in the current context. Work on entity tracking [Kim 2023] finds
that models pretrained only on text track the state of entities poorly, while work on binding mechanisms
[Feng 2023] shows that sufficiently large models represent in-context bindings through specific internal
mechanisms; our results add that a bound variable can be well represented and still fail to reach the output.

**Implications for monitoring critical systems.** The setting is minimal, and we do not claim direct
transfer to deployed systems. Three practical implications nonetheless follow for any role in which a
language model's output is taken as evidence about a physical process. First, the correspondence between
output and internal state should be measured, not assumed: a fluent explanation may express less than
the model has registered, or something other than it. Second, where ground truth is available, reading
internal states with calibrated probes may be more faithful than reading the model's answer — a suggestion we
have not tested in a monitoring setting. Third, names
carry priors: in our data the meaning of a word outweighed in-context evidence about the thing it named.
In a monitoring setting, labels of components, alarms or states could likewise pull an answer toward
what the label suggests rather than what the evidence shows — a failure mode worth testing explicitly
before a model's explanations are put in front of operators.

**Methodological lessons.** Several early results were misleading, and each
was caught by a specific control (Section 4): two false positives (shortcuts through the agent's own
closed-loop behavior, and a frequency cue created by survivorship), one overstated significance
(intervals computed over episodes rather than over the words a probe generalizes to), one false
negative (a readout artifact consistent with massive activations [Sun 2024]), one null result
that reflected information lost by a symbolic interface, and a condition against random weights that
silently required a sign, caught before any data were computed. We list these not as
contributions in themselves but as risks that apply to any evaluation of language models embedded in
agents, where the data a model sees are shaped by the agent that produces them.

**Limitations.** Models are frozen and small: replication covers 1–1.5B parameters, and Pythia extends to
6.9B for some analyses. We did not study instruction-tuned models, whose answer channel is trained to
report and may behave differently — the most important open question for the practical implications
above. The environment is a toy designed to separate mechanisms, not to resemble a real process. Probes
are linear; high linear decodability indicates accessible information, but low decodability does not establish its absence; answers are read as a logit difference between two tokens rather than as free text. The
cross-episode in-context test was run on Pythia only. Code was implemented and run with an AI coding
assistant under the author's direction; all preregistrations, code and results are released.

**Future work.** Three directions follow directly: instruction-tuned and larger models; the minimal
trainable bridge from internal state to answer — how much plasticity a frozen model needs to report what
it represents; and the same protocol on real sensor streams from an operating plant.

## Disclosure

Code implementation and experiment execution were performed with an AI coding assistant (Claude Code)
under the author's direction; hypotheses, preregistrations, decisions to stop or continue each line,
and interpretation are the author's. All code, preregistration commits, and results are released.

## References

- Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. arXiv:2204.01691.
- Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. arXiv:2304.13734.
- Belinkov, Y. (2021). Probing Classifiers: Promises, Shortcomings, and Advances. arXiv:2102.12452.
- Burns, C., et al. (2022). Discovering Latent Knowledge in Language Models Without Supervision. arXiv:2212.03827.
- Dave, A. J., et al. (2024). Integrating LLMs for Explainable Fault Diagnosis in Complex Systems. arXiv:2402.06695.
- Elazar, Y., et al. (2020). Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. arXiv:2006.00995.
- Feng, J., & Steinhardt, J. (2023). How do Language Models Bind Entities in Context? arXiv:2310.17191.
- Gekhman, Z., et al. (2025). Inside-Out: Hidden Factual Knowledge in LLMs. arXiv:2503.15299.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. arXiv:1909.03368.
- Hu, J., & Levy, R. (2023). Prompting is not a substitute for probability measurements in large language models. arXiv:2305.13264.
- Huang, W., et al. (2022). Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. arXiv:2201.07207.
- Kim, N., & Schuster, S. (2023). Entity Tracking in Language Models. arXiv:2305.02363.
- Li, S., et al. (2022). Pre-Trained Language Models for Interactive Decision-Making. arXiv:2202.01771.
- Li, B., & Zhao, C. (2026). S2S-FDD: Bridging Industrial Time Series and Natural Language for Explainable Zero-shot Fault Diagnosis. arXiv:2603.08048.
- Liu, Y., et al. (2025). Evaluation of Large Language Models for Numeric Anomaly Detection in Power Systems. arXiv:2511.21371.
- Orgad, H., et al. (2024). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. arXiv:2410.02707.
- Reid, M., et al. (2022). Can Wikipedia Help Offline Reinforcement Learning? arXiv:2201.12122.
- Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). Massive Activations in Large Language Models. arXiv:2402.17762.
- Zhong, Z., & Andreas, J. (2024). Algorithmic Capabilities of Random Transformers. arXiv:2410.04368.

## Figure captions

**Figure F1 — Probe vs. answer for the latent episode sign.** AUC against the episode sign of a linear probe at the question position (blue) and of the model's answer, the logit difference " A" − " B" (orange), for three zero-shot questions. Filled markers: pretrained; hollow: the same architecture with random weights and matched embedding norms. flip-2, 32-step event-word window; 10 sets × 100 episodes per model; mean and 95% t-interval over sets. Data: A28 (protocol of A22). File: `figures/en/F1_probe_vs_answer.pdf`.

**Figure F2 — Meaningful vs. arbitrary event words.** Balanced accuracy of a linear probe for the episode sign at the final window position, with meaningful event words (A, B, none / up, down, same; blue) or arbitrary single-token words (mean of two word sets; orange); filled: pretrained, hollow: random weights. 10 sets × 200 episodes per model; mean and 95% t-interval over sets. Data: A28 (protocols of A3 and A4). File: `figures/en/F2_words_vs_random.pdf`.

**Figure F3 — Word meaning by position.** AUC of a linear probe for "the kind word denotes food", tested on words never seen in training, read at the kind word's own position (blue) and at the final window position (orange); filled: pretrained, hollow: random weights. Pythia 1B / 2.8B / 6.9B from A24 and Qwen2.5-1.5B / OLMo-2-1B from A28 were computed on different RTX 5090 instances; the comparison across these groups is descriptive. 200 episodes, ≈56 distinct test words; 95% cluster-bootstrap intervals over words. Random weights: one network per model; across independent random networks the AUC spreads over ≈0.36–0.63 (simulation). Dashed line: preregistered threshold 0.8. File: `figures/en/F3_meaning_by_position.pdf`.

**Figure F4 — Meaning against experience in synthetic windows.** (a) On conflict windows (kind words whose meaning contradicts the outcomes): the meaning effect, |d| on the same windows with outcomes permuted (blue); the shift toward experience, d(outcomes) − d(permuted) (orange); the same shift for random weights (grey); d is the logit difference between the true-food word and the poison word. (b) Share of conflict windows answered toward experience, with outcomes (orange) and with outcomes permuted (hatched). Sentence interface, 2–6 events per window, 10 sets × 648 windows; mean and 95% t-interval over sets. Pythia-1B, Qwen2.5-1.5B, OLMo-2-1B: A28 (one instance); Pythia-6.9B: A27 (a different instance). File: `figures/en/F4_synthetic_binding.pdf`.
