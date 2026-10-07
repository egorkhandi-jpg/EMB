## Appendix A Replication and tokenizer transfer

<!-- Источник: A28, docs/RESULTS_2026-10-06_replication.md; replicate.py. -->

**Models.** Qwen2.5-1.5B (`Qwen/Qwen2.5-1.5B`) and OLMo-2-1B (`allenai/OLMo-2-0425-1B`), both without access restrictions. Pythia-1B was run through the same transferred code on the same machine as a reference; it was not part of the replication family. For every model, the random-weight control uses the same architecture, a network reseeded per set (one network, seed 1, for the meaning probe) and input embeddings rescaled to the pretrained mean row norm. All runs use fp32 with TF32 disabled.

**Start token.**
- Each model's end-of-document token, `<|endoftext|>`, is used as the start token: 0 in Pythia, 100257 in OLMo-2 and 151643 in Qwen2.5.
- In Pythia and OLMo-2 this is the BOS token; Qwen2.5 has no BOS by default.

**Single-token words.**
- Every word used by the protocols is a single token with a leading space in all three tokenizers. This covers the event symbols, both arbitrary word sets and all meaning pools (30 food, 30 harmful, 30 neutral and 11 direct words). No substitutions were needed.
- Token ids are taken from each tokenizer: for the event symbols A, B, none / up, down, same they are 362, 425, 6857 / 705, 1495, 1852 in Qwen2.5 and 362, 426, 7000 / 709, 1523, 1890 in OLMo-2.
- In Pythia they are 329, 378, 5293 / 598, 1066, 1072, exactly the constants of the original scripts.

**Verification.**
- All inputs of all five protocols were decoded and inspected for the three tokenizers.
- Hidden states are read as `base_model(...).last_hidden_state` (after the final normalization). On a model outside the family this matched the last element of `hidden_states` exactly, and logits matched hidden state × output embedding exactly.
- Pythia-1B through the transferred code reproduced the original results:

| Protocol | Transferred code | Original |
| --- | --- | --- |
| Meaning probe: kind position / final position / direct words | 0.983 / 0.526 / 1.000 | 0.983 / 0.526 / 1.000 |
| Synthetic windows: d on conflict windows, with / without binding | −0.262 / −0.284 | −0.262 / −0.284 |
| Zero-shot answer AUC / probe AUC | 0.61 / 0.99 | 0.61 / 0.98–0.99 |

**Protocols and family.**
- **R1:** sign probe, event words.
- **R2:** arbitrary words, non-inferiority margin 0.02 plus superiority over random weights.
- **R3:** expression gap, three questions.
- **R4:** meaning by position, thresholds 0.9 and 0.8.
- **R5:** synthetic windows, shift toward experience versus permuted outcomes and versus random weights.
- **Tests:** R1, R2 and R5 used one-sided exact paired permutation tests over 10 sets, with Holm correction over six tests (three checks × two models). R3 and R4 are thresholds, as in the original reports.

**Deviations from the original protocols, recorded before computation.**
- R1 tests "pretrained > random" directly and one-sided. The original A3 test was two-sided, on the change of this difference relative to the projector interface.
- Probe features are standardized on the training part of each fold, as in the original A3/A4.
- In R3 the probe is fitted at the position of each of the three questions.

**Results.**
- **Both new families:** R2–R5 replicated.
- **R1:** replicated on OLMo-2. On Qwen2.5 it did not replicate with the original event words, although it did with arbitrary words.

**Compute.** One rented RTX 5090 for the whole family. The queue ran for 2.56 h without failures; the instance was rented for 3 h 16 min in total, including model downloads.

## Appendix B Registry of experiments used in this paper

<!-- Источник: docs/ATTEMPTS.md; метки и даты — git. Хэши — открытого репозитория github.com/egorkhandi-jpg/EMB (локальные → опубликованные: docs/COMMIT_MAP.md; A28 — по фильтру публикации, 2026-10-07). -->

Every experiment below was preregistered in the repository before computation, with all outcomes and their interpretation written in advance. The tag marks the code version used; commit hashes refer to the public repository (see Code and data availability); dates are commit dates.

| Id | Question | Section | Code tag (commit, date) | Outcome |
| --- | --- | --- | --- | --- |
| A2 | Pretrained vs. random, continuous projector, Pythia 70M–1B | 3.1 | `v2-scale` (6af7ab7, 2026-10-01) | pretrained ≤ random; 160M significantly lower |
| A3 | Interface control: start token, event words | 3.1 | `v2-interface` (5476edf, 2026-10-02) | supported (b − a, c − a) |
| A4 | Meaning of event words: arbitrary words, embedding swaps | 3.2 | `v3-words` (ebab915, 2026-10-02) | machinery, not meaning |
| A13 | Out-of-loop evaluation of closed-loop readouts | 4 | `v3-replay` (bcffc98, 2026-10-03) | closed-loop advantages vanish out of the loop |
| A22 | Report without examples; probe vs. answer | 3.3 | `v4-report` (1f2be62, 2026-10-04) | expression gap; few-shot = copying |
| A23 | In-context examples from other episodes (Pythia only) | 3.3 | `v4-icl` (9cc7aea, 2026-10-04) | no in-context learning |
| A24 | Word meaning in language and at positions in the event window | 3.4 | `v5-meaning` (bfb0c62, 2026-10-05) | K1 without K2; intervals refined 2026-10-07 |
| A25 | Self/world separation through an event alphabet (Pythia-160M) | 4 | f114250 (2026-10-05) | null; alphabet loses information |
| A26 | Meaning vs. experience on simulated bodies (Pythia 1B–6.9B) | 3.5, 4 | `v5-bind` (ac18bce, 2026-10-06) | formal B > 0, explained post hoc by frequency |
| A27 | Meaning vs. experience on synthetic windows | 3.5, 4 | stop rule 1719022; preregistration 6462459; amendment before computation d0e7b01 (`v5-synth`); runs efedb20 (2026-10-06) | H2 without H1 |
| A28 | Replication on Qwen2.5 and OLMo-2 | all | `v7-replication` (ec88b6f, 2026-10-06) | R2–R5 on both; R1 on OLMo-2 |

The full registry of the project, with 57 entries, also lists environment variants that were abandoned, lines closed by stopping rules, and analyses outside the scope of this paper (https://github.com/egorkhandi-jpg/EMB/blob/v1.1-preprint/docs/ATTEMPTS.md).

## Appendix C Compute and costs

<!-- Источник: FINAL_REPORT п. 10; RESULTS A24, A26, A27, A28 (итоги по панели — со слов автора). -->

**Hardware.**
- **Laptop:** an RTX 5050 Laptop GPU (8 GB) ran the experiments up to 410M and A22 at 1B.
- **Rented RTX 5090 (32 GB):** experiments with Pythia-1B and long contexts or larger models, at $0.475 per hour.
- **One machine per family:** all conditions of one preregistered family were computed on one machine. Agreement between machines was checked before each new instance on fixed inputs: Pythia-160M hidden states differed by at most 2.1·10⁻³, within the laptop's own CPU–GPU spread.

**Rented GPU time for the experiments in this paper** (billed time includes model downloads and setup):

| Experiment | Queue time | Billed time | Cost |
| --- | --- | --- | --- |
| A23 | 0.85 h | ≈ 1.2 h | ≈ $0.55 |
| A24 | 0.60 h | 1 h 11 min | $0.56 |
| A26 | 0.87 h | 1 h 6 min | $0.52 |
| A27 | 0.65 h | 58 min | $0.46 |
| A28 | 2.56 h | 3 h 16 min | $1.55 |
| **Total** | **5.53 h** | **≈ 7.7 h** | **≈ $3.6** |

Laptop experiments (A2–A4, A13, A22, A25) took from minutes to a few hours each and were run through a queue runner with thermal pauses.
