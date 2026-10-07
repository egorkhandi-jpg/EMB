# Knowing More Than It Can Say: Frozen Language Models in Minimal Bodies

Code, stored results and preregistrations for the preprint

> Egor Grigorev (2026). *Knowing More Than It Can Say: Frozen Language Models in Minimal Bodies.* Preprint.
> ORCID [0009-0000-2725-5466](https://orcid.org/0009-0000-2725-5466)

- **Preprint:** URL pending
- **DOI:** pending
- **Release:** [`v1.1-preprint`](https://github.com/egorkhandi-jpg/EMB/tree/v1.1-preprint) — the state of the repository the preprint refers to. PDFs: [`docs/paper/paper_en.pdf`](docs/paper/paper_en.pdf) (English) and [`docs/paper/paper_zh.pdf`](docs/paper/paper_zh.pdf) (Chinese translation, AI-assisted; the English text is authoritative). LaTeX sources: [`docs/paper/latex/`](docs/paper/latex/) and [`docs/paper/zh/`](docs/paper/zh/).

## Abstract

Frozen pretrained language models are increasingly used as components of embodied agents, yet it is unclear what pretraining contributes in such settings and how much of what the model internally represents is available through its output. We study this in a minimal two-dimensional foraging environment where an agent's body events are presented to a frozen language model as a short sequence of discrete tokens, and compare each pretrained model against the same architecture with random weights and matched embedding norms. Across three model families (Pythia 160M–6.9B, Qwen2.5-1.5B, OLMo-2-1B), we find that: (i) where pretraining helps, the advantage comes from sequence-processing machinery rather than word meaning — arbitrary event words work as well as or better than meaningful ones, and in one family (Qwen2.5) the advantage appears only with arbitrary words; (ii) a latent episode variable that is never stated explicitly in the context is linearly decodable from hidden states with AUC 0.93–0.99, but the model's own answer to a direct question recovers it only at AUC 0.61–0.64, and in Pythia, in-context examples from other episodes do not close this gap; (iii) the meaning of a word (food vs. harmful) is fully present at that word's position but reaches the final position, where an episode-level readout would use it, at most weakly (0.97–1.00 vs. 0.51–0.62); and (iv) when word meaning conflicts with in-context experience, a statistically reliable trace of the experience reaches the answer but is substantially weaker than the meaning effect (4–28% of it) and leaves the answers essentially unchanged. All primary analyses were preregistered in commits made before computation, use exact permutation or bootstrap tests with Holm correction, and include controls for confounds that misled or would have misled our conclusions during the project, such as survivorship-induced frequency cues. Taken together, the results suggest that frozen language models in this regime represent substantially more about their embodied situation than they can express, and that the bottleneck lies in routing and expression rather than in representation.

## Experiments in the paper

Each report below contains the preregistration, committed before any computation, followed by the results. Reports are in Russian. **Prereg. commit** is the commit hash in this repository. Commits are dated, so the order "preregistration before results" can be checked from the history.

| Id | Question | Paper section | Tag | Prereg. commit | Report | Scripts | Results |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | Pretrained vs. random, continuous projector, Pythia 70M–1B | 3.1 | `v2-scale` | 6af7ab7 | [RESULTS_2026-10-01_flip.md](docs/RESULTS_2026-10-01_flip.md) | `probe_scale.py`, `summarize_probe.py` | `results_scale/` |
| A3 | Interface control: start token, event words | 3.1 | `v2-interface` | 5476edf | [RESULTS_2026-10-01_flip.md](docs/RESULTS_2026-10-01_flip.md) | `probe_interface.py` | `results_interface/` |
| A4 | Meaning of event words: arbitrary words, embedding swaps | 3.2 | `v3-words` | ebab915 | [RESULTS_2026-10-01_flip.md](docs/RESULTS_2026-10-01_flip.md) | `probe_words.py`, `bayes_ceiling.py` | `results_words/` |
| A13 | Out-of-loop evaluation of closed-loop readouts | 4 | `v3-replay` | bcffc98 | [RESULTS_2026-10-01_flip.md](docs/RESULTS_2026-10-01_flip.md) | `body_head3.py`, `summarize_replay.py` | `results_bodyhead3/` |
| A22 | Report without examples; probe vs. answer | 3.3 | `v4-report` | 1f2be62 | [RESULTS_2026-10-04_report.md](docs/RESULTS_2026-10-04_report.md) | `report_body.py`, `summarize_report.py` | `results_report/` |
| A23 | In-context examples from other episodes (Pythia) | 3.3 | `v4-icl` | 9cc7aea | [RESULTS_2026-10-04_report.md](docs/RESULTS_2026-10-04_report.md) | `report_icl.py`, `summarize_icl.py` | `results_report_vast/` |
| A24 | Word meaning in language and at positions in the event window | 3.4 | `v5-meaning` | bfb0c62 | [RESULTS_2026-10-05_meaning.md](docs/RESULTS_2026-10-05_meaning.md) | `meaning_probe.py`, `summarize_meaning.py`, `f3_check.py` | `results_meaning/` |
| A25 | Self/world separation through an event alphabet (Pythia-160M) | 4 | — | f114250 | [RESULTS_2026-10-05_geometry.md](docs/RESULTS_2026-10-05_geometry.md) | `geo3_precheck.py`, `geo_lm.py`, `env_geo.py` | `results_geo3/` |
| A26 | Meaning vs. experience on simulated bodies (Pythia 1B–6.9B) | 3.5, 4 | `v5-bind` | ac18bce | [RESULTS_2026-10-06_bind.md](docs/RESULTS_2026-10-06_bind.md) | `meaning_bind.py`, `summarize_bind.py`, `bind_posthoc.py` (post hoc) | `results_bind/` |
| A27 | Meaning vs. experience on synthetic windows | 3.5, 4 | `v5-synth` | stop rule 1719022; preregistration 6462459; amendment before computation d0e7b01; runs efedb20 | [RESULTS_2026-10-06_bind.md](docs/RESULTS_2026-10-06_bind.md), section A27 | `meaning_synth.py`, `summarize_synth.py`, `synth_descr.py` | `results_synth/` |
| A28 | Replication on Qwen2.5 and OLMo-2 | all | `v7-replication` | ec88b6f | [RESULTS_2026-10-06_replication.md](docs/RESULTS_2026-10-06_replication.md) | `replicate.py`, `summarize_replication.py` | `results_replication/` |

The full registry of the project has 57 entries: [`docs/ATTEMPTS.md`](docs/ATTEMPTS.md), in Russian. Besides the experiments above, it lists abandoned environment variants, lines closed by stopping rules, and analyses outside the scope of the paper. Their code, results and reports are kept in the repository.

**About hashes inside reports and results.** The project was developed in a private repository. This public repository is the same history with one file removed, so its commit hashes differ from the private ones. The hashes written inside the reports and in the `git_commit` field of `results_*/**/*.json` are the private ones. [`docs/COMMIT_MAP.md`](docs/COMMIT_MAP.md) maps every private hash to its public counterpart. Tags have the same names in both.

## Reproducing the figures from the stored results

No GPU and no model downloads are needed:

```bash
python make_figures.py --lang en    # figures/en/F1–F4 (.pdf, .png)
python make_figures.py --lang ru    # figures/F1–F4, Russian labels
python make_figures.py --lang zh    # figures/zh/F1–F4, Chinese labels (needs NotoSansCJKsc-Regular.otf in docs/paper/zh/fonts/, not in the repository)
python f3_check.py                  # F3: cluster bootstrap over words, spread of random networks
```

| Figure | What | Stored results |
| --- | --- | --- |
| F1 | Probe vs. answer for the latent episode sign | `results_replication/` (A28, protocol of A22) |
| F2 | Meaningful vs. arbitrary event words | `results_replication/` (A28, protocols of A3, A4) |
| F3 | Word meaning by position | `results_meaning/` (A24), `results_replication/` (A28) |
| F4 | Meaning against experience in synthetic windows | `results_replication/` (A28), `results_synth/` (A27, Pythia-6.9B) |

## Recomputing the results

The scripts accept a Hugging Face model id or a local path. Each script documents its protocol and command line in its header docstring (in Russian). The commands are the ones used for the paper.

**A28 — replication family (F1, F2, and parts of F3, F4).** RTX 5090, queue time 2.56 h for the three models:

```bash
python replicate.py --model Qwen/Qwen2.5-1.5B --chunk 256
python replicate.py --model allenai/OLMo-2-0425-1B --chunk 256
python replicate.py --model EleutherAI/pythia-1b --chunk 256
python summarize_replication.py results_replication > results_replication/summary.txt
```

To run single protocols, use `--parts r12 r3 r4 r5`:
- `r12` — protocols of A3/A4;
- `r3` — A22;
- `r4` — A24;
- `r5` — A27.

`--show` decodes the inputs of all protocols without computing anything.

**A24 — word meaning by position (F3, Pythia):**

```bash
python meaning_probe.py --model EleutherAI/pythia-1b --bs 16
python meaning_probe.py --model EleutherAI/pythia-2.8b --bs 16
python meaning_probe.py --model EleutherAI/pythia-6.9b --bs 4
python summarize_meaning.py results_meaning > results_meaning/summary.txt
```

**A27 — synthetic windows (F4, Pythia-6.9B):**

```bash
python meaning_synth.py --model EleutherAI/pythia-6.9b --ifaces s c --bs 16
python meaning_synth.py --model EleutherAI/pythia-1b --ifaces s --bs 32
python summarize_synth.py results_synth > results_synth/summary.txt
```

## Environment

- **Software:**
  - Python 3.12–3.13;
  - `torch` 2.11 (CUDA 12.8), `transformers` 5.18, `accelerate` 1.15, `tokenizers` 0.23, `safetensors` 0.8, `huggingface_hub` 1.33;
  - `numpy` 2.5, `scikit-learn` 1.9, `scipy` 1.18;
  - `matplotlib` 3.11 for the figures.
- **Models:** `EleutherAI/pythia-70m` … `EleutherAI/pythia-6.9b`, `Qwen/Qwen2.5-1.5B`, `allenai/OLMo-2-0425-1B`.
- **Precision:** all runs use fp32 with TF32 disabled. This is set in `stand_gpu.py`, which every experiment imports.
- **Hardware:**
  - a laptop RTX 5050 (8 GB) ran experiments up to Pythia-410M, and A22 at 1B;
  - a rented RTX 5090 (32 GB) ran experiments with Pythia-1B and long contexts or larger models (A23, A24, A26–A28).
- **One machine per family:** all conditions of one comparison family were computed on one machine. On fixed inputs, Pythia-160M hidden states agree across the two machines to within 2.1·10⁻³.

## Repository layout

| Path | What |
| --- | --- |
| `*.py`, `*.sh` | environments, stand, analysis and summary scripts |
| `results_*/` | stored results: JSON, summaries, and raw logits where analyses need them |
| `figures/`, `figures/en/`, `figures/zh/` | figures F1–F4 (Russian, English, Chinese labels), produced by `make_figures.py` |
| `docs/RESULTS_*.md` | reports with preregistrations, in Russian |
| `docs/ATTEMPTS.md` | registry of all attempts, in Russian |
| `docs/paper/` | paper drafts (Markdown), LaTeX build (`build_latex.py`), `latex/paper.tex`, `latex/paper.pdf` |
| `docs/COMMIT_MAP.md` | private → public commit hashes |

Internal working documents and remote-machine tooling were removed from the tree after the last commit published before the preprint. They remain in the earlier history; the list is in [`.public_exclude`](.public_exclude).

## License

- **Code:** Apache License 2.0 ([`LICENSE`](LICENSE)).
- **Paper text and figures:** CC BY 4.0 ([`LICENSE-text.md`](LICENSE-text.md)). This covers `docs/paper/` and `figures/`.

## Citation

```
Egor Grigorev (2026). Knowing More Than It Can Say: Frozen Language Models in Minimal Bodies.
Preprint. Code: https://github.com/egorkhandi-jpg/EMB, release v1.1-preprint. DOI: pending.
```

## Disclosure

Code implementation and experiment execution were performed with an AI coding assistant (Claude Code) under the author's direction; hypotheses, preregistrations, decisions to stop or continue each line, and interpretation are the author's. All code, preregistration commits, and results are released.
