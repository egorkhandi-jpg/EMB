"""Сборка docs/paper/draft_full.md из частей (2026-10-07). Аннотация и раскрытие — из docs/PAPER_OUTLINE.md (разделы 1, 4),
введение, Setup+Results, Pitfalls, Discussion — из docs/paper/draft_*.md; References — только цитируемые в тексте работы
(проверяется, что у каждой ссылки [Author Year] есть запись); подписи к рисункам F1–F4 (английские, по figures/en/).
Комментарии-источники <!-- … --> сохраняются. Печатает объём по разделам (слова без комментариев).

python docs/paper/build_draft.py
"""
import os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
P = lambda *a: os.path.join(ROOT, *a)
read = lambda p: open(p, encoding='utf-8').read().replace('\r\n', '\n')

REFS = {   # ключ [Author Year] → запись (по docs/RELATED_WORK.md; год — первой версии arXiv)
    'Ahn 2022': 'Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. arXiv:2204.01691.',
    'Azaria 2023': 'Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It\'s Lying. arXiv:2304.13734.',
    'Belinkov 2021': 'Belinkov, Y. (2021). Probing Classifiers: Promises, Shortcomings, and Advances. arXiv:2102.12452.',
    'Burns 2022': 'Burns, C., et al. (2022). Discovering Latent Knowledge in Language Models Without Supervision. arXiv:2212.03827.',
    'Dave 2024': 'Dave, A. J., et al. (2024). Integrating LLMs for Explainable Fault Diagnosis in Complex Systems. arXiv:2402.06695.',
    'Elazar 2020': 'Elazar, Y., et al. (2020). Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. arXiv:2006.00995.',
    'Feng 2023': 'Feng, J., & Steinhardt, J. (2023). How do Language Models Bind Entities in Context? arXiv:2310.17191.',
    'Gekhman 2025': 'Gekhman, Z., et al. (2025). Inside-Out: Hidden Factual Knowledge in LLMs. arXiv:2503.15299.',
    'Hewitt 2019': 'Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. arXiv:1909.03368.',
    'Hu 2023': 'Hu, J., & Levy, R. (2023). Prompting is not a substitute for probability measurements in large language models. arXiv:2305.13264.',
    'Huang 2022': 'Huang, W., et al. (2022). Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. arXiv:2201.07207.',
    'Kim 2023': 'Kim, N., & Schuster, S. (2023). Entity Tracking in Language Models. arXiv:2305.02363.',
    'Li 2022': 'Li, S., et al. (2022). Pre-Trained Language Models for Interactive Decision-Making. arXiv:2202.01771.',
    'Li 2026': 'Li, B., & Zhao, C. (2026). S2S-FDD: Bridging Industrial Time Series and Natural Language for Explainable Zero-shot Fault Diagnosis. arXiv:2603.08048.',
    'Liu 2025': 'Liu, Y., et al. (2025). Evaluation of Large Language Models for Numeric Anomaly Detection in Power Systems. arXiv:2511.21371.',
    'Orgad 2024': 'Orgad, H., et al. (2024). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. arXiv:2410.02707.',
    'Reid 2022': 'Reid, M., et al. (2022). Can Wikipedia Help Offline Reinforcement Learning? arXiv:2201.12122.',
    'Sun 2024': 'Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). Massive Activations in Large Language Models. arXiv:2402.17762.',
    'Zhong 2024': 'Zhong, Z., & Andreas, J. (2024). Algorithmic Capabilities of Random Transformers. arXiv:2410.04368.',
}

CAPTIONS = '''**Figure F1 — Probe vs. answer for the latent episode sign.** AUC against the episode sign of a linear probe at the question position (blue) and of the model's answer, the logit difference " A" − " B" (orange), for three zero-shot questions. Filled markers: pretrained; hollow: the same architecture with random weights and matched embedding norms. flip-2, 32-step event-word window; 10 sets × 100 episodes per model; mean and 95% t-interval over sets. Data: A28 (protocol of A22). File: `figures/en/F1_probe_vs_answer.pdf`.

**Figure F2 — Meaningful vs. arbitrary event words.** Balanced accuracy of a linear probe for the episode sign at the final window position, with meaningful event words (A, B, none / up, down, same; blue) or arbitrary single-token words (mean of two word sets; orange); filled: pretrained, hollow: random weights. 10 sets × 200 episodes per model; mean and 95% t-interval over sets. Data: A28 (protocols of A3 and A4). File: `figures/en/F2_words_vs_random.pdf`.

**Figure F3 — Word meaning by position.** AUC of a linear probe for "the kind word denotes food", tested on words never seen in training, read at the kind word's own position (blue) and at the final window position (orange); filled: pretrained, hollow: random weights. Pythia 1B / 2.8B / 6.9B from A24 and Qwen2.5-1.5B / OLMo-2-1B from A28 were computed on different RTX 5090 instances; the comparison across these groups is descriptive. 200 episodes, ≈56 distinct test words; 95% cluster-bootstrap intervals over words. Random weights: one network per model; across independent random networks the AUC spreads over ≈0.36–0.63 (simulation). Dashed line: preregistered threshold 0.8. File: `figures/en/F3_meaning_by_position.pdf`.

**Figure F4 — Meaning against experience in synthetic windows.** (a) On conflict windows (kind words whose meaning contradicts the outcomes): the meaning effect, |d| on the same windows with outcomes permuted (blue); the shift toward experience, d(outcomes) − d(permuted) (orange); the same shift for random weights (grey); d is the logit difference between the true-food word and the poison word. (b) Share of conflict windows answered toward experience, with outcomes (orange) and with outcomes permuted (hatched). Sentence interface, 2–6 events per window, 10 sets × 648 windows; mean and 95% t-interval over sets. Pythia-1B, Qwen2.5-1.5B, OLMo-2-1B: A28 (one instance); Pythia-6.9B: A27 (a different instance). File: `figures/en/F4_synthetic_binding.pdf`.
'''


def words(t):
    t = re.sub(r'<!--.*?-->', '', t, flags=re.S)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.,'’%–/+-]*", t))


def main():
    outline = read(P('docs', 'PAPER_OUTLINE.md'))
    abstract = outline.split('## 1. Аннотация\n', 1)[1].split('\n### Сверка', 1)[0].strip()
    disclosure = outline.split('## 4. Раскрытие\n', 1)[1].split('\n## ', 1)[0].strip()
    intro = read(P('docs', 'paper', 'draft_intro.md')).strip()
    sr = read(P('docs', 'paper', 'draft_setup_results.md'))
    sr = sr.split('## 2 Setup', 1)[1]; setup, results = sr.split('## 3 Results', 1)
    setup, results = '## 2 Setup' + setup.rstrip(), '## 3 Results' + results.rstrip()
    pitfalls = read(P('docs', 'paper', 'draft_pitfalls.md')).strip()
    discussion = read(P('docs', 'paper', 'draft_discussion.md')).strip()
    availability = read(P('docs', 'paper', 'draft_availability.md')).strip()
    body = '\n\n'.join([intro, setup, results, pitfalls, discussion])
    keys = []
    for grp in re.findall(r'\[([A-Z][^\]]*?\d{4}[^\]]*)\]', body + abstract):
        for k in grp.split(';'):
            k = k.strip()
            if re.match(r'^[A-Z][A-Za-z\-]+ \d{4}$', k) and k not in keys:
                keys.append(k)
    missing = [k for k in keys if k not in REFS]
    assert not missing, f'нет записи в References: {missing}'
    refs = '\n'.join(f'- {REFS[k]}' for k in sorted(keys))
    doc = ('# Knowing More Than It Can Say — full draft (focus 1)\n\n'
           '<!-- Собрано build_draft.py 2026-10-07 из docs/PAPER_OUTLINE.md и docs/paper/draft_*.md. Комментарии-источники сохранены; в итоговом тексте будут удалены. -->\n\n'
           '## Abstract\n\n' + abstract + '\n\n' + body + '\n\n## Disclosure\n\n' + disclosure + '\n\n## Code and data availability\n\n' + availability +
           '\n\n## References\n\n' + refs + '\n\n## Figure captions\n\n' + CAPTIONS)
    open(P('docs', 'paper', 'draft_full.md'), 'w', encoding='utf-8', newline='\n').write(doc)
    parts = [('Abstract', abstract.split('\n\n', 1)[1] if abstract.startswith('Title') else abstract), ('1 Introduction', intro), ('2 Setup', setup), ('3 Results', results),
             ('4 Pitfalls and controls', pitfalls), ('5 Discussion', discussion), ('Disclosure', disclosure), ('Code and data availability', availability),('Figure captions', CAPTIONS)]
    total = 0
    for nm, t in parts:
        n = words(t); total += n; print(f'{nm:26s} {n:5d}')
    print(f'{"Итого (без References)":26s} {total:5d}; ссылок в References: {len(keys)}')


if __name__ == '__main__':
    main()
