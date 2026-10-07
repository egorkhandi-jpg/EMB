"""Графики для публикации (фокус 1) по готовым результатам; новых расчётов нет.
--lang ru → figures/F*.pdf, .png (подписи по-русски); --lang en → figures/en/F*.pdf, .png (язык работы).
F1 зонд против ответа (знак эпизода; протокол A22, A28): Pythia-1B, Qwen2.5-1.5B, OLMo-2-1B, 10 наборов.
F2 осмысленные / произвольные слова / случайные веса (протоколы A3/A4, A28): три семейства, 10 наборов.
F3 зонд смысла «еда / вред» по позициям (позиция символа вида / последняя позиция окна): Pythia 1B / 2.8B / 6.9B (A24)
   и Qwen2.5-1.5B, OLMo-2-1B (A28); один прогон из 200 эпизодов; интервалы — кластерный бутстреп по словам вида A
   (исправление 2026-10-07: прежний бутстреп по эпизодам занижал интервалы — обобщение идёт на новые слова; f3_check.py).
F4 синтетические окна (протокол A27): сдвиг к опыту против эффекта смысла и доля лживых окон с ответом к опыту;
   Pythia-1B, Qwen2.5-1.5B, OLMo-2-1B (A28), Pythia-6.9B (A27, другой экземпляр 5090 — помечено); 10 наборов.
Интервалы по наборам — 95 % t-интервал среднего (n = 10).

python make_figures.py --lang ru
python make_figures.py --lang en
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T95 = 2.262                                                     # t(0,975; 9)
C1, C2, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
REP = 'results_replication'
MODELS = [('pythia-1b', 'Pythia-1B'), ('Qwen2.5-1.5B', 'Qwen2.5-1.5B'), ('OLMo-2-0425-1B', 'OLMo-2-1B')]
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42})

TX = {
    'ru': {
        'sets': 'наборов', 'n10': 'n = 10',
        'f1_probe_p': 'зонд на позиции вопроса — предобученная', 'f1_probe_r': 'зонд — случайные веса',
        'f1_ans_p': 'ответ (логит «A» − «B») — предобученная', 'f1_ans_r': 'ответ — случайные веса',
        'f1_q': ['Food:', 'Safe to\neat:', 'Good\nkind:'], 'f1_y': 'AUC против знака эпизода',
        'f1_t': 'F1. Знак тела в представлении есть, в ответе — меньше половины (flip-2, окно 32, вопросы без примеров; 95 % ДИ по наборам)',
        'f2_s': ['предобученная — осмысленные слова (A, B, none / up, down, same)', 'случайные веса — осмысленные слова',
                 'предобученная — произвольные слова (среднее двух наборов)', 'случайные веса — произвольные слова'],
        'f2_y': 'точность зонда знака\n(сбалансированная)',
        'f2_t': 'F2. Произвольные слова не хуже осмысленных\n(flip-2, окно 32, зонд по последней позиции; 95 % ДИ по наборам)',
        'f3_s': ['предобученная — позиция символа вида', 'случайные веса — позиция символа вида',
                 'предобученная — последняя позиция окна', 'случайные веса — последняя позиция окна'],
        'f3_y': 'AUC зонда «слово вида — еда»',
        'f3_t': 'F3. Смысл слова есть на его позиции, но не на последней позиции окна\n(200 эпизодов, ≈ 56 слов; 95 % ДИ — кластерный бутстреп по словам)',
        'f3_note': 'Случайные веса — одна сеть на модель; разброс AUC между независимыми случайными сетями ≈ 0,36–0,63 (моделирование, f3_check.py).',
        'f4_a': ['эффект смысла: |d| на окнах без связки', 'сдвиг к опыту: d(исходы) − d(без связки)', 'сдвиг к опыту у случайных весов'],
        'f4_b': ['с исходами (связка есть)', 'исходы переставлены (связки нет)'],
        'f4_ya': 'логиты (слово истинной еды − слово яда)', 'f4_ta': 'а) лживые окна: смысл против опыта',
        'f4_yb': 'доля лживых окон,\nгде ответ — к опыту', 'f4_tb': 'б) ответ почти не меняется',
        'f4_t': 'F4. Синтетические окна без частотного канала: связка «слово ↔ исход» есть, но мала и ответа не меняет\n'
                '(интерфейс-предложения, n = 2–6 событий; 95 % ДИ по наборам)',
        'f4_note': '* Pythia-6.9B — A27, другой экземпляр RTX 5090; остальные — A28, один экземпляр.',
    },
    'en': {
        'sets': 'sets', 'n10': 'n = 10',
        'f1_probe_p': 'probe at question position — pretrained', 'f1_probe_r': 'probe — random weights',
        'f1_ans_p': 'answer (logit "A" − "B") — pretrained', 'f1_ans_r': 'answer — random weights',
        'f1_q': ['Food:', 'Safe to\neat:', 'Good\nkind:'], 'f1_y': 'AUC vs. episode sign',
        'f1_t': 'The body state is represented but poorly reported (flip-2, 32-step window, zero-shot questions; 95% CI over sets)',
        'f2_s': ['pretrained — meaningful words (A, B, none / up, down, same)', 'random weights — meaningful words',
                 'pretrained — arbitrary words (mean of two sets)', 'random weights — arbitrary words'],
        'f2_y': 'sign-probe accuracy\n(balanced)',
        'f2_t': 'Arbitrary event words work as well as meaningful ones\n(flip-2, 32-step window, probe at final position; 95% CI over sets)',
        'f3_s': ['pretrained — kind-word position', 'random weights — kind-word position',
                 'pretrained — final window position', 'random weights — final window position'],
        'f3_y': 'AUC of probe "kind word is food"',
        'f3_t': 'Word meaning is present at the word, not at the final position\n(200 episodes, ≈56 words; 95% CI — cluster bootstrap over words)',
        'f3_note': 'Random weights: one network per model; AUC spread across independent random networks ≈ 0.36–0.63 (simulation, f3_check.py).',
        'f4_a': ['meaning effect: |d| without binding', 'shift toward experience: d(outcomes) − d(no binding)', 'shift toward experience, random weights'],
        'f4_b': ['with outcomes (binding present)', 'outcomes permuted (no binding)'],
        'f4_ya': 'logits (true-food word − poison word)', 'f4_ta': 'a) conflict windows: meaning vs. experience',
        'f4_yb': 'share of conflict windows\nanswered toward experience', 'f4_tb': 'b) answers barely change',
        'f4_t': 'Synthetic windows without a frequency cue: a word–outcome trace exists but is small and does not change answers\n'
                '(sentence interface, 2–6 events; 95% CI over sets)',
        'f4_note': '* Pythia-6.9B — A27, a different RTX 5090 instance; others — A28, one instance.',
    },
}


def ci(x):
    x = np.asarray(x, float); return x.mean(), T95 * x.std(ddof=1) / np.sqrt(len(x))


def jload(p):
    return json.load(open(p, encoding='utf-8'))


def find(d, prefix, suffix=''):
    f = [x for x in os.listdir(d) if x.startswith(prefix) and x.endswith(suffix + '.json')]
    assert len(f) == 1, (d, prefix, suffix, f); return os.path.join(d, f[0])


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def dot(ax, x, val, err, color, label=None, hollow=False, marker='o'):
    """Точка с 95 % интервалом (ось не от нуля — поэтому точки, а не столбцы)."""
    lo, hi = (err if isinstance(err, tuple) else (err, err))
    ax.errorbar(x, val, yerr=[[lo], [hi]], fmt=marker, ms=6.5, mfc='white' if hollow else color, mec=color, mew=1.4,
                ecolor=color, elinewidth=1.2, capsize=2.5, label=label, zorder=4)


def f1(L, out):
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=True)
    for ax, (d, name) in zip(axes, MODELS):
        P = jload(find(os.path.join(REP, d), 'report_', '_pretrained'))['results']
        R = jload(find(os.path.join(REP, d), 'report_', '_random'))['results']
        qs = ['zs1', 'zs2', 'zs3']; x = np.arange(3)
        for j, q in enumerate(qs):
            for off, src, key, col, hol, lab in ((-0.22, P, 'probe_auc', C1, False, L['f1_probe_p']), (-0.07, R, 'probe_auc', C1, True, L['f1_probe_r']),
                                                 (0.07, P, 'auc', C2, False, L['f1_ans_p']), (0.22, R, 'auc', C2, True, L['f1_ans_r'])):
                m, e = ci([r['modes'][q][key] for r in src])
                dot(ax, j + off, m, e, col, lab if (j == 0 and ax is axes[0]) else None, hollow=hol)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_xticks(x, L['f1_q'])
        ax.set_title(f"{name}  (n = {len(P)} {L['sets']})", fontsize=9, color=INK)
        ax.set_ylim(0.4, 1.02)
    axes[0].set_ylabel(L['f1_y'])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.suptitle(L['f1_t'], fontsize=9.5, y=1.03)
    save(fig, out, 'F1_probe_vs_answer')


def f2(L, out):
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(MODELS)); w = 0.2
    for i, (d, name) in enumerate(MODELS):
        r = jload(os.path.join(REP, d, 'r12.json'))['results']
        g = lambda wts, c: np.array([next(z['bacc'] for z in r if z['set'] == s and z['weights'] == wts and z['cond'] == c) for s in range(1, 11)])
        series = [('pretrained', ['c'], C1, False), ('random', ['c'], C1, True), ('pretrained', ['c_arb1', 'c_arb2'], C2, False), ('random', ['c_arb1', 'c_arb2'], C2, True)]
        for j, (wts, cs, col, hol) in enumerate(series):
            v = np.mean([g(wts, c) for c in cs], 0); m, e = ci(v)
            dot(ax, i + (j - 1.5) * w, m, e, col, L['f2_s'][j] if i == 0 else None, hollow=hol)
    ax.set_xticks(x, [f"{n}\n(n = 10 {L['sets']})" for _, n in MODELS]); ax.set_ylim(0.88, 1.0)
    ax.set_ylabel(L['f2_y'])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)
    ax.set_title(L['f2_t'], fontsize=9.5)
    save(fig, out, 'F2_words_vs_random')


def auc(s, t):
    p, n = s[t], s[~t]; return ((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())


def boot_cluster(dec, y, unit, B=2000, seed=0):
    dec, y, unit = np.asarray(dec), np.asarray(y, bool), np.asarray(unit)
    uu = np.unique(unit); grp = {u: np.nonzero(unit == u)[0] for u in uu}; rng = np.random.RandomState(seed); out = []
    for _ in range(B):
        idx = np.concatenate([grp[u] for u in rng.choice(uu, len(uu))])
        if len(np.unique(y[idx])) == 2:
            out.append(auc(dec[idx], y[idx]))
    return auc(dec, y), np.percentile(out, 2.5), np.percentile(out, 97.5)


def f3(L, out):
    from f3_check import episode_words
    wA, _, a_food, _ = episode_words()
    src = [('results_meaning/meaning_1b.json', 'Pythia-1B\n(A24)'), ('results_meaning/meaning_2.8b.json', 'Pythia-2.8B\n(A24)'),
           ('results_meaning/meaning_6.9b.json', 'Pythia-6.9B\n(A24)'), (find(os.path.join(REP, 'Qwen2.5-1.5B'), 'meaning_'), 'Qwen2.5-1.5B\n(A28)'),
           (find(os.path.join(REP, 'OLMo-2-0425-1B'), 'meaning_'), 'OLMo-2-1B\n(A28)')]
    fig, ax = plt.subplots(figsize=(7.4, 3.1)); x = np.arange(len(src))
    for i, (p, name) in enumerate(src):
        r = jload(p)
        for j, (off, wts, pos, col, hol) in enumerate(((-0.24, 'pretrained', 'kindA', C1, False), (-0.08, 'random', 'kindA', C1, True),
                                                       (0.08, 'pretrained', 'last', C2, False), (0.24, 'random', 'last', C2, True))):
            c2 = r[wts]['check2']; ep = np.array(c2['episode'])
            assert (a_food[ep].astype(int) == np.array(c2['label'])).all()
            a, lo, hi = boot_cluster(c2[f'dec_{pos}'], c2['label'], wA[ep], seed=i)
            dot(ax, i + off, a, (a - lo, hi - a), col, L['f3_s'][j] if i == 0 else None, hollow=hol)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':'); ax.axhline(0.8, color=MUTED, lw=0.6, ls='--')
    ax.set_xticks(x, [n for _, n in src]); ax.set_ylim(0.1, 1.05); ax.set_ylabel(L['f3_y'])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7.5)
    ax.set_title(L['f3_t'], fontsize=9.5)
    fig.text(0.5, -0.33, L['f3_note'], fontsize=7, color=MUTED, ha='center')
    save(fig, out, 'F3_meaning_by_position')


def f4(L, out):
    src = [(find(os.path.join(REP, 'pythia-1b'), 'synth_'), 'Pythia-1B'), (find(os.path.join(REP, 'Qwen2.5-1.5B'), 'synth_'), 'Qwen2.5-1.5B'),
           (find(os.path.join(REP, 'OLMo-2-0425-1B'), 'synth_'), 'OLMo-2-1B'), ('results_synth/synth_6.9b.json', 'Pythia-6.9B*')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.1))
    x = np.arange(len(src)); GRAY = '#8a8984'
    for i, (p, name) in enumerate(src):
        S = jload(p)['sets']
        g = lambda w, v, k, n=None: np.array([(S[f'{w}/s/{s}']['measures'][v]['by_n'][n] if n else S[f'{w}/s/{s}']['measures'][v])[k] for s in range(1, 11)])
        Lp_, Lpp = g('pretrained', 'q1', 'd_lying'), g('pretrained', 'q1_perm', 'd_lying'); Lr, Lrp = g('random', 'q1', 'd_lying'), g('random', 'q1_perm', 'd_lying')
        for j, (off, v, col) in enumerate(((-0.27, -Lpp, C1), (-0.09, Lp_ - Lpp, C2), (0.09, Lr - Lrp, GRAY))):
            m, e = ci(v)
            a1.bar(i + off, m, width=0.17, color=col, edgecolor=col, linewidth=1.0, label=L['f4_a'][j] if i == 0 else None)
            a1.errorbar(i + off, m, yerr=e, fmt='none', ecolor=INK, elinewidth=0.9, capsize=2)
        fr = np.mean([g('pretrained', 'q1', 'frac_lying_to_experience', n) for n in ('2', '4', '6')], 0)
        frp = np.mean([g('pretrained', 'q1_perm', 'frac_lying_to_experience', n) for n in ('2', '4', '6')], 0)
        for j, (off, v, col, h) in enumerate(((-0.1, fr, C2, None), (0.1, frp, C1, '///'))):
            m, e = ci(v)
            a2.bar(i + off, m, width=0.18, color='white' if h else col, edgecolor=col, hatch=h, linewidth=1.0, label=L['f4_b'][j] if i == 0 else None)
            a2.errorbar(i + off, m, yerr=e, fmt='none', ecolor=INK, elinewidth=0.9, capsize=2)
    a1.axhline(0, color=MUTED, lw=0.8)
    for a in (a1, a2):
        a.set_xticks(x, [f"{n}\n({L['n10']})" for _, n in src], fontsize=8)
    a1.set_ylabel(L['f4_ya']); a1.set_title(L['f4_ta'], fontsize=9)
    a2.set_ylabel(L['f4_yb']); a2.set_ylim(0, 0.6); a2.set_title(L['f4_tb'], fontsize=9)
    a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=7.5); a2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=7.5)
    fig.suptitle(L['f4_t'], fontsize=9.5, y=1.08)
    fig.text(0.5, -0.36, L['f4_note'], fontsize=7, color=MUTED, ha='center')
    save(fig, out, 'F4_synthetic_binding')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--lang', choices=['ru', 'en'], default='ru'); a = ap.parse_args()
    out = 'figures' if a.lang == 'ru' else os.path.join('figures', 'en')
    L = TX[a.lang]
    f1(L, out); f2(L, out); f3(L, out); f4(L, out); print('готово:', out, sorted(x for x in os.listdir(out) if '.' in x))
