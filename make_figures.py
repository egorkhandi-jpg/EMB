"""Графики для публикации (фокус 1) по готовым результатам; новых расчётов нет. Вывод — figures/F*.pdf и .png.
F1 зонд против ответа (знак эпизода; A22-протокол, A28): Pythia-1B, Qwen2.5-1.5B, OLMo-2-1B, 10 наборов.
F2 осмысленные / произвольные слова / случайные веса (A3/A4-протокол, A28): три семейства, 10 наборов.
F3 зонд смысла «еда / вред» по позициям (позиция символа вида / последняя позиция окна): Pythia 1B / 2.8B / 6.9B (A24)
   и Qwen2.5-1.5B, OLMo-2-1B (A28); один прогон из 200 эпизодов, интервалы — бутстреп по эпизодам.
F4 синтетические окна (A27-протокол): сдвиг к опыту против эффекта смысла и доля лживых окон с ответом к опыту;
   Pythia-1B, Qwen2.5-1.5B, OLMo-2-1B (A28), Pythia-6.9B (A27, другой экземпляр 5090 — помечено); 10 наборов.
Интервалы по наборам — 95 % t-интервал среднего (n = 10).

python make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'figures'
T95 = 2.262                                                     # t(0,975; 9)
C1, C2, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b', '#52514e'
REP = 'results_replication'
MODELS = [('pythia-1b', 'Pythia-1B'), ('Qwen2.5-1.5B', 'Qwen2.5-1.5B'), ('OLMo-2-0425-1B', 'OLMo-2-1B')]
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42})


def ci(x):
    x = np.asarray(x, float); return x.mean(), T95 * x.std(ddof=1) / np.sqrt(len(x))


def jload(p):
    return json.load(open(p, encoding='utf-8'))


def find(d, prefix, suffix=''):
    f = [x for x in os.listdir(d) if x.startswith(prefix) and x.endswith(suffix + '.json')]
    assert len(f) == 1, (d, prefix, suffix, f); return os.path.join(d, f[0])


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def dot(ax, x, val, err, color, label=None, hollow=False, marker='o'):
    """Точка с 95 % интервалом (ось не от нуля — поэтому точки, а не столбцы)."""
    lo, hi = (err if isinstance(err, tuple) else (err, err))
    ax.errorbar(x, val, yerr=[[lo], [hi]], fmt=marker, ms=6.5, mfc='white' if hollow else color, mec=color, mew=1.4,
                ecolor=color, elinewidth=1.2, capsize=2.5, label=label, zorder=4)


# ---------------- F1 ----------------
def f1():
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=True)
    for ax, (d, name) in zip(axes, MODELS):
        P = jload(find(os.path.join(REP, d), 'report_', '_pretrained'))['results']
        R = jload(find(os.path.join(REP, d), 'report_', '_random'))['results']
        qs = ['zs1', 'zs2', 'zs3']; x = np.arange(3)
        for j, q in enumerate(qs):
            for off, src, key, col, hol, lab in ((-0.22, P, 'probe_auc', C1, False, 'зонд на позиции вопроса — предобученная'),
                                                 (-0.07, R, 'probe_auc', C1, True, 'зонд — случайные веса'),
                                                 (0.07, P, 'auc', C2, False, 'ответ (логит «A» − «B») — предобученная'),
                                                 (0.22, R, 'auc', C2, True, 'ответ — случайные веса')):
                m, e = ci([r['modes'][q][key] for r in src])
                dot(ax, j + off, m, e, col, lab if (j == 0 and ax is axes[0]) else None, hollow=hol)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_xticks(x, ['Food:', 'Safe to\neat:', 'Good\nkind:'])
        ax.set_title(f'{name}  (n = {len(P)} наборов)', fontsize=9, color=INK)
        ax.set_ylim(0.4, 1.02)
    axes[0].set_ylabel('AUC против знака эпизода')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.suptitle('F1. Знак тела в представлении есть, в ответе — меньше половины (flip-2, окно 32, вопросы без примеров; 95 % ДИ по наборам)', fontsize=9.5, y=1.03)
    save(fig, 'F1_probe_vs_answer')


# ---------------- F2 ----------------
def f2():
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(MODELS)); w = 0.2
    for i, (d, name) in enumerate(MODELS):
        r = jload(os.path.join(REP, d, 'r12.json'))['results']
        g = lambda wts, c: np.array([next(z['bacc'] for z in r if z['set'] == s and z['weights'] == wts and z['cond'] == c) for s in range(1, 11)])
        series = [('pretrained', ['c'], C1, False, 'предобученная — осмысленные слова (A, B, none / up, down, same)'),
                  ('random', ['c'], C1, True, 'случайные веса — осмысленные слова'),
                  ('pretrained', ['c_arb1', 'c_arb2'], C2, False, 'предобученная — произвольные слова (среднее двух наборов)'),
                  ('random', ['c_arb1', 'c_arb2'], C2, True, 'случайные веса — произвольные слова')]
        for j, (wts, cs, col, hol, lab) in enumerate(series):
            v = np.mean([g(wts, c) for c in cs], 0); m, e = ci(v)
            dot(ax, i + (j - 1.5) * w, m, e, col, lab if i == 0 else None, hollow=hol)
    ax.set_xticks(x, [f'{n}\n(n = 10 наборов)' for _, n in MODELS]); ax.set_ylim(0.88, 1.0)
    ax.set_ylabel('точность зонда знака\n(сбалансированная)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)
    ax.set_title('F2. Произвольные слова не хуже осмысленных\n(flip-2, окно 32, зонд по последней позиции; 95 % ДИ по наборам)', fontsize=9.5)
    save(fig, 'F2_words_vs_random')


# ---------------- F3 ----------------
def boot_auc(dec, y, ep, B=2000, seed=0):
    dec, y, ep = np.asarray(dec), np.asarray(y, bool), np.asarray(ep)
    def auc(s, t):
        p, n = s[t], s[~t]; return ((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())
    ue = np.unique(ep); groups = {e: np.nonzero(ep == e)[0] for e in ue}; rng = np.random.RandomState(seed); out = []
    for _ in range(B):
        idx = np.concatenate([groups[e] for e in rng.choice(ue, len(ue))])
        if len(np.unique(y[idx])) == 2:
            out.append(auc(dec[idx], y[idx]))
    return auc(dec, y), np.percentile(out, 2.5), np.percentile(out, 97.5)


def f3():
    src = [('results_meaning/meaning_1b.json', 'Pythia-1B\n(A24)'), ('results_meaning/meaning_2.8b.json', 'Pythia-2.8B\n(A24)'),
           ('results_meaning/meaning_6.9b.json', 'Pythia-6.9B\n(A24)'), (find(os.path.join(REP, 'Qwen2.5-1.5B'), 'meaning_'), 'Qwen2.5-1.5B\n(A28)'),
           (find(os.path.join(REP, 'OLMo-2-0425-1B'), 'meaning_'), 'OLMo-2-1B\n(A28)')]
    fig, ax = plt.subplots(figsize=(7.4, 3.1)); x = np.arange(len(src))
    for i, (p, name) in enumerate(src):
        r = jload(p)
        for off, wts, pos, col, hol, lab in ((-0.24, 'pretrained', 'kindA', C1, False, 'предобученная — позиция символа вида'),
                                             (-0.08, 'random', 'kindA', C1, True, 'случайные веса — позиция символа вида'),
                                             (0.08, 'pretrained', 'last', C2, False, 'предобученная — последняя позиция окна'),
                                             (0.24, 'random', 'last', C2, True, 'случайные веса — последняя позиция окна')):
            c2 = r[wts]['check2']
            a, lo, hi = boot_auc(c2[f'dec_{pos}'], c2['label'], c2['episode'], seed=i)
            dot(ax, i + off, a, (a - lo, hi - a), col, lab if i == 0 else None, hollow=hol)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':'); ax.axhline(0.8, color=MUTED, lw=0.6, ls='--')
    ax.set_xticks(x, [n for _, n in src]); ax.set_ylim(0.2, 1.05); ax.set_ylabel('AUC зонда «слово вида — еда»')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7.5)
    ax.set_title('F3. Смысл слова есть на его позиции, но не на последней позиции окна (200 эпизодов; 95 % ДИ — бутстреп по эпизодам)', fontsize=9.5)
    save(fig, 'F3_meaning_by_position')


# ---------------- F4 ----------------
def f4():
    src = [(find(os.path.join(REP, 'pythia-1b'), 'synth_'), 'Pythia-1B'), (find(os.path.join(REP, 'Qwen2.5-1.5B'), 'synth_'), 'Qwen2.5-1.5B'),
           (find(os.path.join(REP, 'OLMo-2-0425-1B'), 'synth_'), 'OLMo-2-1B'), ('results_synth/synth_6.9b.json', 'Pythia-6.9B*')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.1))
    x = np.arange(len(src))
    for i, (p, name) in enumerate(src):
        S = jload(p)['sets']
        g = lambda w, v, k, n=None: np.array([(S[f'{w}/s/{s}']['measures'][v]['by_n'][n] if n else S[f'{w}/s/{s}']['measures'][v])[k] for s in range(1, 11)])
        L, Lp = g('pretrained', 'q1', 'd_lying'), g('pretrained', 'q1_perm', 'd_lying'); Lr, Lrp = g('random', 'q1', 'd_lying'), g('random', 'q1_perm', 'd_lying')
        for off, v, col, h, lab in ((-0.27, -Lp, C1, None, 'эффект смысла: |d| на окнах без связки'), (-0.09, L - Lp, C2, None, 'сдвиг к опыту: d(исходы) − d(без связки)'),
                                    (0.09, Lr - Lrp, GRAY, None, 'сдвиг к опыту у случайных весов')):
            m, e = ci(v)
            a1.bar(i + off, m, width=0.17, color=col, edgecolor=col, linewidth=1.0, label=lab if i == 0 else None)
            a1.errorbar(i + off, m, yerr=e, fmt='none', ecolor=INK, elinewidth=0.9, capsize=2)
        fr = np.mean([g('pretrained', 'q1', 'frac_lying_to_experience', n) for n in ('2', '4', '6')], 0)
        frp = np.mean([g('pretrained', 'q1_perm', 'frac_lying_to_experience', n) for n in ('2', '4', '6')], 0)
        for off, v, col, h, lab in ((-0.1, fr, C2, None, 'с исходами (связка есть)'), (0.1, frp, C1, '///', 'исходы переставлены (связки нет)')):
            m, e = ci(v)
            a2.bar(i + off, m, width=0.18, color='white' if h else col, edgecolor=col, hatch=h, linewidth=1.0, label=lab if i == 0 else None)
            a2.errorbar(i + off, m, yerr=e, fmt='none', ecolor=INK, elinewidth=0.9, capsize=2)
    a1.axhline(0, color=MUTED, lw=0.8)
    for a in (a1, a2):
        a.set_xticks(x, [f'{n}\n(n = 10)' for _, n in src], fontsize=8)
    a1.set_ylabel('логиты (слово истинной еды − слово яда)'); a1.set_title('а) лживые окна: смысл против опыта', fontsize=9)
    a2.set_ylabel('доля лживых окон, где ответ — к опыту'); a2.set_ylim(0, 0.6); a2.set_title('б) ответ почти не меняется', fontsize=9)
    a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=7.5); a2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=7.5)
    fig.suptitle('F4. Синтетические окна без частотного канала: связка «слово ↔ исход» есть, но мала и ответа не меняет '
                 '(интерфейс-предложения, n = 2–6 событий; 95 % ДИ по наборам)', fontsize=9.5, y=1.04)
    fig.text(0.5, -0.36, '* Pythia-6.9B — A27, другой экземпляр RTX 5090; остальные — A28, один экземпляр.', fontsize=7, color=MUTED, ha='center')
    save(fig, 'F4_synthetic_binding')


if __name__ == '__main__':
    f1(); f2(); f3(); f4(); print('готово:', sorted(os.listdir(OUT)))
