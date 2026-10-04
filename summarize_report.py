"""Сводка «Звено 4 — отчёт и тело» по пререгистрации (docs/RESULTS_2026-10-04_report.md).
Q1: AUC пред > AUC случ; Q2: AUC пред > AUC копирования; zero-shot (среднее трёх шаблонов) и few-shot; 160M и 1B;
односторонние точные парные перестановки по 10 наборам, Холм по восьми.

python summarize_report.py [results_report]
"""
import itertools, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_report'
MODES = ['zs1', 'zs2', 'zs3', 'fs', 'fsx', 'zs1_sh', 'fs_sh', 'en']


def perm1(d):
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float(((s * d).mean(1) >= d.mean() - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def load(tag):
    p = os.path.join(D, f'report_{tag}.json')
    if not os.path.exists(p):
        return None
    r = json.load(open(p, encoding='utf-8'))
    return {x['set']: x for x in r['results']}, r['runs']


def val(x, mode, key='auc'):
    if mode == 'zs':
        return float(np.mean([x['modes'][m][key] for m in ('zs1', 'zs2', 'zs3')]))
    return x['modes'][mode][key]


tests = []
for model in ('160m', '1b'):
    P, R = load(f'{model}_pretrained'), load(f'{model}_random')
    if not P:
        continue
    P, runs = P
    print(f'=== {model} | прогоны пред: {[(r["git_commit"][:8], r["git_dirty"]) for r in runs]}' + (f' | случ: {[(r["git_commit"][:8], r["git_dirty"]) for r in R[1]]}' if R else ''))
    R = R[0] if R else {}
    ss = sorted(set(P) & set(R)) if R else sorted(P)
    print(f"  копирование AUC {np.mean([P[s]['auc_copy'] for s in ss]):.3f} | «последний ответ примера» AUC {np.mean([P[s]['auc_lastanswer'] for s in ss]):.3f} (точн. {np.mean([P[s]['acc_lastanswer'] for s in ss]):.3f}) | запросов {np.mean([P[s]['n_queries'] for s in ss]):.0f}")
    for mode in MODES:
        pa = [val(P[s], mode) for s in ss]; ra = [val(R[s], mode) for s in ss] if R else [np.nan]
        line = f"  {mode:7s} AUC пред {np.mean(pa):.3f} ± {np.std(pa):.3f} | случ {np.nanmean(ra):.3f} | точн. пред {np.mean([val(P[s], mode, 'acc') for s in ss]):.3f}"
        if 'probe_auc' in P[ss[0]]['modes'][mode]:
            pp = np.mean([P[s]['modes'][mode]['probe_auc'] for s in ss]); rp = np.mean([R[s]['modes'][mode]['probe_auc'] for s in ss]) if R else np.nan
            line += f" | зонд пред {pp:.3f}, случ {rp:.3f} | доля от зонда пред {(np.mean(pa) - 0.5) / (pp - 0.5):.2f}"
        print(line)
    zs_sd = np.mean([np.std([P[s]['modes'][m]['auc'] for m in ('zs1', 'zs2', 'zs3')]) for s in ss])
    print(f'  разброс по шаблонам zero-shot (ст. откл. внутри набора, среднее): {zs_sd:.3f}')
    if R:
        for mode, lab in (('zs', 'zero-shot'), ('fs', 'few-shot')):
            pa = np.array([val(P[s], mode) for s in ss]); ra = np.array([val(R[s], mode) for s in ss]); ca = np.array([P[s]['auc_copy'] for s in ss])
            tests.append((f'{model} {lab} Q1 пред − случ', pa - ra)); tests.append((f'{model} {lab} Q2 пред − копирование', pa - ca))
if tests:
    ps = [perm1(d) for _, d in tests]; adj = holm(ps)
    print('\nОСНОВНАЯ СЕМЬЯ (односторонние, Холм по восьми):')
    for (nm, d), p_, a_ in zip(tests, ps, adj):
        print(f'  {nm:34s} Δ AUC = {d.mean():+.4f}  выше в {int((d > 0).sum())}/{len(d)}  p = {p_:.4f}  p(Холм) = {a_:.4f}')
