"""Сводка A23 (few-shot с примерами из других эпизодов) по пререгистрации — docs/RESULTS_2026-10-04_report.md, раздел A23.
Проверка утечки: среднее AUC «последнего ответа примера» в [0,45; 0,55], иначе — СТОП без толкования.
R1: AUC пред > AUC случ; R2: AUC пред > AUC «последнего ответа»; R3: AUC пред > AUC пред с перемешанным текущим окном;
k = 4, 160M и 1B, односторонние точные парные перестановки по 10 наборам, Холм по шести.

python summarize_icl.py results_report > results_report/summary_icl.txt
"""
import itertools, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_report'
MODES = ['c4', 'c2', 'c8', 'c4_shcur', 'c4_shex', 'c4x', 'c4_en']


def perm1(d):
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float(((s * d).mean(1) >= d.mean() - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def load(tag):
    p = os.path.join(D, f'icl_{tag}.json')
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p, encoding='utf-8'))
    return {x['set']: x for x in r['results']}, r['runs']


tests, leak_ok = [], True
for model in ('160m', '1b'):
    P, rp = load(f'{model}_pretrained'); R, rr = load(f'{model}_random')
    if not P:
        continue
    ss = sorted(set(P) & set(R)) if R else sorted(P)
    print(f'=== {model} | прогоны: пред {[(x["git_commit"][:8], x["git_dirty"]) for x in rp]}' + (f', случ {[(x["git_commit"][:8], x["git_dirty"]) for x in rr]}' if R else ''))
    la = [P[s]['auc_lastanswer'] for s in ss]; ma = [P[s]['auc_majority'] for s in ss]
    ok = 0.45 <= np.mean(la) <= 0.55
    leak_ok &= ok
    print(f"  ПРОВЕРКА УТЕЧКИ: «последний ответ примера» AUC {np.mean(la):.3f} (наборы {min(la):.3f}–{max(la):.3f}), «большинство» {np.mean(ma):.3f} → {'в норме' if ok else 'ВНЕ [0,45; 0,55] — СТОП'}")
    print(f"  копирование вида последнего съеденного: {np.mean([P[s]['auc_copy'] for s in ss]):.3f}; запросов {np.mean([P[s]['n_queries'] for s in ss]):.0f}")
    for mode in MODES:
        pa = [P[s]['modes'][mode]['auc'] for s in ss]; ra = [R[s]['modes'][mode]['auc'] for s in ss] if R else [np.nan]
        line = (f"  {mode:9s} длина {P[ss[0]]['modes'][mode]['len']:4d} | AUC пред {np.mean(pa):.3f} ± {np.std(pa):.3f} | случ {np.nanmean(ra):.3f} | "
                f"точн. пред {np.mean([P[s]['modes'][mode]['acc'] for s in ss]):.3f}")
        if 'probe_auc' in P[ss[0]]['modes'][mode]:
            pp = np.mean([P[s]['modes'][mode]['probe_auc'] for s in ss]); rpb = np.mean([R[s]['modes'][mode]['probe_auc'] for s in ss]) if R else np.nan
            line += f" | зонд пред {pp:.3f}, случ {rpb:.3f} | доля от зонда пред {(np.mean(pa) - 0.5) / (pp - 0.5):.2f}"
        print(line)
    if R:
        pa = np.array([P[s]['modes']['c4']['auc'] for s in ss])
        tests += [(f'{model} R1 пред − случ', pa - np.array([R[s]['modes']['c4']['auc'] for s in ss])),
                  (f'{model} R2 пред − «последний ответ»', pa - np.array(la)),
                  (f'{model} R3 пред − перемешанное текущее окно', pa - np.array([P[s]['modes']['c4_shcur']['auc'] for s in ss]))]
if tests:
    ps = [perm1(d) for _, d in tests]; adj = holm(ps)
    print('\nОСНОВНАЯ СЕМЬЯ (k = 4, односторонние, Холм по шести)' + ('' if leak_ok else ' — НЕ ТОЛКУЕТСЯ: конструкция протекает') + ':')
    for (nm, d), p_, a_ in zip(tests, ps, adj):
        print(f'  {nm:42s} Δ AUC = {d.mean():+.4f}  выше в {int((d > 0).sum())}/{len(d)}  p = {p_:.4f}  p(Холм) = {a_:.4f}')
