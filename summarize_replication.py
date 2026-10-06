"""Сводка репликации (docs/RESULTS_2026-10-06_replication.md). Семья — R1, R2, R5 × {Qwen2.5-1.5B, OLMo-2-0425-1B}: точные парные
перестановки по 10 наборам, односторонние; в R2, R5 p — больший из двух условий; Холм по шести. R3, R4 — пороговые.
Pythia-1B — ориентир, описательно (те же тесты без поправки).

python summarize_replication.py results_replication > results_replication/summary.txt
"""
import itertools, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_replication'
FAMILY = ['Qwen2.5-1.5B', 'OLMo-2-0425-1B']; REF = 'pythia-1b'
SIGNS = np.array(list(itertools.product([1, -1], repeat=10)))
perm = lambda x: float(((SIGNS * np.asarray(x, float)).mean(1) >= np.mean(x) - 1e-12).mean())


def holm(ps):
    o = np.argsort(ps); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def load(m, name):
    p = os.path.join(D, m, name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def r12(m):
    r = load(m, 'r12.json')
    if not r:
        return None
    g = lambda w, c: np.array([next(x['bacc'] for x in r['results'] if x['set'] == s and x['weights'] == w and x['cond'] == c) for s in range(1, 11)])
    pc, rc = g('pretrained', 'c'), g('random', 'c')
    pa, ra = (g('pretrained', 'c_arb1') + g('pretrained', 'c_arb2')) / 2, (g('random', 'c_arb1') + g('random', 'c_arb2')) / 2
    return {'pc': pc, 'rc': rc, 'pa': pa, 'ra': ra, 'p1': perm(pc - rc), 'p2': max(perm(pa - pc + 0.02), perm(pa - ra))}


def r3(m):
    out = {}
    for w in ('pretrained', 'random'):
        f = [x for x in os.listdir(os.path.join(D, m)) if x.startswith('report_') and x.endswith(f'_{w}.json')] if os.path.isdir(os.path.join(D, m)) else []
        if not f:
            return None
        R = json.load(open(os.path.join(D, m, f[0]), encoding='utf-8'))['results']
        out[w] = {md: (np.mean([x['modes'][md]['auc'] for x in R]), np.mean([x['modes'][md]['probe_auc'] for x in R]), len(R)) for md in ('zs1', 'zs2', 'zs3')}
    return out


def r4(m):
    f = [x for x in os.listdir(os.path.join(D, m)) if x.startswith('meaning_') and x.endswith('.json')] if os.path.isdir(os.path.join(D, m)) else []
    return json.load(open(os.path.join(D, m, f[0]), encoding='utf-8')) if f else None


def r5(m):
    f = [x for x in os.listdir(os.path.join(D, m)) if x.startswith('synth_') and x.endswith('.json')] if os.path.isdir(os.path.join(D, m)) else []
    if not f:
        return None
    S = json.load(open(os.path.join(D, m, f[0]), encoding='utf-8'))['sets']
    g = lambda w, v, k, n=None: np.array([(S[f'{w}/s/{i}']['measures'][v]['by_n'][n] if n else S[f'{w}/s/{i}']['measures'][v])[k] for i in range(1, 11)])
    L, Lp, Lr, Lrp = g('pretrained', 'q1', 'd_lying'), g('pretrained', 'q1_perm', 'd_lying'), g('random', 'q1', 'd_lying'), g('random', 'q1_perm', 'd_lying')
    fr = np.mean([g('pretrained', 'q1', 'frac_lying_to_experience', n) for n in ('2', '4', '6')], 0)
    frp = np.mean([g('pretrained', 'q1_perm', 'frac_lying_to_experience', n) for n in ('2', '4', '6')], 0)
    return {'L': L, 'Lp': Lp, 'sp': L - Lp, 'sr': Lr - Lrp, 'p': max(perm(L - Lp), perm((L - Lp) - (Lr - Lrp))), 'fr': fr, 'frp': frp}


res = {m: {'r12': r12(m), 'r3': r3(m), 'r4': r4(m), 'r5': r5(m)} for m in FAMILY + [REF]}
tests = []
for m in FAMILY:
    a, c = res[m]['r12'], res[m]['r5']
    tests += [(m, 'R1', a['p1'] if a else 1.0), (m, 'R2', a['p2'] if a else 1.0), (m, 'R5', c['p'] if c else 1.0)]
adj = holm([t[2] for t in tests]); verdict = {}
print('СЕМЬЯ (Холм по шести; односторонние точные перестановки по 10 наборам):')
for (m, k, p), pa in zip(tests, adj):
    verdict[(m, k)] = pa < 0.05
    print(f'  {m:16s} {k}: p {p:.4f}, p(Холм) {pa:.4f} → {"воспроизвелось" if pa < 0.05 else "нет"}')

print('\nПО МОДЕЛЯМ (Pythia-1B — ориентир, без поправки):')
for m in FAMILY + [REF]:
    print(f'=== {m}')
    a = res[m]['r12']
    if a:
        print(f"  R1: пред(c) {a['pc'].mean():.3f} ± {a['pc'].std():.3f}, случ(c) {a['rc'].mean():.3f} ± {a['rc'].std():.3f}; p {a['p1']:.4f}")
        print(f"  R2: пред(c_arb) {a['pa'].mean():.3f}, пред(c) {a['pc'].mean():.3f} (Δ {np.mean(a['pa'] - a['pc']):+.3f}, граница −0,02), случ(c_arb) {a['ra'].mean():.3f}; p {a['p2']:.4f}")
    b = res[m]['r3']
    if b:
        ok = []
        for md, (au, pr, n) in b['pretrained'].items():
            cond = (au - 0.5) <= 0.5 * (pr - 0.5); ok.append(cond)
            print(f"  R3 {md}: AUC ответа {au:.3f} (случ {b['random'][md][0]:.3f}), AUC зонда {pr:.3f} (случ {b['random'][md][1]:.3f}) → "
                  f"(ответ − 0,5) {au - 0.5:+.3f} ≤ 0,5·(зонд − 0,5) {0.5 * (pr - 0.5):.3f}: {'да' if cond else 'нет'}")
        verdict[(m, 'R3')] = all(ok); print(f"  R3 итог (все три вопроса): {'выполнено' if all(ok) else 'нет'}")
    c4 = res[m]['r4']
    if c4:
        P, Q = c4['pretrained'], c4['random']
        kA, la = P['check2']['auc_kindA'], P['check2']['auc_last']
        verdict[(m, 'R4')] = kA >= 0.9 and la < 0.8
        print(f"  R4: позиция символа вида {kA:.3f} (случ {Q['check2']['auc_kindA']:.3f}), последняя позиция {la:.3f} (случ {Q['check2']['auc_last']:.3f}) → "
              f"{'выполнено' if verdict[(m, 'R4')] else 'нет'}; положительный контроль (прямые слова, зонд) {P['check1']['auc_probe_direct']:.3f}, "
              f"зонд по словам (проверка 1) {P['check1']['auc_probe']:.3f} (случ {Q['check1']['auc_probe']:.3f})")
    e = res[m]['r5']
    if e:
        print(f"  R5: d лож {e['L'].mean():+.3f}, переставл. {e['Lp'].mean():+.3f}, сдвиг пред {e['sp'].mean():+.3f}, сдвиг случ {e['sr'].mean():+.3f}; p {e['p']:.4f}; "
              f"доля лживых окон к опыту {e['fr'].mean():.3f} (переставл. {e['frp'].mean():.3f})")
    if m == REF:
        for k, v in (('R1', a and a['p1'] < 0.05), ('R2', a and a['p2'] < 0.05), ('R5', e and e['p'] < 0.05)):
            verdict[(m, k)] = bool(v)

print('\nТОЛКОВАНИЕ (записано заранее):')
for k in ('R1', 'R2', 'R3', 'R4', 'R5'):
    n = sum(bool(verdict.get((m, k))) for m in FAMILY)
    t = {2: 'на обеих моделях — утверждение о замороженных LM порядка 1–2B', 1: 'на одной — утверждение с оговоркой о семействе',
         0: 'ни на одной — утверждение только о Pythia'}[n]
    print(f"  {k}: {t} ({', '.join(m for m in FAMILY if verdict.get((m, k))) or '—'}); Pythia-1B (ориентир): {'да' if verdict.get((REF, k)) else 'нет'}")
