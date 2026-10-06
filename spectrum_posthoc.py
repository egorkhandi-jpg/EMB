"""Ось 1, версия 1 — анализ ПОСЛЕ расчёта (не пререгистрирован, в критерии не входит): артефакт нижней границы сетки τ.
При «часто» (и «средне») лучший τ у L1 / L2 упирается в нижнюю границу сетки 3,2, а уточнение L3 уходит до 1,8; тогда L3 − L1
при частых свидетельствах — выигрыш от более короткой постоянной, а не от слоёв. Здесь L1, L2, L3 подбираются на сетке,
расширенной вниз до 10^−0,5 (≈ 0,32), на тех же мирах (те же посевы), и пересчитываются критерии 4, 5, 6 и L2 − L1, L3 − L1.

python spectrum_posthoc.py > results_spectrum/posthoc.txt
"""
import math
import numpy as np
import env_spectrum as es

GRID = 10 ** (-0.5 + 0.25 * np.arange(23))           # 0,32 … 100 000
COARSE = 10 ** (-0.5 + 0.5 * np.arange(12))


def fit_ext(W):
    t1, _ = es.best(W, GRID[:, None]); t2, _ = es.best(W, es.monotone(GRID, 2)); t3c, _ = es.best(W, es.monotone(COARSE, 3))
    near = np.array([c for c in __import__('itertools').product(*[[t * 10 ** d for d in (-0.25, 0, 0.25)] for t in t3c]) if c[0] <= c[1] <= c[2]])
    t3, _ = es.best(W, near)
    return {'L1': t1, 'L2': t2, 'L3': t3}


def zi(a, b):
    return (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


print('конфигурация | выигрыш L3 − L1 на встречу: редко / средне / часто, z редко − часто | L2 − L1 (энергия, z) при редко | доля L1 (6) | τ L3 при редко, |ln τh|')
for ci, cfg in enumerate(es.configs()):
    G, row = {}, {}
    for fi, (fn, c) in enumerate(es.FREQ.items()):
        q = c * cfg['h_f']
        Wtr = es.world(cfg, q, es.N_TRAIN, es.seeds(ci, fi, 'train')); Wte = es.world(cfg, q, es.N_TEST, es.seeds(ci, fi, 'test'))
        t = fit_ext(Wtr)
        sc = {k: es.score(Wte, es.cascade(Wte, [t[k]])[0]) for k in ('L1', 'L2', 'L3')}
        G[fn] = sc['L3'][1] - sc['L1'][1]
        if fn == 'rare':
            B = es.score(Wte, es.bayes(Wte, cfg))[0]; Rr = es.score(Wte, es.reactive(Wte, es.seeds(ci, fi, 'test') + 1))[0]
            d = sc['L2'][0] - sc['L1'][0]
            row = {'l2l1': (d.mean(), d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-300)),
                   'share': (sc['L1'][0].mean() - Rr.mean()) / (B.mean() - Rr.mean()), 'tau': t['L3'],
                   'lt': [abs(math.log(t['L3'][0] * cfg['h_f'])), abs(math.log(t['L3'][1] * cfg['h_m']))]}
    print(f"{cfg['name']:22s} | {G['rare'].mean():+.4f} / {G['mid'].mean():+.4f} / {G['often'].mean():+.4f}, z {zi(G['rare'], G['often']):.1f} | "
          f"{row['l2l1'][0]:+.4f} z {row['l2l1'][1]:.1f} | {row['share']:.2f} | {[round(x) for x in row['tau']]}, {' '.join(f'{x:.2f}' for x in row['lt'])}", flush=True)
