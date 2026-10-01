"""Отношение сигнал/шум шагов цепочки частичных решений для ES: разность двух эвристик,
попарно по эпизоду, при 4 эпизодах на особь: SNR = mean(d) / (sd(d)/2). Варианты среды —
подменой констант модуля (черновик)."""
import sys, numpy as np
sys.path.insert(0, r'C:\embodied-lm-stand')
import env_flip as ef
def run(pol, n=300, fit='base'):
    b = ef.FlipBodies(n, [999000 + k for k in range(n)], True, 'flip-2'); hist = []; Es = []
    for _ in range(ef.T):
        s = b.sense(); hist.append(s); a = pol(s, hist, b); a[~b.alive] = 0; b.step(a)
        Es.append(np.where(b.alive, b.E, 0))
    return b.fitness() if fit == 'base' else np.mean(Es, 0) * 2      # «средняя энергия», масштаб ~ как base
stand = lambda s, h, b: np.tile([0, -1.0], (len(s), 1))
steps = [('без памяти − стоять', ef.greedy_any, stand), ('окно 2 − без памяти', ef.learner(2), ef.greedy_any),
         ('окно 8 − окно 2', ef.learner(8), ef.learner(2)), ('окно 32 − окно 8', ef.learner(32), ef.learner(8))]
def scan(label, fit='base'):
    cache = {}
    def F(p):
        if id(p) not in cache: cache[id(p)] = run(p, fit=fit)
        return cache[id(p)]
    out = []
    for nm, a_, b_ in steps:
        d = F(a_) - F(b_); out.append(f'{d.mean():+.3f}/SNR {d.mean() / (d.std() / 2):.2f}')
    print(f'{label:34s} ' + ' | '.join(out), flush=True)
print('шаги: ' + ' | '.join(s[0] for s in steps))
base = dict(N_PER=ef.N_PER, E0=ef.E0, T=ef.T)
scan('flip-2 как есть')
scan('flip-2, приспособленность = ср. энергия', fit='mean')
ef.N_PER = 4; scan('по 4 предмета каждого вида'); scan('по 4 предмета + ср. энергия', fit='mean')
ef.N_PER = base['N_PER']; ef.E0 = 0.9; scan('E0 = 0,9'); ef.E0 = base['E0']
