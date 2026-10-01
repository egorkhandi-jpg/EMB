import sys, numpy as np
sys.path.insert(0, r'C:\embodied-lm-stand')
import env_flip as ef
def run(pol, n=300, death=True, T=100):
    ef.T = T
    b = ef.FlipBodies(n, [999000 + k for k in range(n)], True, 'flip-2')
    b.t_rev = np.array([r.randint(int(.3 * T), int(.7 * T)) for r in b.rs])   # разворот в той же доле эпизода
    hist = []; Es = []
    for _ in range(T):
        s = b.sense(); hist.append(s); a = pol(s, hist, b); a[~b.alive] = 0; b.step(a)
        if not death: b.E = np.maximum(b.E, 0); b.alive[:] = True
        Es.append(np.where(b.alive, b.E, 0))
    ef.T = 100
    return np.mean(Es, 0) * 2
stand = lambda s, h, b: np.tile([0, -1.0], (len(s), 1))
steps = [('без памяти − стоять', ef.greedy_any, stand), ('окно 2 − без памяти', ef.learner(2), ef.greedy_any),
         ('окно 8 − окно 2', ef.learner(8), ef.learner(2)), ('окно 32 − окно 8', ef.learner(32), ef.learner(8)),
         ('окно 32 − мешок 32', ef.learner(32), ef.bag_learner(32)), ('оракул − окно 32', ef.oracle, ef.learner(32))]
def scan(label, **kw):
    c = {}
    F = lambda p: c.setdefault(id(p), run(p, **kw))
    print(f'{label:36s} ' + ' | '.join(f'{(F(a)-F(b)).mean():+.3f}/{(F(a)-F(b)).mean()/((F(a)-F(b)).std()/2):.1f}' for _, a, b in steps), flush=True)
print('приспособленность = ср. энергия×2; Δ/SNR при 4 эпизодах; шаги: ' + ' | '.join(s[0] for s in steps))
scan('смерть, 2 предмета, T 100')
scan('без смерти, 2 предмета, T 100', death=False)
ef.N_PER = 4
scan('без смерти, 4 предмета, T 100', death=False)
scan('без смерти, 4 предмета, T 200', death=False, T=200)
