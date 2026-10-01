import sys, numpy as np
sys.path.insert(0, r'C:\embodied-lm-stand')
import env_flip as ef
rng = np.random.RandomState(0)
for rev in (False, True):
    print('разворот' if rev else 'постоянный знак')
    for P in (0.30, 0.20, 0.15, 0.10):
        ef.POISON = P
        pe = {k: ef.run(p, n_ep=200, reversal=rev, per_episode=True) for k, p in
              [('стоять', lambda s, h, b: np.tile([0, -1.0], (len(s), 1))), ('без памяти', ef.greedy_any),
               ('окно 2', ef.learner(2)), ('окно 8', ef.learner(8)), ('окно 32', ef.learner(32)),
               ('мешок 32', ef.bag_learner(32)), ('оракул', ef.oracle)]}
        d = pe['окно 32'] - pe['без памяти']; d2 = pe['без памяти'] - pe['стоять']
        print(f'  яд {P:.2f}: ' + ' '.join(f'{k} {v.mean():.3f}' for k, v in pe.items()) +
              f' | без памяти − стоять {d2.mean():+.3f} (z {d2.mean()/(d2.std()/np.sqrt(200)):.1f})'
              f' | окно32 − без памяти {d.mean():+.3f} (z {d.mean()/(d.std()/np.sqrt(200)):.1f})')
