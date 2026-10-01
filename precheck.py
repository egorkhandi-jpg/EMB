"""Предварительные проверки стенда: есть ли барьер, есть ли выгода, различим ли эффект.
python precheck.py [--episodes 20]"""
import argparse, numpy as np
from stand_gpu import Bodies, T
ap = argparse.ArgumentParser(); ap.add_argument('--episodes', type=int, default=20); a = ap.parse_args()
seeds = [999000 + k for k in range(a.episodes)]
def run(fn):
    b = Bodies(len(seeds), seeds)
    for _ in range(T):
        s = b.sense(); act = fn(s); act[~b.alive] = 0; b.step(act)
    f = b.fitness(); return f.mean(), f.std()
rng = np.random.RandomState(0)
res = {'стоять': run(lambda s: np.tile([0, -1.0], (len(s), 1))),
       'случайно': run(lambda s: rng.uniform(-1, 1, (len(s), 2))),
       'жадно к пище': run(lambda s: np.stack([np.clip(np.arctan2(s[:, 1], s[:, 0]) / 0.6, -1, 1), np.ones(len(s))], 1))}
for k, (m, sd) in res.items(): print(f'{k:14s} {m:.3f} ± {sd:.3f}')
gap = res['жадно к пище'][0] - res['случайно'][0]
print(f"барьер: {'есть' if res['случайно'][0] < 0.8 else 'НЕТ'} | выгода (жадно − случайно): {gap:.3f} | "
      f"различимость (выгода / разброс случайной): {gap / (res['случайно'][1] + 1e-9):.1f}")
