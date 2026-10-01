"""Черновик: «решения без навигации». Каждый такт предлагается предмет вида A или B (50/50);
действие: съесть / пропустить. Знак эпизода и разворот как в flip-2 (такт 30-69). Энергия
дрейфует, приспособленность — ср. энергия×2 (смерть: E <= 0). Кадр органа: [предложен A,
предложен B, съел A, съел B, тренд энергии, энергия]."""
import numpy as np
T, GAIN, DRIFT, E0 = 100, 0.35, 0.012, 0.6
def episode_batch(policy, n, poison, death, seed0=999000):
    rs = [np.random.RandomState(seed0 + k) for k in range(n)]
    goodA = np.array([r.rand() < .5 for r in rs]); trev = np.array([r.randint(30, 70) for r in rs])
    offers = np.array([[r.rand() < .5 for _ in range(T)] for r in rs])          # True = A
    E = np.full(n, E0); prevE = E.copy(); alive = np.ones(n, bool); ate = np.zeros((n, 2)); hist = []; Es = []
    for t in range(T):
        goodA ^= trev == t
        offA = offers[:, t]
        s = np.stack([offA, ~offA, ate[:, 0], ate[:, 1], (E - prevE) * 10, E], 1).astype(float); hist.append(s)
        prevE = E.copy()
        eat = policy(s, hist, goodA) & alive
        good = offA == goodA
        E = np.minimum(E - DRIFT + eat * np.where(good, GAIN, -poison), 1.0)
        ate[:] = 0; ate[eat & offA, 0] = 1; ate[eat & ~offA, 1] = 1
        if death: alive &= E > 0
        else: E = np.maximum(E, 0)
        Es.append(np.where(alive, E, 0))
    return np.mean(Es, 0) * 2
def learner(N):
    def pol(s, hist, goodA):
        H = np.stack(hist[-N - 1:-1], 1) if len(hist) > 1 else None             # прошлые кадры
        eat = np.ones(len(s), bool)
        if H is None: return eat
        for k in range(len(s)):
            # исход еды из кадра t виден в кадре t+1 (флаг «съел» и тренд) — оба в прошлом окне
            ev = np.where(H[k, :, 2:4].sum(1) > 0)[0]
            if len(ev):
                fr = H[k, ev[-1]]; gA = (fr[2] > 0) == (fr[4] > 0)
                eat[k] = bool(s[k, 0]) == gA
        return eat
    return pol
def bag(N):
    def pol(s, hist, goodA):
        H = np.stack(hist[-N - 1:-1], 1) if len(hist) > 1 else None
        eat = np.ones(len(s), bool)
        if H is None: return eat
        for k in range(len(s)):
            ev = H[k, H[k, :, 2:4].sum(1) > 0]
            if len(ev):
                v = np.where((ev[:, 2] > 0) == (ev[:, 4] > 0), 1, -1).sum()
                if v: eat[k] = bool(s[k, 0]) == (v > 0)
        return eat
    return pol
pols = {'не есть': lambda s, h, g: np.zeros(len(s), bool), 'есть всё (без памяти)': lambda s, h, g: np.ones(len(s), bool),
        'окно 1': learner(1), 'окно 2': learner(2), 'окно 8': learner(8), 'окно 32': learner(32), 'мешок 32': bag(32),
        'оракул': lambda s, h, g: s[:, 0].astype(bool) == g}
steps = [('есть всё', 'не есть'), ('окно 1', 'есть всё (без памяти)'), ('окно 2', 'окно 1'), ('окно 8', 'окно 2'),
         ('окно 32', 'окно 8'), ('окно 32', 'мешок 32'), ('оракул', 'окно 32')]
for poison in (0.20, 0.30):
    for death in (True, False):
        F = {k: episode_batch(p, 300, poison, death) for k, p in pols.items()}
        F['есть всё'] = F['есть всё (без памяти)']
        print(f'яд {poison:.2f}, {"со смертью" if death else "без смерти"}: ' + ' '.join(f'{k} {F[k].mean():.2f}' for k in pols))
        print('   Δ/SNR(4 эп.): ' + ' | '.join(f'{a}−{b}: {(F[a]-F[b]).mean():+.3f}/{(F[a]-F[b]).mean()/((F[a]-F[b]).std()/2):.1f}' for a, b in steps))
