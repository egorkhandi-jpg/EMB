"""Черновик: решения без навигации + редкая обратная связь. Съеденное действует с вероятностью p,
иначе энергия меняется только дрейфом (событие неинформативно)."""
import numpy as np
T, GAIN, DRIFT, E0 = 100, 0.35, 0.012, 0.6
def batch(policy, n, poison, death, p, seed0=999000):
    rs = [np.random.RandomState(seed0 + k) for k in range(n)]
    goodA = np.array([r.rand() < .5 for r in rs]); trev = np.array([r.randint(30, 70) for r in rs])
    offers = np.array([[r.rand() < .5 for _ in range(T)] for r in rs]); fire = np.array([[r.rand() < p for _ in range(T)] for r in rs])
    E = np.full(n, E0); prevE = E.copy(); alive = np.ones(n, bool); ate = np.zeros((n, 2)); hist = []; Es = []
    for t in range(T):
        goodA ^= trev == t
        offA = offers[:, t]
        s = np.stack([offA, ~offA, ate[:, 0], ate[:, 1], (E - prevE) * 10, E], 1).astype(float); hist.append(s)
        prevE = E.copy()
        eat = policy(s, hist, goodA) & alive
        good = offA == goodA
        E = np.minimum(E - DRIFT + (eat & fire[:, t]) * np.where(good, GAIN, -poison), 1.0)
        ate[:] = 0; ate[eat & offA, 0] = 1; ate[eat & ~offA, 1] = 1
        if death: alive &= E > 0
        else: E = np.maximum(E, 0)
        Es.append(np.where(alive, E, 0))
    return np.mean(Es, 0) * 2
def vote(N, last=None):
    """Голоса информативных событий в окне N (|тренд| заметно больше дрейфа); last — сколько последних учитывать."""
    def pol(s, hist, g):
        eat = np.ones(len(s), bool)
        if len(hist) < 2: return eat
        H = np.stack(hist[-N - 1:-1], 1)
        for k in range(len(s)):
            m = (H[k, :, 2:4].sum(1) > 0) & (np.abs(H[k, :, 4]) > 0.5)
            ev = H[k, m]
            if last: ev = ev[-last:]
            if len(ev):
                v = np.where((ev[:, 2] > 0) == (ev[:, 4] > 0), 1, -1).sum()
                if v: eat[k] = bool(s[k, 0]) == (v > 0)
        return eat
    return pol
pols = {'не есть': lambda s, h, g: np.zeros(len(s), bool), 'есть всё': lambda s, h, g: np.ones(len(s), bool),
        'посл. 2': vote(2, 1), 'посл. 8': vote(8, 1), 'посл. 32': vote(32, 1), 'мешок 32': vote(32), 'оракул': lambda s, h, g: s[:, 0].astype(bool) == g}
steps = [('есть всё', 'не есть'), ('посл. 2', 'есть всё'), ('посл. 8', 'посл. 2'), ('посл. 32', 'посл. 8'), ('посл. 32', 'мешок 32'), ('оракул', 'посл. 32')]
for p in (0.5, 0.25, 0.15):
    for death in (False, True):
        F = {k: batch(f, 300, 0.20, death, p) for k, f in pols.items()}
        print(f'p {p:.2f}, {"со смертью" if death else "без смерти"}: ' + ' '.join(f'{k} {F[k].mean():.2f}' for k in pols))
        print('   Δ/SNR(4 эп.): ' + ' | '.join(f'{a}−{b}: {(F[a]-F[b]).mean():+.3f}/{(F[a]-F[b]).mean()/((F[a]-F[b]).std()/2+1e-9):.1f}' for a, b in steps), flush=True)
