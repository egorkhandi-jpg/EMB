"""Прототип (черновик, не стенд): рефлекторный слой. Агент выдаёт (предпочтение вида, скорость);
зашитый рефлекс рулит к ближайшему предмету выбранного вида: a0 > 0 -> A, иначе B."""
import sys, json, numpy as np, torch
sys.path.insert(0, r'C:\embodied-lm-stand')
import env_flip as ef, stand_flip as sf
from env_flip import FlipBodies, T
torch.set_num_threads(4)
def reflex(s, a):
    kind = (a[:, 0] <= 0).astype(int)
    st = ef._steer(s, 0); stB = ef._steer(s, 1); st[kind == 1] = stB[kind == 1]
    st[:, 1] = a[:, 1]; return st
def rollout(pol, TH, seeds, eps_env='flip-2'):
    P, E = TH.shape[0], len(seeds)
    THb = torch.as_tensor(np.repeat(TH, E, 0).astype(np.float32))
    b = FlipBodies(P * E, list(seeds) * P, True, eps_env); pol.reset(P * E)
    for _ in range(T):
        if not b.alive.any(): break
        s = b.sense(); a = reflex(s, pol.act(THb, torch.as_tensor(s)).numpy()); a[~b.alive] = 0; b.step(a)
    return b.fitness().reshape(P, E).mean(1), b
def train(pol, seed, gens=40, pairs=16, eps=4):
    rng = np.random.RandomState(seed); th = pol.init(rng); m = np.zeros_like(th); v = np.zeros_like(th); H = []
    for g in range(gens):
        es = [seed * 100000 + g * 10 + k for k in range(eps)]
        nz = rng.randn(pairs, pol.n).astype(np.float32)
        f, _ = rollout(pol, np.concatenate([th + .05 * nz, th - .05 * nz]), es)
        r = f.argsort().argsort() / (len(f) - 1) - .5
        gr = ((r[:pairs] - r[pairs:])[:, None] * nz).sum(0) / (2 * pairs * .05)
        m = .9 * m + .1 * gr; v = .999 * v + .001 * gr ** 2
        th = (th + .03 * (m / (1 - .9 ** (g + 1))) / (np.sqrt(v / (1 - .999 ** (g + 1))) + 1e-8)).astype(np.float32); H.append(f.mean())
    f, b = rollout(pol, th[None], sf.TEST_SEEDS)
    return f[0], b.n_good.mean(), b.n_bad.mean(), np.array(H)
if __name__ == '__main__':
    # ориентиры с рефлексом: всегда A на полной скорости (без памяти)
    b = FlipBodies(100, sf.TEST_SEEDS, True, 'flip-2')
    for _ in range(T):
        s = b.sense(); a = reflex(s, np.tile([1., 1.], (100, 1))); a[~b.alive] = 0; b.step(a)
    print(f'без памяти с рефлексом (всегда A): {b.fitness().mean():.3f}', flush=True)
    for name, mk in [('linwin N8', lambda: sf.WindowNet(8, 0, 'cpu')), ('flywin N8', lambda: sf.WindowNet(8, 32, 'cpu')),
                     ('flywin N32', lambda: sf.WindowNet(32, 32, 'cpu')), ('gru', lambda: sf.GRUFly(56, 'cpu'))]:
        for s in (1, 2, 3):
            t, fo, po, H = train(mk(), s)
            print(f'{name:10s} s{s} тест {t:.3f} пища/яд {fo:.1f}/{po:.1f} | кривая по 5: ' + ' '.join(f'{H[i:i+5].mean():.2f}' for i in range(0, 40, 5)), flush=True)
