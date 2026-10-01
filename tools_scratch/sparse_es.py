"""Черновик: ES на варианте «решения без навигации, срабатывает с p = 0,5, без смерти».
Действие a0 > 0 — съесть предложенный предмет. Орган: [предложен A, предложен B, съел A, съел B, тренд, E]."""
import sys, numpy as np, torch
sys.path.insert(0, r'C:\embodied-lm-stand')
import stand_flip as sf
sf.S_DIM = 6
torch.set_num_threads(4)
T, GAIN, DRIFT, E0, POISON, P = 100, 0.35, 0.012, 0.6, 0.20, 0.5
class Decide:
    def __init__(self, B, seeds):
        rs = [np.random.RandomState(s) for s in seeds]; self.B = B
        self.goodA = np.array([r.rand() < .5 for r in rs]); self.trev = np.array([r.randint(30, 70) for r in rs])
        self.off = np.array([[r.rand() < .5 for _ in range(T)] for r in rs]); self.fire = np.array([[r.rand() < P for _ in range(T)] for r in rs])
        self.E = np.full(B, E0); self.prevE = self.E.copy(); self.ate = np.zeros((B, 2)); self.t = 0; self.Es = []; self.alive = np.ones(B, bool)
        self.ng = np.zeros(B); self.nb = np.zeros(B)
    def sense(self):
        self.goodA ^= self.trev == self.t
        o = self.off[:, self.t]
        s = np.stack([o, ~o, self.ate[:, 0], self.ate[:, 1], (self.E - self.prevE) * 10, self.E], 1).astype(np.float32)
        self.prevE = self.E.copy(); return s
    def step(self, a):
        o = self.off[:, self.t]; eat = a[:, 0] > 0; good = o == self.goodA
        self.E = np.clip(self.E - DRIFT + (eat & self.fire[:, self.t]) * np.where(good, GAIN, -POISON), 0, 1)
        self.ate[:] = 0; self.ate[eat & o, 0] = 1; self.ate[eat & ~o, 1] = 1
        self.ng += eat & good; self.nb += eat & ~good; self.Es.append(self.E.copy()); self.t += 1
    def fitness(self): return np.mean(self.Es, 0) * 2
def rollout(pol, TH, seeds):
    Pn, E = TH.shape[0], len(seeds); THb = torch.as_tensor(np.repeat(TH, E, 0).astype(np.float32))
    b = Decide(Pn * E, list(seeds) * Pn); pol.reset(Pn * E)
    for _ in range(T):
        s = torch.as_tensor(b.sense()); b.step(pol.act(THb, s).numpy())
    return b.fitness().reshape(Pn, E).mean(1), b
def train(pol, seed, gens=40, pairs=16, eps=4):
    rng = np.random.RandomState(seed); th = pol.init(rng); m = np.zeros_like(th); v = np.zeros_like(th); H = []
    for g in range(gens):
        es = [seed * 100000 + g * 10 + k for k in range(eps)]; nz = rng.randn(pairs, pol.n).astype(np.float32)
        f, _ = rollout(pol, np.concatenate([th + .05 * nz, th - .05 * nz]), es)
        r = f.argsort().argsort() / (len(f) - 1) - .5; gr = ((r[:pairs] - r[pairs:])[:, None] * nz).sum(0) / (2 * pairs * .05)
        m = .9 * m + .1 * gr; v = .999 * v + .001 * gr ** 2
        th = (th + .03 * (m / (1 - .9 ** (g + 1))) / (np.sqrt(v / (1 - .999 ** (g + 1))) + 1e-8)).astype(np.float32); H.append(f.mean())
    f, b = rollout(pol, th[None], sf.TEST_SEEDS); return f[0], b.ng.mean(), b.nb.mean(), np.array(H)
print('ориентиры (300 эп.): не есть 0,29; есть всё 1,28; посл. 2 1,59; посл. 8 1,75; посл. 32 1,83; мешок 32 1,77; оракул 1,90')
for name, mk in [('linwin N8', lambda: sf.WindowNet(8, 0, 'cpu')), ('flywin N2', lambda: sf.WindowNet(2, 32, 'cpu')),
                 ('flywin N8', lambda: sf.WindowNet(8, 32, 'cpu')), ('flywin N32', lambda: sf.WindowNet(32, 32, 'cpu')), ('gru', lambda: sf.GRUFly(56, 'cpu'))]:
    for s in (1, 2, 3):
        t, g_, b_, H = train(mk(), s)
        print(f'{name:10s} s{s} тест {t:.3f} пища/яд {g_:.1f}/{b_:.1f} | кривая по 5: ' + ' '.join(f'{H[i:i+5].mean():.2f}' for i in range(0, 40, 5)), flush=True)
