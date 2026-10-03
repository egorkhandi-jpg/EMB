"""Стенд пилота D: «LM + рефлекс» с интерфейсом c (события такта словами + BOS), flip-2 с разворотом.

Зачем. П. 4: с событиями словами + BOS предобученная Pythia-160M линейно хранит знак эпизода лучше
случайной (зонд 0,950 против 0,813). Пилоты flip-1/flip-2 (ES по проектору и голове) не нашли
различения ни у одного агента; П3 показал, что голове не хватало второго произведения «знак ×
направление к виду». Здесь второе произведение снимает зашитый рефлекс, проектора нет, LM видит
только события словами, ES учит только линейную голову — ей достаточно прочитать знак.
Изменение агента, а не новая версия среды (docs/ATTEMPTS.md).

Голова выдаёт (предпочтение вида, скорость): a0 > 0 -> к ближайшему A, иначе к ближайшему B
(рефлекс по органу чувств), a1 — скорость. На тесте пишется выбор вида на каждом такте.

Условия (--cond), у всех один вход — события такта (вид съеденного A/B/нет, изменение энергии
рост/падение/нет по порогу |тренд| > 0,5) за окно из N тактов:
  lm      — BOS + 2 токена на такт (--words c | c_arb1), Pythia, --weights pretrained|random
            (случайная: та же архитектура, своя случайная матрица эмбеддингов, приведённая к средней
            норме предобученной); скрытое состояние последней позиции, фиксированная стандартизация
            по случайному поведению; линейная голова 2D + 2;
  quad    — случайные признаки второго порядка (A·x) ⊙ (B·x) ширины 768 по one-hot окну событий
            (6 на такт), фиксированная стандартизация; линейная голова;
  flywin  — MLP (32) на one-hot окне событий;
  gru     — GRU (H = 56) на one-hot событиях такта.

python stand_words.py --cond lm --weights pretrained --words c --N 32 --seed 1
python stand_words.py --cond lm ... --init-probe results_readout/probe_pretrained_c_s1.npz   # п. 3: BC + ES

--init-probe (ДОБАВЛЕНО 2026-10-02, п. 3): голова инициализируется линейным зондом знака, обученным
с учителем (readout.py): логит зонда w·(h − μ_з)/σ_з + b переводится в стандартизацию головы
(h − μ)/σ: веса w·σ/σ_з, смещение b + Σ w(μ − μ_з)/σ_з; скорость — смещение 2 (tanh ≈ 0,96).
"""
import argparse, json, math, os, time
import numpy as np
import torch

import env_flip as ef
from env_flip import FlipBodies, T
from stand_gpu import git_info

TEST_SEEDS = [999000 + k for k in range(100)]
ENV, REVERSAL = 'flip-2', True
BOS = 0
WORDS = {'c': ((329, 378, 5293), (598, 1066, 1072)),          # ' A' ' B' ' none' | ' up' ' down' ' same'
         'c_arb1': ((5798, 2829, 8805), (8281, 6951, 3369)),  # ' cat' ' table' ' stone' | ' river' ' chair' ' door'
         'c_arb2': ((18067, 9005, 20098), (29514, 10329, 3497))}  # ' lamp' ' cloud' ' brick' | ' pencil' ' garden' ' window' (как в A)


def events(s):
    """Индексы событий такта: вид (0 A, 1 B, 2 нет) и энергия (0 рост, 1 падение, 2 нет)."""
    kind = np.full(len(s), 2); kind[s[:, 10] > 0] = 0; kind[s[:, 11] > 0] = 1
    en = np.full(len(s), 2); en[s[:, 7] > 0.5] = 0; en[s[:, 7] < -0.5] = 1
    return kind, en


def onehot(s):
    kind, en = events(s)
    x = np.zeros((len(s), 6), np.float32)
    x[np.arange(len(s)), kind] = 1; x[np.arange(len(s)), 3 + en] = 1
    return x


def reflex(s, a):
    kind = (a[:, 0] <= 0).astype(int)
    st = ef._steer(s, 0); stB = ef._steer(s, 1); st[kind == 1] = stB[kind == 1]
    st[:, 1] = a[:, 1]
    return st


class Standardized:
    """Фиксированная стандартизация признаков по случайному поведению (20 эпизодов, все такты)."""

    def fit_std(self, rng):
        b = FlipBodies(20, [555000 + k for k in range(20)], REVERSAL, ENV)
        self.reset(20); H = []
        for _ in range(T):
            H.append(self.features(b.sense()))
            b.step(rng.uniform(-1, 1, (20, 2)))
        H = torch.cat(H)
        self.mu, self.sd = H.mean(0), H.std(0) + 1e-6

    def head(self, TH, h):
        B, D = h.shape
        h = (h - self.mu) / self.sd
        W = TH[:, :D * 2].view(B, D, 2)
        return torch.tanh(torch.einsum('bd,bdo->bo', h, W) + TH[:, D * 2:D * 2 + 2])


class LMWords(Standardized):
    def __init__(self, model, weights, words, N, device, net_seed, chunk=256):
        from transformers import AutoModel, AutoConfig
        pre_norm = None
        if weights == 'random':
            pre = AutoModel.from_pretrained(model, dtype=torch.float32)
            pre_norm = pre.get_input_embeddings().weight.norm(dim=1).mean().item(); del pre
            torch.manual_seed(net_seed)
            self.lm = AutoModel.from_config(AutoConfig.from_pretrained(model)).to(torch.float32)
            with torch.no_grad():
                E = self.lm.get_input_embeddings().weight
                E.mul_(pre_norm / E.norm(dim=1).mean())
        else:
            self.lm = AutoModel.from_pretrained(model, dtype=torch.float32)
        self.lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.E = self.lm.get_input_embeddings().weight
        self.D = self.lm.config.hidden_size
        self.N, self.device, self.chunk = N, device, chunk
        (tA, tB, tN), (tU, tD, tS) = WORDS[words]
        self.kind_tok = torch.tensor([tA, tB, tN], device=device)
        self.en_tok = torch.tensor([tU, tD, tS], device=device)
        self.n = self.D * 2 + 2

    def reset(self, B):
        self.buf = []

    def features(self, s):
        kind, en = events(s)
        tok = torch.stack([self.kind_tok[torch.as_tensor(kind, device=self.device)],
                           self.en_tok[torch.as_tensor(en, device=self.device)]], 1)       # (B, 2)
        self.buf.append(tok); self.buf = self.buf[-self.N:]
        ids = torch.stack(self.buf, 1).reshape(len(s), -1)                                  # (B, 2n)
        ids = torch.cat([torch.full((len(s), 1), BOS, device=self.device, dtype=torch.long), ids], 1)
        out = []
        for i in range(0, len(s), self.chunk):
            with torch.no_grad():
                h = self.lm(inputs_embeds=self.E[ids[i:i + self.chunk]]).last_hidden_state[:, -1]
            if not torch.isfinite(h).all():
                raise FloatingPointError('нечисловое скрытое состояние')
            out.append(h.float())
        return torch.cat(out)

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:self.D * 2] = rng.randn(self.D * 2) * 0.01
        self.fit_std(rng)
        return th

    def act(self, TH, s):
        return self.head(TH, self.features(s))


class QuadWords(Standardized):
    def __init__(self, N, width, device, seed):
        self.N, self.D, self.device = N, width, device
        r = np.random.RandomState(2000 + seed)
        I = N * 6
        self.A = torch.as_tensor(r.randn(I, width).astype(np.float32) / math.sqrt(I), device=device)
        self.Bm = torch.as_tensor(r.randn(I, width).astype(np.float32) / math.sqrt(I), device=device)
        self.n = width * 2 + 2

    def reset(self, B):
        self.buf = torch.zeros(B, self.N, 6, device=self.device)

    def features(self, s):
        self.buf = torch.cat([self.buf[:, 1:], torch.as_tensor(onehot(s), device=self.device)[:, None]], 1)
        x = self.buf.reshape(len(s), -1)
        x = (x - 1.0 / 3) * 2      # центрирование one-hot (доля единиц в блоке из трёх — 1/3)
        return (x @ self.A) * (x @ self.Bm)

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:self.D * 2] = rng.randn(self.D * 2) * 0.01
        self.fit_std(rng)
        return th

    def act(self, TH, s):
        return self.head(TH, self.features(s))


class FlyWords:
    def __init__(self, N, H, device):
        self.N, self.H, self.device = N, H, device
        self.I = N * 6
        self.n = self.I * H + H + H * 2 + 2

    def reset(self, B):
        self.buf = torch.zeros(B, self.N, 6, device=self.device)

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:self.I * self.H] = rng.randn(self.I * self.H) / math.sqrt(self.I)
        o = self.I * self.H + self.H
        th[o:o + self.H * 2] = rng.randn(self.H * 2) * 0.1
        return th

    def act(self, TH, s):
        self.buf = torch.cat([self.buf[:, 1:], torch.as_tensor(onehot(s), device=self.device)[:, None]], 1)
        B, H, I = len(s), self.H, self.I
        x = self.buf.reshape(B, I); i = 0
        W1 = TH[:, i:i + I * H].view(B, I, H); i += I * H
        b1 = TH[:, i:i + H]; i += H
        W2 = TH[:, i:i + H * 2].view(B, H, 2); i += H * 2
        z = torch.tanh(torch.einsum('bi,bih->bh', x, W1) + b1)
        return torch.tanh(torch.einsum('bh,bho->bo', z, W2) + TH[:, i:i + 2])


class GRUWords:
    def __init__(self, H, device):
        self.H, self.device = H, device
        self.g = 3 * ((6 + H) * H + H)
        self.n = self.g + H * 2 + 2

    def reset(self, B):
        self.h = torch.zeros(B, self.H, device=self.device)

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:self.g] = rng.randn(self.g) / math.sqrt(6 + self.H)
        th[self.g:self.g + self.H * 2] = rng.randn(self.H * 2) * 0.1
        return th

    def act(self, TH, s):
        x0 = torch.as_tensor(onehot(s), device=self.device)
        B, H, I = len(s), self.H, 6 + self.H
        blk = I * H + H
        def gate(k, x):
            o = k * blk
            return torch.einsum('bi,bih->bh', x, TH[:, o:o + I * H].view(B, I, H)) + TH[:, o + I * H:o + blk]
        x = torch.cat([x0, self.h], 1)
        z, r = torch.sigmoid(gate(0, x)), torch.sigmoid(gate(1, x))
        c = torch.tanh(gate(2, torch.cat([x0, r * self.h], 1)))
        self.h = (1 - z) * self.h + z * c
        Wo = TH[:, self.g:self.g + H * 2].view(B, H, 2)
        return torch.tanh(torch.einsum('bh,bho->bo', self.h, Wo) + TH[:, self.g + H * 2:])


def rollout(pol, TH, seeds, device, stats=False):
    P, E = TH.shape[0], len(seeds)
    THb = torch.as_tensor(np.repeat(TH, E, axis=0).astype(np.float32), device=device)
    b = FlipBodies(P * E, list(seeds) * P, REVERSAL, ENV)
    pol.reset(P * E)
    log = []
    for _ in range(T):
        if not b.alive.any():
            break
        s = b.sense()
        a = pol.act(THb, s).cpu().numpy()
        if stats:   # (выбран A, A сейчас пища, жив, кадр-событие, после разворота)
            log.append(np.stack([a[:, 0] > 0, b.good_A.copy(), b.alive.copy(), s[:, 10:12].sum(1) > 0, b.t_rev < b.t], 1))
        a = reflex(s, a)
        a[~b.alive] = 0
        b.step(a)
    if stats:
        b.choice_log = np.stack(log, 1)
        return b.fitness().reshape(P, E), b
    return b.fitness().reshape(P, E).mean(1)


def choice_stats(log):
    chA, goodA, alive, ev, after_rev = (log[:, :, i].astype(bool) for i in range(5))
    B, T_ = chA.shape
    correct = chA == goodA
    informed = np.zeros_like(alive); k = np.full((B, T_), 99)
    for i in range(B):
        last = -1; rev_seen = False; last_post = -1
        for t in range(T_):
            if after_rev[i, t] and not rev_seen:
                rev_seen = True; last_post = -1
            if ev[i, t]:
                last = t; last_post = t
            informed[i, t] = alive[i, t] and (last_post >= 0 if rev_seen else last >= 0)
            k[i, t] = t - last if last >= 0 else 99
    m = lambda mask: float(correct[mask].mean()) if mask.any() else None
    return {'all': m(alive), 'informed': m(informed), 'first': float(correct[:, 0].mean()),
            'by_k': [m(informed & (k == kk)) for kk in range(11)], 'frac_choose_A': float(chA[alive].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=['lm', 'quad', 'flywin', 'gru'], required=True)
    ap.add_argument('--model', default='EleutherAI/pythia-160m')
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--words', choices=sorted(WORDS), default='c')
    ap.add_argument('--N', type=int, default=32)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gens', type=int, default=40)
    ap.add_argument('--pairs', type=int, default=16)
    ap.add_argument('--eps', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_reflex')
    ap.add_argument('--init-probe', default=None, help='npz зонда из readout.py: голова стартует с него')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cond == 'lm':
        tag = f"{a.model.split('/')[-1]}_{a.weights}_{a.words}_N{a.N}"
        pol = LMWords(a.model, a.weights, a.words, a.N, a.device, a.seed)
    elif a.cond == 'quad':
        tag = f'quad768_N{a.N}'; pol = QuadWords(a.N, 768, a.device, a.seed)
    elif a.cond == 'flywin':
        tag = f'flywin_N{a.N}'; pol = FlyWords(a.N, 32, a.device)
    else:
        tag = 'gru_H56'; pol = GRUWords(56, a.device)
    name = f'{tag}_s{a.seed}'
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: прогон с незакоммиченным кодом стенда (HEAD {commit[:8]})', flush=True)
    rng = np.random.RandomState(a.seed)
    th = pol.init(rng)
    if a.init_probe:
        z = np.load(a.init_probe)
        mu, sd = pol.mu.cpu().numpy(), pol.sd.cpu().numpy()
        w, bias = z['w'][:-1], float(z['w'][-1])
        D = pol.D
        W = th[:D * 2].reshape(D, 2).copy()
        W[:, 0] = w * sd / z['sd']; W[:, 1] = 0
        th[:D * 2] = W.reshape(-1)
        th[D * 2] = bias + float((w * (mu - z['mu']) / z['sd']).sum())
        th[D * 2 + 1] = 2.0
        tag += '_bc'; name = f'{tag}_s{a.seed}'
    m = np.zeros_like(th); v = np.zeros_like(th)
    hist, t0 = [], time.time()
    with open(os.path.join(a.out, f'log_{name}.jsonl'), 'w') as log:
        for g in range(a.gens):
            env_seeds = [a.seed * 100000 + g * 10 + k for k in range(a.eps)]
            noise = rng.randn(a.pairs, pol.n).astype(np.float32)
            TH = np.concatenate([th + a.sigma * noise, th - a.sigma * noise])
            f = rollout(pol, TH, env_seeds, a.device)
            ranks = f.argsort().argsort() / (len(f) - 1) - 0.5
            grad = ((ranks[:a.pairs] - ranks[a.pairs:])[:, None] * noise).sum(0) / (2 * a.pairs * a.sigma)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            mh, vh = m / (1 - 0.9 ** (g + 1)), v / (1 - 0.999 ** (g + 1))
            th = (th + a.lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
            hist.append(float(f.mean()))
            rec = {'gen': g, 'mean': round(float(f.mean()), 4), 'max': round(float(f.max()), 4), 'sec': round(time.time() - t0)}
            print(json.dumps(rec), file=log, flush=True)
            print(name, rec, flush=True)
    fe, b = rollout(pol, th[None], TEST_SEEDS, a.device, stats=True)
    fe = fe[0]
    choice = choice_stats(b.choice_log)
    np.save(os.path.join(a.out, f'theta_{name}.npy'), th)
    np.savez_compressed(os.path.join(a.out, f'choice_{name}.npz'), log=b.choice_log.astype(np.bool_))
    with open(os.path.join(a.out, f'result_{name}.json'), 'w') as fh:
        json.dump({'tag': tag, 'cond': a.cond, 'seed': a.seed, 'model': a.model if a.cond == 'lm' else None,
                   'weights': a.weights if a.cond == 'lm' else None, 'words': a.words if a.cond == 'lm' else None,
                   'N': a.N if a.cond != 'gru' else None, 'env_version': ENV, 'reversal': REVERSAL, 'reflex': True,
                   'hist': hist, 'test': float(fe.mean()), 'test_per_episode': [round(float(x), 4) for x in fe],
                   'test_food': float(b.n_good.mean()), 'test_poison': float(b.n_bad.mean()), 'choice': choice,
                   'sec': round(time.time() - t0), 'n_params': int(pol.n), 'dtype': 'torch.float32',
                   'init_probe': a.init_probe, 'git_commit': commit, 'git_dirty': dirty, 'args': vars(a)}, fh)
    print(name, 'test', round(float(fe.mean()), 4), f'(пища {b.n_good.mean():.2f}, яд {b.n_bad.mean():.2f}, '
          f'выбор верен на информированных тактах {choice["informed"]})', 'за', round(time.time() - t0), 'с')


if __name__ == '__main__':
    main()
