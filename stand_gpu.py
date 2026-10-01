"""Стенд: замороженная LM в минимальном теле (версия под GPU).

Тело: 2D-арена, энергия дрейфует вниз без входа, пища восстанавливает.
Орган чувств (зашит): направление/дистанция до пищи, тренды, порог энергии, стена.
Проектор (эволюционирует): 8 признаков -> K векторов в пространстве эмбеддингов LM,
  нормированных на средний норм токена.
LM (заморожена): любая causal-модель из transformers с поддержкой inputs_embeds.
Стандартизация скрытого состояния (--center, по умолчанию включена): статистика
  снимается один раз на реальных ощущениях при случайном поведении и не эволюционирует.
Голова действия (эволюционирует): линейная со скрытого состояния последней позиции.
Отбор: OpenAI-ES, антитетические пары, ранговая нормировка, Adam.

Контроли: --cond fly (MLP без LM), --cond lin (линейная политика прямо на 8 признаках,
  18 параметров), --weights random (та же LM со случайными весами; сеть сеется от --net-seed,
  по умолчанию = --seed, чтобы разброс между случайными сетями попадал в выборку).

ИЗМЕНЕНО 2026-10-01 (перед GPU-серией):
- тип по умолчанию float32 и на CUDA (было fp16): Qwen2.5 в fp16 переполняется, а NaN
  проходит tanh -> clip -> тело тихо умирает без исключения. Теперь _hidden падает на
  первом нечисловом значении. bf16/fp16 — только явно через --dtype;
- случайная сеть сеялась константой 12345 для всех посевов (одна сеть на всю серию);
- добавлен контроль lin: на CPU 5 посевов дают тест 1,333–1,443 (ср. 1,395) при жадном 1,46,
  т.е. задачу почти целиком решает линейное чтение признаков органа;
- в результат пишется n_params: Adam шагает ~lr по каждой координате, поэтому сдвиг выхода
  за поколение растёт с числом параметров головы, и «поколение до 1,2» отражает размерность,
  а не только представление (выведено, не проверено).

Примеры:
  python stand_gpu.py --cond fly --seed 1
  python stand_gpu.py --cond lm --model EleutherAI/pythia-70m --weights pretrained --seed 1
  python stand_gpu.py --cond lm --model Qwen/Qwen2.5-1.5B --weights random --seed 1 --chunk 128
"""
import argparse, json, math, os, time
import numpy as np
import torch

# ---------------- тело ----------------
T = 100
N_FOOD = 3
EAT_R = 0.07
FOOD_GAIN = 0.35
DRIFT = 0.012
E0 = 0.6
VMAX = 0.03
TURN = 0.6
S_DIM = 8


class Bodies:
    def __init__(self, B, seeds):
        self.B = B
        self.rs = [np.random.RandomState(s) for s in seeds]
        self.pos = np.stack([r.rand(2) for r in self.rs])
        self.head = np.array([r.rand() * 2 * np.pi for r in self.rs])
        self.food = np.stack([r.rand(N_FOOD, 2) for r in self.rs])
        self.E = np.full(B, E0)
        self.alive = np.ones(B, bool)
        self.t_alive = np.zeros(B)
        self.prev_d = None
        self.prev_E = self.E.copy()

    def _nearest(self):
        d = np.linalg.norm(self.food - self.pos[:, None, :], axis=2)
        i = d.argmin(1)
        vec = self.food[np.arange(self.B), i] - self.pos
        return vec, d[np.arange(self.B), i], i

    def sense(self):
        vec, d, _ = self._nearest()
        ang = np.arctan2(vec[:, 1], vec[:, 0]) - self.head
        if self.prev_d is None:
            self.prev_d = d.copy()
        dtrend = (self.prev_d - d) * 10
        etrend = (self.E - self.prev_E) * 10
        hx, hy = np.cos(self.head), np.sin(self.head)
        tx = np.where(hx > 0, (1 - self.pos[:, 0]) / np.maximum(hx, 1e-6), self.pos[:, 0] / np.maximum(-hx, 1e-6))
        ty = np.where(hy > 0, (1 - self.pos[:, 1]) / np.maximum(hy, 1e-6), self.pos[:, 1] / np.maximum(-hy, 1e-6))
        wall = np.clip(np.minimum(tx, ty), 0, 0.2) / 0.2
        s = np.stack([np.cos(ang), np.sin(ang), d * 2, dtrend, self.E, etrend,
                      (self.E < 0.3).astype(float), 1 - wall], 1)
        self.prev_d = d
        self.prev_E = self.E.copy()
        return s.astype(np.float32)

    def step(self, act):
        a = np.clip(act, -1, 1)
        self.head = self.head + a[:, 0] * TURN
        v = (a[:, 1] + 1) / 2 * VMAX
        self.pos = np.clip(self.pos + v[:, None] * np.stack([np.cos(self.head), np.sin(self.head)], 1), 0, 1)
        self.E = self.E - DRIFT
        _, d, i = self._nearest()
        eat = (d < EAT_R) & self.alive
        self.E = np.where(eat, np.minimum(self.E + FOOD_GAIN, 1.0), self.E)
        for b in np.where(eat)[0]:
            self.food[b, i[b]] = self.rs[b].rand(2)
        self.alive &= self.E > 0
        self.t_alive += self.alive

    def fitness(self):
        return self.t_alive / T + 0.5 * np.where(self.alive, self.E, 0)


# ---------------- политики ----------------
class LMPolicy:
    def __init__(self, model, weights, center, device, dtype, chunk, layer, K=2, net_seed=12345):
        from transformers import AutoModel, AutoConfig
        self.device, self.dtype, self.chunk, self.layer = device, dtype, chunk, layer
        if weights == 'pretrained':
            self.lm = AutoModel.from_pretrained(model, dtype=dtype)
        else:
            cfg = AutoConfig.from_pretrained(model)
            torch.manual_seed(net_seed)
            self.lm = AutoModel.from_config(cfg).to(dtype)
        self.lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.D = self.lm.config.hidden_size
        self.K = K
        self.center = center
        self.mu = None
        emb = self.lm.get_input_embeddings().weight
        self.tok_norm = emb.float().norm(dim=1).mean().item()
        self.n_proj = S_DIM * self.D * K + self.D * K
        self.n = self.n_proj + self.D * 2 + 2

    def _hidden(self, v):
        outs = []
        for i in range(0, v.shape[0], self.chunk):
            x = v[i:i + self.chunk].to(self.dtype)
            with torch.no_grad():
                if self.layer == -1:
                    h = self.lm(inputs_embeds=x).last_hidden_state[:, -1]
                else:
                    h = self.lm(inputs_embeds=x, output_hidden_states=True).hidden_states[self.layer][:, -1]
            if not torch.isfinite(h).all():
                raise FloatingPointError(f'нечисловое скрытое состояние (dtype={self.dtype}): '
                                         f'{(~torch.isfinite(h)).sum().item()} из {h.numel()}')
            outs.append(h.float())
        return torch.cat(outs)

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:S_DIM * self.D * self.K] = rng.randn(S_DIM * self.D * self.K) / math.sqrt(S_DIM)
        th[self.n_proj:self.n_proj + self.D * 2] = rng.randn(self.D * 2) * 0.01
        if self.center:
            b = Bodies(20, [555000 + k for k in range(20)]); S = []
            for _ in range(25):
                S.append(b.sense()); b.step(rng.uniform(-1, 1, (20, 2)))
            S = torch.as_tensor(np.concatenate(S), device=self.device)
            TH = torch.as_tensor(np.repeat(th[None], len(S), 0), device=self.device)
            h = self._hidden(self.project(TH, S))
            self.mu, self.sd = h.mean(0), h.std(0) + 1e-6
        return th

    def project(self, TH, s):
        B = s.shape[0]
        W = TH[:, :S_DIM * self.D * self.K].view(B, S_DIM, self.D * self.K)
        b = TH[:, S_DIM * self.D * self.K:self.n_proj]
        v = (torch.einsum('bs,bsd->bd', s, W) + b).view(B, self.K, self.D)
        return v / (v.norm(dim=2, keepdim=True) + 1e-6) * self.tok_norm

    def act(self, TH, s):
        B = s.shape[0]
        h = self._hidden(self.project(TH, s))
        if self.center:
            h = (h - self.mu) / self.sd
        else:
            h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6)
        Wh = TH[:, self.n_proj:self.n_proj + self.D * 2].view(B, self.D, 2)
        bh = TH[:, self.n_proj + self.D * 2:]
        return torch.tanh(torch.einsum('bd,bdo->bo', h, Wh) + bh)


class LinPolicy:
    """Контроль: действие = tanh(W s + b) прямо на признаках органа."""

    def __init__(self, device):
        self.device = device
        self.n = S_DIM * 2 + 2

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:S_DIM * 2] = rng.randn(S_DIM * 2) * 0.1
        return th

    def act(self, TH, s):
        B = s.shape[0]
        W = TH[:, :S_DIM * 2].view(B, S_DIM, 2)
        return torch.tanh(torch.einsum('bs,bso->bo', s, W) + TH[:, S_DIM * 2:])


class FlyPolicy:
    H = 32

    def __init__(self, device):
        self.device = device
        self.n = S_DIM * self.H + self.H + self.H * 2 + 2

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:S_DIM * self.H] = rng.randn(S_DIM * self.H) / math.sqrt(S_DIM)
        th[S_DIM * self.H + self.H:S_DIM * self.H + self.H + self.H * 2] = rng.randn(self.H * 2) * 0.1
        return th

    def act(self, TH, s):
        B, H, i = s.shape[0], self.H, 0
        W1 = TH[:, i:i + S_DIM * H].view(B, S_DIM, H); i += S_DIM * H
        b1 = TH[:, i:i + H]; i += H
        W2 = TH[:, i:i + H * 2].view(B, H, 2); i += H * 2
        b2 = TH[:, i:i + 2]
        z = torch.tanh(torch.einsum('bs,bsh->bh', s, W1) + b1)
        return torch.tanh(torch.einsum('bh,bho->bo', z, W2) + b2)


def rollout(pol, TH, seeds, device):
    P, E = TH.shape[0], len(seeds)
    THb = torch.as_tensor(np.repeat(TH, E, axis=0).astype(np.float32), device=device)
    bodies = Bodies(P * E, list(seeds) * P)
    for _ in range(T):
        if not bodies.alive.any():
            break
        s = torch.as_tensor(bodies.sense(), device=device)
        a = pol.act(THb, s).cpu().numpy()
        a[~bodies.alive] = 0
        bodies.step(a)
    return bodies.fitness().reshape(P, E).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=['fly', 'lin', 'lm'], required=True)
    ap.add_argument('--model', default='EleutherAI/pythia-70m')
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--no-center', action='store_true', help='отключить стандартизацию скрытого состояния')
    ap.add_argument('--layer', type=int, default=-1, help='слой съёма состояния (-1 = последний)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--net-seed', type=int, default=None, help='посев случайной сети (по умолчанию = --seed; 12345 = как в первой серии)')
    ap.add_argument('--gens', type=int, default=40)
    ap.add_argument('--pairs', type=int, default=16)
    ap.add_argument('--eps', type=int, default=2)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default=None)
    ap.add_argument('--chunk', type=int, default=256, help='размер пакета прямого прохода LM')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()

    if a.device == 'cpu':
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    dtype = getattr(torch, a.dtype) if a.dtype else torch.float32
    net_seed = a.seed if a.net_seed is None else a.net_seed
    os.makedirs(a.out, exist_ok=True)
    tag = a.cond if a.cond != 'lm' else f"{a.model.split('/')[-1]}_{a.weights}{'' if a.no_center else '_c'}{'' if a.layer == -1 else f'_L{a.layer}'}"
    name = f'{tag}_s{a.seed}'

    rng = np.random.RandomState(a.seed)
    if a.cond == 'fly':
        pol = FlyPolicy(a.device)
    elif a.cond == 'lin':
        pol = LinPolicy(a.device)
    else:
        pol = LMPolicy(a.model, a.weights, not a.no_center, a.device, dtype, a.chunk, a.layer, net_seed=net_seed)
    th = pol.init(rng)
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
    test = float(rollout(pol, th[None], [999000 + k for k in range(20)], a.device)[0])
    np.save(os.path.join(a.out, f'theta_{name}.npy'), th)
    with open(os.path.join(a.out, f'result_{name}.json'), 'w') as fh:
        json.dump({'tag': tag, 'seed': a.seed, 'model': a.model, 'hist': hist, 'test': test,
                   'sec': round(time.time() - t0), 'n_params': int(pol.n), 'dtype': str(dtype),
                   'net_seed': net_seed if a.cond == 'lm' and a.weights == 'random' else None, 'args': vars(a)}, fh)
    print(name, 'test', round(test, 4), 'за', round(time.time() - t0), 'с')


if __name__ == '__main__':
    main()
