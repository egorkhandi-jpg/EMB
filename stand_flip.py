"""Стенд этапа 2: среда «смена знака» (env_flip.py), политики с памятью.

Зачем. Серия v1-gpu-fp32 (docs/RESULTS_2026-10-01_gpu.md): на исходной арене предобученная
Pythia-70M неотличима от случайной (p 0,20), а скрытое состояние LM имеет эффективную
размерность 4–6 из 512 при 3,5 у входа — два псевдотокена без контекста дают гладкое
отображение входа, знаниям не к чему приложиться. Здесь агенту надо вывести скрытую
переменную эпизода (какой вид — пища) из последовательности событий.

Условия (--cond), у всех один орган env_flip (12 признаков) и одна ES:
  lm       — проектор -> K псевдотокенов на такт, в LM подаются последние N тактов (K·N позиций),
             голова со скрытого состояния последней позиции; --weights pretrained|random;
  lm_shuf  — то же, но порядок ПРОШЛЫХ тактов в окне перемешивается на каждом шаге (пары внутри
             такта сохраняются), текущий такт остаётся последним. Текущий такт не перемешивается
             сознательно: иначе контроль ломал бы ещё и «где сейчас», и падение нельзя было бы
             приписать порядку истории;
  flywin   — MLP (32 скрытых) на конкатенации окна из N тактов (начало эпизода — нули);
  linwin   — линейная политика на том же окне;
  gru      — рекуррентная муха: GRU, память в состоянии, окна нет (--N не используется).
Окно пересчитывается целиком на каждом такте (без KV-кэша): для Pythia-70M это не узкое место.

Отбор: OpenAI-ES, антитетические пары, общие посевы среды на поколение, ранги, Adam.
Тест — 100 отложенных эпизодов (посевы 999000+k), общих для всех агентов; приспособленность
по каждому эпизоду пишется в результат (test_per_episode) — для попарных сравнений и формы
распределения.
Среда — --env (версии в env_flip.ENVS) и --reversal; версия пишется в результат.
ИСПРАВЛЕНО 2026-10-01: статистика стандартизации LM снималась на flip-1 без разворота
независимо от условия; теперь — на той же среде и с тем же разворотом, что у прогона.

Примеры:
  python stand_flip.py --cond flywin --N 32 --seed 1
  python stand_flip.py --cond lm --weights random --N 32 --seed 1
  python stand_flip.py --cond gru --seed 1
"""
import argparse, json, math, os, time
import numpy as np
import torch

import env_flip as ef
from env_flip import FlipBodies, S_DIM, T
from stand_gpu import git_info

TEST_SEEDS = [999000 + k for k in range(100)]


# ---------------- политики ----------------
class LMWindow:
    def __init__(self, model, weights, N, shuffle, device, dtype, chunk, net_seed, K=2,
                 env=ef.ENV_VERSION, reversal=False):
        self.env, self.reversal = env, reversal
        from transformers import AutoModel, AutoConfig
        self.device, self.dtype, self.chunk = device, dtype, chunk
        self.N, self.shuffle, self.K = N, shuffle, K
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
        self.tok_norm = self.lm.get_input_embeddings().weight.float().norm(dim=1).mean().item()
        self.n_proj = S_DIM * self.D * K + self.D * K
        self.n = self.n_proj + self.D * 2 + 2
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(net_seed + 777)
        self.mu = None

    def reset(self, B):
        self.buf = []

    def project(self, TH, s):
        B = s.shape[0]
        W = TH[:, :S_DIM * self.D * self.K].view(B, S_DIM, self.D * self.K)
        b = TH[:, S_DIM * self.D * self.K:self.n_proj]
        v = (torch.einsum('bs,bsd->bd', s, W) + b).view(B, self.K, self.D)
        return v / (v.norm(dim=2, keepdim=True) + 1e-6) * self.tok_norm

    def hidden(self, TH, s):
        """Сдвинуть окно и вернуть сырое скрытое состояние последней позиции."""
        self.buf.append(self.project(TH, s))
        self.buf = self.buf[-self.N:]
        fr = torch.stack(self.buf, 1)                                  # (B, n, K, D)
        B, n = fr.shape[:2]
        if self.shuffle and n > 2:
            perm = torch.rand(B, n - 1, generator=self.gen, device=self.device).argsort(1)
            past = fr[torch.arange(B, device=self.device)[:, None], perm]
            fr = torch.cat([past, fr[:, -1:]], 1)
        seq = fr.reshape(B, n * self.K, self.D)
        outs = []
        for i in range(0, B, self.chunk):
            with torch.no_grad():
                h = self.lm(inputs_embeds=seq[i:i + self.chunk].to(self.dtype)).last_hidden_state[:, -1]
            if not torch.isfinite(h).all():
                raise FloatingPointError(f'нечисловое скрытое состояние (dtype={self.dtype})')
            outs.append(h.float())
        return torch.cat(outs)

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:S_DIM * self.D * self.K] = rng.randn(S_DIM * self.D * self.K) / math.sqrt(S_DIM)
        th[self.n_proj:self.n_proj + self.D * 2] = rng.randn(self.D * 2) * 0.01
        # Фиксированная стандартизация: статистика по всем тактам 20 эпизодов при случайном
        # поведении, с тем же окном и тем же перемешиванием, что у условия.
        b = FlipBodies(20, [555000 + k for k in range(20)], self.reversal, self.env)
        TH = torch.as_tensor(np.repeat(th[None], 20, 0), device=self.device)
        self.reset(20); H = []
        for _ in range(T):
            H.append(self.hidden(TH, torch.as_tensor(b.sense(), device=self.device)))
            b.step(rng.uniform(-1, 1, (20, 2)))
        H = torch.cat(H)
        self.mu, self.sd = H.mean(0), H.std(0) + 1e-6
        return th

    def act(self, TH, s):
        B = s.shape[0]
        h = (self.hidden(TH, s) - self.mu) / self.sd
        self.last_h = h
        Wh = TH[:, self.n_proj:self.n_proj + self.D * 2].view(B, self.D, 2)
        return torch.tanh(torch.einsum('bd,bdo->bo', h, Wh) + TH[:, self.n_proj + self.D * 2:])


class WindowNet:
    """Окно из N тактов сырых признаков; H=0 — линейная политика, иначе MLP с H скрытыми."""

    def __init__(self, N, H, device):
        self.N, self.H, self.device = N, H, device
        self.I = N * S_DIM
        self.n = self.I * 2 + 2 if H == 0 else self.I * H + H + H * 2 + 2

    def reset(self, B):
        self.buf = torch.zeros(B, self.N, S_DIM, device=self.device)

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        if self.H == 0:
            th[:self.I * 2] = rng.randn(self.I * 2) * 0.1 / math.sqrt(self.N)
        else:
            th[:self.I * self.H] = rng.randn(self.I * self.H) / math.sqrt(self.I)
            o = self.I * self.H + self.H
            th[o:o + self.H * 2] = rng.randn(self.H * 2) * 0.1
        return th

    def act(self, TH, s):
        self.buf = torch.cat([self.buf[:, 1:], s[:, None]], 1)
        x = self.buf.reshape(s.shape[0], self.I)
        B, H, I = s.shape[0], self.H, self.I
        if H == 0:
            return torch.tanh(torch.einsum('bi,bio->bo', x, TH[:, :I * 2].view(B, I, 2)) + TH[:, I * 2:])
        i = 0
        W1 = TH[:, i:i + I * H].view(B, I, H); i += I * H
        b1 = TH[:, i:i + H]; i += H
        W2 = TH[:, i:i + H * 2].view(B, H, 2); i += H * 2
        z = torch.tanh(torch.einsum('bi,bih->bh', x, W1) + b1)
        self.last_h = z
        return torch.tanh(torch.einsum('bh,bho->bo', z, W2) + TH[:, i:i + 2])


class GRUFly:
    """Рекуррентная муха. H=56 -> 11 706 параметров, сопоставимо с flywin при N=32 (12 386)."""

    def __init__(self, H, device):
        self.H, self.device = H, device
        self.g = 3 * ((S_DIM + H) * H + H)
        self.n = self.g + H * 2 + 2

    def reset(self, B):
        self.h = torch.zeros(B, self.H, device=self.device)

    def init(self, rng):
        th = np.zeros(self.n, np.float32)
        th[:self.g] = rng.randn(self.g) / math.sqrt(S_DIM + self.H)
        th[self.g:self.g + self.H * 2] = rng.randn(self.H * 2) * 0.1
        return th

    def act(self, TH, s):
        B, H, I = s.shape[0], self.H, S_DIM + self.H
        blk = I * H + H
        def gate(k, x):
            o = k * blk
            return torch.einsum('bi,bih->bh', x, TH[:, o:o + I * H].view(B, I, H)) + TH[:, o + I * H:o + blk]
        x = torch.cat([s, self.h], 1)
        z, r = torch.sigmoid(gate(0, x)), torch.sigmoid(gate(1, x))
        c = torch.tanh(gate(2, torch.cat([s, r * self.h], 1)))
        self.h = (1 - z) * self.h + z * c
        self.last_h = self.h
        Wo = TH[:, self.g:self.g + H * 2].view(B, H, 2)
        return torch.tanh(torch.einsum('bh,bho->bo', self.h, Wo) + TH[:, self.g + H * 2:])


def rollout(pol, TH, seeds, device, reversal, env, stats=False):
    P, E = TH.shape[0], len(seeds)
    THb = torch.as_tensor(np.repeat(TH, E, axis=0).astype(np.float32), device=device)
    bodies = FlipBodies(P * E, list(seeds) * P, reversal, env)
    pol.reset(P * E)
    for _ in range(T):
        if not bodies.alive.any():
            break
        s = torch.as_tensor(bodies.sense(), device=device)
        a = pol.act(THb, s).cpu().numpy()
        a[~bodies.alive] = 0
        bodies.step(a)
    if stats:
        return bodies.fitness().reshape(P, E), bodies
    return bodies.fitness().reshape(P, E).mean(1)


def make_policy(a, dtype, net_seed):
    if a.cond in ('lm', 'lm_shuf'):
        return LMWindow(a.model, a.weights, a.N, a.cond == 'lm_shuf', a.device, dtype, a.chunk, net_seed,
                        env=a.env, reversal=a.reversal)
    if a.cond == 'flywin':
        return WindowNet(a.N, 32, a.device)
    if a.cond == 'linwin':
        return WindowNet(a.N, 0, a.device)
    return GRUFly(a.H, a.device)


def make_tag(a):
    if a.cond in ('lm', 'lm_shuf'):
        return f"{a.model.split('/')[-1]}_{a.weights}{'_shuf' if a.cond == 'lm_shuf' else ''}_N{a.N}"
    if a.cond == 'gru':
        return f'gru_H{a.H}'
    return f'{a.cond}_N{a.N}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=['lm', 'lm_shuf', 'flywin', 'linwin', 'gru'], required=True)
    ap.add_argument('--model', default='EleutherAI/pythia-70m')
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--N', type=int, default=32, help='длина окна в тактах')
    ap.add_argument('--H', type=int, default=56, help='размер состояния GRU')
    ap.add_argument('--reversal', action='store_true', help='знак меняется один раз внутри эпизода')
    ap.add_argument('--env', choices=sorted(ef.ENVS), default=ef.ENV_VERSION, help='версия среды (сила яда)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--net-seed', type=int, default=None)
    ap.add_argument('--gens', type=int, default=40)
    ap.add_argument('--pairs', type=int, default=16)
    ap.add_argument('--eps', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--out', default='results_flip')
    a = ap.parse_args()

    if a.device == 'cpu':
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    dtype = getattr(torch, a.dtype)
    net_seed = a.seed if a.net_seed is None else a.net_seed
    os.makedirs(a.out, exist_ok=True)
    tag = make_tag(a) + ('_rev' if a.reversal else '') + ('' if a.env == 'flip-1' else '_' + a.env.replace('-', ''))
    name = f'{tag}_s{a.seed}'
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: прогон с незакоммиченным кодом стенда (HEAD {commit[:8]})', flush=True)

    rng = np.random.RandomState(a.seed)
    pol = make_policy(a, dtype, net_seed)
    th = pol.init(rng)
    m = np.zeros_like(th); v = np.zeros_like(th)
    hist, t0 = [], time.time()
    with open(os.path.join(a.out, f'log_{name}.jsonl'), 'w') as log:
        for g in range(a.gens):
            env_seeds = [a.seed * 100000 + g * 10 + k for k in range(a.eps)]
            noise = rng.randn(a.pairs, pol.n).astype(np.float32)
            TH = np.concatenate([th + a.sigma * noise, th - a.sigma * noise])
            f = rollout(pol, TH, env_seeds, a.device, a.reversal, a.env)
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
    fe, b = rollout(pol, th[None], TEST_SEEDS, a.device, a.reversal, a.env, stats=True)
    fe = fe[0]
    np.save(os.path.join(a.out, f'theta_{name}.npy'), th)
    with open(os.path.join(a.out, f'result_{name}.json'), 'w') as fh:
        json.dump({'tag': tag, 'cond': a.cond, 'seed': a.seed, 'model': a.model if a.cond.startswith('lm') else None,
                   'N': a.N if a.cond != 'gru' else None, 'reversal': a.reversal, 'env_version': a.env,
                   'hist': hist, 'test': float(fe.mean()), 'test_per_episode': [round(float(x), 4) for x in fe], 'test_food': float(b.n_good.mean()), 'test_poison': float(b.n_bad.mean()),
                   'sec': round(time.time() - t0), 'n_params': int(pol.n), 'dtype': str(dtype),
                   'net_seed': net_seed if a.cond.startswith('lm') and a.weights == 'random' else None,
                   'git_commit': commit, 'git_dirty': dirty, 'args': vars(a)}, fh)
    print(name, 'test', round(float(fe.mean()), 4), f'(пища {b.n_good.mean():.2f}, яд {b.n_bad.mean():.2f})',
          'за', round(time.time() - t0), 'с')


if __name__ == '__main__':
    main()
