"""Эволюция параметров проигрывания на записанном опыте — п. 1A (решение Егора 2026-10-04 после v3-replay).
Пререгистрация — docs/RESULTS_2026-10-01_flip.md, раздел «Эволюция правила проигрывания и устойчивость A15».

Опыт: жизни агента lm pretrained c с проигрыванием (параметры сетки A15), записанные `body_head3.py record`
(100 эпизодов на посев): жизнь 1 — эпизоды 0–49 (обучение), жизнь 2 — эпизоды 50–99 того же посева (оценка).
Особь — правило проигрывания u ∈ R⁴ → (η, B, K, метки):
    η = 2·σ(u₀) ∈ (0, 2) — шаг медленных (нормированный шаг по мини-пакету устойчив при 0 < η < 2, правило 12);
    B = clip(round(e^{u₁}), 1, 50) — буфер в эпизодах (50 — вся жизнь 1);
    K = clip(round(e^{u₂}), 1, 10) — проходов на паузу;
    метки = all, если u₃ ≥ 0, иначе event.
Медленные веса учатся на жизни 1 только в паузах после эпизодов (проигрывание, мини-пакеты по 32, логистическая потеря,
нормированный шаг), как в A15. Шаг быстрой части на записанном опыте не влияет на медленные веса и не отбирается.
Приспособленность — только наблюдаемое агентом: точность предсказания меток тела «последний исход» (такты, где метка
есть) замороженными медленными весами на жизни 2 того же посева; среднее по посевам отбора. Не истинный знак.
ES: простой шаг θ += lr·g (без Adam), антитетические пары, ранговая нормировка. Случайный поиск — то же число оценок.

python evo_replay.py --sel 201 ... 216 --test 301 ... 310
"""
import argparse, json, math, os, time
import numpy as np
import torch

from stand_gpu import git_info

MB, BMAX, KMAX, E1 = 32, 50, 10, 50


def decode(u):
    u = np.asarray(u, float)
    return {'eta': float(2 / (1 + math.exp(-u[0]))), 'B': int(np.clip(round(math.exp(u[1])), 1, BMAX)),
            'K': int(np.clip(round(math.exp(u[2])), 1, KMAX)), 'rule': 'all' if u[3] >= 0 else 'event'}


class Data:
    """Записанный опыт посевов на видеокарте: признаки жизни 1 (плоско), записи для правил all / event, жизнь 2."""

    def __init__(self, paths, device):
        Zs, feat, y, ep, blk, Zev, Yev, Mev = [], [], [], [], {}, [], [], []
        off_f = off_e = 0
        for si, p in enumerate(paths):
            r = np.load(p)
            z = r['z'][:E1]; T = z.shape[1]; d = z.shape[2]
            Zs.append(torch.as_tensor(z.reshape(-1, d)))
            for rule in ('all', 'event'):
                if rule == 'all':
                    m = r['m_all'][:E1] > 0; e_idx, t_idx = np.nonzero(m); fi = e_idx * T + t_idx; yy = r['bel'][:E1][m]
                else:
                    m = r['m_ev'][:E1] > 0; e_idx, t_idx = np.nonzero(m); fi = e_idx * T + t_idx - 1; yy = r['yA'][:E1][m]
                o = np.argsort(e_idx, kind='stable'); e_idx, fi, yy = e_idx[o], fi[o], yy[o]
                starts = np.searchsorted(e_idx, np.arange(E1 + 1))             # записи эпизода e: [starts[e], starts[e+1])
                blk[(si, rule)] = off_e + starts
                feat.append(fi + off_f); y.append(yy); off_e += len(fi)
            off_f += z.shape[0] * T
            Zev.append(torch.as_tensor(r['z'][E1:2 * E1].reshape(-1, d))); Yev.append(torch.as_tensor(r['bel'][E1:2 * E1].reshape(-1)))
            Mev.append(torch.as_tensor(r['m_all'][E1:2 * E1].reshape(-1)))
        self.Z = torch.cat(Zs).to(device); self.N2 = (self.Z * self.Z).sum(1) + 1
        self.feat = torch.as_tensor(np.concatenate(feat), device=device); self.y = torch.as_tensor(np.concatenate(y), dtype=torch.float32, device=device)
        self.blk = blk; self.S = len(paths); self.d = self.Z.shape[1]; self.device = device
        self.Zev = torch.stack(Zev).to(device); self.Yev = torch.stack(Yev).to(device); self.Mev = torch.stack(Mev).to(device) > 0


def fitness(data, params, gen_seed):
    """params — список словарей особей; возвращает точность (особи × посевы) на жизни 2."""
    M, S, dev = len(params), data.S, data.device
    H = M * S
    mem = np.repeat(np.arange(M), S); sd = np.tile(np.arange(S), M)
    eta = torch.as_tensor([params[i]['eta'] for i in mem], dtype=torch.float32, device=dev)
    K = torch.as_tensor([params[i]['K'] for i in mem], device=dev)
    w = torch.zeros(H, data.d, device=dev); b = torch.zeros(H, device=dev)
    g = torch.Generator(device=dev).manual_seed(gen_seed)
    sd_t = torch.as_tensor(sd, device=dev)
    for e in range(E1):
        lo = np.array([data.blk[(sd[h], params[mem[h]]['rule'])][max(0, e - params[mem[h]]['B'] + 1)] for h in range(H)])
        hi = np.array([data.blk[(sd[h], params[mem[h]]['rule'])][e + 1] for h in range(H)])
        n = torch.as_tensor(hi - lo, device=dev); lo_t = torch.as_tensor(lo, device=dev)
        L = int(math.ceil(max(int(n.max()), 1) / MB) * MB)
        ar = torch.arange(L, device=dev)
        for k in range(int(K.max())):
            keys = torch.rand(S, L, generator=g, device=dev)[sd_t]                # общие случайные числа у особей одного посева
            keys = torch.where(ar[None] < n[:, None], keys, torch.full_like(keys, 2.0))
            pos = keys.argsort(1)
            valid = (ar[None] < n[:, None]) & (K[:, None] > k)
            ids = (lo_t[:, None] + pos).clamp_max(len(data.feat) - 1)
            for j in range(0, L, MB):
                v = valid[:, j:j + MB].float()
                if not v.any():
                    break
                ent = ids[:, j:j + MB]; fr = data.feat[ent]
                z = data.Z[fr]; yy = data.y[ent]
                err = (yy - torch.sigmoid((z * w[:, None]).sum(-1) + b[:, None])) * v
                norm = (v * data.N2[fr]).sum(1).clamp_min(1.0)
                w += (eta / norm)[:, None] * (err[:, :, None] * z).sum(1)
                b += eta * err.sum(1) / norm
    acc = torch.zeros(M, S, device=dev)
    for s in range(S):
        hs = np.nonzero(sd == s)[0]
        pred = (data.Zev[s] @ w[hs].T + b[hs]) >= 0                                  # (такты, особи)
        m = data.Mev[s]
        acc[:, s] = (pred[m] == (data.Yev[s][m, None] > 0.5)).float().mean(0)
    return acc.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sel', type=int, nargs='+', default=list(range(201, 217)))
    ap.add_argument('--test', type=int, nargs='+', default=list(range(301, 311)))
    ap.add_argument('--tag', default='pretrained_c')
    ap.add_argument('--rec', default='results_evo/record')
    ap.add_argument('--pairs', type=int, default=16)
    ap.add_argument('--gens', type=int, default=80)
    ap.add_argument('--sigma', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=0.3)
    ap.add_argument('--u0', type=float, nargs=4, default=[math.log(0.1 / 1.9), math.log(3), 0.0, 0.0])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_evo')
    a = ap.parse_args()
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    rec = lambda s: os.path.join(a.rec, f'rec_{a.tag}_s{s}.npz')
    t0 = time.time()
    sel = Data([rec(s) for s in a.sel], a.device)
    print(f'отбор: {sel.S} посевов, записей {len(sel.feat)}, загрузка {time.time() - t0:.0f} с', flush=True)
    rng = np.random.RandomState(a.seed)
    th = np.array(a.u0, float); hist = []
    for gnum in range(a.gens):
        eps = rng.randn(a.pairs, 4)
        U = np.concatenate([th + a.sigma * eps, th - a.sigma * eps])
        f = fitness(sel, [decode(u) for u in U], 1000 + gnum).mean(1)
        ranks = f.argsort().argsort() / (len(f) - 1) - 0.5
        grad = ((ranks[:a.pairs] - ranks[a.pairs:])[:, None] * eps).sum(0) / (2 * a.pairs * a.sigma)
        th = th + a.lr * grad
        hist.append({'gen': gnum, 'center': th.tolist(), 'center_dec': decode(th), 'mean_f': float(f.mean()), 'max_f': float(f.max()),
                     'best_dec': decode(U[f.argmax()])})
        print(f"{time.strftime('%H:%M:%S')} пок. {gnum}: средняя {f.mean():.4f} макс {f.max():.4f} | центр {decode(th)} | {time.time() - t0:.0f} с", flush=True)
    # случайный поиск: то же число оценок
    rs, rs_f = [], []
    for gnum in range(a.gens):
        P = [{'eta': float(math.exp(rng.uniform(math.log(0.01), math.log(1.99)))), 'B': int(round(math.exp(rng.uniform(0, math.log(BMAX))))),
              'K': int(round(math.exp(rng.uniform(0, math.log(KMAX))))), 'rule': ['all', 'event'][rng.randint(2)]} for _ in range(2 * a.pairs)]
        rs += P; rs_f += fitness(sel, P, 5000 + gnum).mean(1).tolist()
    best_rs = rs[int(np.argmax(rs_f))]
    print(f'случайный поиск: лучшая {max(rs_f):.4f} у {best_rs} | {time.time() - t0:.0f} с', flush=True)
    es_final = decode(th)
    f_es_sel = float(fitness(sel, [es_final], 9999).mean())
    del sel; torch.cuda.empty_cache() if a.device == 'cuda' else None
    test = Data([rec(s) for s in a.test], a.device)
    cand = {'es_final': es_final, 'rs_best': best_rs, 'grid_a15': {'eta': 1.0, 'B': 50, 'K': 3, 'rule': 'all'},
            'grid_a15_event': {'eta': 1.0, 'B': 50, 'K': 3, 'rule': 'event'}, 'a15_random_c': {'eta': 0.3, 'B': 10, 'K': 3, 'rule': 'all'}}
    ft = fitness(test, list(cand.values()), 77777)
    out = {'sel': a.sel, 'test': a.test, 'args': vars(a), 'hist': hist, 'es_final': es_final, 'es_final_u': th.tolist(), 'es_final_sel': f_es_sel,
           'rs_best': best_rs, 'rs_best_sel': float(max(rs_f)), 'rs_all': [dict(p, f=fv) for p, fv in zip(rs, rs_f)],
           'test_fitness': {k: ft[i].tolist() for i, k in enumerate(cand)}, 'candidates': cand,
           'git_commit': commit, 'git_dirty': dirty, 'sec': round(time.time() - t0)}
    for i, k in enumerate(cand):
        print(f'тест (отложенные посевы): {k:15s} {cand[k]} → {ft[i].mean():.4f} ± {ft[i].std():.4f}', flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f'evo_A_{a.tag}.json'), 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
