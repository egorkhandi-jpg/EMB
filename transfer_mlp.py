"""Перенос считывания на новые символы видов: линейная голова против MLP и разнообразие обучающих наборов
(п. А дополнения Егора 2026-10-04; повод — Pacela et al., ICML 2026, arXiv 2603.28744: при суперпозиции признак может
быть нелинейно отделим в активациях, особенно вне распределения). Пререгистрация — docs/RESULTS_2026-10-04_sleep.md,
раздел «Перенос: нелинейная голова и несколько наборов символов».

Наборы символов видов (только токены видов A / B; ' none' и символы энергии те же; все однотокенные):
  обучающий пул (5): c ' A'/' B', p1 ' cat'/' table', p2 ' lamp'/' cloud', p3 ' violin'/' carpet', p4 ' bottle'/' feather';
  тестовые (3, никогда не в обучении): t1 ' candle'/' tunnel', t2 ' basket'/' mirror', t3 ' jacket'/' bridge'.
Пары (стандартизованное скрытое состояние, метка тела «последний исход»): жизни A11 «all, full» (онлайн-голова от тела,
η условия), 10 жизней (посевы 1–10) на каждый набор пула. Стандартизация — одна, по набору c (агент её не пересчитывает).
Тест вне контура: слепая политика, 100 эпизодов 999000+k, точность «A — пища» на информированных тактах, по каждому набору.
Повтор r = 1..10, k ∈ {1, 2, 3, 5}: k наборов пула (случайная выборка, при k = 5 — все), пары из жизни r каждого набора,
всего 3000 пар поровну между наборами (объём данных не растёт с k). Головы:
  линейная — логистическая регрессия L2, λ по CV с группами-эпизодами (как A14);
  MLP — 1 скрытый слой 256 (ReLU), AdamW (lr 1e-3), затухание весов из {1e-4, 1e-2, 1} и ранняя остановка — по точности на отложенных
        20 % эпизодов обучающих пар (терпение 10 эпох, до 300); тестовые наборы в подборе не участвуют.
Точность в распределении — среднее по k обучающим наборам, на новых — среднее по трём тестовым.
Перенос T = (точность на новых − 0,5) / (точность в распределении − 0,5).

python transfer_mlp.py --cond pretrained_c
"""
import argparse, copy, json, math, os, time
import numpy as np
import torch

import stand_words as sw
import body_head as bh
import body_head2 as bh2
import body_head3 as bh3
from readout import fit_probe
from transfer_symbols import traj, QuadNew
from stand_gpu import git_info

KIND = {'c': (329, 378), 'p1': (5798, 2829), 'p2': (18067, 9005), 'p3': (36608, 23656), 'p4': (11996, 22676),
        't1': (28725, 16583), 't2': (11281, 11472), 't3': (18553, 9729)}
POOL, TEST = ['c', 'p1', 'p2', 'p3', 'p4'], ['t1', 't2', 't3']
ETA = {'pretrained_c': 0.3, 'random_c': 0.03, 'quad': 0.3}          # η A11 «all, full» (results_bodyhead2/tune_*)
KS, REPS, NPAIRS = [1, 2, 3, 5], 10, 3000
WD = (1e-4, 1e-2, 1.0)


def feat_for(base, cond, name):
    if cond == 'quad':
        f = base if name == 'c' else QuadNew(base, 100 + list(KIND).index(name))
    else:
        f = copy.copy(base)
        f.kind_tok = torch.tensor([*KIND[name], 5293], device=base.kind_tok.device)
    f.mu, f.sd = base.mu, base.sd
    return f


def fit_linear(X, y, g, dev):
    mu, sd, w, cv, lam = fit_probe(X, y.astype(float), g, dev)
    W = (w[:-1] / sd).cpu().numpy(); b = float((w[-1] - (mu / sd * w[:-1]).sum()).cpu())
    return lambda Z: Z @ W + b >= 0


def fit_mlp(X, y, g, dev, seed):
    """Затухание весов — по сетке WD, ранняя остановка — по точности на отложенных 20 % эпизодов обучающих наборов."""
    best = None
    for wd in WD:
        pred, vl = _fit_mlp(X, y, g, dev, seed, wd)
        if best is None or vl > best[1]:
            best = (pred, vl, wd)
    best[0].wd = best[2]
    return best[0]


def _fit_mlp(X, y, g, dev, seed, wd):
    rng = np.random.RandomState(seed); ug = np.unique(g); rng.shuffle(ug)
    va = np.isin(g, ug[:max(1, len(ug) // 5)]); tr = ~va
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    tri = torch.as_tensor(np.nonzero(tr)[0], device=dev); vai = torch.as_tensor(np.nonzero(va)[0], device=dev)
    best, best_state, bad = -1.0, None, 0                               # по точности на отложенной части (цель — точность)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(300):
        net.train()
        perm = tri[torch.randperm(len(tri), generator=gen).to(dev)]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad(); lossf(net(Xt[idx]).squeeze(1), yt[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(Xt[vai]).squeeze(1) >= 0) == (yt[vai] > 0.5)).float().mean())
        if vl > best + 1e-4:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(best_state); net.eval()
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=dev); sd_t = torch.as_tensor(sd, dtype=torch.float32, device=dev)

    def pred(Z):
        with torch.no_grad():
            return (net((torch.as_tensor(Z, dtype=torch.float32, device=dev) - mu_t) / sd_t).squeeze(1) >= 0).cpu().numpy()
    return pred, best


def acc(pred, d):
    p = pred(d['Z']); m = d['INF']
    return float((p[m] == d['GA'][m]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=list(ETA), default='pretrained_c')
    ap.add_argument('--out', default='results_transfer')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    commit, dirty = git_info(); t0 = time.time()
    if a.cond == 'quad':
        base = sw.QuadWords(32, 768, dev, 1)
    else:
        base = sw.LMWords('EleutherAI/pythia-160m', 'pretrained' if a.cond == 'pretrained_c' else 'random', 'c', 32, dev, 1)
    base.fit_std(np.random.RandomState(0))
    tests, pairs = {}, {}
    for name in POOL + TEST:
        f = feat_for(base, a.cond, name)
        tests[name] = traj(f, sw.TEST_SEEDS, base.mu, base.sd, dev)
        if name in POOL:                                              # жизни A11 «all, full» (одна сеть на все 10 жизней)
            head = bh2.Head(10, base.mu.shape[0], [ETA[a.cond]] * 10, dev, 'full')
            _, _, pr = bh3.live_c(f, [[500000 + 1000 * s + e for e in range(50)] for s in range(1, 11)], head, 'all', dev, collect=True)
            pairs[name] = [{'z': np.stack([p[0] for p in pl]), 'y': np.array([p[1] for p in pl]), 'ep': np.array([p[4] for p in pl])} for pl in pr]
        print(f"{time.strftime('%H:%M:%S')} набор {name}: тест готов" + (f", пар на жизнь {np.mean([len(p['y']) for p in pairs[name]]):.0f}" if name in pairs else '')
              + f' | {time.time() - t0:.0f} с', flush=True)
    rows = []
    for r in range(REPS):
        for k in KS:
            rng = np.random.RandomState(1000 * r + k)
            sets = POOL if k == 5 else list(rng.choice(POOL, k, replace=False))
            X, y, g = [], [], []
            for j, nm in enumerate(sets):
                p = pairs[nm][r]; n = min(len(p['y']), NPAIRS // k)
                idx = rng.choice(len(p['y']), n, replace=False)
                X.append(p['z'][idx]); y.append(p['y'][idx]); g.append(p['ep'][idx] + 1000 * j)
            X, y, g = np.concatenate(X), np.concatenate(y).astype(float), np.concatenate(g)
            row = {'rep': r, 'k': k, 'sets': sets, 'n_pairs': int(len(y))}
            for hn, pred in (('linear', fit_linear(X, y, g, dev)), ('mlp', fit_mlp(X, y, g, dev, 10 * r + k))):
                a_in = float(np.mean([acc(pred, tests[nm]) for nm in sets])); per_new = {nm: acc(pred, tests[nm]) for nm in TEST}
                a_new = float(np.mean(list(per_new.values())))
                row[hn] = {'acc_in': a_in, 'acc_new': a_new, 'per_new': per_new, 'T': (a_new - 0.5) / (a_in - 0.5) if a_in > 0.5 else None}
            rows.append(row)
            print(f"{time.strftime('%H:%M:%S')} r{r} k{k} {sets}: линейная {row['linear']['acc_in']:.3f} → {row['linear']['acc_new']:.3f} | "
                  f"MLP {row['mlp']['acc_in']:.3f} → {row['mlp']['acc_new']:.3f} | {time.time() - t0:.0f} с", flush=True)
    summ = {}
    for hn in ('linear', 'mlp'):
        for k in KS:
            rr = [x[hn] for x in rows if x['k'] == k]
            ai, an = np.mean([x['acc_in'] for x in rr]), np.mean([x['acc_new'] for x in rr])
            summ[f'{hn}_k{k}'] = {'acc_in': float(ai), 'acc_new': float(an), 'T': float(np.mean([x['T'] for x in rr if x['T'] is not None])),
                                  'threshold': float(0.8 * (ai - 0.5) + 0.5), 'transfer': bool(an >= 0.8 * (ai - 0.5) + 0.5)}
            print(f"{a.cond} {hn} k={k}: в распределении {ai:.3f}, на новых {an:.3f}, T {summ[f'{hn}_k{k}']['T']:.3f}, порог {0.8 * (ai - 0.5) + 0.5:.3f} → "
                  f"{'перенос' if summ[f'{hn}_k{k}']['transfer'] else 'нет'}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({'cond': a.cond, 'rows': rows, 'summary': summ, 'git_commit': commit, 'git_dirty': dirty, 'sec': round(time.time() - t0)},
              open(os.path.join(a.out, f'transfer_mlp_{a.cond}.json'), 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
