"""Анализ обученных агентов этапа 2 (пререгистрация v2, docs/RESULTS_2026-10-01_flip.md).

1) Зонд: логистическая регрессия «текущий знак эпизода» по признакам —
   agent: скрытое состояние LM (стандартизованное) / состояние GRU / скрытый слой flywin;
   body:  позиция, курс (cos, sin), энергия —
   как функция k = 0..10 тактов (а) после первого съеденного предмета до разворота и
   (б) после первого съеденного предмета после разворота. Кросс-валидация по эпизодам (5 частей),
   признаки стандартизуются и сжимаются до 32 главных компонент на обучающей части.
   Мера П3 — площадь под кривой точности по k = 0..10 (среднее по k).
2) Телепортация: в такт ПОСЛЕ первого съеденного предмета тело переносится в случайную точку
   со случайным курсом. В такт события нельзя: кадр-событие несёт исход через тренд энергии,
   и вмешательство до его чтения мерило бы порчу сигнала, а не память в теле.
   Тот же перенос у оракула — геометрическая цена переноса без всякой памяти.
3) Подмена энергии: в тот же момент энергия (и её предыдущее значение — чтобы не создать
   ложный тренд) заменяется медианой энергии в момент первого события по эпизодам.

python analyze_flip.py results_flip2 [--episodes 400] [--out results_flip2/analysis.json]
"""
import argparse, glob, json, os
import numpy as np
import torch

import env_flip as ef
from env_flip import FlipBodies, T
import stand_flip as sf

K_MAX = 10


def load_agent(r, device):
    a = argparse.Namespace(**r['args'])
    a.device = device
    pol = sf.make_policy(a, getattr(torch, a.dtype), r['net_seed'] if r['net_seed'] is not None else a.seed)
    pol.init(np.random.RandomState(a.seed))          # воспроизводит стандартизацию LM, как в обучении
    th = np.load(r['_theta'])
    return a, pol, th


def run_episodes(pol, th, a, seeds, device, intervene=None, e_med=None, oracle=False):
    """Прогон с записью признаков. intervene: None | 'teleport' | 'energy'."""
    B = len(seeds)
    b = FlipBodies(B, seeds, a.reversal, a.env)
    TH = torch.as_tensor(np.repeat(th[None], B, 0), device=device) if not oracle else None
    if not oracle:
        pol.reset(B)
    rs = np.random.RandomState(4242)
    first_eat = np.full(B, -1)
    first_eat_post = np.full(B, -1)
    rec = {'agent': [], 'body': [], 'sign': [], 'alive': []}
    e_at_first = np.full(B, np.nan)
    for t in range(T):
        if intervene is not None:
            hit = (first_eat >= 0) & (first_eat == t - 1)       # такт после кадра-события
            if hit.any():
                if intervene == 'teleport':
                    b.pos[hit] = rs.rand(hit.sum(), 2)
                    b.head[hit] = rs.rand(hit.sum()) * 2 * np.pi
                else:
                    b.E[hit] = e_med
                    b.prev_E[hit] = e_med
        s = b.sense()
        ate = s[:, 10:12].sum(1) > 0
        new = ate & (first_eat < 0)
        first_eat[new] = t
        e_at_first[new] = b.E[new]
        post = ate & (first_eat_post < 0) & (b.t_rev >= 0) & (b.t_rev < b.t)
        first_eat_post[post] = t
        if oracle:
            act = ef.oracle(s, None, b)
            feat = np.zeros((B, 1), np.float32)
        else:
            act = pol.act(TH, torch.as_tensor(s, device=device)).cpu().numpy()
            feat = pol.last_h.cpu().numpy() if hasattr(pol, 'last_h') else np.zeros((B, 1), np.float32)
        rec['agent'].append(feat)
        rec['body'].append(np.stack([b.pos[:, 0], b.pos[:, 1], np.cos(b.head), np.sin(b.head), b.E], 1))
        rec['sign'].append(b.good_A.copy())
        rec['alive'].append(b.alive.copy())
        act[~b.alive] = 0
        b.step(act)
    out = {k: np.stack(v, 1) for k, v in rec.items()}           # (B, T, ...)
    out.update(fitness=b.fitness(), first_eat=first_eat, first_eat_post=first_eat_post,
               t_rev=b.t_rev, e_at_first=e_at_first)
    return out


def probe_curve(X, y, groups, folds=5, pcs=32, lam=1.0):
    """Точность логистической регрессии с CV по эпизодам; X (n, d), y (n,), groups — эпизод."""
    if len(y) < 20 or y.min() == y.max():
        return np.nan
    ug = np.unique(groups); rng = np.random.RandomState(0); rng.shuffle(ug)
    acc = []
    for f in range(folds):
        te_g = ug[f::folds]; te = np.isin(groups, te_g); tr = ~te
        if te.sum() == 0 or tr.sum() < 10 or y[tr].min() == y[tr].max():
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if Xtr.shape[1] > pcs:
            U, S, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
            P = Vt[:pcs].T; m0 = Xtr.mean(0)
            Xtr, Xte = (Xtr - m0) @ P, (Xte - m0) @ P
        Xt = torch.tensor(np.c_[Xtr, np.ones(len(Xtr))], dtype=torch.float64)
        yt = torch.tensor(y[tr], dtype=torch.float64)
        w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w, yt) + lam * (w[:-1] ** 2).sum() / len(yt)
            l.backward(); return l
        opt.step(closure)
        p = (torch.tensor(np.c_[Xte, np.ones(len(Xte))]) @ w.detach()).numpy() > 0
        acc.append((p == y[te].astype(bool)).mean())
    return float(np.mean(acc)) if acc else np.nan


def probe(out, key, anchor):
    """Кривая точности по k тактов после якоря (first_eat до разворота / first_eat_post)."""
    curve = []
    B = len(out['fitness'])
    for k in range(K_MAX + 1):
        X, y, g = [], [], []
        for i in range(B):
            t0 = out[anchor][i]
            if t0 < 0:
                continue
            t = t0 + k
            if t >= T or not out['alive'][i, t]:
                continue
            if anchor == 'first_eat' and out['t_rev'][i] >= 0 and t > out['t_rev'][i]:
                continue                                         # до разворота знак не менялся
            X.append(out[key][i, t]); y.append(out['sign'][i, t]); g.append(i)
        curve.append(probe_curve(np.array(X), np.array(y, float), np.array(g)) if X else np.nan)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--episodes', type=int, default=400)
    ap.add_argument('--only', default='', help='подстрока тега — анализировать только такие')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    a0 = ap.parse_args()
    seeds = [999000 + k for k in range(a0.episodes)]
    out_path = a0.out or os.path.join(a0.folder, 'analysis.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    # эталон: оракул без вмешательства и с телепортацией — цена переноса без памяти
    dummy = argparse.Namespace(reversal=True, env='flip-2')
    o0 = run_episodes(None, None, dummy, seeds, a0.device, oracle=True)
    o1 = run_episodes(None, None, dummy, seeds, a0.device, intervene='teleport', oracle=True)
    res['_oracle'] = {'fitness': float(o0['fitness'].mean()), 'teleport': float(o1['fitness'].mean())}
    print(f"оракул: {o0['fitness'].mean():.3f}, с телепортацией {o1['fitness'].mean():.3f}", flush=True)
    for f in sorted(glob.glob(os.path.join(a0.folder, 'result_*.json'))):
        r = json.load(open(f))
        name = os.path.basename(f)[7:-5]
        if a0.only not in name or name in res:
            continue
        r['_theta'] = os.path.join(a0.folder, f'theta_{name}.npy')
        a, pol, th = load_agent(r, a0.device)
        base = run_episodes(pol, th, a, seeds, a0.device)
        e_med = float(np.nanmedian(base['e_at_first']))
        tel = run_episodes(pol, th, a, seeds, a0.device, intervene='teleport')
        ene = run_episodes(pol, th, a, seeds, a0.device, intervene='energy', e_med=e_med)
        has_eat = base['first_eat'] >= 0
        item = {'cond': r['cond'], 'N': r['N'], 'seed': r['seed'], 'tag': r['tag'],
                'test_reported': r['test'], 'test_repro': float(base['fitness'][:100].mean()),
                'fitness': float(base['fitness'].mean()),
                'teleport': float(tel['fitness'].mean()), 'energy': float(ene['fitness'].mean()),
                'teleport_drop_eaters': float((base['fitness'] - tel['fitness'])[has_eat].mean()),
                'energy_drop_eaters': float((base['fitness'] - ene['fitness'])[has_eat].mean()),
                'frac_eat': float(has_eat.mean()), 'e_med': e_med}
        for key in ('agent', 'body'):
            if key == 'agent' and base['agent'].shape[2] == 1:
                continue
            for anchor in ('first_eat', 'first_eat_post'):
                c = probe(base, key, anchor)
                item[f'probe_{key}_{anchor}'] = c
                item[f'auc_{key}_{anchor}'] = float(np.nanmean(c))
        res[name] = item
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in item.items()
                     if not k.startswith('probe_')}, flush=True)
        json.dump(res, open(out_path, 'w'), ensure_ascii=False, indent=1)
    json.dump(res, open(out_path, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
