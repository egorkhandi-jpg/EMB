"""H4: направление компетентности копий на незнакомом (решение Егора 2026-10-04; пререгистрация —
docs/RESULTS_2026-10-04_copies.md, раздел H4). Новая серия копий на ноутбуке (посевы 9–16, results_copies_laptop).
Медленные головы копий после возврата; набор rev3 (три разворота за эпизод, слепая политика, 100 эпизодов 999000+k).
d_hi = выбор на информированных тактах у h_hi − среднее s1, s2; d_lo — то же у h_lo.
H4a: d_hi > 0; H4b: d_lo < 0 — односторонние точные перестановочные тесты по 8 посевам, Холм по двум.

python h4_copies.py
"""
import itertools, json, os
from types import SimpleNamespace
import numpy as np
import torch

import body_head as bh
import stand_words as sw
from latent_copies import blind_set, DIFF
from stand_gpu import git_info

D = 'results_copies_laptop'


def perm(d, side):
    """Точный перестановочный тест знаков: side = +1 (среднее > 0), −1 (< 0), 0 (двусторонний)."""
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d)))); m = (s * d).mean(1); o = d.mean()
    if side == 0:
        return float((np.abs(m) >= abs(o) - 1e-12).mean())
    return float((side * m >= side * o - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    commit, dirty = git_info()
    r = json.load(open(os.path.join(D, 'copies_pretrained_c.json'), encoding='utf-8'))
    seeds = [x['seed'] for x in r['results']]
    W = torch.as_tensor(np.load(os.path.join(D, f'w_pretrained_c_bat_{seeds[0]}s1.npy')), device=dev)
    Bv = torch.as_tensor(np.load(os.path.join(D, f'b_pretrained_c_bat_{seeds[0]}s1.npy')), device=dev)
    feat = bh.make_feat(SimpleNamespace(cond='lm', model='EleutherAI/pythia-160m', weights='pretrained', words='c', device=dev), 1, 32)
    out = {'git_commit': commit, 'git_dirty': dirty, 'seeds': seeds, 'sets': {}}
    for kind in ('rev3', 'base'):
        bs = blind_set(feat, kind, sw.TEST_SEEDS, dev)
        P = (torch.einsum('tnd,ld->tnl', bs['Z'], W) + Bv) >= 0
        m = bs['INF']; q = (P[m] == bs['GA'][m][:, None]).float().mean(0).cpu().numpy().reshape(len(seeds), 5)
        s = q[:, :2].mean(1); d_hi = q[:, 2] - s; d_lo = q[:, 3] - s; d_de = q[:, 4] - s
        Pa = P[bs['AL']]; Dm = (Pa[:, :, None] != Pa[:, None, :]).float().mean(0).cpu().numpy()
        de = [float(np.mean([Dm[5 * i + a, 5 * i + b] for a, b in DIFF])) for i in range(len(seeds))]; ds = [float(Dm[5 * i, 5 * i + 1]) for i in range(len(seeds))]
        out['sets'][kind] = {'q': q.tolist(), 'd_hi': d_hi.tolist(), 'd_lo': d_lo.tolist(), 'd_dense': d_de.tolist(),
                             'p_hi_one': perm(d_hi, 1), 'p_lo_one': perm(d_lo, -1), 'p_hi_two': perm(d_hi, 0), 'p_lo_two': perm(d_lo, 0),
                             'p_dense_two': perm(d_de, 0), 'D_env': de, 'D_ss': ds, 'p_Denv_two': perm(np.subtract(de, ds), 0)}
        o = out['sets'][kind]
        print(f"{kind}: d_hi {d_hi.mean():+.4f} ({(d_hi > 0).sum()}/8, p одност. {o['p_hi_one']:.4f}) | d_lo {d_lo.mean():+.4f} ({(d_lo < 0).sum()}/8 ниже, "
              f"p одност. {o['p_lo_one']:.4f}) | d_dense {d_de.mean():+.4f} ({(d_de > 0).sum()}/8, p {o['p_dense_two']:.4f}) | D_env − D_ss "
              f"{np.mean(np.subtract(de, ds)):+.4f} (p {o['p_Denv_two']:.4f})", flush=True)
    o = out['sets']['rev3']; adj = holm([o['p_hi_one'], o['p_lo_one']])
    out['H4a'] = {'d': float(np.mean(o['d_hi'])), 'p': o['p_hi_one'], 'p_holm': adj[0], 'pass': bool(np.mean(o['d_hi']) > 0 and adj[0] < 0.05)}
    out['H4b'] = {'d': float(np.mean(o['d_lo'])), 'p': o['p_lo_one'], 'p_holm': adj[1], 'pass': bool(np.mean(o['d_lo']) < 0 and adj[1] < 0.05)}
    print('H4a:', out['H4a'], '\nH4b:', out['H4b'], flush=True)
    json.dump(out, open(os.path.join(D, 'h4.json'), 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
