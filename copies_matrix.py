"""Матрица «копия × среда» и H5 (п. Б дополнения Егора 2026-10-04). Пререгистрация — docs/RESULTS_2026-10-04_copies.md,
раздел «H5». Серия копий на ноутбуке (посевы 9–16, results_copies_laptop); веса после 3-й жизни расхождения (cp3) и после
возврата (cpret).
Для каждой копии c ∈ {s1, s2, h_hi, h_lo, dense} и среды e ∈ {base, h_hi, h_lo, dense} — замороженная медленная голова
(без быстрой части):
  вне контура — выбор на информированных тактах на общем наборе слепой политики среды e (100 эпизодов 999000+k);
  в контуре — тест 20 эпизодов в среде e: выбор на информированных тактах, приспособленность.
A(e) = мера копии из e в среде e − среднее s1, s2 в среде e, e ∈ {h_hi, h_lo, dense}.
H5: среднее A(e) по трём средам > 0 после 3-й жизни (вне контура), односторонний точный тест по 8 посевам.
Семья: H4a, H4b (h4_copies.py) и H5 — Холм по трём.

python copies_matrix.py
"""
import itertools, json, os, time
from types import SimpleNamespace
import numpy as np
import torch

from env_flip import T
import stand_words as sw
import body_head as bh
from copies import VarBodies, ENV_OF, COPIES
from h4_copies import perm, holm
from stand_gpu import git_info

D = 'results_copies_laptop'
ENVS = ['base', 'h_hi', 'h_lo', 'dense']
ROWS = 200


def run_env(feat, env, W, b, seeds, dev, frozen_head):
    """Слепая политика (frozen_head = False) или замороженная голова (True) в среде env; L = W.shape[0] голов × len(seeds)."""
    L = W.shape[0] if frozen_head else 1
    n = L * len(seeds)
    revs, nper = [ENV_OF[env][0]] * n, [ENV_OF[env][1]] * n
    bd = VarBodies(n, [s for _ in range(L) for s in seeds], revs, nper); feat.reset(n)
    Wr = W.repeat_interleave(len(seeds), 0) if frozen_head else None; br = b.repeat_interleave(len(seeds), 0) if frozen_head else None
    last_post = np.full(n, -1); n_rev = np.zeros(n, int)
    Z, INF, GA, COR = [], [], [], np.zeros(n); CNT = np.zeros(n)
    for t in range(T):
        s = bd.sense()
        has_ev, _ = bh.labels(s)
        z = bh.std(feat, feat.features(s))
        nr = bd.revs_passed(); last_post[nr > n_rev] = -1; n_rev = nr
        last_post[has_ev] = t
        inf = bd.alive & (last_post >= 0)
        if frozen_head:
            chA = (((z * Wr).sum(1) + br) >= 0).cpu().numpy()
            COR += (chA == bd.good_A) & inf; CNT += inf
        else:
            Z.append(z); INF.append(inf); GA.append(bd.good_A.copy())
            chA = s[:, 2] <= s[:, 5]
        act = sw.reflex(s, np.stack([np.where(chA, 1.0, -1.0), np.ones(n)], 1)); act[~bd.alive] = 0
        bd.step(act)
    if frozen_head:
        acc = (COR / np.maximum(CNT, 1)).reshape(L, len(seeds)); fit = bd.fitness().reshape(L, len(seeds))
        cnt = CNT.reshape(L, len(seeds))
        return (acc * cnt).sum(1) / np.maximum(cnt.sum(1), 1), fit.mean(1)
    tt = lambda x: torch.as_tensor(np.stack(x), device=dev)
    return {'Z': torch.stack(Z), 'INF': tt(INF), 'GA': tt(GA)}


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    commit, dirty = git_info(); t0 = time.time()
    r = json.load(open(os.path.join(D, 'copies_pretrained_c.json'), encoding='utf-8'))
    seeds = [x['seed'] for x in r['results']]; S = len(seeds)
    feat = bh.make_feat(SimpleNamespace(cond='lm', model='EleutherAI/pythia-160m', weights='pretrained', words='c', device=dev), 1, 32)
    out = {'git_commit': commit, 'git_dirty': dirty, 'seeds': seeds, 'envs': ENVS, 'copies': COPIES, 'cp': {}}
    blind = {e: run_env(feat, e, None, None, sw.TEST_SEEDS, dev, False) for e in ENVS}
    print(f'общие наборы слепой политики готовы | {time.time() - t0:.0f} с', flush=True)
    for cp in ('cp3', 'cpret'):
        W = torch.as_tensor(np.load(os.path.join(D, f'w_pretrained_c_bat_{seeds[0]}s1_{cp}.npy')), device=dev)
        B = torch.as_tensor(np.load(os.path.join(D, f'b_pretrained_c_bat_{seeds[0]}s1_{cp}.npy')), device=dev)
        Mout = np.zeros((S, 5, len(ENVS))); Min = np.zeros((S, 5, len(ENVS))); Mfit = np.zeros((S, 5, len(ENVS)))
        for ei, e in enumerate(ENVS):
            bs = blind[e]
            P = (torch.einsum('tnd,ld->tnl', bs['Z'], W) + B) >= 0
            m = bs['INF']; q = (P[m] == bs['GA'][m][:, None]).float().mean(0).cpu().numpy()
            Mout[:, :, ei] = q.reshape(S, 5)
            per = max(1, ROWS // 20); accs, fits = [], []
            for l0 in range(0, W.shape[0], per):
                a_, f_ = run_env(feat, e, W[l0:l0 + per], B[l0:l0 + per], sw.TEST_SEEDS[:20], dev, True); accs.append(a_); fits.append(f_)
            Min[:, :, ei] = np.concatenate(accs).reshape(S, 5); Mfit[:, :, ei] = np.concatenate(fits).reshape(S, 5)
            print(f'{cp} среда {e}: вне контура по копиям ' + ' '.join(f'{COPIES[c]} {Mout[:, c, ei].mean():.3f}' for c in range(5))
                  + ' | в контуре ' + ' '.join(f'{COPIES[c]} {Min[:, c, ei].mean():.3f}' for c in range(5)) + f' | {time.time() - t0:.0f} с', flush=True)
        A = {}
        for nm, M in (('out', Mout), ('in', Min), ('fit', Mfit)):
            Ae = {e: (M[:, COPIES.index(e), ENVS.index(e)] - M[:, :2, ENVS.index(e)].mean(1)).tolist() for e in ('h_hi', 'h_lo', 'dense')}
            Am = np.mean([Ae[e] for e in Ae], 0)
            A[nm] = {'A_e': Ae, 'A_mean': Am.tolist(), 'p_one': perm(Am, 1), 'p_two': perm(Am, 0)}
        out['cp'][cp] = {'out': Mout.tolist(), 'in': Min.tolist(), 'fit': Mfit.tolist(), 'A': A}
        print(f"{cp}: A(e) вне контура " + ' '.join(f"{e} {np.mean(A['out']['A_e'][e]):+.4f}" for e in A['out']['A_e'])
              + f" | среднее {np.mean(A['out']['A_mean']):+.4f} (выше в {sum(x > 0 for x in A['out']['A_mean'])}/{S}, p одност. {A['out']['p_one']:.4f})"
              + f" | в контуре среднее {np.mean(A['in']['A_mean']):+.4f} (p {A['in']['p_one']:.4f})", flush=True)
    h4 = json.load(open(os.path.join(D, 'h4.json'), encoding='utf-8'))
    ps = [h4['H4a']['p'], h4['H4b']['p'], out['cp']['cp3']['A']['out']['p_one']]; adj = holm(ps)
    ds = [h4['H4a']['d'], h4['H4b']['d'], float(np.mean(out['cp']['cp3']['A']['out']['A_mean']))]
    ok = [ds[0] > 0 and adj[0] < 0.05, ds[1] < 0 and adj[1] < 0.05, ds[2] > 0 and adj[2] < 0.05]
    out['family'] = {nm: {'d': d, 'p': p, 'p_holm': a_, 'pass': o} for nm, d, p, a_, o in zip(('H4a', 'H4b', 'H5'), ds, ps, adj, ok)}
    print('СЕМЬЯ (Холм по трём):', {k: (round(v['d'], 4), round(v['p'], 4), round(v['p_holm'], 4), v['pass']) for k, v in out['family'].items()}, flush=True)
    json.dump(out, open(os.path.join(D, 'matrix.json'), 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
