"""Латентная индивидуальность копий (решение Егора 2026-10-04 после v4-sleep). Без обучения, по готовым весам.
Пререгистрация — docs/RESULTS_2026-10-04_copies.md, раздел «Латентная индивидуальность».

Медленные головы 40 копий после возврата (results_copies_vast, lm pretrained c, 8 посевов × 5 копий, RTX 5090).
Общие наборы состояний слепой политики (рефлекс к ближайшему предмету любого вида, 100 эпизодов 999000+k) из вариантов
среды, новых для всех копий (только оценка):
  base  — flip-2 (сверка с контрольной точкой «возврат»);
  rev3  — три разворота за эпизод (такты из [15, 40), [40, 65), [65, 90));
  ratio — соотношение видов 3:1 (какой вид в большинстве — монетка эпизода);
  rare  — по одному предмету каждого вида и задержка появления 20 тактов (длинные серии без событий).
Скрытые состояния Pythia-160M считаются на ноутбуке один раз на набор; копии сравниваются между собой на одних и тех же
состояниях (все копии — с одной машины, A18).
Статистика — как в H1/H2: D_env (9 пар копий из разных сред) − D_ss (s1–s2), доля несовпадающих выборов на живых тактах.

python latent_copies.py
"""
import itertools, json, os, time
from types import SimpleNamespace
import numpy as np
import torch

import env_flip as ef
from env_flip import T
import stand_words as sw
import body_head as bh
from stand_gpu import git_info

C = ['s1', 's2', 'h_hi', 'h_lo', 'dense']
ENV = {'s1': 'base', 's2': 'base', 'h_hi': 'hi', 'h_lo': 'lo', 'dense': 'dense'}
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
DIFF = [(i, j) for i, j in PAIRS if ENV[C[i]] != ENV[C[j]]]
NMAX, FAR = 4, 100.0


class NovelBodies(ef.FlipBodies):
    """flip-2 с вариантом: base | rev3 | ratio | rare. Базовые особи тянут из генератора то же, что FlipBodies."""

    def __init__(self, B, seeds, kind):
        super().__init__(B, seeds, True, 'flip-2')
        it = np.full((B, 2 * NMAX, 2), FAR)
        it[:, :2] = self.items[:, :2]; it[:, NMAX:NMAX + 2] = self.items[:, 2:4]
        rt = np.stack([self.t_rev, np.full(B, -1), np.full(B, -1)], 1)
        self.delay = 0; self.timer = np.zeros((B, 2 * NMAX), int)
        for b in range(B):
            r = self.rs[b]
            if kind == 'rev3':
                rt[b] = [r.randint(15, 40), r.randint(40, 65), r.randint(65, 90)]
            elif kind == 'ratio':                       # большинство — 3 предмета, меньшинство — 1
                majA = r.rand() < 0.5
                extra = r.rand(1, 2)
                if majA:
                    it[b, 2] = extra[0]; it[b, NMAX + 1] = FAR
                else:
                    it[b, NMAX + 2] = extra[0]; it[b, 1] = FAR
            elif kind == 'rare':
                it[b, 1] = FAR; it[b, NMAX + 1] = FAR
        if kind == 'rare':
            self.delay = 20
        self.items, self.rev_times = it, rt
        self.t_rev = np.full(B, 10 ** 6)

    def sense(self):
        ef.N_PER, keep = NMAX, ef.N_PER
        try:
            return super().sense()
        finally:
            ef.N_PER = keep

    def step(self, act):
        self.good_A ^= (self.rev_times == self.t).any(1)
        before = self.items.copy()
        ef.N_PER, keep = NMAX, ef.N_PER
        try:
            super().step(act)
        finally:
            ef.N_PER = keep
        if self.delay:
            moved = np.any(self.items != before, axis=2) & (before[:, :, 0] < FAR)     # только что съеденные
            self.items[moved] = FAR; self.timer[moved] = self.delay
            back = (self.timer == 1)
            for b, j in zip(*np.nonzero(back)):
                self.items[b, j] = self.rs[b].rand(2)
            self.timer = np.maximum(self.timer - 1, 0)

    def revs_passed(self):
        return ((self.rev_times >= 0) & (self.rev_times < self.t)).sum(1)


def blind_set(feat, kind, seeds, device):
    n = len(seeds)
    b = NovelBodies(n, seeds, kind); feat.reset(n)
    last_post = np.full(n, -1); n_rev = np.zeros(n, int)
    Z, AL, INF, GA, EV = [], [], [], [], []
    for t in range(T):
        s = b.sense()
        has_ev, _ = bh.labels(s)
        Z.append(bh.std(feat, feat.features(s)))
        nr = b.revs_passed(); last_post[nr > n_rev] = -1; n_rev = nr
        last_post[has_ev] = t
        AL.append(b.alive.copy()); INF.append(b.alive & (last_post >= 0)); GA.append(b.good_A.copy()); EV.append(has_ev & b.alive)
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(n)], 1))
        act[~b.alive] = 0
        b.step(act)
    tt = lambda x: torch.as_tensor(np.stack(x), device=device)
    return {'Z': torch.stack(Z), 'AL': tt(AL), 'INF': tt(INF), 'GA': tt(GA), 'EV': tt(EV)}


def perm_paired(d):
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    commit, dirty = git_info(); t0 = time.time()
    W = torch.as_tensor(np.load('results_copies_vast/w_pretrained_c_bat_1s1.npy'), device=dev)    # (40, d): посевы 1–8 × 5 копий
    Bv = torch.as_tensor(np.load('results_copies_vast/b_pretrained_c_bat_1s1.npy'), device=dev)
    seeds = json.load(open('results_copies_vast/copies_pretrained_c.json'))['runs'][0]['seeds']
    feat = bh.make_feat(SimpleNamespace(cond='lm', model='EleutherAI/pythia-160m', weights='pretrained', words='c', device=dev), 1, 32)
    out = {'git_commit': commit, 'git_dirty': dirty, 'seeds': seeds, 'sets': {}}
    for kind in ('base', 'rev3', 'ratio', 'rare'):
        bs = blind_set(feat, kind, sw.TEST_SEEDS, dev)
        P = (torch.einsum('tnd,ld->tnl', bs['Z'], W) + Bv) >= 0
        res = {'events_per_alive_tact': float(bs['EV'].float().sum() / bs['AL'].float().sum()), 'alive_tacts': int(bs['AL'].sum()),
               'informed_frac': float(bs['INF'].float().sum() / bs['AL'].float().sum())}
        for lab, mask in (('all', bs['AL']), ('informed', bs['INF'])):
            Pm = P[mask]                                                             # (такты, 40)
            D = (Pm[:, :, None] != Pm[:, None, :]).float().mean(0).cpu().numpy()
            de, ds, dl = [], [], []
            for si in range(len(seeds)):
                idx = list(range(5 * si, 5 * si + 5))
                M = D[np.ix_(idx, idx)]
                de.append(float(np.mean([M[i, j] for i, j in DIFF]))); ds.append(float(M[0, 1]))
            q = (Pm == bs['GA'][mask][:, None]).float().mean(0).cpu().numpy()
            res[lab] = {'D_env': de, 'D_ss': ds, 'delta': list(np.subtract(de, ds)), 'p': perm_paired(np.subtract(de, ds)),
                        'quality_by_copy': [float(np.mean(q[c::5])) for c in range(5)]}
        out['sets'][kind] = res
        r = res['all']
        print(f"{kind:6s}: событий на живой такт {res['events_per_alive_tact']:.3f}, информ. {res['informed_frac']:.2f} | D_env {np.mean(r['D_env']):.4f} D_ss {np.mean(r['D_ss']):.4f} "
              f"Δ {np.mean(r['delta']):+.4f} (выше в {sum(x > 0 for x in r['delta'])}/8, p {r['p']:.4f}) | качество "
              + ' '.join(f'{C[c]} {res["informed"]["quality_by_copy"][c]:.3f}' for c in range(5)) + f' | {time.time() - t0:.0f} с', flush=True)
    novel = ['rev3', 'ratio', 'rare']
    ps = [out['sets'][k]['all']['p'] for k in novel]; adj = holm(ps)
    out['H3'] = {k: {'delta': float(np.mean(out['sets'][k]['all']['delta'])), 'p': p, 'p_holm': a} for k, p, a in zip(novel, ps, adj)}
    out['H3_pass'] = any(out['H3'][k]['delta'] > 0 and out['H3'][k]['p_holm'] < 0.05 for k in novel)
    print('H3 (Холм по трём новым наборам):', {k: (round(v['delta'], 4), round(v['p'], 4), round(v['p_holm'], 4)) for k, v in out['H3'].items()},
          '→', 'ВЫПОЛНЕНА' if out['H3_pass'] else 'не выполнена', flush=True)
    os.makedirs('results_latent', exist_ok=True)
    json.dump(out, open('results_latent/latent.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
