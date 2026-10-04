"""Этап 3: копии обученного агента (решение Егора 2026-10-04 после v3-evo).
Пререгистрация — docs/RESULTS_2026-10-04_copies.md.

Агент: lm pretrained c (или random c), flip-2 с разворотом, рефлекс, двухскоростная голова; медленные веса учатся
только проигрыванием в паузе после эпизода по правилу A17 (η ≈ 2, B 50, K 10, метки all), быстрая часть онлайн (0,3).
Родитель живёт одну жизнь (50 эпизодов) в базовой среде. В конце жизни — 5 полных копий (медленные веса + буфер
проигрывания, быстрая часть обнулена). Фаза расхождения — 3 жизни: s1, s2 — базовая среда, разные потоки случайности;
h_hi — разворот вдвое чаще (два за эпизод), h_lo — вдвое реже (в половине эпизодов), dense — +2 предмета каждого вида.
Фаза возврата — 1 жизнь в базовой среде, одинаковые эпизоды у всех копий.
Контрольные точки: после родителя, после каждой жизни расхождения, после возврата. Меры:
  поведенческое расхождение — доля несовпадающих выборов вида замороженными медленными головами двух копий на общем
  наборе состояний слепой политики (100 тестовых эпизодов базовой среды, все такты, где агент жив);
  расхождение весов — 1 − косинус медленных весов; качество — выбор на информированных тактах вне контура,
  тест в контуре (20 эпизодов, замороженная медленная голова).
Копии считаются пакетом (одна LM на все жизни пакета); --sequential — каждая копия отдельно (сверка).

python copies.py --weights pretrained --seeds 1 2 3 4 5 6 7 8
"""
import argparse, json, math, os, time
from types import SimpleNamespace
import numpy as np
import torch

import env_flip as ef
from env_flip import T
import stand_words as sw
import body_head as bh
import body_head2 as bh2
from stand_gpu import git_info

COPIES = ['s1', 's2', 'h_hi', 'h_lo', 'dense']
ENV_OF = {'base': ('base', 2), 's1': ('base', 2), 's2': ('base', 2), 'h_hi': ('hi', 2), 'h_lo': ('lo', 2), 'dense': ('base', 4)}
NMAX = 4                                   # предметов каждого вида в массиве (лишние — далеко за ареной)
FAR = 100.0
MB = 32
EVAL_ROWS = 200


class VarBodies(ef.FlipBodies):
    """flip-2 с разворотом и параметрами по особи: rev ∈ {base, hi, lo}, n_per ∈ {2, 4}.
    Базовые особи (base, 2) тянут из своего генератора то же, что FlipBodies, — их эпизоды совпадают с flip-2.
    hi — два разворота за эпизод (такты из [20, 45) и [55, 80)); lo — разворот базовой среды в половине эпизодов."""

    def __init__(self, B, seeds, revs, nper):
        super().__init__(B, seeds, True, 'flip-2')
        it = np.full((B, 2 * NMAX, 2), FAR)
        it[:, :2] = self.items[:, :2]; it[:, NMAX:NMAX + 2] = self.items[:, 2:4]
        rt = np.stack([self.t_rev, np.full(B, -1)], 1)
        for b in range(B):
            r = self.rs[b]
            if nper[b] == 4:
                extra = r.rand(4, 2); it[b, 2:4] = extra[:2]; it[b, NMAX + 2:NMAX + 4] = extra[2:]
            if revs[b] == 'hi':
                rt[b] = [r.randint(20, 45), r.randint(55, 80)]
            elif revs[b] == 'lo' and r.rand() < 0.5:
                rt[b] = [-1, -1]
        self.items, self.rev_times = it, rt
        self.t_rev = np.full(B, 10 ** 6)               # не используется; развороты — по rev_times

    def sense(self):
        ef.N_PER, keep = NMAX, ef.N_PER
        try:
            return super().sense()
        finally:
            ef.N_PER = keep

    def step(self, act):
        self.good_A ^= (self.rev_times == self.t).any(1)
        ef.N_PER, keep = NMAX, ef.N_PER
        try:
            super().step(act)                          # внутри good_A ^= (t_rev == t) — всегда ложь
        finally:
            ef.N_PER = keep
        # съеденный «далёкий» предмет невозможен: расстояние ≫ EAT_R

    def revs_passed(self):
        return ((self.rev_times >= 0) & (self.rev_times < self.t)).sum(1)


class Replay:
    """Медленные веса (L, d), буфер последних B эпизодов по каждой жизни, свой генератор у каждой жизни."""

    def __init__(self, L, d, prm, gens, device):
        self.eta, self.B, self.K = prm['eta'], prm['B'], prm['K']
        self.buf = []                                  # список эпизодов: (Z (L,T,d), Y (L,T), M (L,T), N (L,T))
        self.gens = gens; self.device = device

    def clone_rows(self, rows):
        """Новый Replay по строкам rows (копии особей): буфер копируется."""
        r = Replay.__new__(Replay)
        r.eta, r.B, r.K, r.device = self.eta, self.B, self.K, self.device
        r.buf = [tuple(x[rows].clone() for x in ep) for ep in self.buf]
        r.gens = None
        return r

    def after_episode(self, head, ep):
        Z, Y, M = ep['z'], ep['bel'], ep['m_all']
        self.buf.append((Z, Y, M, (Z * Z).sum(-1) + 1))
        del self.buf[:-self.B]
        Zb, Yb, Mb, Nb = (torch.cat([x[k] for x in self.buf], 1) for k in range(4))
        L, n = Zb.shape[:2]
        rows = torch.arange(L, device=Zb.device)[:, None]
        for _ in range(self.K):
            perm = torch.stack([torch.randperm(n, generator=g) for g in self.gens]).to(Zb.device)     # (L, n)
            for k in range(0, n, MB):
                idx = perm[:, k:k + MB]
                z, y, m = Zb[rows, idx], Yb[rows, idx], Mb[rows, idx]
                if not m.any():
                    continue
                e = (y - torch.sigmoid((z * head.w[:, None]).sum(-1) + head.b[:, None])) * m
                norm = (m * Nb[rows, idx]).sum(1).clamp_min(1.0)
                head.w += (self.eta / norm)[:, None] * (e[:, :, None] * z).sum(1)
                head.b += self.eta * e.sum(1) / norm


def live(feat, head, rep, envs, seeds_per_life, device):
    """Жизни пакетом: envs — имя среды по жизни, seeds_per_life — посевы эпизодов. Онлайн-статистика по жизни."""
    L, n_ep = len(envs), len(seeds_per_life[0])
    revs = [ENV_OF[e][0] for e in envs]; nper = [ENV_OF[e][1] for e in envs]
    corr = np.zeros(L); cnt = np.zeros(L); fit = np.zeros(L); food = np.zeros(L); pois = np.zeros(L)
    for e in range(n_ep):
        head.reset_fast()
        bodies = VarBodies(L, [seeds_per_life[i][e] for i in range(L)], revs, nper)
        feat.reset(L)
        belief = np.zeros(L, bool); has_belief = np.zeros(L, bool)
        last_post = np.full(L, -1); n_rev = np.zeros(L, int)
        EP = {'z': [], 'bel': [], 'm_all': []}
        for t in range(T):
            s = bodies.sense()
            has_ev, yA = bh.labels(s)
            belief = np.where(has_ev, yA, belief); has_belief |= has_ev
            z = bh.std(feat, feat.features(s))
            m_all = has_belief & bodies.alive
            m = torch.as_tensor(m_all, dtype=torch.float32, device=device)
            if m.any():
                head.update(z, torch.as_tensor(belief, dtype=torch.float32, device=device), m)    # медленный шаг 0 — только быстрая
            nr = bodies.revs_passed(); last_post[nr > n_rev] = -1; n_rev = nr
            last_post[has_ev] = t
            informed = bodies.alive & (last_post >= 0)
            chA = (head.logit(z) >= 0).cpu().numpy()
            corr += (chA == bodies.good_A) & informed; cnt += informed
            EP['z'].append(z); EP['bel'].append(torch.as_tensor(belief, dtype=torch.float32, device=device))
            EP['m_all'].append(m)
            act = sw.reflex(s, np.stack([np.where(chA, 1.0, -1.0), np.ones(L)], 1))
            act[~bodies.alive] = 0
            bodies.step(act)
        fit += bodies.fitness(); food += bodies.n_good; pois += bodies.n_bad
        rep.after_episode(head, {k: torch.stack(v, 1) for k, v in EP.items()})
    return {'online_informed': (corr / np.maximum(cnt, 1)).tolist(), 'fitness': (fit / n_ep).tolist(),
            'food': (food / n_ep).tolist(), 'poison': (pois / n_ep).tolist()}


def blind_set(feat, seeds, device):
    """Общий набор состояний: слепая политика в базовой среде; Z (T, n, d), живые и информированные такты, знак."""
    n = len(seeds)
    b = ef.FlipBodies(n, seeds, True, 'flip-2'); feat.reset(n)
    last_post = np.full(n, -1); rev_seen = np.zeros(n, bool)
    Z, AL, INF, GA = [], [], [], []
    for t in range(T):
        s = b.sense()
        has_ev, _ = bh.labels(s)
        Z.append(bh.std(feat, feat.features(s)))
        after = b.t_rev < b.t
        newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1
        last_post[has_ev] = t
        AL.append(b.alive.copy()); INF.append(b.alive & (last_post >= 0)); GA.append(b.good_A.copy())
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(n)], 1))
        act[~b.alive] = 0
        b.step(act)
    tt = lambda x: torch.as_tensor(np.stack(x), device=device)
    return {'Z': torch.stack(Z), 'AL': tt(AL), 'INF': tt(INF), 'GA': tt(GA)}


def checkpoint(feat, head, parent_w, bs, test_seeds, device):
    """Меры по всем жизням пакета (строки head): предсказания на общем наборе, расхождения, качество."""
    W, b = head.w, head.b
    P = (torch.einsum('tnd,ld->tnl', bs['Z'], W) + b) >= 0                   # (T, n, L)
    al, inf, ga = bs['AL'], bs['INF'], bs['GA']
    L = W.shape[0]
    Pa = P[al]; Pi = P[inf]                                                    # (такты, L)
    beh = (Pa[:, :, None] != Pa[:, None, :]).float().mean(0).cpu().numpy()
    beh_inf = (Pi[:, :, None] != Pi[:, None, :]).float().mean(0).cpu().numpy()
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (Wn @ Wn.T).cpu().numpy()
    cos_par = (Wn @ (parent_w / parent_w.norm(dim=1, keepdim=True).clamp_min(1e-12)).T).diagonal().cpu().numpy() if parent_w is not None else None
    q_out = (Pi == ga[inf][:, None]).float().mean(0).cpu().numpy()
    # тест в контуре: замороженная медленная голова, test_seeds эпизодов на жизнь, все жизни одним пакетом
    n = len(test_seeds); per = max(1, EVAL_ROWS // n)                         # куски ≤ EVAL_ROWS строк: иначе видеопамять
    q_in = []
    for l0 in range(0, L, per):
        ls = list(range(l0, min(L, l0 + per))); m = len(ls) * n
        h1 = bh2.Head(m, 1, [0.0] * m, device, 'dual')
        h1.w = W[ls].repeat_interleave(n, 0).clone(); h1.b = b[ls].repeat_interleave(n, 0).clone(); h1.wf = torch.zeros_like(h1.w)
        _, st = bh2.live(feat, [[s] for _ in ls for s in test_seeds], h1, 'event', device, frozen=True)
        lg = st['log'][0]; fit = np.array(st['fitness']); food = np.array(st['food']); pois = np.array(st['poison'])
        for k in range(len(ls)):
            sl = slice(k * n, (k + 1) * n)
            q_in.append({'choice_informed': sw.choice_stats(lg[sl])['informed'], 'test': float(fit[sl].mean()),
                         'food': float(food[sl].mean()), 'poison': float(pois[sl].mean())})
    return {'beh': beh.tolist(), 'beh_informed': beh_inf.tolist(), 'one_minus_cos': (1 - cos).tolist(),
            'one_minus_cos_parent': (1 - cos_par).tolist() if cos_par is not None else None,
            'out_of_loop': q_out.tolist(), 'in_loop': q_in, 'w_norm': W.norm(dim=1).tolist()}


def run_seeds(a, seeds, feat, prm, bs, out, commit, dirty, sequential=False):
    """Родитель → 5 копий → 3 жизни расхождения → возврат, для группы посевов (одна сеть на группу)."""
    S = len(seeds); d = feat.mu.shape[0]; t0 = time.time()
    tests = sw.TEST_SEEDS[:a.test_eps]
    head = bh2.Head(S, d, [0.0] * S, a.device, 'dual')
    rep = Replay(S, d, prm, [torch.Generator().manual_seed(9000 + 100 * s) for s in seeds], a.device)
    on_par = live(feat, head, rep, ['base'] * S, [[500000 + 1000 * s + e for e in range(a.life)] for s in seeds], a.device)
    cp0 = checkpoint(feat, head, None, bs, tests, a.device)
    print(f"{time.strftime('%H:%M:%S')} родители {seeds}: онлайн {np.round(on_par['online_informed'], 3).tolist()}, вне контура "
          f"{np.round(cp0['out_of_loop'], 3).tolist()} | {time.time() - t0:.0f} с", flush=True)
    parent_w = head.w.clone(); parent_b = head.b.clone()
    rows = torch.as_tensor(np.repeat(np.arange(S), len(COPIES)), device=a.device)
    envs = [c for _ in seeds for c in COPIES]
    groups = [list(range(len(envs)))] if not sequential else [[i] for i in range(len(envs))]
    res_by_life = {i: [] for i in range(len(envs))}
    for grp in groups:
        g = torch.as_tensor(grp, device=a.device)
        L = len(grp)
        h = bh2.Head(L, d, [0.0] * L, a.device, 'dual')
        h.w = parent_w[rows[g]].clone(); h.b = parent_b[rows[g]].clone()
        r = rep.clone_rows(rows[g])
        r.gens = [torch.Generator().manual_seed(9100 + 100 * seeds[i // len(COPIES)] + i % len(COPIES)) for i in grp]
        pw = parent_w[rows[g]]
        cps, ons = [], []
        for j in range(a.div_lives):
            sp = [[600000 + 10000 * seeds[i // len(COPIES)] + 1000 * (i % len(COPIES)) + a.life * j + e for e in range(a.life)] for i in grp]
            ons.append(live(feat, h, r, [envs[i] for i in grp], sp, a.device))
            cps.append(checkpoint(feat, h, pw, bs, tests, a.device))
        sp = [[700000 + 1000 * seeds[i // len(COPIES)] + e for e in range(a.life)] for i in grp]
        ons.append(live(feat, h, r, ['base'] * L, sp, a.device))
        cps.append(checkpoint(feat, h, pw, bs, tests, a.device))
        for k, i in enumerate(grp):
            res_by_life[i] = {'cps': cps, 'k': k, 'ons': ons, 'w': [None]}
        nm = f"{a.tag}_{'seq' if sequential else 'bat'}_{seeds[grp[0] // len(COPIES)]}{COPIES[grp[0] % len(COPIES)]}"
        np.save(os.path.join(out, f'w_{nm}.npy'), h.w.cpu().numpy()); np.save(os.path.join(out, f'b_{nm}.npy'), h.b.cpu().numpy())
        print(f"{time.strftime('%H:%M:%S')} копии {[envs[i] for i in grp][:5]}{'…' if L > 5 else ''} посевов "
              f"{sorted({seeds[i // len(COPIES)] for i in grp})}: готово | {time.time() - t0:.0f} с", flush=True)
    # раскладка по посевам: матрицы 5×5 по каждой контрольной точке
    out_seeds = []
    for si, s in enumerate(seeds):
        idx = list(range(si * len(COPIES), (si + 1) * len(COPIES)))
        cps_s = []
        for c in range(a.div_lives + 1):
            if not sequential:
                cp = res_by_life[idx[0]]['cps'][c]
                sub = lambda M: np.array(M)[np.ix_(idx, idx)].tolist()
                cps_s.append({'beh': sub(cp['beh']), 'beh_informed': sub(cp['beh_informed']), 'one_minus_cos': sub(cp['one_minus_cos']),
                              'one_minus_cos_parent': [cp['one_minus_cos_parent'][i] for i in idx],
                              'out_of_loop': [cp['out_of_loop'][i] for i in idx], 'in_loop': [cp['in_loop'][i] for i in idx]})
            else:
                cps_s.append({'per_copy': [{kk: (vv[0] if isinstance(vv, list) and kk in ('out_of_loop', 'in_loop', 'one_minus_cos_parent') else None)
                                            for kk, vv in res_by_life[i]['cps'][c].items()} for i in idx]})
        ons = [[{kk: res_by_life[i]['ons'][ph][kk][res_by_life[i]['k']] for kk in res_by_life[i]['ons'][ph]} for i in idx]
               for ph in range(a.div_lives + 1)]
        out_seeds.append({'seed': s, 'parent_online': on_par['online_informed'][si], 'parent_fitness': on_par['fitness'][si],
                          'parent_cp': {'out_of_loop': cp0['out_of_loop'][si], 'in_loop': cp0['in_loop'][si]},
                          'checkpoints': cps_s, 'online': ons, 'git_commit': commit})
    return out_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--words', default='c')
    ap.add_argument('--model', default='EleutherAI/pythia-160m')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 9)))
    ap.add_argument('--life', type=int, default=50)
    ap.add_argument('--div-lives', type=int, default=3)
    ap.add_argument('--test-eps', type=int, default=20)
    ap.add_argument('--params', default='results_evo/evo_A_pretrained_c.json')
    ap.add_argument('--sequential', action='store_true', help='сверка: каждая копия отдельной жизнью')
    ap.add_argument('--name', default='')
    ap.add_argument('--require-check', default=None, help='JSON сверки (copies_check.py): без pass = true серия не запускается')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_copies')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    if a.require_check:
        ck = json.load(open(a.require_check)) if os.path.exists(a.require_check) else {'pass': False}
        if not ck['pass']:
            raise SystemExit(f'сверка пакет/последовательно не пройдена или не сделана ({a.require_check}) — серия не запускается')
    prm = json.load(open(a.params))['es_final']
    assert prm['rule'] == 'all' and 0 < prm['eta'] < 2           # правило 12: нормированный шаг по мини-пакету устойчив при 0 < η < 2
    a.tag = f"{a.weights}_{a.words}" + (f'_{a.name}' if a.name else '')
    path = os.path.join(a.out, f"copies_{a.tag}{'_seq' if a.sequential else ''}.json")
    res = json.load(open(path)) if os.path.exists(path) else {'tag': a.tag, 'params': prm, 'copies': COPIES, 'life': a.life,
                                                              'div_lives': a.div_lives, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'seeds': a.seeds, 'sequential': a.sequential})
    todo = [s for s in a.seeds if s not in {x['seed'] for x in res['results']}]
    shared = a.weights == 'pretrained'
    t0 = time.time()
    for grp in ([todo] if shared else [[s] for s in todo]):
        if not grp:
            continue
        feat = bh.make_feat(SimpleNamespace(cond='lm', model=a.model, weights=a.weights, words=a.words, device=a.device), grp[0], 32)
        bs = blind_set(feat, sw.TEST_SEEDS, a.device)
        res['results'] += run_seeds(a, grp, feat, prm, bs, a.out, commit, dirty, a.sequential)
        json.dump(res, open(path, 'w'), ensure_ascii=False)
        del feat, bs; torch.cuda.empty_cache() if a.device == 'cuda' else None
    res['sec'] = round(time.time() - t0)
    json.dump(res, open(path, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
