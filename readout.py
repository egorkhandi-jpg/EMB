"""Три уровня считывания знака при рефлексе (п. 1 решения Егора 2026-10-02 по пилоту D).
flip-2 с разворотом, N = 32, рефлекторный слой, тест — те же 100 эпизодов, что в пилоте D.

Зачем. Пилот D: головы LM, найденные ES, выбирают вид верно на 0,57–0,71 информированных тактов,
а линейный зонд с учителем на том же скрытом состоянии читает знак на 0,95 (П3). Различение
пища / яд — у 1 агента из 18. Неизвестно, где потери: в представлении, в обучении ES или в теле.

Уровни:
  oracle_sign — рефлексу подаётся истинный текущий знак (потолок тела);
  probe_head  — голова = линейный зонд знака, обученный с учителем на скрытом состоянии LM:
                раунд 1 — траектории со слепой к знаку головой (рефлекс к ближайшему предмету
                любого вида), раунд 2 — траектории самого зонда (одна итерация переобучения на
                собственных траекториях, итоговый зонд — на объединении);
  es_head     — головы из пилота D (повтор теста на тех же эпизодах; результат должен совпасть);
  es_head_fs  — описательно: выбор вида — от головы ES, скорость — полная (как у oracle/probe),
                чтобы отделить вклад выученной скорости.
У oracle_sign и probe_head скорость полная. Главная мера — точность выбора вида на
информированных тактах (после хотя бы одного кадра-события с последнего разворота).

python readout.py --seeds 1 2 3 [--out results_readout]
"""
import argparse, json, os, time
import numpy as np
import torch

import env_flip as ef
from env_flip import FlipBodies, T
import stand_words as sw
from probe_fast import _newton, _bacc, LAMBDAS
from stand_gpu import git_info

LM_CONDS = [('pretrained', 'c'), ('random', 'c'), ('pretrained', 'c_arb1')]


def run(get_choice, seeds, feat=None, speed=None, collect=False, es=None):
    """Эпизоды с рефлексом. get_choice(s, h, b) -> выбран A (bool[B]); feat — признаки LM или None;
    es — (pol, TH): голова ES выдаёт и выбор, и скорость (speed=None) либо только выбор."""
    B = len(seeds)
    b = FlipBodies(B, seeds, True, 'flip-2')
    if feat is not None:
        feat.reset(B)
    log, H, Y, INF, EP = [], [], [], [], []
    last_post = np.full(B, -1); rev_seen = np.zeros(B, bool)
    for t in range(T):
        s = b.sense()
        after = b.t_rev < b.t
        newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1
        ev = s[:, 10:12].sum(1) > 0
        last_post[ev] = t
        informed = b.alive & (last_post >= 0)
        h = feat.features(s) if feat is not None else None
        if es is not None:
            pol, TH = es
            a = pol.head(TH, h) if hasattr(pol, 'head') else pol.act(TH, s)
            a = a.cpu().numpy()
            chA = a[:, 0] > 0
            sp = a[:, 1] if speed is None else np.full(B, speed)
        else:
            chA = get_choice(s, h, b)
            sp = np.full(B, 1.0 if speed is None else speed)
        log.append(np.stack([chA, b.good_A.copy(), b.alive.copy(), ev, after], 1))
        if collect:
            H.append(h.cpu().numpy()); Y.append(b.good_A.copy()); INF.append(informed.copy()); EP.append(np.arange(B))
        act = sw.reflex(s, np.stack([np.where(chA, 1.0, -1.0), sp], 1))
        act[~b.alive] = 0
        b.step(act)
    b.choice_log = np.stack(log, 1)
    out = {'b': b}
    if collect:
        m = np.concatenate(INF)
        out['data'] = (np.concatenate(H)[m], np.concatenate(Y)[m].astype(float), np.concatenate(EP)[m])
    return out


def fit_probe(X, y, g, device, folds=5):
    """λ — по CV с группами-эпизодами (сбалансированная точность), затем обучение на всём.
    Возвращает (mu, sd, w, точность CV)."""
    ug = np.unique(g); r = np.random.RandomState(0); r.shuffle(ug)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device); yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    acc = {l: [] for l in LAMBDAS}; pred = {l: np.zeros(len(y), bool) for l in LAMBDAS}
    for f in range(folds):
        te = np.isin(g, ug[f::folds]); tr = ~te
        trt = torch.as_tensor(tr, device=device)
        mu, sd = Xt[trt].mean(0), Xt[trt].std(0, unbiased=False) + 1e-6
        X1 = torch.cat([(Xt - mu) / sd, torch.ones(len(y), 1, device=device)], 1)
        for l in LAMBDAS:
            w = _newton(X1[trt], yt[trt], l, torch.zeros(X1.shape[1], device=device))
            pred[l][te] = (X1[torch.as_tensor(te, device=device)] @ w).cpu().numpy() > 0
    accs = {l: _bacc(pred[l], y) for l in LAMBDAS}
    best = max(LAMBDAS, key=lambda l: (accs[l], -LAMBDAS.index(l)))
    mu, sd = Xt.mean(0), Xt.std(0, unbiased=False) + 1e-6
    X1 = torch.cat([(Xt - mu) / sd, torch.ones(len(y), 1, device=device)], 1)
    w = _newton(X1, yt, best, torch.zeros(X1.shape[1], device=device))
    return mu, sd, w, float(accs[best]), best


def apply_probe(P, X):
    mu, sd, w = P[:3]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=mu.device)
    return ((((Xt - mu) / sd) @ w[:-1] + w[-1]) > 0).cpu().numpy()


def summarize(b):
    fe = b.fitness()
    st = sw.choice_stats(b.choice_log)
    return {'test': float(fe.mean()), 'test_per_episode': [round(float(x), 4) for x in fe],
            'food': float(b.n_good.mean()), 'poison': float(b.n_bad.mean()), 'choice': st}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--episodes', type=int, default=200, help='эпизодов на раунд обучения зонда')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_readout')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'readout.json')
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    res = json.load(open(path)) if os.path.exists(path) else {'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'seeds': a.seeds, 'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    done = {(x['level'], x['cond'], x['seed']) for x in res['results']}
    t0 = time.time()

    def save():
        json.dump(res, open(path, 'w'), ensure_ascii=False)

    if ('oracle_sign', '-', 0) not in done:
        o = run(lambda s, h, b: b.good_A.copy(), sw.TEST_SEEDS)
        res['results'].append({'level': 'oracle_sign', 'cond': '-', 'seed': 0, **summarize(o['b'])}); save()
        print('oracle_sign', res['results'][-1]['choice']['informed'], res['results'][-1]['test'], flush=True)
    blind = lambda s, h, b: s[:, 2] <= s[:, 5]           # к ближайшему предмету любого вида
    for seed in a.seeds:
        for weights, words in LM_CONDS:
            cond = f'{weights}_{words}'
            if all((lv, cond, seed) in done for lv in ('probe_head', 'es_head', 'es_head_fs')):
                continue
            pol = sw.LMWords('EleutherAI/pythia-160m', weights, words, 32, a.device, seed)
            th = pol.init(np.random.RandomState(seed))           # воспроизводит стандартизацию пилота D
            tag = f'pythia-160m_{weights}_{words}_N32_s{seed}'
            th = np.load(os.path.join('results_reflex', f'theta_{tag}.npy'))
            TH = torch.as_tensor(np.repeat(th[None], len(sw.TEST_SEEDS), 0), device=a.device)
            for lv, sp in (('es_head', None), ('es_head_fs', 1.0)):
                if (lv, cond, seed) in done:
                    continue
                r = run(None, sw.TEST_SEEDS, feat=pol, speed=sp, es=(pol, TH))
                res['results'].append({'level': lv, 'cond': cond, 'seed': seed, **summarize(r['b']),
                                       'git_commit': commit}); save()
            if ('probe_head', cond, seed) not in done:
                r1 = run(blind, [60000 + 1000 * seed + k for k in range(a.episodes)], feat=pol, collect=True)
                X1, y1, g1 = r1['data']
                P1 = fit_probe(X1, y1, g1, a.device)
                probe1 = lambda s, h, b: apply_probe(P1, h.cpu().numpy())
                r2 = run(probe1, [70000 + 1000 * seed + k for k in range(a.episodes)], feat=pol, collect=True)
                X2, y2, g2 = r2['data']
                acc_p1_on_own = float(_bacc(apply_probe(P1, X2), y2))   # зонд раунда 1 на собственных траекториях
                P2own = fit_probe(X2, y2, g2, a.device)                    # CV-точность на собственных траекториях
                P = fit_probe(np.concatenate([X1, X2]), np.concatenate([y1, y2]),
                              np.concatenate([g1, g2 + 10 ** 6]), a.device)
                np.savez(os.path.join(a.out, f'probe_{cond}_s{seed}.npz'), mu=P[0].cpu().numpy(), sd=P[1].cpu().numpy(),
                         w=P[2].cpu().numpy(), lam=P[4])          # для п. 3 (голова с зонда + ES)
                rt = run(lambda s, h, b: apply_probe(P, h.cpu().numpy()), sw.TEST_SEEDS, feat=pol)
                res['results'].append({'level': 'probe_head', 'cond': cond, 'seed': seed, **summarize(rt['b']),
                                       'probe_acc_blind_cv': P1[3], 'probe_acc_own_cv': P2own[3],
                                       'probe1_on_own': acc_p1_on_own, 'probe_acc_union_cv': P[3],
                                       'lambda': P[4], 'n_blind': int(len(y1)), 'n_own': int(len(y2)),
                                       'git_commit': commit}); save()
            last = [x for x in res['results'] if x['cond'] == cond and x['seed'] == seed]
            print(f'{time.strftime("%H:%M:%S")} {cond} s{seed}: ' + '; '.join(
                f"{x['level']} выбор {x['choice']['informed']:.3f} тест {x['test']:.3f} пища/яд {x['food']:.1f}/{x['poison']:.1f}" for x in last)
                + f' | {time.time() - t0:.0f} с', flush=True)
            del pol; torch.cuda.empty_cache() if a.device == 'cuda' else None
    save()


if __name__ == '__main__':
    main()
