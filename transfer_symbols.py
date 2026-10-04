"""Перенос медленного считывания на новые символы видов (решение Егора 2026-10-04; решает судьбу S2).
Пререгистрация — docs/RESULTS_2026-10-04_sleep.md, раздел «Перенос считывания на новые символы».

Голова: пакетная логистическая регрессия до сходимости (как зонд п. 15 / A14) на парах (стандартизованное скрытое
состояние, метка тела «последний исход») жизней A14 (flip-2, правило A11 all/full, символы видов X = ' A' / ' B').
Тест — замороженная голова вне контура: слепая политика на 100 тестовых эпизодах (999000+k), точность «A — пища»
на информированных тактах против истинного знака:
  X  — те же символы (ориентир, ≈ 0,90 у pretrained c);
  Y1 — новые символы видов ' candle' / ' tunnel' (однотокенные, при обучении не встречались), основной;
  Y2 — ' basket' / ' mirror', описательно.
Символ «нет события» (' none') и символы энергии те же. Стандартизация — та же, что при обучении (по X; агент не
пересчитывает её при смене видов); описательно — и с пересчётом по Y.
Ориентир «обучено прямо на Y»: та же регрессия на траекториях слепой политики с символами Y (200 эпизодов 60000+k,
метки тела), и для честного сравнения — то же на X.
lm random c — сеть посева s (как в A14); quad — новые символы = новые случайные строки проекции для позиций видов.

python transfer_symbols.py
"""
import json, math, os, time
from types import SimpleNamespace
import numpy as np
import torch

import env_flip as ef
from env_flip import T
import stand_words as sw
import body_head as bh
from readout import fit_probe
from stand_gpu import git_info

sw.WORDS.setdefault('y1', ((28725, 16583, 5293), (598, 1066, 1072)))     # ' candle' ' tunnel' ' none' | ' up' ' down' ' same'
sw.WORDS.setdefault('y2', ((11281, 11472, 5293), (598, 1066, 1072)))     # ' basket' ' mirror'
PAIRS = 'results_bodyhead3/pairs'


class QuadNew(sw.QuadWords):
    """quad с новыми символами видов: строки проекций для позиций «вид A» и «вид B» каждого такта — новые случайные."""

    def __init__(self, base, which):
        self.__dict__.update({k: v for k, v in base.__dict__.items() if k not in ('A', 'Bm')})
        r = np.random.RandomState(9000 + which)
        A, Bm = base.A.clone(), base.Bm.clone(); I, D = A.shape
        idx = [6 * j + c for j in range(base.N) for c in (0, 1)]
        A[idx] = torch.as_tensor(r.randn(len(idx), D).astype(np.float32) / math.sqrt(I), device=A.device)
        Bm[idx] = torch.as_tensor(r.randn(len(idx), D).astype(np.float32) / math.sqrt(I), device=A.device)
        self.A, self.Bm = A, Bm


def make(cond, words, seed, dev):
    if cond == 'quad':
        return sw.QuadWords(32, 768, dev, seed)
    return sw.LMWords('EleutherAI/pythia-160m', 'pretrained' if cond == 'pretrained_c' else 'random', words, 32, dev, seed)


def traj(feat, seeds, mu, sd, dev):
    """Слепая политика: стандартизованные (mu, sd) признаки, метки тела, знак, маски."""
    n = len(seeds)
    b = ef.FlipBodies(n, seeds, True, 'flip-2'); feat.reset(n)
    belief = np.zeros(n, bool); has_b = np.zeros(n, bool); last_post = np.full(n, -1); rev_seen = np.zeros(n, bool)
    Z, BEL, HB, INF, GA, EP = [], [], [], [], [], []
    for t in range(T):
        s = b.sense()
        has_ev, yA = bh.labels(s)
        belief = np.where(has_ev, yA, belief); has_b |= has_ev
        Z.append(((feat.features(s) - mu) / sd).cpu().numpy())
        after = b.t_rev < b.t; newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1; last_post[has_ev] = t
        BEL.append(belief.copy()); HB.append(has_b & b.alive); INF.append(b.alive & (last_post >= 0)); GA.append(b.good_A.copy()); EP.append(np.arange(n))
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(n)], 1)); act[~b.alive] = 0
        b.step(act)
    st = lambda x: np.stack(x).reshape(-1, *np.stack(x).shape[2:])
    return {'Z': st(Z), 'BEL': st(BEL), 'HB': st(HB), 'INF': st(INF), 'GA': st(GA), 'EP': st(EP)}


def fit(X, y, g, dev):
    mu, sd, w, cv, lam = fit_probe(X, y.astype(float), g, dev)
    return (w[:-1] / sd).cpu().numpy(), float((w[-1] - (mu / sd * w[:-1]).sum()).cpu()), cv


def acc(head, d):
    w, b, _ = head
    p = d['Z'] @ w + b >= 0; m = d['INF']
    return float((p[m] == d['GA'][m]).mean())


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    commit, dirty = git_info(); t0 = time.time()
    out = {'git_commit': commit, 'git_dirty': dirty, 'conds': {}}
    for cond in ('pretrained_c', 'random_c', 'quad'):
        rows = []
        shared = cond == 'pretrained_c'
        cache = {}
        for s in range(1, 11):
            key = 1 if shared else s
            if key not in cache:
                fX = make(cond, 'c', key, dev); fX.fit_std(np.random.RandomState(0))
                fY = {}
                for nm, w in (('y1', 'y1'), ('y2', 'y2')):
                    f = QuadNew(fX, 1 if nm == 'y1' else 2) if cond == 'quad' else make(cond, w, key, dev)
                    f.mu, f.sd = fX.mu, fX.sd
                    fY[nm] = f
                tests = {'X': traj(fX, sw.TEST_SEEDS, fX.mu, fX.sd, dev)}
                for nm, f in fY.items():
                    tests[nm] = traj(f, sw.TEST_SEEDS, fX.mu, fX.sd, dev)
                # пересчёт стандартизации по Y (описательно)
                f1 = fY['y1']; mu_old, sd_old = f1.mu, f1.sd; f1.fit_std(np.random.RandomState(0))
                tests['y1_restd'] = traj(f1, sw.TEST_SEEDS, f1.mu, f1.sd, dev); f1.mu, f1.sd = mu_old, sd_old
                trainX = traj(fX, [60000 + k for k in range(200)], fX.mu, fX.sd, dev)
                trainY = traj(fY['y1'], [60000 + k for k in range(200)], fX.mu, fX.sd, dev)
                hX = fit(trainX['Z'][trainX['HB']], trainX['BEL'][trainX['HB']], trainX['EP'][trainX['HB']], dev)
                hY = fit(trainY['Z'][trainY['HB']], trainY['BEL'][trainY['HB']], trainY['EP'][trainY['HB']], dev)
                cache[key] = (tests, {'blind_X': {'X': acc(hX, tests['X']), 'y1': acc(hX, tests['y1'])},
                                      'blind_Y1': {'y1': acc(hY, tests['y1']), 'X': acc(hY, tests['X'])}})
                del fX, fY; torch.cuda.empty_cache() if dev == 'cuda' else None
            tests, blind = cache[key]
            p = np.load(os.path.join(PAIRS, f'pairs_{cond}_all_full_s{s}.npz'))
            h = fit(p['z'], p['body'], p['ep'], dev)
            r = {'seed': s, **{f'acc_{k}': acc(h, v) for k, v in tests.items()}, 'blind_ref': blind, 'cv': h[2]}
            rows.append(r)
            print(f"{time.strftime('%H:%M:%S')} {cond} s{s}: X {r['acc_X']:.3f} | Y1 {r['acc_y1']:.3f} | Y2 {r['acc_y2']:.3f} | Y1 с пересчётом станд. "
                  f"{r['acc_y1_restd']:.3f} | слепое обучение: X→X {blind['blind_X']['X']:.3f}, X→Y1 {blind['blind_X']['y1']:.3f}, Y1→Y1 {blind['blind_Y1']['y1']:.3f} "
                  f"| {time.time() - t0:.0f} с", flush=True)
        aX = np.mean([r['acc_X'] for r in rows]); aY = np.mean([r['acc_y1'] for r in rows])
        thr = 0.8 * (aX - 0.5) + 0.5
        out['conds'][cond] = {'rows': rows, 'acc_X': aX, 'acc_y1': aY, 'acc_y2': float(np.mean([r['acc_y2'] for r in rows])),
                              'threshold': thr, 'transfer': bool(aY >= thr)}
        print(f'{cond}: X {aX:.3f}, Y1 {aY:.3f}, порог переноса {thr:.3f} → {"ПЕРЕНОС ЕСТЬ" if aY >= thr else "переноса нет"}', flush=True)
    out['sec'] = round(time.time() - t0)
    os.makedirs('results_transfer', exist_ok=True)
    json.dump(out, open('results_transfer/transfer.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
