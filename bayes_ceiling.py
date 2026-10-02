"""Байесовский потолок и ориентир (B) для П3-зондов знака эпизода (п. 4, A–C).

На тех же состояниях (10 наборов по 200 эпизодов, flip-2 с разворотом, политика, слепая к знаку;
такты, где знак определим) и той же цели — текущий знак эпизода:
  hmm_N   — прямой проход скрытой марковской модели по дискретным событиям последних N кадров
            (вид съеденного, исход ' up' / ' same' / ' down' по порогу |тренд| > 0,5, как в интерфейсе c).
            Состояние — (знак, был ли разворот). Разворот — один раз, в такт из [REV_LO, REV_HI)
            равномерно (параметры среды), поэтому риск зависит от такта: 1 / (REV_HI − t).
            Правдоподобия исходов P(исход | вид пища / яд) — частоты по всем данным (свойство среды,
            не метки зонда: «same» при еде бывает, когда пища съедена при полной энергии).
            Априорно на начале окна: знак 50/50, «разворот уже был» — по равномерному закону.
            hmm_all — то же по всей истории эпизода.
  last_N  — эвристика «последний исход по каждому виду» в окне: вывод по более свежему событию.
Мера — сбалансированная точность «P(A — пища) > 0,5» против истинного знака.

python bayes_ceiling.py [--sets 10] [--episodes 200] → results_words/bayes_ceiling.json
"""
import argparse, json, os
import numpy as np

import probe_untrained as pu
import env_flip as ef
from probe_fast import _bacc

NS = (2, 8, 32)


def outcome(et):
    return np.where(et > 0.5, 0, np.where(et < -0.5, 2, 1))       # 0 up, 1 same, 2 down


def likelihoods(S, sign_all):
    """P(исход | съеденный вид — пища) и P(исход | яд) — частоты по всем кадрам-событиям."""
    ateA, ateB = S[:, :, 10] > 0, S[:, :, 11] > 0
    o = outcome(S[:, :, 7])
    good = (ateA & sign_all) | (ateB & ~sign_all)
    bad = (ateA & ~sign_all) | (ateB & sign_all)
    pf = np.array([(o[good] == j).sum() for j in range(3)], float) + 1
    pp = np.array([(o[bad] == j).sum() for j in range(3)], float) + 1
    return pf / pf.sum(), pp / pp.sum()


def hmm_posterior(S_ep, t, N, lf, lp):
    """P(A — пища в кадре t) по кадрам [t−N+1, t] одного эпизода (N = None — вся история)."""
    start = 0 if N is None else max(0, t - N + 1)
    # состояния: 0 — A пища, без разворота; 1 — B пища, без; 2 — A пища, был; 3 — B пища, был
    flipped0 = np.clip((start - ef.REV_LO) / (ef.REV_HI - ef.REV_LO), 0, 1)   # P(t_rev ≤ start − 1)
    alpha = np.array([0.5 * (1 - flipped0), 0.5 * (1 - flipped0), 0.5 * flipped0, 0.5 * flipped0])
    for f in range(start, t + 1):
        if f > start:     # переход f−1 → f: разворот в шаге f−1 с риском 1/(REV_HI − (f−1))
            s = f - 1
            h = 1.0 / (ef.REV_HI - s) if ef.REV_LO <= s < ef.REV_HI else 0.0
            a0, a1, a2, a3 = alpha
            alpha = np.array([a0 * (1 - h), a1 * (1 - h), a2 + a1 * h, a3 + a0 * h])
        fr = S_ep[f]
        if fr[10] > 0 or fr[11] > 0:
            o = outcome(fr[7])
            ate_A = fr[10] > 0
            l_Agood = lf[o] if ate_A else lp[o]      # если A — пища: съел A -> пища, съел B -> яд
            l_Bgood = lp[o] if ate_A else lf[o]
            alpha = alpha * np.array([l_Agood, l_Bgood, l_Agood, l_Bgood])
        alpha = alpha / alpha.sum()
    return alpha[0] + alpha[2]


def last_outcome(S_ep, t, N):
    start = max(0, t - N + 1)
    for f in range(t, start - 1, -1):
        fr = S_ep[f]
        if fr[10] > 0 or fr[11] > 0:
            o = outcome(fr[7])
            if o == 1:
                continue                                       # «same» неинформативно для эвристики
            good_eaten = o == 0
            return 1.0 if (fr[10] > 0) == good_eaten else 0.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--out', default='results_words/bayes_ceiling.json')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = {'sets': a.sets, 'episodes': a.episodes, 'results': []}
    for seed in range(1, a.sets + 1):
        D = pu.collect('flip-2', True, a.episodes, seed0=31000 + 1000 * seed)
        lf, lp = likelihoods(D['S'], D['sign'])
        ep, tt = np.where(D['valid'])
        y = D['sign'][ep, tt]
        k = D['k'][ep, tt]
        out = {'seed': seed, 'lik_food': lf.round(4).tolist(), 'lik_poison': lp.round(4).tolist(), 'n': int(len(y))}
        for nm, fn in [(f'hmm_{N}', (lambda N: lambda e, t: hmm_posterior(D['S'][e], t, N, lf, lp))(N)) for N in NS] + \
                      [('hmm_all', lambda e, t: hmm_posterior(D['S'][e], t, None, lf, lp))] + \
                      [(f'last_{N}', (lambda N: lambda e, t: last_outcome(D['S'][e], t, N))(N)) for N in NS]:
            p = np.array([fn(e, t) for e, t in zip(ep, tt)])
            pred = p > 0.5
            # ничья (0,5 — нет информации): угадывание с равной вероятностью, детерминированно по такту
            tie = p == 0.5
            pred[tie] = (tt[tie] % 2) == 0
            out[nm] = float(_bacc(pred, y))
            out[nm + '_curve'] = [_bacc(pred[k == kk], y[k == kk]) for kk in range(11)]
        res['results'].append(out)
        print(seed, {kk: round(v, 3) for kk, v in out.items() if isinstance(v, float)}, flush=True)
    for nm in [f'hmm_{N}' for N in NS] + ['hmm_all'] + [f'last_{N}' for N in NS]:
        v = np.array([r[nm] for r in res['results']])
        res[nm + '_mean'] = float(v.mean()); res[nm + '_sd'] = float(v.std())
        print(f'{nm:8s} {v.mean():.3f} ± {v.std():.3f}')
    json.dump(res, open(a.out, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
