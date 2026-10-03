"""ES с простым шагом с нуля (п. 2 решения Егора 2026-10-03; описательно, 3 посева, без серии).
lm pretrained c, Pythia-160M, flip-2 с разворотом, N = 32, рефлекс. Голова с нуля (как в пилоте D: веса
выбора и скорости N(0, 0,01), смещения 0), 16 пар × 4 эпизода, σ 0,05, простой шаг θ += lr·g без Adam,
lr ∈ {0,002; 0,005}, 120 поколений. Вопрос: находит ли ES считывание сама.
Зачем: A9 — Adam с шагом 0,03 дрейфует (разрушает даже голову-зонд), простой шаг lr 0,002 удерживает её.
Тест (100 эпизодов пилота D) — на поколениях 40, 80, 120.

python es_simple.py --lr 0.002 --seed 1
"""
import argparse, json, os, time
import numpy as np
import torch

import stand_words as sw
import readout as ro
from stand_gpu import git_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gens', type=int, default=120)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_es_simple')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    pol = sw.LMWords('EleutherAI/pythia-160m', 'pretrained', 'c', 32, a.device, a.seed)
    rng = np.random.RandomState(a.seed)
    th = pol.init(rng)
    pairs, sigma = 16, 0.05
    hist, checks, t0 = [], {}, time.time()
    name = f'lr{a.lr}_s{a.seed}'
    for g in range(a.gens):
        eps = rng.randn(pairs, pol.n).astype(np.float32)
        TH = np.concatenate([th + sigma * eps, th - sigma * eps])
        f = sw.rollout(pol, TH, [a.seed * 100000 + g * 10 + k for k in range(4)], a.device)
        ranks = f.argsort().argsort() / (len(f) - 1) - 0.5
        grad = ((ranks[:pairs] - ranks[pairs:])[:, None] * eps).sum(0) / (2 * pairs * sigma)
        th = (th + a.lr * grad).astype(np.float32)
        hist.append(float(f.mean()))
        if (g + 1) in (40, 80, 120):
            TT = torch.as_tensor(np.repeat(th[None], len(sw.TEST_SEEDS), 0), device=a.device)
            s = ro.summarize(ro.run(None, sw.TEST_SEEDS, feat=pol, speed=None, es=(pol, TT))['b'])
            checks[g + 1] = {'choice_informed': s['choice']['informed'], 'test': s['test'], 'food': s['food'], 'poison': s['poison']}
            print(f"{name} поколение {g + 1}: выбор {s['choice']['informed']:.3f} тест {s['test']:.3f} пища/яд {s['food']:.1f}/{s['poison']:.1f} | {time.time() - t0:.0f} с", flush=True)
    np.save(os.path.join(a.out, f'theta_{name}.npy'), th)
    json.dump({'lr': a.lr, 'seed': a.seed, 'gens': a.gens, 'hist': hist, 'checks': checks, 'sec': round(time.time() - t0),
               'git_commit': commit, 'git_dirty': dirty}, open(os.path.join(a.out, f'es_simple_{name}.json'), 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
