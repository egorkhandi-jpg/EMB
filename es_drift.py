"""Почему ES разрушает голову-зонд (п. 1 плана 2026-10-03). lm pretrained c, Pythia-160M, flip-2 с разворотом,
N = 32, рефлекс; посевы 1–3. Пререгистрация — docs/RESULTS_2026-10-01_flip.md, раздел «Дрейф ES и голова от тела».

П. 3 (BC + ES): ES, стартуя с головы-зонда (выбор 0,965), ухудшила её в 9 из 9 прогонов (→ 0,785).
  --mode noise — ES от головы-зонда, приспособленность заменена чистым шумом N(0, 1) (те же σ 0,05, lr 0,03,
                 Adam, 16 пар, 40 поколений). Падение выбора ≈ как в BC + ES — это дрейф, не сигнал.
  --mode eval  — обучающая приспособленность головы-зонда до ES и головы после BC + ES (results_bc) на тех же
                 160 обучающих эпизодах, что видела ES (посевы s·100000 + g·10 + k, g < 40, k < 4).
                 Выше после ES при худшем выборе — цель рассогласована; ниже — дрейф.
  --mode sgd   — ES от головы-зонда с простым шагом θ += lr·g (без Adam), lr = 0,002 — описательно.

python es_drift.py --mode noise|eval|sgd --seed 1
"""
import argparse, json, os, time
import numpy as np
import torch

import stand_words as sw
import readout as ro
from stand_gpu import git_info


def probe_init(pol, seed):
    th = pol.init(np.random.RandomState(seed))
    z = np.load(f'results_readout/probe_pretrained_c_s{seed}.npz')
    mu, sd = pol.mu.cpu().numpy(), pol.sd.cpu().numpy(); w, bias = z['w'][:-1], float(z['w'][-1]); D = pol.D
    W = th[:D * 2].reshape(D, 2).copy(); W[:, 0] = w * sd / z['sd']; W[:, 1] = 0
    th[:D * 2] = W.reshape(-1); th[D * 2] = bias + float((w * (mu - z['mu']) / z['sd']).sum()); th[D * 2 + 1] = 2.0
    return th


def evaluate(pol, th, seeds, device):
    TH = torch.as_tensor(np.repeat(th[None], len(seeds), 0), device=device)
    r = ro.run(None, seeds, feat=pol, speed=None, es=(pol, TH))
    return ro.summarize(r['b'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['noise', 'eval', 'sgd'], required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_drift')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    pol = sw.LMWords('EleutherAI/pythia-160m', 'pretrained', 'c', 32, a.device, a.seed)
    th0 = probe_init(pol, a.seed)
    out = {'mode': a.mode, 'seed': a.seed, 'git_commit': commit, 'git_dirty': dirty}
    t0 = time.time()
    if a.mode == 'eval':
        train_seeds = [a.seed * 100000 + g * 10 + k for g in range(40) for k in range(4)]
        th1 = np.load(f'results_bc/theta_pythia-160m_pretrained_c_N32_bc_s{a.seed}.npy')
        for nm, th in (('before_es', th0), ('after_es', th1)):
            s = evaluate(pol, th, train_seeds, a.device)
            out[nm] = {'train_fitness': s['test'], 'choice_informed': s['choice']['informed'], 'food': s['food'], 'poison': s['poison']}
        out['rel_drift_bc'] = float(np.linalg.norm(th1 - th0) / np.linalg.norm(th0))
        s = evaluate(pol, th1, sw.TEST_SEEDS, a.device)
        out['after_es_test'] = {'test': s['test'], 'choice_informed': s['choice']['informed']}
    else:
        rng = np.random.RandomState(a.seed)
        noise_rng = np.random.RandomState(10 ** 6 + a.seed)
        pairs, sigma, gens = 16, 0.05, 40
        lr = 0.03 if a.mode == 'noise' else 0.002
        th = th0.copy(); m = np.zeros_like(th); v = np.zeros_like(th); hist = []; drift = []
        for g in range(gens):
            eps = rng.randn(pairs, pol.n).astype(np.float32)
            TH = np.concatenate([th + sigma * eps, th - sigma * eps])
            if a.mode == 'noise':
                f = noise_rng.randn(2 * pairs)
            else:
                f = sw.rollout(pol, TH, [a.seed * 100000 + g * 10 + k for k in range(4)], a.device)
            ranks = f.argsort().argsort() / (len(f) - 1) - 0.5
            grad = ((ranks[:pairs] - ranks[pairs:])[:, None] * eps).sum(0) / (2 * pairs * sigma)
            if a.mode == 'noise':
                m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
                th = (th + lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)).astype(np.float32)
            else:
                th = (th + lr * grad).astype(np.float32)
            hist.append(float(np.mean(f))); drift.append(float(np.linalg.norm(th - th0) / np.linalg.norm(th0)))
        s = evaluate(pol, th, sw.TEST_SEEDS, a.device)
        out.update({'lr': lr, 'hist': hist, 'rel_drift': drift, 'test': s['test'], 'choice_informed': s['choice']['informed'],
                    'food': s['food'], 'poison': s['poison'], 'choice_by_k': s['choice']['by_k']})
        np.save(os.path.join(a.out, f'theta_{a.mode}_s{a.seed}.npy'), th)
    out['sec'] = round(time.time() - t0)
    json.dump(out, open(os.path.join(a.out, f'drift_{a.mode}_s{a.seed}.json'), 'w'), ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k not in ('hist', 'rel_drift', 'choice_by_k')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
