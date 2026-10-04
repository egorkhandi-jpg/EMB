"""Калибровка скорости медленного обучения для эвристики S2 (среда со сном) по кривой обучения медленной головы.
Источник — записанный опыт агента A15 (results_evo/record, 26 жизней × 100 эпизодов, lm pretrained c): медленные веса
учатся проигрыванием (правило A17: η 1,9996, B 50, K 10, метки «последний исход»; для сравнения — сетка A15:
η 1, B 50, K 3) на эпизодах 0–49 по одному. После каждого эпизода замороженная медленная голова оценивается на
эпизодах 50–99 того же посева: точность «A — пища» против истинного знака на информированных тактах.
Это записанные траектории в контуре (не слепая политика); мера — аналог «вне контура» для калибровки.
Выход: кривая точность(эпизоды, события), эпизоды и события до 0,85; C₀ эвристики M2 из условия
0,5 + 0,5·(1 − e^{−C/C₀}) = 0,85 → C₀ = C(0,85) / ln(1/0,3).

python calib_c0.py
"""
import json, math, os, time
import numpy as np
import torch

import stand_gpu  # noqa: F401  — fp32 без TF32
from stand_gpu import git_info

MB, E1 = 32, 50
REC = 'results_evo/record'
SEEDS = list(range(201, 217)) + list(range(301, 311))


def curve(prm, dev):
    accs = np.zeros((len(SEEDS), E1)); evs = np.zeros((len(SEEDS), E1))
    for si, s in enumerate(SEEDS):
        r = np.load(os.path.join(REC, f'rec_pretrained_c_s{s}.npz'))
        Z = torch.as_tensor(r['z'][:E1], device=dev); Y = torch.as_tensor(r['bel'][:E1], device=dev); M = torch.as_tensor(r['m_all'][:E1], device=dev)
        Ze = torch.as_tensor(r['z'][E1:2 * E1].reshape(-1, Z.shape[-1]), device=dev)
        sg = r['sign'][E1:2 * E1].reshape(-1) > 0.5; inf = r['inf'][E1:2 * E1].reshape(-1) > 0.5
        evs[si] = np.cumsum(r['m_ev'][:E1].sum(1))
        w = torch.zeros(Z.shape[-1], device=dev); b = torch.zeros((), device=dev)
        g = torch.Generator().manual_seed(1000 + s)
        for e in range(E1):
            lo = max(0, e - prm['B'] + 1)
            z = Z[lo:e + 1].reshape(-1, Z.shape[-1]); y = Y[lo:e + 1].reshape(-1); m = M[lo:e + 1].reshape(-1)
            keep = m > 0; z, y = z[keep], y[keep]; n2 = (z * z).sum(1) + 1
            for _ in range(prm['K']):
                perm = torch.randperm(len(y), generator=g).to(dev)
                for k in range(0, len(y), MB):
                    i = perm[k:k + MB]
                    err = y[i] - torch.sigmoid(z[i] @ w + b)
                    nrm = n2[i].sum()
                    w += prm['eta'] * (err[:, None] * z[i]).sum(0) / nrm; b += prm['eta'] * err.sum() / nrm
            pred = ((Ze @ w + b) >= 0).cpu().numpy()
            accs[si, e] = (pred[inf] == sg[inf]).mean()
    return accs, evs


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    commit, dirty = git_info(); t0 = time.time()
    out = {'source': 'results_evo/record (агент A15, lm pretrained c, 26 жизней × 100 эпизодов), оценка на эпизодах 50–99, информированные такты',
           'git_commit': commit, 'git_dirty': dirty}
    for name, prm in (('A17', {'eta': 1.999609452367605, 'B': 50, 'K': 10}), ('A15', {'eta': 1.0, 'B': 50, 'K': 3})):
        acc, ev = curve(prm, dev)
        m = acc.mean(0); evm = ev.mean(0)
        hit = np.nonzero(m >= 0.85)[0]
        e85 = int(hit[0]) + 1 if len(hit) else None
        c85 = float(evm[hit[0]]) if len(hit) else None
        per_seed = [int(np.nonzero(a >= 0.85)[0][0]) + 1 if (a >= 0.85).any() else None for a in acc]
        c0 = c85 / math.log(1 / 0.3) if c85 else None
        out[name] = {'params': prm, 'acc_mean_by_episode': m.tolist(), 'acc_sd_by_episode': acc.std(0).tolist(), 'events_mean_cum': evm.tolist(),
                     'episodes_to_085': e85, 'events_to_085': c85, 'episodes_to_085_per_seed': per_seed, 'C0': c0}
        print(f"{name}: точность по эпизодам 1, 2, 3, 5, 10, 20, 50: " + ' '.join(f'{m[i]:.3f}' for i in (0, 1, 2, 4, 9, 19, 49))
              + f" | до 0,85 — эпизодов {e85}, событий {c85}; по посевам {per_seed} | C₀ = {c0}", flush=True)
    out['sec'] = round(time.time() - t0)
    os.makedirs('results_sleep2', exist_ok=True)
    json.dump(out, open('results_sleep2/calib_c0.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
