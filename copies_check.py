"""Сверка пакетного и последовательного режима copies.py на одном посеве (до серии; пререгистрация —
docs/RESULTS_2026-10-04_copies.md). Сравнивает итоговые медленные веса каждой копии (после возврата), посчитанные
пакетом (5 копий в одном прямом проходе) и по одной: поведенческое расхождение на общем наборе состояний слепой политики
и 1 − косинус. Критерий: расхождение «пакет − по одной» каждой копии ≤ max(0,25 · D(s1, s2) пакета; 0,005).
Пишет results_copies/check.json с полем pass; copies.py --require-check не запускает серию без pass = true.

python copies_check.py --seed 1
"""
import argparse, json, os
from types import SimpleNamespace
import numpy as np
import torch

import body_head as bh
import stand_words as sw
from copies import COPIES, blind_set
from stand_gpu import git_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tag', default='pretrained_c_check')
    ap.add_argument('--out', default='results_copies')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    Wb = np.load(os.path.join(a.out, f'w_{a.tag}_bat_{a.seed}s1.npy'))                       # (5, d)
    Ws = np.stack([np.load(os.path.join(a.out, f'w_{a.tag}_seq_{a.seed}{c}.npy'))[0] for c in COPIES])
    rb = json.load(open(os.path.join(a.out, f'copies_{a.tag}.json')))
    feat = bh.make_feat(SimpleNamespace(cond='lm', model='EleutherAI/pythia-160m', weights='pretrained', words='c', device=a.device), a.seed, 32)
    bs = blind_set(feat, sw.TEST_SEEDS, a.device)
    bb = np.load(os.path.join(a.out, f'b_{a.tag}_bat_{a.seed}s1.npy')); bsq = np.stack([np.load(os.path.join(a.out, f'b_{a.tag}_seq_{a.seed}{c}.npy'))[0] for c in COPIES])
    pred = lambda W, b: ((torch.einsum('tnd,ld->tnl', bs['Z'], torch.as_tensor(W, device=a.device)) + torch.as_tensor(b, device=a.device)) >= 0)[bs['AL']]
    Pb, Ps = pred(Wb, bb), pred(Ws, bsq)
    d_bs = (Pb != Ps).float().mean(0).cpu().numpy()
    cos = (Wb * Ws).sum(1) / np.linalg.norm(Wb, axis=1) / np.linalg.norm(Ws, axis=1)
    D_ss = float((Pb[:, 0] != Pb[:, 1]).float().mean())
    D_ss_rec = rb['results'][0]['checkpoints'][-1]['beh'][0][1]
    thr = max(0.25 * D_ss, 0.005)
    ok = bool((d_bs <= thr).all())
    commit, dirty = git_info()
    res = {'seed': a.seed, 'beh_batch_vs_seq': d_bs.tolist(), 'one_minus_cos_batch_vs_seq': (1 - cos).tolist(), 'D_s1s2_batch': D_ss,
           'D_s1s2_recorded': D_ss_rec, 'threshold': thr, 'pass': ok, 'git_commit': commit, 'git_dirty': dirty}
    json.dump(res, open(os.path.join(a.out, 'check.json'), 'w'), ensure_ascii=False)
    print('пакет − по одной (поведенческое):', np.round(d_bs, 4).tolist(), '| 1 − cos:', np.round(1 - cos, 5).tolist())
    print(f'D(s1, s2) пакета {D_ss:.4f} (в записи {D_ss_rec:.4f}); порог {thr:.4f} → {"ПРОЙДЕНА" if ok else "НЕ ПРОЙДЕНА"}')


if __name__ == '__main__':
    main()
