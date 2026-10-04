"""Согласие машин: скрытые состояния Pythia-160M (pretrained, слова c, окно 32) на фиксированном входе.
Вход — 40 тактов слепой политики в 16 эпизодах flip-2 (посевы 999000+k), всё детерминировано numpy.
Пишет raw h и стандартизованное z (стандартизация — fit_std, как в стенде) в npz; сравнение — --compare a.npz b.npz.

python tools_remote/agree_features.py --out feats_laptop.npz
python tools_remote/agree_features.py --compare feats_laptop.npz feats_vast.npz
"""
import argparse, os, sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--compare', nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        x, y = np.load(a.compare[0]), np.load(a.compare[1])
        for k in ('h', 'z', 'mu', 'sd'):
            d = np.abs(x[k] - y[k])
            print(f'{k}: max |Δ| {d.max():.3e}, средн. |Δ| {d.mean():.3e}, max |значение| {np.abs(x[k]).max():.3e}, '
                  f'max отн. {(d / (np.abs(x[k]) + 1e-6)).max():.3e}')
        return
    import torch
    import stand_gpu  # noqa: F401  — fp32 без TF32
    import body_head as bh, env_flip as ef, stand_words as sw
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    feat = bh.make_feat(SimpleNamespace(cond='lm', model='EleutherAI/pythia-160m', weights='pretrained', words='c', device=dev), 1, 32)
    n = 16
    b = ef.FlipBodies(n, [999000 + k for k in range(n)], True, 'flip-2'); feat.reset(n)
    H = []
    for _ in range(40):
        s = b.sense()
        H.append(feat.features(s).cpu().numpy())
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(n)], 1)); act[~b.alive] = 0
        b.step(act)
    H = np.stack(H); mu, sd = feat.mu.cpu().numpy(), feat.sd.cpu().numpy()
    np.savez(a.out, h=H, z=(H - mu) / sd, mu=mu, sd=sd, torch=torch.__version__, device=torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu',
             matmul_precision=torch.get_float32_matmul_precision())
    print('записано', a.out, H.shape, torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', torch.get_float32_matmul_precision())


if __name__ == '__main__':
    main()
