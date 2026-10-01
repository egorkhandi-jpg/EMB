"""Замер: насколько 16-битный тип портит стандартизованное скрытое состояние.
Стандартизация растягивает малую входозависимую часть h вместе с ошибкой округления,
поэтому ошибка у предобученной модели может быть на порядок больше, чем у случайной
(2026-10-01, Pythia-70M: bf16 23,6 % против 1,4 %). Запускать перед серией в 16 битах.
python dtype_check.py --model Qwen/Qwen2.5-1.5B [--dtypes bfloat16 float16]"""
import argparse, time, numpy as np, torch
from stand_gpu import LMPolicy, Bodies
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='EleutherAI/pythia-70m')
ap.add_argument('--dtypes', nargs='+', default=['bfloat16', 'float16'])
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
S = []; b = Bodies(32, [777000 + k for k in range(32)]); r = np.random.RandomState(1)
for _ in range(8): S.append(b.sense()); b.step(r.uniform(-1, 1, (32, 2)))
S = torch.as_tensor(np.concatenate(S), device=a.device)
for w in ['pretrained', 'random']:
    ref = None
    for name in ['float32'] + a.dtypes:
        dt = getattr(torch, name)
        try:
            pol = LMPolicy(a.model, w, True, a.device, dt, 256, -1, net_seed=1)
            th = pol.init(np.random.RandomState(0))
            TH = torch.as_tensor(np.repeat(th[None], len(S), 0), device=a.device)
            h = (pol._hidden(pol.project(TH, S)) - pol.mu) / pol.sd
        except (FloatingPointError, torch.OutOfMemoryError) as e:
            print(f'{w:10s} {name:9s} ОШИБКА: {e}'); continue
        if ref is None: ref = h
        err = ((h - ref).norm(dim=1) / ref.norm(dim=1)).mean().item()
        if a.device == 'cuda': torch.cuda.synchronize()
        t = time.time()
        for _ in range(20): pol._hidden(pol.project(TH[:64], S[:64]))
        if a.device == 'cuda': torch.cuda.synchronize()
        print(f'{w:10s} {name:9s} ошибка стандартизованного h к fp32: {err:7.4f}   проход x64: {(time.time() - t) / 20 * 1000:6.1f} мс')
        del pol; torch.cuda.empty_cache() if a.device == 'cuda' else None
