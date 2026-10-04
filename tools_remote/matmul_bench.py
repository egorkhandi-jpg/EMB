"""Пропускная способность fp32: умножение матриц n×n (по умолчанию 8192, 50 повторов), без TF32.
python tools_remote/matmul_bench.py [--n 8192] [--reps 50]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import stand_gpu  # noqa: F401  — fp32 без TF32

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=8192); ap.add_argument('--reps', type=int, default=50)
a = ap.parse_args()
x = torch.randn(a.n, a.n, device='cuda'); y = torch.randn(a.n, a.n, device='cuda')
for _ in range(3):
    x @ y
torch.cuda.synchronize(); t = time.time()
for _ in range(a.reps):
    z = x @ y
torch.cuda.synchronize(); dt = time.time() - t
print(f'{torch.cuda.get_device_name(0)}: {2 * a.n ** 3 * a.reps / dt / 1e12:.1f} TFLOPS fp32 ({a.reps} × {a.n}², {dt:.2f} с, '
      f'precision {torch.get_float32_matmul_precision()}, allow_tf32 {torch.backends.cuda.matmul.allow_tf32})')
