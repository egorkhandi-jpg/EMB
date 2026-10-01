"""Механистический анализ агента.
1) чувствительность скрытого состояния ко входу по слоям (сырое и после стандартизации);
2) косинус векторов проектора к ближайшему токену против случайных направлений и пар токен-токен.
python analyze_gpu.py --model EleutherAI/pythia-70m --weights pretrained --theta results/theta_pythia-70m_pretrained_c_s1.npy"""
import argparse, numpy as np, torch
from stand_gpu import Bodies, LMPolicy
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='EleutherAI/pythia-70m'); ap.add_argument('--weights', default='pretrained')
ap.add_argument('--theta', default=None); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
dtype = torch.float32  # как в stand_gpu.py с 2026-10-01; fp16 давал риск тихого NaN
pol = LMPolicy(a.model, a.weights, False, a.device, dtype, 256, -1)
th = pol.init(np.random.RandomState(0)) if a.theta is None else np.load(a.theta)
def cosm(h):
    hn = h / h.norm(dim=1, keepdim=True); n = len(h); return ((hn @ hn.T).sum() - n) / (n * (n - 1))
b = Bodies(20, [777000 + k for k in range(20)]); S = []; r = np.random.RandomState(1)
for _ in range(20): S.append(b.sense()); b.step(r.uniform(-1, 1, (20, 2)))
S = torch.as_tensor(np.concatenate(S), device=a.device)
TH = torch.as_tensor(np.repeat(th[None], len(S), 0), device=a.device)
v = pol.project(TH, S)
with torch.no_grad():
    hs = pol.lm(inputs_embeds=v.to(dtype), output_hidden_states=True).hidden_states
print('== чувствительность (средний косинус между состояниями при разных входах; 1 = вход не влияет) ==')
half = len(S) // 2
for L in range(1, len(hs)):
    h = hs[L][:, -1].float(); ref, test = h[:half], h[half:]
    print(f'слой {L:2d}: сырое {cosm(test):.3f} | центрированное {cosm((test - ref.mean(0)) / (ref.std(0) + 1e-6)):.3f}')
E = pol.lm.get_input_embeddings().weight.float(); En = E / E.norm(dim=1, keepdim=True)
idx = torch.randperm(E.shape[0])[:1000]; tt = En[idx] @ En.T; tt[torch.arange(1000), idx] = -1
rnd = torch.randn(1000, pol.D, device=E.device); rnd = rnd / rnd.norm(dim=1, keepdim=True)
vn = v.reshape(-1, pol.D).float(); vn = vn / vn.norm(dim=1, keepdim=True)
print('== косинус к ближайшему токену ==')
print(f'проектор {(vn @ En.T).max(1).values.mean():.3f} | случайные направления {(rnd @ En.T).max(1).values.mean():.3f} | токен-токен {tt.max(1).values.mean():.3f}')
