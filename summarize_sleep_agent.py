"""Сводка «Настоящий агент в S2» по пререгистрации (docs/RESULTS_2026-10-04_sleep.md, с дополнением 2026-10-04).
P1: F(θ = 6, k = 5) > F(онлайн без сна); P2: F(θ = 6) > F(θ = 1). Парные по 200 жизням, односторонние, Монте-Карло
100 000 перестановок знаков, Холм по двум.

python summarize_sleep_agent.py results_sleep_agent > results_sleep_agent/summary.txt
"""
import glob, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_sleep_agent'
R = {json.load(open(f, encoding='utf-8'))['tag']: json.load(open(f, encoding='utf-8')) for f in glob.glob(os.path.join(D, '*.json'))}
F = {k: np.array(v['f']) for k, v in R.items()}


def perm_one(d, n=100000, seed=0):
    d = np.asarray(d, float); rng = np.random.RandomState(seed); obs = d.mean()
    s = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float(((s * d).mean(1) >= obs - 1e-12).mean() * n + 1) / (n + 1)


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def row(tag):
    v = R[tag]
    return (f"{tag:26s} F {v['f_mean']:.4f} ± {np.std(v['f']) / np.sqrt(len(v['f'])):.4f} | сон {v['sleep_frac']:.3f} | снов {v['n_sleeps']:.1f} | "
            f"выбор {v['acc_awake']:.3f} (медл. {v['acc_slow']:.3f}) | пища/яд {v['food']:.1f}/{v['poison']:.1f} | на нуле {v['floor_frac']:.3f} | "
            f"{v.get('git_commit', '')[:8]}{' ГРЯЗН' if v.get('git_dirty') else ''}")


print('Ориентиры:')
for t in ('oracle_thNone_k5', 'online_thNone_k5', 'free_th2_k5', 'never_thNone_k5'):
    if t in R:
        print('  ' + row(t))
for k in (5, 10):
    print(f'\nКривая F(θ), k = {k}:')
    for th in (1, 2, 4, 6, 8, 16):
        t = f'pressure_th{th}_k{k}'
        if t in R:
            print('  ' + row(t))
print('\nБыстрые веса (описательно):')
for tau in (10, 20):
    for t in (f'never_thNone_k5_tau{tau}', f'pressure_th1_k5_tau{tau}', f'pressure_th6_k5_tau{tau}'):
        if t in R:
            print('  ' + row(t))
if all(t in F for t in ('pressure_th6_k5', 'online_thNone_k5', 'pressure_th1_k5')):
    d1 = F['pressure_th6_k5'] - F['online_thNone_k5']; d2 = F['pressure_th6_k5'] - F['pressure_th1_k5']
    ps = [perm_one(d1, seed=1), perm_one(d2, seed=2)]; adj = holm(ps)
    print('\nОСНОВНАЯ СЕМЬЯ (односторонние, Холм по двум):')
    for nm, d, p, a in zip(('P1: F(θ=6) − F(онлайн без сна)', 'P2: F(θ=6) − F(θ=1)'), (d1, d2), ps, adj):
        print(f'  {nm:34s} Δ = {d.mean():+.4f} (z {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):.2f}, лучше в {int((d > 0).sum())}/{len(d)})  p = {p:.5f}  p(Холм) = {a:.5f}')
    curve = {th: F[f'pressure_th{th}_k5'].mean() for th in (1, 2, 4, 6, 8, 16) if f'pressure_th{th}_k5' in F}
    best = max(curve, key=curve.get)
    print(f'  максимум кривой k = 5: θ = {best} ({curve[best]:.4f}); онлайн без сна {F["online_thNone_k5"].mean():.4f}; не спать {F["never_thNone_k5"].mean():.4f}' if 'never_thNone_k5' in F else '')
print('\nКачество медленного считывания по сезонам (θ = 6, k = 5; онлайн без сна):')
for t in ('pressure_th6_k5', 'online_thNone_k5'):
    if t in R:
        print(f"  {t}: " + ' '.join(f'{s}:{a:.2f}' for s, a in list(R[t]['acc_slow_by_season'].items())[:8]))
