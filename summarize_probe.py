"""Сводка П3 без обучения: python summarize_probe.py [results_probe/probe.json]
Таблицы сбалансированной точности (среднее ± ст.откл по посевам), кривые по тактам после
последнего события и основная семья тестов по пререгистрации (Холм, точные парные
перестановки смен знака по 10 посевам)."""
import itertools, json, sys
import numpy as np

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results_probe/probe.json'))
R = r['results']
ORDER = ['raw_current', 'raw_window', 'bilinear', 'random_mlp', 'lm_random_last', 'lm_pretrained_last',
         'lm_random_mid', 'lm_pretrained_mid']


def vals(env, feat, N, target):
    return {x['seed']: x['bacc'] for x in R if x['env'] == env and x['features'] == feat and x['N'] == N and x['target'] == target}


def perm_paired(d):
    """Точный двусторонний тест смен знака для парных разностей."""
    d = np.asarray(d, float); obs = abs(d.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())


print(f"код {r['git_commit'][:8]}{' (грязное дерево)' if r['git_dirty'] else ''}, эпизодов {r['episodes']}, посевов {r['seeds']}, {r.get('sec', '?')} с")
for env in ('flip-2_rev', 'flip-1_const'):
    if env not in r:
        continue
    print(f"\n=== {env}: {r[env]}")
    for target in ('sign', 'edible'):
        print(f"цель {target}:")
        print(f"{'признаки':22s}" + ''.join(f'{"N=" + str(N):>16s}' for N in (2, 8, 32)))
        for feat in ORDER:
            row = f'{feat:22s}'
            for N in (2, 8, 32):
                v = vals(env, feat, 1 if feat == 'raw_current' else N, target)
                v = np.array(list(v.values()))
                row += f'{"—":>16s}' if len(v) == 0 else (f'{v.mean():10.3f}      ' if len(v) == 1 else f'{v.mean():9.3f} ± {v.std():.3f}')
            print(row)
        print(f"кривая по тактам после события, N = 32 (k = 0…10), среднее по посевам:")
        for feat in ('raw_window', 'bilinear', 'random_mlp', 'lm_random_last', 'lm_pretrained_last'):
            cs = np.array([x['curve_k'] for x in R if x['env'] == env and x['features'] == feat and x['N'] == 32 and x['target'] == target], float)
            print(f'  {feat:20s} ' + ' '.join(f'{c:.2f}' for c in np.nanmean(cs, 0)))

# основная семья
env, t = 'flip-2_rev', 'sign'
P32, R32 = vals(env, 'lm_pretrained_last', 32, t), vals(env, 'lm_random_last', 32, t)
P2, R2 = vals(env, 'lm_pretrained_last', 2, t), vals(env, 'lm_random_last', 2, t)
M32 = vals(env, 'random_mlp', 32, t)
seeds = sorted(set(P32) & set(R32) & set(P2) & set(R2) & set(M32))
tests = [('предобученная − случайная LM, N = 32', [P32[s] - R32[s] for s in seeds]),
         ('контраст (пред − случ) N = 32 против N = 2', [(P32[s] - R32[s]) - (P2[s] - R2[s]) for s in seeds]),
         ('случайная LM − случайный MLP, N = 32', [R32[s] - M32[s] for s in seeds])]
ps = [perm_paired(d) for _, d in tests]
order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
for rank, i in enumerate(order):
    run = max(run, min(1, (m - rank) * ps[i])); adj[i] = run
print(f'\nосновная семья ({env}, цель «знак», последний слой, {len(seeds)} посевов; Холм):')
for (nm, d), p, pa in zip(tests, ps, adj):
    print(f'  {nm:46s} Δ = {np.mean(d):+.4f} ± {np.std(d):.4f}   p = {p:.4f}   p(Холм) = {pa:.4f}')
pos = np.mean(tests[0][1]) > 0 and adj[0] < 0.05 and np.mean(tests[2][1]) > 0 and adj[2] < 0.05
print('критерий положительного сигнала звена 2 (пред > случ И случ > MLP при N = 32):', 'ВЫПОЛНЕН' if pos else 'не выполнен')
