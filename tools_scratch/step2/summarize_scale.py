"""Сводка шага 1 (шкала размеров): python summarize_scale.py [results_scale]
Таблицы точности по размерам × признакам × N, основная семья по пререгистрации (Холм:
пред − случ при N = 32 для каждого размера + тренд по log10 числа параметров), сверка повтора
70M с П3 (results_probe/probe.json)."""
import itertools, json, os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else 'results_scale'
SIZES = [('pythia-70m', 7.0e7), ('pythia-160m', 1.6e8), ('pythia-410m', 4.1e8), ('pythia-1b', 1.0e9)]
FEATS = ['raw_current', 'count', 'bilinear', 'random_mlp', 'mlp_std', 'quad', 'lm_random_last', 'lm_pretrained_last',
         'lm_random_mid', 'lm_pretrained_mid']
R = {}
for m, _ in SIZES:
    p = os.path.join(d, f'probe_{m}.json')
    if os.path.exists(p):
        R[m] = json.load(open(p))


def vals(m, env, feat, N, target='sign'):
    return {x['seed']: x['bacc'] for x in R[m]['results']
            if x['env'] == env and x['features'] == feat and x['N'] == N and x['target'] == target}


def perm(dd):
    dd = np.asarray(dd, float); obs = abs(dd.mean())
    s = np.array(list(itertools.product([1, -1], repeat=len(dd))))
    return float((np.abs((s * dd).mean(1)) >= obs - 1e-12).mean())


def cell(v):
    v = np.array(list(v.values()))
    if len(v) == 0:
        return '—'
    return f'{v.mean():.3f}' if len(v) == 1 else f'{v.mean():.3f} ± {v.std():.3f} (n={len(v)})'


for env in ('flip-2_rev', 'flip-1_const'):
    for target in ('sign', 'edible'):
        print(f'\n=== {env}, цель {target}')
        for m, _ in SIZES:
            if m not in R:
                continue
            print(f'-- {m} (код {R[m]["git_commit"][:8]}{", грязное" if R[m]["git_dirty"] else ""})')
            for f in FEATS:
                row = [cell(vals(m, env, f, 1 if f == 'raw_current' else N, target)) for N in (2, 8, 32)]
                if f == 'raw_current':
                    row = [row[0], '', '']
                print(f'   {f:20s} ' + ' | '.join(f'{c:>24s}' for c in row))

# прирост LM над контролями, flip-2, N = 32, знак
print('\n=== прирост LM над контролями (flip-2, N = 32, знак, последний слой; среднее по посевам)')
for m, _ in SIZES:
    if m not in R:
        continue
    c = np.mean(list(vals(m, 'flip-2_rev', 'count', 32).values()))
    out = []
    for lmw in ('random', 'pretrained'):
        L = vals(m, 'flip-2_rev', f'lm_{lmw}_last', 32)
        for ctrl in ('quad', 'mlp_std'):
            C = vals(m, 'flip-2_rev', ctrl, 32)
            s = sorted(set(L) & set(C))
            out.append(f'{lmw} − {ctrl} {np.mean([L[x] - C[x] for x in s]):+.3f}')
        out.append(f'{lmw} − count {np.mean(list(L.values())) - c:+.3f}')
    print(f'   {m:12s} ' + '; '.join(out))

# сверка повтора 70M с П3
if 'pythia-70m' in R and os.path.exists('results_probe/probe.json'):
    old = json.load(open('results_probe/probe.json'))['results']
    diffs = []
    for x in R['pythia-70m']['results']:
        if x['features'].startswith('lm_'):
            o = [y['bacc'] for y in old if all(y[k] == x[k] for k in ('env', 'seed', 'features', 'N', 'target'))]
            if o:
                diffs.append(abs(o[0] - x['bacc']))
    print(f'\nсверка повтора 70M с П3: {len(diffs)} совпавших ячеек LM, макс. |Δ| = {max(diffs):.4f}, средн. |Δ| = {np.mean(diffs):.4f}')

# основная семья
print('\n=== основная семья (flip-2, знак, последний слой, N = 32; Холм)')
tests, slopes_by_seed = [], {}
for m, n in SIZES:
    if m not in R:
        continue
    P, Q = vals(m, 'flip-2_rev', 'lm_pretrained_last', 32), vals(m, 'flip-2_rev', 'lm_random_last', 32)
    s = sorted(set(P) & set(Q))
    dd = [P[x] - Q[x] for x in s]
    tests.append((f'пред − случ, {m}', dd))
    for x in s:
        slopes_by_seed.setdefault(x, []).append((np.log10(n), P[x] - Q[x]))
sl = [np.polyfit([a for a, _ in v], [b for _, b in v], 1)[0] for v in slopes_by_seed.values() if len(v) == len(R)]
if len(R) >= 2:
    tests.append((f'тренд по log10 размера ({len(R)} размеров), наклон на декаду', sl))
ps = [perm(dd) for _, dd in tests]
order = np.argsort(ps); adj = np.zeros(len(ps)); run = 0
for rank, i in enumerate(order):
    run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
for (nm, dd), p, pa in zip(tests, ps, adj):
    print(f'   {nm:58s} Δ = {np.mean(dd):+.4f} ± {np.std(dd):.4f} (n={len(dd)})  p = {p:.4f}  p(Холм) = {pa:.4f}')
if tests:
    big = [i for i, (nm, _) in enumerate(tests) if 'pythia-1b' in nm]
    tr = len(tests) - 1
    ok = (len(R) == 4 and np.mean(tests[tr][1]) > 0 and adj[tr] < 0.05 and big and np.mean(tests[big[0]][1]) > 0 and adj[big[0]] < 0.05)
    print('правило вывода «преимущество знаний появляется с размером»:', 'ВЫПОЛНЕНО' if ok else 'не выполнено')
