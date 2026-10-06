"""A27, описательно (в пререгистрации — «зависимость от n, правдивые окна, 6.9B c, 1B s»; в семью не входит):
сдвиги к опыту «исходы − переставленные» по условиям и n с точными парными перестановками по 10 наборам, без поправки;
относительный размер — сдвиг / |d на переставленных|.

python synth_descr.py results_synth > results_synth/descr.txt
"""
import itertools, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_synth'
SIGNS = np.array(list(itertools.product([1, -1], repeat=10)))
perm_p = lambda x: float(((SIGNS * np.asarray(x)).mean(1) >= np.mean(x) - 1e-12).mean())

for m, f in (('6.9b', 's'), ('6.9b', 'c'), ('1b', 's')):
    r = json.load(open(os.path.join(D, f'synth_{m}.json'), encoding='utf-8'))['sets']
    print(f'=== {m} [{f}] (q1; сдвиг = исходы − переставленные; p — односторонний, без поправки)')
    for wts in ('pretrained', 'random'):
        S = lambda v, k, n=None: np.array([(r[f'{wts}/{f}/{i}']['measures'][v]['by_n'][n] if n else r[f'{wts}/{f}/{i}']['measures'][v])[k] for i in range(1, 11)])
        row = []
        for k, nm in (('d_lying', 'лож'), ('d_truthful', 'правд'), ('d_neutral', 'нейтр'), ('auc_neutral', 'AUC нейтр')):
            x = S('q1', k) - S('q1_perm', k)
            rel = f", {x.mean() / abs(S('q1_perm', k).mean()) * 100:.0f} % от |d перест|" if k in ('d_lying', 'd_truthful') else ''
            row.append(f"{nm} {x.mean():+.3f} (p {perm_p(x):.4f}{rel})")
        print(f"  {wts[:4]}: " + '; '.join(row))
        if wts == 'pretrained':
            for k, nm in (('d_lying', 'сдвиг лож'), ('auc_neutral', 'сдвиг AUC нейтр'), ('frac_lying_to_experience', 'доля лживых к опыту (исходы − перест.)')):
                print(f"     {nm} по n: " + ' | '.join(f"n={n}: {(S('q1', k, n) - S('q1_perm', k, n)).mean():+.3f} (p {perm_p(S('q1', k, n) - S('q1_perm', k, n)):.3f})" for n in ('2', '4', '6')))
