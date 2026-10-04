"""Сводка v4-copies (H1, H2, описательно) по пререгистрации — docs/RESULTS_2026-10-04_copies.md."""
import itertools, json, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
D = sys.argv[1] if len(sys.argv) > 1 else 'results_copies'
C = ['s1', 's2', 'h_hi', 'h_lo', 'dense']
ENV = {'s1': 'base', 's2': 'base', 'h_hi': 'hi', 'h_lo': 'lo', 'dense': 'dense'}
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
DIFF = [(i, j) for i, j in PAIRS if ENV[C[i]] != ENV[C[j]]]
VAR = [(i, j) for i, j in PAIRS if i >= 2 and j >= 2]
CPN = ['жизнь 1', 'жизнь 2', 'жизнь 3', 'возврат']


def perm_paired(d):
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


ck = json.load(open(f'{D}/check.json')) if os.path.exists(f'{D}/check.json') else None
if ck:
    print(f"Сверка (посев {ck['seed']}): пакет − по одной {np.round(ck['beh_batch_vs_seq'], 4).tolist()}, 1 − cos {np.round(ck['one_minus_cos_batch_vs_seq'], 6).tolist()}; "
          f"D(s1, s2) {ck['D_s1s2_batch']:.4f}, порог {ck['threshold']:.4f} → {'пройдена' if ck['pass'] else 'НЕ пройдена'}")

for tag in ('pretrained_c', 'random_c'):
    p = f'{D}/copies_{tag}.json'
    if not os.path.exists(p):
        continue
    R = json.load(open(p))['results']
    print(f'\n=== {tag}, посевов {len(R)} ===')
    print('родитель: онлайн {:.3f}, вне контура {:.3f}, в контуре {:.3f} (тест {:.3f})'.format(
        np.mean([r['parent_online'] for r in R]), np.mean([r['parent_cp']['out_of_loop'] for r in R]),
        np.mean([r['parent_cp']['in_loop']['choice_informed'] for r in R]), np.mean([r['parent_cp']['in_loop']['test'] for r in R])))
    stat = {}
    for key, lab in (('beh', 'поведенческое (все живые такты)'), ('beh_informed', 'поведенческое (информ. такты)'), ('one_minus_cos', '1 − cos весов')):
        print(f'  {lab}: D_env / D_ss / D_вар-вар по контрольным точкам')
        for c in range(4):
            de = [np.mean([r['checkpoints'][c][key][i][j] for i, j in DIFF]) for r in R]
            ds = [r['checkpoints'][c][key][0][1] for r in R]
            dv = [np.mean([r['checkpoints'][c][key][i][j] for i, j in VAR]) for r in R]
            print(f'    {CPN[c]:8s}: {np.mean(de):.4f} / {np.mean(ds):.4f} / {np.mean(dv):.4f}  Δ(env − ss) {np.mean(np.subtract(de, ds)):+.4f} (выше в {sum(x > y for x, y in zip(de, ds))}/{len(R)})')
            if key == 'beh':
                stat[c] = np.subtract(de, ds)
    if len(R) >= 8:
        ps = [perm_paired(stat[2]), perm_paired(stat[3])]; adj = holm(ps)
        print('  ГИПОТЕЗЫ (Холм):')
        for nm, d, p_, a_ in zip(('H1 после расхождения', 'H2 после возврата'), (stat[2], stat[3]), ps, adj):
            print(f'    {nm:22s} Δ = {np.mean(d):+.4f}  p = {p_:.4f}  p(Холм) = {a_:.4f}  (выше в {sum(x > 0 for x in d)}/{len(d)})')
    print('  1 − cos с родителем (по копиям):')
    for c in range(4):
        print(f'    {CPN[c]:8s}: ' + ', '.join(f"{C[k]} {np.mean([r['checkpoints'][c]['one_minus_cos_parent'][k] for r in R]):.4f}" for k in range(5)))
    print('  пары копий после жизни 3 / после возврата (поведенческое):')
    for i, j in PAIRS:
        print(f"    {C[i]}–{C[j]}: {np.mean([r['checkpoints'][2]['beh'][i][j] for r in R]):.4f} / {np.mean([r['checkpoints'][3]['beh'][i][j] for r in R]):.4f}")
    print('  качество: вне контура / в контуре (выбор, тест) по копиям')
    for c in range(4):
        print(f'    {CPN[c]:8s}: ' + '; '.join(f"{C[k]} {np.mean([r['checkpoints'][c]['out_of_loop'][k] for r in R]):.3f} / "
                                              f"{np.mean([r['checkpoints'][c]['in_loop'][k]['choice_informed'] for r in R]):.3f}, {np.mean([r['checkpoints'][c]['in_loop'][k]['test'] for r in R]):.3f}" for k in range(5)))
    sp = [np.mean([r['checkpoints'][2]['in_loop'][k]['test'] for k in (2, 3, 4)]) - np.mean([r['checkpoints'][2]['in_loop'][k]['test'] for k in (0, 1)]) for r in R]
    so = [np.mean([r['checkpoints'][2]['out_of_loop'][k] for k in (2, 3, 4)]) - np.mean([r['checkpoints'][2]['out_of_loop'][k] for k in (0, 1)]) for r in R]
    print(f'  специализация после жизни 3 (варианты − s): тест в контуре {np.mean(sp):+.3f} (p {perm_paired(sp):.3f}), вне контура {np.mean(so):+.3f} (p {perm_paired(so):.3f})')
    print('  онлайн-выбор в жизнях (жизнь 1, 2, 3, возврат):')
    for k in range(5):
        print(f"    {C[k]:6s}: " + ' '.join(f"{np.mean([r['online'][ph][k]['online_informed'] for r in R]):.3f}" for ph in range(4)))
