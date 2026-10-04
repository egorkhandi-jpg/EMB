"""Сводка опыта со слотом (вариант F) по пререгистрации — docs/RESULTS_2026-10-02_slot.md, дополнение 2.
P1: slot_organ − noslot > 0; P2: slot_organ − slot_random > 0 (односторонние); P3: slot_organ − fast (двусторонний);
вне контура, k > 4; точные парные перестановки по 10 посевам, Холм по трём.

python summarize_slot.py [results_slot]
"""
import itertools, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_slot'


def perm(d, side):
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d)))); m = (s * d).mean(1); o = d.mean()
    return float((np.abs(m) >= abs(o) - 1e-12).mean()) if side == 0 else float((side * m >= side * o - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


for w in ('pretrained', 'random'):
    p = os.path.join(D, f'slot_{w}.json')
    if not os.path.exists(p):
        continue
    r = json.load(open(p, encoding='utf-8'))
    print(f"=== {w} | прогоны: {[(x['git_commit'][:8], x['git_dirty']) for x in r['runs']]}")
    M = {}
    for x in r['results']:
        M.setdefault(x['agent'], {})[x['seed']] = x
    for ag, X in M.items():
        v = list(X.values())
        g = lambda wh, k: float(np.mean([x[wh][k] for x in v if x[wh][k] is not None]))
        print(f"  {ag:12s} вне контура: k>4 {g('out_of_loop', 'k_gt4'):.3f}  k≤4 {g('out_of_loop', 'k_le4'):.3f} | "
              f"в контуре: k>4 {g('in_loop', 'k_gt4'):.3f}  k≤4 {g('in_loop', 'k_le4'):.3f}  тест {g('in_loop', 'test'):.3f}  "
              f"пища/яд {g('in_loop', 'food'):.2f}/{g('in_loop', 'poison'):.2f}  (n = {len(v)})")
    if w == 'pretrained' and all(a in M for a in ('slot_organ', 'noslot', 'slot_random', 'fast')):
        for where in ('out_of_loop', 'in_loop'):
            ss = sorted(set(M['slot_organ']) & set(M['noslot']) & set(M['slot_random']) & set(M['fast']))
            q = lambda a: np.array([M[a][s][where]['k_gt4'] for s in ss])
            ds = [q('slot_organ') - q('noslot'), q('slot_organ') - q('slot_random'), q('slot_organ') - q('fast')]
            ps = [perm(ds[0], 1), perm(ds[1], 1), perm(ds[2], 0)]; adj = holm(ps)
            lab = 'ОСНОВНАЯ СЕМЬЯ (вне контура, k > 4)' if where == 'out_of_loop' else 'то же в контуре (описательно)'
            print(f'  {lab}, Холм по трём:')
            for nm, d, p_, a_ in zip(('P1 slot_organ − noslot (одност.)', 'P2 slot_organ − slot_random (одност.)', 'P3 slot_organ − fast (двуст.)'), ds, ps, adj):
                print(f'    {nm:38s} Δ = {d.mean():+.4f}  выше в {int((d > 0).sum())}/{len(d)}  p = {p_:.4f}  p(Холм) = {a_:.4f}')
