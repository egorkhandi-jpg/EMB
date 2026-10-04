"""Сводка v3-evo (п. 1A, 1B, 2) по пререгистрации — docs/RESULTS_2026-10-01_flip.md."""
import itertools, json, os
import numpy as np

os.chdir(r'C:\embodied-lm-stand')
E, R3 = 'results_evo', 'results_bodyhead3'
load = lambda p: json.load(open(p)) if os.path.exists(p) else None
ms = lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}'


def perm_paired(d):
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


A = load(f'{E}/evo_A_pretrained_c.json')
if A:
    print('П. 1A. ES (центр по поколениям: η, B, K, метки; средняя / макс приспособленность популяции):')
    for h in A['hist'][::10] + [A['hist'][-1]]:
        c = h['center_dec']
        print(f"  пок. {h['gen']:2d}: η {c['eta']:.3f} B {c['B']} K {c['K']} {c['rule']} | {h['mean_f']:.4f} / {h['max_f']:.4f}")
    print(f"  итог ES: {A['es_final']} (отбор {A['es_final_sel']:.4f}); случайный поиск: {A['rs_best']} (отбор {A['rs_best_sel']:.4f})")
    rs = A['rs_all']; f = np.array([r['f'] for r in rs])
    top = np.argsort(-f)[:10]
    print('  случайный поиск, верхние 10: ' + '; '.join(f"η {rs[i]['eta']:.2f} B {rs[i]['B']} K {rs[i]['K']} {rs[i]['rule']} {f[i]:.4f}" for i in top))
    print(f"  случайный поиск: event — лучшая {f[[r['rule'] == 'event' for r in rs]].max():.4f}, all — {f[[r['rule'] == 'all' for r in rs]].max():.4f}")
    print('  тест уровня A (отложенные посевы 301–310):')
    tf = A['test_fitness']
    for k, v in tf.items():
        print(f"    {k:15s} {A['candidates'][k]} → {ms(v)}")
    d = np.array(tf['es_final']) - np.array(tf['grid_a15'])
    print(f"    ES − сетка A15: {d.mean():+.4f} (p {perm_paired(d):.4f}, ES выше в {int((d > 0).sum())}/10)")
    d = np.array(tf['es_final']) - np.array(tf['rs_best'])
    print(f"    ES − случайный поиск: {d.mean():+.4f} (p {perm_paired(d):.4f})")


def rep(path):
    r = load(path)
    return {x['seed']: x for x in r['results']} if r else {}, r


print('\nП. 1B. Контур, эволюционированное правило: замороженная в контуре / вне контура; тест, пища/яд; онлайн жизни; онлайн на тесте в/вне')
RES = {}
for c, path in (('pretrained_c (эволюция)', f'{E}/replay_pretrained_c_replay_evo.json'), ('random_c (эволюция)', f'{E}/replay_random_c_replay_evo.json'),
                ('pretrained_c (сетка A15)', f'{R3}/replay_pretrained_c_replay.json'), ('random_c (сетка A15)', f'{R3}/replay_random_c_replay.json')):
    X, r = rep(path)
    if not X:
        continue
    RES[c] = X; v = list(X.values())
    print(f"  {c:26s} η {r['eta']:.3f} B {r['B']} K {r.get('K', 3)} {r.get('rule', 'all')}: {ms([x['frozen_test']['choice_informed'] for x in v])} / "
          f"{ms([x['frozen_out_of_loop']['acc'] for x in v])} | тест {np.mean([x['frozen_test']['test'] for x in v]):.3f}, "
          f"{np.mean([x['frozen_test']['food'] for x in v]):.2f}/{np.mean([x['frozen_test']['poison'] for x in v]):.2f} | онлайн {np.mean([x['life_online']['choice_informed'] for x in v]):.3f} | "
          f"на тесте {np.mean([x['online_test_in_loop']['choice_informed'] for x in v]):.3f} / {np.mean([x['online_test_out_of_loop']['acc'] for x in v]):.3f}")
if 'pretrained_c (эволюция)' in RES and 'random_c (эволюция)' in RES:
    o = lambda c: {s: x['frozen_out_of_loop']['acc'] for s, x in RES[c].items()}
    Pe, Pg, Re = o('pretrained_c (эволюция)'), o('pretrained_c (сетка A15)'), o('random_c (эволюция)')
    ss = sorted(set(Pe) & set(Pg) & set(Re))
    ds = [[Pe[s] - Pg[s] for s in ss], [Pe[s] - Re[s] for s in ss]]
    ps = [perm_paired(d) for d in ds]; adj = holm(ps)
    print('  ОСНОВНАЯ СЕМЬЯ (вне контура, Холм):')
    for nm, d, p_, a_ in zip(('эволюция − сетка A15 (pretrained c)', 'pretrained c − random c (эволюция)'), ds, ps, adj):
        print(f'    {nm:38s} Δ = {np.mean(d):+.4f}  p = {p_:.4f}  p(Холм) = {a_:.4f}  (выше в {sum(x > 0 for x in d)}/{len(d)})')
    ii = lambda c: {s: x['frozen_test']['choice_informed'] for s, x in RES[c].items()}
    print(f"  в контуре: эволюция − сетка {np.mean([ii('pretrained_c (эволюция)')[s] - ii('pretrained_c (сетка A15)')[s] for s in ss]):+.4f}; "
          f"пред − случ {np.mean([ii('pretrained_c (эволюция)')[s] - ii('random_c (эволюция)')[s] for s in ss]):+.4f}")

print('\nП. 2. Устойчивость A15 (правило A15 по условию): замороженная в контуре / вне контура')
for nm, pre, rnd in (('c_arb3, 160M', 'pretrained_c_arb3_replay_a15', 'random_c_arb3_replay_a15'),
                     ('слова c, 410M', '410m_pretrained_c_replay_a15', '410m_random_c_replay_a15')):
    P, _ = rep(f'{E}/replay_{pre}.json'); Rr, _ = rep(f'{E}/replay_{rnd}.json')
    if not P or not Rr:
        continue
    ss = sorted(set(P) & set(Rr))
    for lab, key in (('в контуре', lambda x: x['frozen_test']['choice_informed']), ('вне контура', lambda x: x['frozen_out_of_loop']['acc'])):
        d = [key(P[s]) - key(Rr[s]) for s in ss]
        print(f"  {nm:14s} {lab:11s}: пред {ms([key(P[s]) for s in ss])}, случ {ms([key(Rr[s]) for s in ss])}, Δ {np.mean(d):+.4f} (p {perm_paired(d):.4f}, выше в {sum(x > 0 for x in d)}/{len(d)})")
    print(f"  {nm:14s} поведение пред: тест {np.mean([P[s]['frozen_test']['test'] for s in ss]):.3f}, пища/яд {np.mean([P[s]['frozen_test']['food'] for s in ss]):.2f}/{np.mean([P[s]['frozen_test']['poison'] for s in ss]):.2f}; "
          f"онлайн жизни пред {np.mean([P[s]['life_online']['choice_informed'] for s in ss]):.3f}, случ {np.mean([Rr[s]['life_online']['choice_informed'] for s in ss]):.3f}")
