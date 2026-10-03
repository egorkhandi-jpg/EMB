"""Сводка v3-replay (п. 1а, 1б, 1в, 2, 3) по пререгистрации — docs/RESULTS_2026-10-01_flip.md."""
import glob, itertools, json, os
import numpy as np

D, A11 = 'results_bodyhead3', 'results_bodyhead2'
CONDS = ['pretrained_c', 'random_c', 'pretrained_c_arb1', 'pretrained_c_arb2', 'quad', 'raw']
VARS = ['event_full', 'all_full', 'all_pca', 'all_dual']
ms = lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}' if len(x) else '—'
load = lambda p: json.load(open(p)) if os.path.exists(p) else None


def perm_paired(d):
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def a11_online(cond, var):
    src = A11 if cond != 'pretrained_c_arb2' else D
    r = load(f'{src}/result_{cond}_{var}.json')
    return {x['seed']: x for x in r['results']} if r else {}


print('П. 1а. Замороженная голова: в контуре (A11 или пересчёт при неточном повторе) / вне контура; повтор точный')
for c in CONDS:
    for v in VARS:
        r = load(f'{D}/repro_{c}_{v}.json')
        if not r:
            continue
        R = r['results']
        inl = [x.get('frozen_test_now', {}).get('choice_informed', x['frozen_test_a11']) for x in R]
        out = [x['frozen_out_of_loop']['acc'] for x in R]
        exact = sum(abs(x['repro']['life_online_now'] - x['repro']['life_online_a11']) <= 0.005 and
                    abs(x['repro']['w_norm_now'] / x['repro']['w_norm_a11'] - 1) <= 0.01 for x in R)
        line = f'  {c:18s} {v:10s} в контуре {ms(inl)} | вне {ms(out)} | Δ {np.mean(out) - np.mean(inl):+.3f} | точных {exact}/{len(R)}'
        if v == 'all_dual':
            line += (f" || онлайн на тесте: в контуре {ms([x['online_test_in_loop']['choice_informed'] for x in R])}, "
                     f"вне {ms([x['online_test_out_of_loop']['acc'] for x in R])}")
        print(line)

print('\nП. 1б. Только быстрая часть: онлайн жизни / на тесте в контуре / вне контура; dual A11 онлайн жизни; кривые (по числу событий жизни)')
for c in CONDS:
    r = load(f'{D}/fastonly_{c}_fastonly.json')
    if not r:
        continue
    R = r['results']; du = a11_online(c, 'all_dual')
    fo = {x['seed']: x['life_online']['choice_informed'] for x in R}
    dd = [fo[s] - du[s]['life_online']['choice']['informed'] for s in fo if s in du]
    cv = lambda items, key: np.nanmean(np.array([[q[1] if q[1] is not None else np.nan for q in it[key]] for it in items], float), 0)
    print(f"  {c:18s} {ms(list(fo.values()))} / {R[0]['online_test_in_loop']['choice_informed']:.3f} / {R[0]['online_test_out_of_loop']['acc']:.3f}"
          f" | dual {ms([du[s]['life_online']['choice']['informed'] for s in fo if s in du])} | быстрая − dual {np.mean(dd) if dd else np.nan:+.3f} "
          f"(быстрая выше в {sum(d > 0 for d in dd)}/{len(dd)})")
    print('      кривая быстрой: ' + ' '.join(f'{v:.2f}' for v in cv(R, 'curve')))
    if du:
        print('      кривая dual   : ' + ' '.join(f'{v:.2f}' for v in cv(list(du.values()), 'curve')))

print('\nП. 1в. c_arb2, dual (A11-протокол): замороженная / онлайн жизни; для сравнения c_arb1')
for c, src in (('pretrained_c_arb1', A11), ('pretrained_c_arb2', D)):
    r = load(f'{src}/result_{c}_all_dual.json')
    if r:
        print(f"  {c:18s} η {r['eta']}: замороженная {ms([x['frozen_test']['choice']['informed'] for x in r['results']])}, "
              f"онлайн жизни {ms([x['life_online']['choice']['informed'] for x in r['results']])}")

print('\nП. 2. Пакетная регрессия на парах жизни (all, full): вне контура (главное) / в контуре; оракул-метки вне контура; CV; метки = знак')
dec = None
for c in ['pretrained_c', 'random_c', 'pretrained_c_arb1', 'quad', 'raw']:
    r = load(f'{D}/repro_{c}_all_full.json')
    if not r or 'batch_body' not in r['results'][0]:
        continue
    R = r['results']
    bo = [x['batch_body']['out_of_loop']['acc'] for x in R]
    print(f"  {c:18s} тело: вне {ms(bo)} / в контуре {ms([x['batch_body']['in_loop']['choice_informed'] for x in R])} "
          f"(CV {np.mean([x['batch_body']['cv_bacc'] for x in R]):.3f}) | оракул: вне {ms([x['batch_oracle']['out_of_loop']['acc'] for x in R])} "
          f"(CV {np.mean([x['batch_oracle']['cv_bacc'] for x in R]):.3f}) | онлайн-голова вне {ms([x['frozen_out_of_loop']['acc'] for x in R])} "
          f"| метки = знак: все {np.mean([x['labels']['agree_all'] for x in R]):.3f}, информ. {np.mean([x['labels']['agree_informed'] for x in R]):.3f}, "
          f"неинформ. {np.nanmean([np.nan if x['labels']['agree_uninformed'] is None else x['labels']['agree_uninformed'] for x in R]):.3f}; пар {np.mean([x['batch_body']['n_pairs'] for x in R]):.0f}")
    if c == 'pretrained_c':
        m = np.mean(bo); mo = np.mean([x['batch_oracle']['out_of_loop']['acc'] for x in R])
        dec = 'виновато правило → п. 3' if m >= 0.85 else ('плох учитель' + (' (и оракул ≤ 0,70: распределение пар)' if mo <= 0.70 else '') if m <= 0.70 else 'промежуточно → СТОП')
        print(f'  ПРАВИЛО (pretrained c, {len(R)} жизней): {m:.3f} → {dec}')

print('\nП. 3. Проигрывание опыта: замороженная в контуре (главная) / вне контура; онлайн жизни; онлайн на тесте в / вне контура')
rep = {}
for c in ['pretrained_c', 'random_c', 'quad', 'pretrained_c_arb1', 'raw']:
    t = load(f'{D}/tune_{c}_replay.json'); r = load(f'{D}/replay_{c}_replay.json')
    if not r:
        continue
    R = r['results']; rep[c] = {x['seed']: x['frozen_test']['choice_informed'] for x in R}
    print(f"  {c:18s} η {r['eta']} B {r['B']} (подбор {t['best']['mean']:.3f}): {ms(list(rep[c].values()))} / вне {ms([x['frozen_out_of_loop']['acc'] for x in R])} "
          f"| онлайн жизни {ms([x['life_online']['choice_informed'] for x in R])} | на тесте {ms([x['online_test_in_loop']['choice_informed'] for x in R])} / "
          f"вне {ms([x['online_test_out_of_loop']['acc'] for x in R])}")
if all(k in rep for k in ('pretrained_c', 'random_c', 'quad')):
    du = {s: x['frozen_test']['choice']['informed'] for s, x in a11_online('pretrained_c', 'all_dual').items()}
    P, Rr, Q = rep['pretrained_c'], rep['random_c'], rep['quad']
    ss = sorted(set(P) & set(Rr) & set(Q) & set(du))
    ds = [[P[s] - du[s] for s in ss], [P[s] - Rr[s] for s in ss], [P[s] - Q[s] for s in ss]]
    ps = [perm_paired(d) for d in ds]; adj = holm(ps)
    print('\n  ОСНОВНАЯ СЕМЬЯ (Холм):')
    for nm, d, p_, a_ in zip(('с проигрыванием − dual A11 (pretrained c)', 'pretrained c − random c', 'pretrained c − quad'), ds, ps, adj):
        print(f'    {nm:42s} Δ = {np.mean(d):+.4f}  p = {p_:.4f}  p(Холм) = {a_:.4f}  (n = {len(d)})')
