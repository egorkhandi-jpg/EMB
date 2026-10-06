"""Сводка узкого вопроса C по пререгистрации (docs/RESULTS_2026-10-06_bind.md).
Основная семья — 3 модели × 2 интерфейса × 3 меры = 18 проверок на вопросе q1; в каждой проверке два условия, p проверки — больший из двух
(пересечение-объединение). Тесты — точные парные перестановки (смена знаков) по 10 наборам, односторонние; Холм по 18.
  M: M_пред > 0,5 и M_пред > M_случ;
  E: E_пред > E_случ и E_пред > F (частотный ориентир на тех же окнах);
  B: B_пред > 0 (логиты) и B_auc_пред > B_auc_случ (шкала логитов у случайной другая — сравнение в AUC).
Метод годен для модели и интерфейса, если у предобученной M_direct (прямые слова, n = 0, q1) ≥ 0,8.

python summarize_bind.py results_bind > results_bind/summary.txt
"""
import glob, itertools, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_bind'
SETS = list(range(1, 11))
SIGNS = np.array(list(itertools.product([1, -1], repeat=len(SETS))))


def perm_p(x):
    """Односторонний точный тест смены знаков: H1 — среднее x > 0."""
    x = np.asarray(x, float)
    return float(((SIGNS * x).mean(1) >= x.mean() - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def get(r, wts, v, f, key):
    return np.array([r['sets'][f'{wts}/{i}']['measures'][v][f][key] for i in SETS], float)


R = {}
for p in sorted(glob.glob(os.path.join(D, 'bind_*.json'))):
    r = json.load(open(p, encoding='utf-8')); R[os.path.basename(p)[5:-5]] = r
order = [m for m in ('1b', '2.8b', '6.9b') if m in R]
tests, valid, verdict = [], {}, {}
for m in order:
    r = R[m]
    missing = [f'{w}/{i}' for w in ('pretrained', 'random') for i in SETS if f'{w}/{i}' not in r['sets']]
    commits = {x['git_commit'][:8] + (' ГРЯЗН' if x['git_dirty'] else '') for x in r['sets'].values()}
    print(f"=== {m} | коммиты {', '.join(sorted(commits))}" + (f' | НЕ ХВАТАЕТ {missing}' if missing else ''))
    if missing:
        continue
    for f in ('c', 's'):
        g = lambda w, k, v='q1': get(r, w, v, f, k)
        Mp, Mr, Ep, Er, F, Bp, Bap, Bar = g('pretrained', 'M'), g('random', 'M'), g('pretrained', 'E'), g('random', 'E'), g('pretrained', 'F'), \
            g('pretrained', 'B'), g('pretrained', 'B_auc'), g('random', 'B_auc')
        md = g('pretrained', 'M_direct').mean(); valid[(m, f)] = md >= 0.8
        print(f"  [{f}] положительный контроль M_direct: пред {md:.3f}, случ {g('random', 'M_direct').mean():.3f} → {'метод годен' if valid[(m, f)] else 'МЕТОД НЕГОДЕН'}")
        print(f"      M  пред {Mp.mean():.3f} ± {Mp.std():.3f} | случ {Mr.mean():.3f} ± {Mr.std():.3f}")
        print(f"      E  пред {Ep.mean():.3f} ± {Ep.std():.3f} | случ {Er.mean():.3f} ± {Er.std():.3f} | частотный {F.mean():.3f} | байес {g('pretrained', 'bayes').mean():.3f}")
        print(f"      B  пред {Bp.mean():+.3f} ± {Bp.std():.3f} логита (случ {g('random', 'B').mean():+.3f}) | B_auc пред {Bap.mean():.3f} ± {Bap.std():.3f}, случ {Bar.mean():.3f} ± {Bar.std():.3f}")
        tests += [(m, f, 'M', max(perm_p(Mp - 0.5), perm_p(Mp - Mr))),
                  (m, f, 'E', max(perm_p(Ep - Er), perm_p(Ep - F))),
                  (m, f, 'B', max(perm_p(Bp), perm_p(Bap - Bar)))]
if tests:
    adj = holm([t[3] for t in tests])
    print('\nОСНОВНАЯ СЕМЬЯ (q1; точные парные перестановки по 10 наборам, односторонние; p проверки — больший из двух; Холм по 18):')
    for (m, f, k, p), pa in zip(tests, adj):
        ok = pa < 0.05; verdict[(m, f, k)] = ok
        print(f"  {m:5s} {f} {k}: p {p:.4f}, p(Холм) {pa:.4f} → {'выполнено' if ok else 'нет'}{'' if valid[(m, f)] else ' (метод негоден)'}")
    print('\nТОЛКОВАНИЕ (по записанному заранее):')
    for m in order:
        for f in ('c', 's'):
            if (m, f, 'M') not in verdict:
                continue
            M_, E_, B_ = (verdict[(m, f, k)] for k in 'MEB')
            if not valid[(m, f)]:
                t = 'метод негоден — не толкуется'
            elif M_ and B_:
                t = 'M и B — смысл связан с опытом тела' + (' (E тоже)' if E_ else ' (E нет)')
            elif M_ and E_:
                t = 'M и E без B — смысл и опыт по отдельности, при конфликте ответ к опыту не сдвигается'
            elif M_:
                t = 'только M — знает слова, опыт не использует'
            elif E_ and B_:
                t = 'E и B без M — вне записанных исходов'
            elif E_:
                t = 'только E — опыт без смысла'
            elif B_:
                t = 'только B — вне записанных исходов'
            else:
                t = 'ничего'
            print(f"  {m:5s} {f}: {t}")

print('\nОПИСАТЕЛЬНО')
for m in order:
    r = R[m]
    if any(f'{w}/{i}' not in r['sets'] for w in ('pretrained', 'random') for i in SETS):
        continue
    for f in ('c', 's'):
        g = lambda w, k, v='q1': get(r, w, v, f, k)
        print(f"--- {m} [{f}]")
        sh = {k: g('pretrained', k, 'q1_sh') for k in ('E', 'F', 'B', 'B_auc')}
        Er_sh, Bar_sh = g('random', 'E', 'q1_sh'), g('random', 'B_auc', 'q1_sh')
        print(f"  перемешанные слоты (q1_sh): E {sh['E'].mean():.3f} (q1 {g('pretrained', 'E').mean():.3f}; случ {Er_sh.mean():.3f}), "
              f"B {sh['B'].mean():+.3f} (q1 {g('pretrained', 'B').mean():+.3f}), B_auc {sh['B_auc'].mean():.3f} (случ {Bar_sh.mean():.3f})")
        print(f"     держится ли при перемешивании (те же условия, без поправки): E p {max(perm_p(sh['E'] - Er_sh), perm_p(sh['E'] - sh['F'])):.4f}; "
              f"B p {max(perm_p(sh['B']), perm_p(sh['B_auc'] - Bar_sh)):.4f} | падение q1 → q1_sh: E p {perm_p(g('pretrained', 'E') - sh['E']):.4f}, "
              f"B p {perm_p(g('pretrained', 'B') - sh['B']):.4f}")
        for v in ('q1', 'q2', 'q3'):
            print(f"  {v}: M {g('pretrained', 'M', v).mean():.3f} (случ {g('random', 'M', v).mean():.3f}), E {g('pretrained', 'E', v).mean():.3f} "
                  f"(случ {g('random', 'E', v).mean():.3f}), B {g('pretrained', 'B', v).mean():+.3f}, B_auc {g('pretrained', 'B_auc', v).mean():.3f} "
                  f"(случ {g('random', 'B_auc', v).mean():.3f}), M_direct {g('pretrained', 'M_direct', v).mean():.3f}")
        for k in ('M', 'E', 'B_auc'):
            x = np.array([g('pretrained', k, v).mean() for v in ('q1', 'q2', 'q3')])
            print(f"     {k} по трём вопросам: среднее {x.mean():.3f}, разброс (макс − мин) {x.max() - x.min():.3f}")
        sets = [r['sets'][f'pretrained/{i}']['measures']['q1'][f] for i in SETS]
        for key, nm in (('B_by_n', 'B по n (логиты, правдивые − лживые)'), ('outweigh_by_n', 'доля лживых окон, где ответ к опыту')):
            row = []
            for n_ in ('1', '2', '3', '4+'):
                x = [s[key][n_] for s in sets if s[key][n_] is not None]
                row.append(f"n={n_}: {np.mean(x):+.3f} ± {np.std(x):.3f}" if x else f"n={n_}: —")
            print(f"  {nm}: " + '; '.join(row))
        mc = [s['M_by_cond'] for s in sets]
        print(f"  M по условиям (n = 0): правдивые {np.mean([x['truthful'] for x in mc if x['truthful'] is not None]):.3f}, "
              f"лживые {np.mean([x['lying'] for x in mc if x['lying'] is not None]):.3f}")
        print(f"  окон на набор: M {np.mean([s['n_M'] for s in sets]):.0f}, E {np.mean([s['n_E'] for s in sets]):.0f}, B {np.mean([s['n_B'] for s in sets]):.0f}; "
              f"n: " + ', '.join(f"{k} {np.mean([s['n_hist'][k] for s in sets]):.0f}" for k in ('0', '1', '2', '3+')))
    sec = [x['sec'] for x in r['sets'].values()]
    print(f"  время {m}: {sum(sec)} с на {len(sec)} прогонов (набор × веса)")
