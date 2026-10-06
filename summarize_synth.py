"""Сводка последней проверки для варианта C (A27) по пререгистрации (docs/RESULTS_2026-10-06_bind.md, раздел A27).
Основная семья — Pythia-6.9B, интерфейс s, вопрос q1; точные парные перестановки (смена знаков) по 10 наборам, односторонние;
в каждой проверке два условия, p проверки — больший из двух; Холм по двум.
  H1: AUC_нейтр(пред, исходы) > AUC_нейтр(пред, переставленные) и > AUC_нейтр(случ, исходы);
  H2: d_лож(пред, исходы) > d_лож(пред, переставленные) и > d_лож(случ, исходы).
Остальное — описательно: по n, правдивые окна, 6.9B c, 1B s, прямые слова при n = 0, три вопроса.

python summarize_synth.py results_synth > results_synth/summary.txt
"""
import glob, itertools, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_synth'
SETS = list(range(1, 11))
SIGNS = np.array(list(itertools.product([1, -1], repeat=len(SETS))))


def perm_p(x):
    x = np.asarray(x, float)
    return float(((SIGNS * x).mean(1) >= x.mean() - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


R = {os.path.basename(p)[6:-5]: json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob(os.path.join(D, 'synth_*.json')))}


def g(m, wts, f, v, key, n=None):
    r = R[m]['sets']
    return np.array([(r[f'{wts}/{f}/{i}']['measures'][v]['by_n'][n] if n else r[f'{wts}/{f}/{i}']['measures'][v])[key] for i in SETS], float)


def have(m, f):
    return m in R and all(f'{w}/{f}/{i}' in R[m]['sets'] for w in ('pretrained', 'random') for i in SETS)


for m in R:
    commits = {x['git_commit'][:8] + (' ГРЯЗН' if x['git_dirty'] else '') for x in R[m]['sets'].values()}
    print(f"{m}: коммиты {', '.join(sorted(commits))}; прогонов {len(R[m]['sets'])}")

if have('6.9b', 's'):
    A, Ap, Ar = g('6.9b', 'pretrained', 's', 'q1', 'auc_neutral'), g('6.9b', 'pretrained', 's', 'q1_perm', 'auc_neutral'), g('6.9b', 'random', 's', 'q1', 'auc_neutral')
    L, Lp, Lr = g('6.9b', 'pretrained', 's', 'q1', 'd_lying'), g('6.9b', 'pretrained', 's', 'q1_perm', 'd_lying'), g('6.9b', 'random', 's', 'q1', 'd_lying')
    p1 = (perm_p(A - Ap), perm_p(A - Ar)); p2 = (perm_p(L - Lp), perm_p(L - Lr))
    adj = holm([max(p1), max(p2)])
    print('\nОСНОВНАЯ СЕМЬЯ (6.9B, s, q1; точные парные перестановки по 10 наборам, односторонние; Холм по двум):')
    print(f"  H1 опыт без смысла: AUC нейтр — пред {A.mean():.3f} ± {A.std():.3f}, переставл. {Ap.mean():.3f} ± {Ap.std():.3f}, случ {Ar.mean():.3f} ± {Ar.std():.3f} | "
          f"p {p1[0]:.4f} / {p1[1]:.4f} → p {max(p1):.4f}, Холм {adj[0]:.4f} → {'ВЫПОЛНЕНО' if adj[0] < 0.05 else 'нет'}")
    print(f"  H2 связывание при конфликте: d лож — пред {L.mean():+.3f} ± {L.std():.3f}, переставл. {Lp.mean():+.3f} ± {Lp.std():.3f}, случ {Lr.mean():+.3f} ± {Lr.std():.3f} | "
          f"p {p2[0]:.4f} / {p2[1]:.4f} → p {max(p2):.4f}, Холм {adj[1]:.4f} → {'ВЫПОЛНЕНО' if adj[1] < 0.05 else 'нет'}")
    h1, h2 = adj[0] < 0.05, adj[1] < 0.05
    t = {(True, True): 'H1 и H2 — модель выводит из опыта и при конфликте сдвигает ответ от смысла к опыту: смысл связан с опытом тела (6.9B, предложения)',
         (True, False): 'H1 без H2 — опыт использует, но смысл при конфликте его перебивает',
         (False, True): 'H2 без H1 — сдвиг только при наличии смысла; толковать осторожно, смотреть по n',
         (False, False): 'ни H1, ни H2 — по правилу остановки вариант C закрывается: языковой канал Pythia 1B–6.9B передаёт лексику (смысл слова, частоту упоминания), но не вывод из опыта тела'}[(h1, h2)]
    print(f"\nТОЛКОВАНИЕ (записано заранее): {t}")
    print(f"  справочно: сдвиг лживых к опыту d − d_перест: пред {np.mean(L - Lp):+.3f} (p {perm_p(L - Lp):.4f}), случ "
          f"{np.mean(Lr - g('6.9b', 'random', 's', 'q1_perm', 'd_lying')):+.3f}; пред − случ по сдвигу p {perm_p((L - Lp) - (Lr - g('6.9b', 'random', 's', 'q1_perm', 'd_lying'))):.4f}")

print('\nОПИСАТЕЛЬНО')
for m, f in (('6.9b', 's'), ('6.9b', 'c'), ('1b', 's')):
    if not have(m, f):
        continue
    print(f"--- {m} [{f}]")
    for wts in ('pretrained', 'random'):
        for v in (('q1', 'q1_perm', 'q2', 'q3') if f == 's' else ('q1', 'q1_perm')):
            a, l, tr, ne = (g(m, wts, f, v, k) for k in ('auc_neutral', 'd_lying', 'd_truthful', 'd_neutral'))
            print(f"  {wts[:4]} {v:7s}: AUC нейтр {a.mean():.3f} ± {a.std():.3f} | d лож {l.mean():+.3f} ± {l.std():.3f} | d правд {tr.mean():+.3f} | d нейтр {ne.mean():+.3f}")
    for v in ('q1', 'q1_perm'):
        row = []
        for n in ('2', '4', '6'):
            a, l, tr, fr = (g(m, 'pretrained', f, v, k, n) for k in ('auc_neutral', 'd_lying', 'd_truthful', 'frac_lying_to_experience'))
            row.append(f"n={n}: AUC {a.mean():.3f}, d лож {l.mean():+.2f}, d правд {tr.mean():+.2f}, к опыту {fr.mean():.2f}")
        print(f"  пред {v} по n — " + ' | '.join(row))
    if f == 's':
        for k in ('auc_neutral', 'd_lying'):
            x = np.array([g(m, 'pretrained', f, v, k).mean() for v in ('q1', 'q2', 'q3')])
            print(f"  три вопроса, {k}: {', '.join(f'{y:+.3f}' for y in x)}; среднее {x.mean():+.3f}, разброс {x.max() - x.min():.3f}")
    n0 = R[m]['n0']
    if f'pretrained/{f}' in n0:
        p0 = n0[f'pretrained/{f}']
        rnd = [n0[f'random/{f}/{i}'] for i in SETS if f'random/{f}/{i}' in n0]
        print('  n = 0 (один вход): ' + '; '.join(f"{q}: прямые {p0[q]['M_direct']:.3f} (случ {np.mean([r[q]['M_direct'] for r in rnd]):.3f}), пулы {p0[q]['M_pools']:.3f} "
                                                 f"(случ {np.mean([r[q]['M_pools'] for r in rnd]):.3f})" for q in ('q1', 'q2', 'q3')))
    sec = [x['sec'] for k, x in R[m]['sets'].items() if f'/{f}/' in k]
    print(f"  время: {sum(sec)} с на {len(sec)} прогонов")
