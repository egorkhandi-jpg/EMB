"""A26, анализ ПОСЛЕ расчёта (не пререгистрирован, в семью не входит): объясняется ли B копированием чаще упомянутого слова.
Частотный ориентир на окнах E — 0,62, не 0,5: у живых тел вид-еда в окне съеден чаще (отбор выживших). В правдивых эпизодах чаще
упоминается слово «еды», в лживых — слово «вреда», и копирование даёт B > 0 без связки «съел → исход».
Проверка: B на окнах с равным числом съеденных предметов двух видов (Δ = 0, n ≥ 2) — упоминания слов равны, копирование вычитается;
связка в этих окнах полностью определяет знак, а после перемешивания слотов знак не выводим (поровну видов, поровну исходов).
Плюс регрессия a ~ условие + разность упоминаний (слово «еды» − слово «вреда») по набору.

python bind_posthoc.py results_bind > results_bind/posthoc.txt
"""
import itertools, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_bind'
W = json.load(open('meaning_words.json', encoding='utf-8'))
SIGNS = np.array(list(itertools.product([1, -1], repeat=10)))
perm_p = lambda x: float(((SIGNS * np.asarray(x)).mean(1) >= np.mean(x) - 1e-12).mean())


def food_ids(model):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')     # токенизатор у семейства общий
    return [tk.encode(' ' + w)[0] for w in W['food']]


FID = food_ids(None)
for m in ('1b', '2.8b', '6.9b'):
    for f in ('c', 's'):
        print(f'=== {m} [{f}]')
        rows = {}
        for wts in ('pretrained', 'random'):
            for v in ('q1', 'q1_sh'):
                B0, Bne, coef, Fn, n0 = [], [], [], [], []
                for i in range(1, 11):
                    z = np.load(os.path.join(D, f'raw_{m}_{wts}_{i}.npz')); meta = z['meta']; lg = z[f'{v}_{f}']
                    cond, n, nA, nB, gA, tA = meta[:, 2], meta[:, 3], meta[:, 4], meta[:, 5], meta[:, 7], meta[:, 8]
                    afw = np.isin(tA, FID); a = np.where(afw, lg[:, 0] - lg[:, 1], lg[:, 1] - lg[:, 0])
                    nf = np.where(gA == 1, nA, nB); npz = np.where(gA == 1, nB, nA)                # съедено вида-еды, вида-яда
                    dmen = np.where(cond == 0, nf - npz, npz - nf)                                  # упоминания: слово «еды» − слово «вреда»
                    sel = (cond < 2) & (n >= 2)
                    s0 = sel & (nf == npz)
                    tr, li = a[s0 & (cond == 0)], a[s0 & (cond == 1)]
                    B0.append(tr.mean() - li.mean()); n0.append(len(tr) + len(li))
                    s1 = sel & (nf != npz); Bne.append(a[s1 & (cond == 0)].mean() - a[s1 & (cond == 1)].mean())
                    X = np.c_[np.ones(sel.sum()), np.where(cond[sel] == 0, 0.5, -0.5), dmen[sel]]
                    coef.append(np.linalg.lstsq(X, a[sel], rcond=None)[0])
                    Fn.append([(nf[(cond < 2) & (n == k)] > npz[(cond < 2) & (n == k)]).mean() - (nf[(cond < 2) & (n == k)] < npz[(cond < 2) & (n == k)]).mean() for k in (1, 2, 3)])
                rows[(wts, v)] = (np.array(B0), np.array(Bne), np.array(coef), np.array(Fn), np.mean(n0))
        P = rows[('pretrained', 'q1')]
        print(f"  частота: P(вид-еда съеден чаще) − P(реже) в правдивых+лживых, n = 1 / 2 / 3: " + ' / '.join(f'{x:+.2f}' for x in P[3].mean(0)))
        print(f"  окон Δ = 0, n ≥ 2 на набор: {P[4]:.0f}")
        for wts in ('pretrained', 'random'):
            for v in ('q1', 'q1_sh'):
                B0, Bne, coef, _, _ = rows[(wts, v)]
                print(f"  {wts[:4]} {v:5s}: B(Δ=0) {B0.mean():+.3f} ± {B0.std():.3f} (p>0 {perm_p(B0):.4f}) | B(Δ≠0) {Bne.mean():+.3f} | "
                      f"регрессия: условие {coef[:, 1].mean():+.3f} (p>0 {perm_p(coef[:, 1]):.4f}), упоминание {coef[:, 2].mean():+.3f} на событие (p>0 {perm_p(coef[:, 2]):.4f})")
        d = rows[('pretrained', 'q1')][0] - rows[('pretrained', 'q1_sh')][0]
        print(f"  B(Δ=0): q1 − q1_sh {d.mean():+.3f} (p>0 {perm_p(d):.4f}); пред − случ (q1) {(rows[('pretrained', 'q1')][0] - rows[('random', 'q1')][0]).mean():+.3f} "
              f"(p>0 {perm_p(rows[('pretrained', 'q1')][0] - rows[('random', 'q1')][0]):.4f})")
