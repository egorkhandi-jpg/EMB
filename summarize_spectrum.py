"""Ось 1, этап 1: критерии по пререгистрации (docs/RESULTS_2026-10-06_spectrum.md). z — по отложенным жизням (200), парные по жизни.
Для каждой конфигурации мира (16):
  1 барьер:        редко — BAYES − L1 > 0, z ≥ 3;
  2 средний слой:  редко — L2 − L1 > 0, z ≥ 3; при меняющейся g (switch и T·h_g ≥ 1) ещё L3 − L2 > 0, z ≥ 3;
  3 согласование:  редко — L3 − L3_mis > 0, z ≥ 3;
  4 масштабы:      постоянные L3 (подбор при «редко»): |ln(τ1 h_f)| ≤ ln 3, |ln(τ2 h_m)| ≤ ln 3, при меняющейся g ещё |ln(τ3 h_g)| ≤ ln 3;
  5 условность:    выигрыш L3 − L1 на встречу убывает: редко > средне > часто (точечно), и редко − часто > 0, z ≥ 3 (независимые жизни);
  6 нет вырождения: редко — (L1 − REACT) / (BAYES − REACT) < 0,8.
Меры 1–3 — средняя энергия за жизнь; 5 — выгода на встречу (средняя энергия аффинна по выгоде на такт, частоты встреч разные).
Версия среды проходит, если хотя бы одна конфигурация выполняет 1–6. Конфигурация для этапа 2 — с наибольшим L3 − L1 (энергия)
на обучающих жизнях при «редко» среди прошедших.

python summarize_spectrum.py results_spectrum > results_spectrum/summary.txt
"""
import json, math, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_spectrum'
R = json.load(open(os.path.join(D, 'spectrum.json'), encoding='utf-8'))['results']
L3 = math.log(3)


def z_paired(a, b):
    d = np.asarray(a) - np.asarray(b); return d.mean(), d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))


def z_indep(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.mean() - b.mean(), (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


names = sorted({k.split('/')[0] for k in R})
passed, rows = [], []
print('конфигурация | 1 барьер | 2 L2−L1 (L3−L2) | 3 L3−mis | 4 τ·h | 5 условность | 6 доля L1 | итог')
for nm in names:
    if any(f'{nm}/{f}' not in R for f in ('rare', 'mid', 'often')):
        print(f'{nm}: НЕ ХВАТАЕТ частот'); continue
    r, md, of = R[f'{nm}/rare'], R[f'{nm}/mid'], R[f'{nm}/often']
    cfg, te, tau = r['cfg'], r['test'], r['taus']
    E = {k: np.array(v['E']) for k, v in te.items()}
    sw = cfg['g'] == 'switch' and 5000 * cfg['h_g'] >= 1                     # g меняется за жизнь хотя бы раз в среднем
    d1, z1 = z_paired(E['BAYES'], E['L1']); c1 = d1 > 0 and z1 >= 3
    d2, z2 = z_paired(E['L2'], E['L1']); c2 = d2 > 0 and z2 >= 3
    d23, z23 = z_paired(E['L3'], E['L2'])
    if sw:
        c2 = c2 and d23 > 0 and z23 >= 3
    d3, z3 = z_paired(E['L3'], E['L3_mis']); c3 = d3 > 0 and z3 >= 3
    lt = [abs(math.log(tau['L3'][0] * cfg['h_f'])), abs(math.log(tau['L3'][1] * cfg['h_m']))] + ([abs(math.log(tau['L3'][2] * cfg['h_g']))] if sw else [])
    c4 = all(x <= L3 for x in lt)
    G = {f: np.array(R[f'{nm}/{f}']['test']['L3']['pay']) - np.array(R[f'{nm}/{f}']['test']['L1']['pay']) for f in ('rare', 'mid', 'often')}
    g5, z5 = z_indep(G['rare'], G['often'])
    c5 = G['rare'].mean() > G['mid'].mean() > G['often'].mean() and g5 > 0 and z5 >= 3
    share = (E['L1'].mean() - E['REACT'].mean()) / (E['BAYES'].mean() - E['REACT'].mean())
    c6 = share < 0.8
    ok = all((c1, c2, c3, c4, c5, c6))
    if ok:
        passed.append((nm, r['train_E']['L3'] - r['train_E']['L1']))
    f = lambda c: '+' if c else '−'
    print(f"{nm:22s} | {f(c1)} {d1:+.3f} z {z1:.1f} | {f(c2)} {d2:+.3f} z {z2:.1f} ({d23:+.3f} z {z23:.1f}) | {f(c3)} {d3:+.3f} z {z3:.1f} [{tau['L3_mis_kind']}] | "
          f"{f(c4)} {' '.join(f'{x:.2f}' for x in lt)} | {f(c5)} {G['rare'].mean():+.4f} / {G['mid'].mean():+.4f} / {G['often'].mean():+.4f} z {z5:.1f} | "
          f"{f(c6)} {share:.2f} | {'ПРОШЛА' if ok else 'нет'}")
    rows.append((nm, c1, c2, c3, c4, c5, c6))

print(f"\nВЕРСИЯ 1: {'ПРОШЛА' if passed else 'НЕ ПРОШЛА'} — конфигураций, выполнивших 1–6: {len(passed)} из {len(rows)}")
if passed:
    best = max(passed, key=lambda x: x[1])
    print(f"  для этапа 2 (наибольший L3 − L1 на обучающих при «редко»): {best[0]} ({best[1]:+.4f})")
print('\nТОЛКОВАНИЕ ПО КОНФИГУРАЦИЯМ (записано заранее):')
for nm, c1, c2, c3, c4, c5, c6 in rows:
    if all((c1, c2, c3, c4, c5, c6)):
        t = '1–6: гипотеза оси 1 поддержана на уровне задачи'
    elif not c2:
        t = '2 не выполнено: один след справляется — гипотеза не подтверждена'
    elif c2 and c3 and not c4:
        t = '2 и 3 без 4: слоёв столько, сколько масштабов, но скорости не совпадают — слабая форма' + ('' if c5 else '; 5 нет: выгода не зависит от частоты свидетельств')
    elif not c5:
        t = '5 не выполнено: выгода не зависит от частоты свидетельств — уточнение гипотезы неверно'
    else:
        t = 'вне записанных исходов: ' + ', '.join(str(i + 1) for i, c in enumerate((c1, c2, c3, c4, c5, c6)) if not c) + ' не выполнены'
    print(f'  {nm:22s}: {t}')

print('\nОПИСАТЕЛЬНО (отложенные жизни): средняя энергия; выгода на встречу; точность на первой встрече после невидимой смены знака; τ')
for nm in names:
    for fr in ('rare', 'mid', 'often'):
        k = f'{nm}/{fr}'
        if k not in R:
            continue
        te, tau = R[k]['test'], R[k]['taus']
        print(f"  {k:28s} " + ' | '.join(f"{a} {np.mean(te[a]['E']):.3f}/{np.mean(te[a]['pay']):+.3f}/{np.mean(te[a]['acc_flip']):.2f}"
                                          for a in ('REACT', 'L1', 'L2', 'L3', 'L3_mis', 'BAYES'))
              + f" | τ L1 {tau['L1'][0]:.0f}; L2 {[round(t) for t in tau['L2']]}; L3 {[round(t) for t in tau['L3']]}")
