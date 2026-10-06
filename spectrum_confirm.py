"""Ось 1 — подтверждающая проверка H_info на свежих параметрах и посевах (docs/RESULTS_2026-10-06_spectrum.md, раздел
«Подтверждающая проверка H_info»). Та же среда env_spectrum.py; ни одно значение параметров не совпадает с сеткой версии 1.
Гипотезы о τ2 (постоянная среднего слоя): H_info √(T_m Δ_m / I_m), H_geo √(T_m Δ_m), H_world T_m; T, Δ, I — как в описательной
проверке (spectrum_tau.deltas).
Агенты L1 (τ1), L2 (τ1 ≤ τ2), сетка τ = 10^(−0,5 + 0,125 k), k = 0…44 (0,32 … 100 000), подбор на 200 обучающих жизнях.
Ячейка участвует, если L2 − L1 > 0 на 200 отложенных жизнях, z ≥ 2 (L1, L2 — лучшие на обучающих).
Плато по допуску: профиль E(τ2) с τ1, переподобранным для каждого τ2; в плато — τ2 со средней энергией ≥ лучшей − ст. ошибка
парной (по жизням) разности с лучшим. Определена, если max / min плато ≤ 10; τ2_fit — среднее геометрическое плато.

python spectrum_confirm.py --check     # проверка кода
python spectrum_confirm.py             # расчёт → results_spectrum/confirm.json, confirm.txt
"""
import itertools, json, math, os, sys, time
import numpy as np
import env_spectrum as es
from spectrum_tau import life_E, deltas

GRID = 10 ** (-0.5 + 0.125 * np.arange(45))
FREQ_C = {'rare07': 1 / 1.4, 'rare15': 1 / 3.0}          # интервал между встречами 0,7× и 1,5× от «редко» версии 1 (2 / h_f)
LN3 = math.log(3)


def worlds():
    out = []
    for hf, r, (ph, pl) in itertools.product((1 / 30, 1 / 80), (1 / 20, 1 / 50), ((0.7, 0.3), (0.85, 0.15))):
        out.append({'name': f"hf{round(1 / hf)}_r{round(1 / r)}_const_p{round(ph * 100)}", 'h_f': hf, 'h_m': hf * r, 'h_g': 0.0,
                    'p_hi': ph, 'p_lo': pl, 'g': 'const'})
    return out


def seed(wi, fi, part):
    return 7_000_000 + 100_000 * wi + 10_000 * fi + (0 if part == 'train' else 5_000)


def plateau(E2, pairs):
    """E2 (пары, жизни) — энергии L2; pairs (пары, 2). Профиль по τ2: лучший τ1 для каждого τ2, допуск — ст. ошибка
    парной разности с лучшим. Возвращает τ2_fit, плато (список τ2), лучшую пару."""
    m = E2.mean(1); t2s = np.unique(pairs[:, 1]); prof = {}
    for t2 in t2s:
        idx = np.nonzero(pairs[:, 1] == t2)[0]; prof[t2] = idx[int(m[idx].argmax())]
    best = max(prof.values(), key=lambda i: m[i]); n = E2.shape[1]
    plat = []
    for t2, i in prof.items():
        d = E2[best] - E2[i]; se = d.std(ddof=1) / math.sqrt(n)
        if m[i] >= m[best] - se:
            plat.append(float(t2))
    plat = np.array(sorted(plat))
    return float(np.exp(np.log(plat).mean())), plat, pairs[best]


def check():
    cfg = worlds()[0]; W = es.world(cfg, FREQ_C['rare07'] * cfg['h_f'], 30, 123)
    combos = np.array([[3.0, 300.0], [10.0, 1000.0], [1.0, 1.0]])
    a = life_E(W, combos); b = es.score(W, es.cascade(W, combos))[0]
    print('энергия по жизням: life_E против score(cascade) — макс. разница', np.abs(a - b).max())
    # прямой пересчёт утечки по тактам для одной жизни
    ch = es.cascade(W, combos[:1])[0]; i = 0; r = np.zeros(es.T); v = W['valid'][i]
    r[W['t'][i][v]] = np.where(ch[i][v] == (W['y'][i][v] == 1), es.FOOD, -es.POISON)
    Ed, s = es.E0, 0.0
    for t in range(es.T):
        Ed = Ed - es.LAM * (Ed - es.E0) + r[t]; s += Ed
    print('прямой пересчёт энергии одной жизни:', round(s / es.T, 9), 'против', round(a[0, i], 9))
    # искусственные равенства: плоский профиль на τ2 ∈ [10, 1000] → центр 100, а не край
    pairs = es.monotone(GRID, 2); n = 50; rng = np.random.RandomState(0)
    base = rng.randn(n) * 0.1; E2 = np.tile(base, (len(pairs), 1)) - 1.0
    flat = (pairs[:, 1] >= 10 - 1e-9) & (pairs[:, 1] <= 1000 + 1e-9)
    E2[flat] = base + 0.0                                                         # точно равные энергии на плато
    fit, plat, bp = plateau(E2, pairs)
    print(f'искусственное плато [10, 1000]: τ2_fit {fit:.1f} (центр — 100), плато {plat.min():.1f}…{plat.max():.1f}, лучшая пара {bp}')


def main():
    t0 = time.time(); pairs = es.monotone(GRID, 2); singles = GRID[:, None]; rows = []
    for wi, cfg in enumerate(worlds()):
        I = (cfg['p_hi'] - cfg['p_lo']) ** 2 / 0.25
        for fi, (fn, c) in enumerate(FREQ_C.items()):
            q = c * cfg['h_f']; ts = time.time()
            Wtr = es.world(cfg, q, es.N_TRAIN, seed(wi, fi, 'train')); Wte = es.world(cfg, q, es.N_TEST, seed(wi, fi, 'test'))
            dm, _, nint, _ = deltas(Wtr)
            E1, E2 = life_E(Wtr, singles), life_E(Wtr, pairs)
            b1, b2 = singles[int(E1.mean(1).argmax())], pairs[int(E2.mean(1).argmax())]
            T1, T2 = life_E(Wte, [b1])[0], life_E(Wte, [b2])[0]
            d = T2 - T1; z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std() > 0 else 0.0
            fit, plat, _ = plateau(E2, pairs)
            Tm = 1 / cfg['h_m']
            rows.append({'world': cfg['name'], 'freq': fn, 'q': q, 'p': [cfg['p_hi'], cfg['p_lo']], 'T_m': Tm, 'Delta_m': dm, 'n_intervals': nint, 'I': I,
                         'L1': b1.tolist(), 'L2': b2.tolist(), 'L2_minus_L1': float(d.mean()), 'z': float(z), 'participates': bool(d.mean() > 0 and z >= 2),
                         'tau2_fit': fit, 'plateau': [float(plat.min()), float(plat.max())], 'n_plateau': int(len(plat)),
                         'determined': bool(plat.max() / plat.min() <= 10),
                         'pred': {'info': math.sqrt(Tm * dm / I), 'geo': math.sqrt(Tm * dm), 'world': Tm}})
            print(f"{time.strftime('%H:%M:%S')} {cfg['name']}/{fn}: {time.time() - ts:.0f} с", flush=True)
    os.makedirs('results_spectrum', exist_ok=True)
    json.dump(rows, open('results_spectrum/confirm.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'всего {time.time() - t0:.0f} с')


def report():
    rows = json.load(open('results_spectrum/confirm.json', encoding='utf-8')); out = []
    P = out.append
    sel = [r for r in rows if r['participates'] and r['determined']]
    P(f"ячеек {len(rows)}; участвуют (L2 − L1 > 0, z ≥ 2) {sum(r['participates'] for r in rows)}; из них определены {len(sel)}")

    def stats(cells, h):
        x = np.log([r['tau2_fit'] for r in cells]); p = np.log([r['pred'][h] for r in cells]); dev = np.abs(x - p)
        sl = np.polyfit(p, x, 1)[0] if len(cells) >= 3 and p.std() > 0 else float('nan')
        rr = np.corrcoef(p, x)[0, 1] if len(cells) >= 3 and p.std() > 0 and x.std() > 0 else float('nan')
        return dev.mean(), (dev <= LN3).mean(), sl, rr

    if len(sel) < 12:
        P(f'участвующих определённых ячеек {len(sel)} < 12 — ПРОВЕРКА НЕИНФОРМАТИВНА')
    else:
        mi, si, sli, ri = stats(sel, 'info'); mw = stats(sel, 'world')[0]
        a, b, c = si >= 0.75, mi < mw, 0.5 <= sli <= 1.5
        P(f"КРИТЕРИИ H_info: а) в ln 3 {si * 100:.0f} % {'+' if a else '−'}; б) ср |ln| {mi:.2f} против {mw:.2f} у H_world {'+' if b else '−'}; "
          f"в) наклон {sli:.2f} {'+' if c else '−'} (r {ri:.2f})")
        t = ('а, б, в выполнены — постоянная среднего слоя определяется компромиссом √(T·Δ/I), а не временем жизни: установлено для этой среды и двухслойного агента (эвристики)'
             if a and b and c else 'б выполнено, а нет — направление верное, точности формулы недостаточно' if b and not a
             else 'б не выполнено — наблюдение версии 1 не подтвердилось, линия закрыта' if not b
             else 'а и б выполнены, в нет — вне записанных исходов')
        P(f'ТОЛКОВАНИЕ (записано заранее): {t}')
    P('\nОПИСАТЕЛЬНО (участвующие и определённые ячейки):')
    for nm, cells in (('все', sel),) + tuple((f, [r for r in sel if r['freq'] == f]) for f in FREQ_C) + \
            tuple((f'p {pp}', [r for r in sel if r['p'][0] == pp]) for pp in (0.7, 0.85)):
        if len(cells) == 0:
            P(f'  {nm}: нет ячеек'); continue
        P(f'  {nm} (n {len(cells)}): ' + ' | '.join(f"{h}: ср |ln| {m:.2f}, в ln3 {s * 100:.0f} %, наклон {sl:.2f}, r {rr:.2f}"
                                                   for h in ('info', 'geo', 'world') for m, s, sl, rr in [stats(cells, h)]))
    if sel:
        x = np.log([r['tau2_fit'] for r in sel]); di = np.abs(x - np.log([r['pred']['info'] for r in sel])); dg = np.abs(x - np.log([r['pred']['geo'] for r in sel]))
        P(f'  H_info против H_geo: |ln| info меньше в {np.mean(di < dg) * 100:.0f} % ячеек; средняя разность |ln| (geo − info) {np.mean(dg - di):+.3f}')
    P('\nЯЧЕЙКИ (τ2_fit [плато] | предсказания info / geo / world | L2 − L1, z | Δ_m, I):')
    for r in rows:
        P(f"  {r['world']:20s} {r['freq']:6s} {r['tau2_fit']:8.1f} [{r['plateau'][0]:.1f}, {r['plateau'][1]:.1f}]{'' if r['determined'] else ' ПЛАТО'}"
          f"{'' if r['participates'] else ' НЕ УЧАСТВУЕТ'} | {r['pred']['info']:.0f} / {r['pred']['geo']:.0f} / {r['pred']['world']:.0f} | "
          f"{r['L2_minus_L1']:+.4f}, z {r['z']:.1f} | Δ_m {r['Delta_m']:.0f} ({r['n_intervals']}), I {r['I']:.2f}")
    open('results_spectrum/confirm.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n'); print('\n'.join(out))


if __name__ == '__main__':
    if '--check' in sys.argv:
        check()
    else:
        if not os.path.exists('results_spectrum/confirm.json'):
            main()
        report()
