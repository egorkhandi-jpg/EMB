"""Ось 1 — описательная проверка уточнения (после данных версии 1, в семью не входит; docs/RESULTS_2026-10-06_spectrum.md).
Гипотезы о постоянных памяти: H_world τ ≈ T, H_geo τ ≈ √(T Δ), H_info τ ≈ √(T Δ / I). Подбор L3 (и L2, L1 описательно) полным перебором
на сетке τ = 10^(−0,5 + 0,25k), k = 0…22, на обучающих жизнях версии 1; бутстреп по жизням (200) → 80 %-й интервал τ;
ячейка определена, если верх ≤ 10 × низ. Δ_m, Δ_g — на тех же траекториях (опознанные смены знака и m).

python spectrum_tau.py results_spectrum
"""
import json, math, os, sys, time
import numpy as np
import env_spectrum as es

D = sys.argv[1] if len(sys.argv) > 1 else 'results_spectrum'
GRID = 10 ** (-0.5 + 0.25 * np.arange(23))
NBOOT, LN3 = 200, math.log(3)


def life_E(W, taus, chunk=600):
    """Средняя энергия за жизнь (C, n) для сочетаний taus (C, L) — без хранения решений."""
    taus = np.asarray(taus, float); out = []
    for a in range(0, len(taus), chunk):
        tc = taus[a:a + chunk]; C, L = tc.shape; n, Kmax = W['y'].shape
        S = np.zeros((C, L, n)); N = np.zeros((C, L, n)); E = np.zeros((C, n))
        for k in range(Kmax):
            v = W['valid'][:, k]
            if not v.any():
                break
            dec = np.exp(-W['dt'][:, k][None, None, :] / tc[:, :, None]); S *= dec; N *= dec
            d = np.zeros((C, n))
            for i in range(L - 1, -1, -1):
                d = (S[:, i] - 0.5 * N[:, i] + es.KAPPA * d) / (N[:, i] + es.KAPPA)
            ch = np.where(d == 0, W['coin'][:, k][None], d > 0)
            r = np.where(ch == (W['y'][:, k] == 1)[None], es.FOOD, -es.POISON) * v[None]
            E += r * W['w'][:, k][None]
            yk = W['y'][:, k] * v; S += yk[None, None, :]; N += v[None, None, :]
        out.append(es.E0 + E)
    return np.concatenate(out)


def recog(change_t, enc_t, T):
    """Моменты опознания смен: первая встреча (enc_t, отсортированы) в [смена, следующая смена)."""
    R = []
    nxt = list(change_t[1:]) + [T]
    for c, c2 in zip(change_t, nxt):
        j = np.searchsorted(enc_t, c)
        if j < len(enc_t) and enc_t[j] < c2:
            R.append(enc_t[j])
    return R


def deltas(W):
    """Δ_m — средний интервал между опознанными сменами знака; Δ_g — между опознанными сменами m."""
    S, M, ENC = W['S'], W['M'], W['ENC']; T = S.shape[1]; im, ig = [], []
    for i in range(S.shape[0]):
        enc = np.nonzero(ENC[i])[0]
        sc = np.nonzero(S[i, 1:] != S[i, :-1])[0] + 1
        Rs = recog(sc, enc, T); im += list(np.diff(Rs))
        mc = np.nonzero(M[i, 1:] != M[i, :-1])[0] + 1
        Rs_arr = np.array(Rs); sc_rec = []                                 # опознанные смены знака: (время смены, время опознания)
        for c in sc:
            j = np.searchsorted(enc, c); nx = sc[sc > c]
            if j < len(enc) and enc[j] < (nx[0] if len(nx) else T):
                sc_rec.append((c, enc[j]))
        Rm = []
        nxm = list(mc[1:]) + [T]
        for c, c2 in zip(mc, nxm):
            cand = [rt for (st, rt) in sc_rec if c <= st < c2]
            if cand:
                Rm.append(cand[0])
        ig += list(np.diff(Rm))
    return (float(np.mean(im)) if im else float('nan')), (float(np.mean(ig)) if ig else float('nan')), len(im), len(ig)


def fit_boot(E, combos, rng):
    """Точечный подбор и бутстреп: возвращает τ (L,), низ и верх 80 %-го интервала по слоям."""
    best = combos[int(E.mean(1).argmax())]
    n = E.shape[1]; picks = []
    for _ in range(NBOOT):
        idx = rng.randint(0, n, n); picks.append(combos[int(E[:, idx].mean(1).argmax())])
    P = np.log10(np.array(picks))
    lo, hi = 10 ** np.percentile(P, 10, axis=0), 10 ** np.percentile(P, 90, axis=0)
    return best, lo, hi


def main():
    t0 = time.time(); rng = np.random.RandomState(2026)
    combos = {1: es.monotone(GRID, 1), 2: es.monotone(GRID, 2), 3: es.monotone(GRID, 3)}
    rows = []
    for ci, cfg in enumerate(es.configs()):
        I = (cfg['p_hi'] - cfg['p_lo']) ** 2 / (0.25)                           # p̄ = 0,5 во всех конфигурациях
        for fi, (fn, c) in enumerate(es.FREQ.items()):
            q = c * cfg['h_f']; ts = time.time()
            W = es.world(cfg, q, es.N_TRAIN, es.seeds(ci, fi, 'train'))
            dm, dg, nm_, ng_ = deltas(W)
            fits = {}
            for L in (1, 2, 3):
                E = life_E(W, combos[L]); fits[L] = fit_boot(E, combos[L], rng)
            layers = [('s', 1 / cfg['h_f'], 1 / q, 1.0), ('m', 1 / cfg['h_m'], dm, I)]
            if cfg['g'] == 'switch':
                layers.append(('g', 1 / cfg['h_g'], dg, I))
            for L in (1, 2, 3):
                best, lo, hi = fits[L]
                for li, (nm, Tw, Dl, Il) in enumerate(layers[:L]):
                    rows.append({'cfg': cfg['name'], 'freq': fn, 'agent': f'L{L}', 'layer': li + 1, 'var': nm, 'T': Tw, 'Delta': Dl, 'I': Il,
                                 'tau': float(best[li]), 'lo': float(lo[li]), 'hi': float(hi[li]), 'determined': bool(hi[li] <= 10 * lo[li]),
                                 'pred': {'world': Tw, 'geo': math.sqrt(Tw * Dl), 'info': math.sqrt(Tw * Dl / Il)}})
            print(f"{time.strftime('%H:%M:%S')} {cfg['name']}/{fn}: Δ_m {dm:.0f} ({nm_}), Δ_g {dg:.0f} ({ng_}) | {time.time() - ts:.0f} с", flush=True)
    json.dump(rows, open(os.path.join(D, 'tau.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'всего {time.time() - t0:.0f} с')


def report():
    rows = json.load(open(os.path.join(D, 'tau.json'), encoding='utf-8'))
    out = []
    P = lambda *a: out.append(' '.join(str(x) for x in a))

    def stats(sel, h):
        x = np.log(np.array([r['tau'] for r in sel])); p = np.log(np.array([r['pred'][h] for r in sel]))
        dev = np.abs(x - p)
        slope = np.polyfit(p, x, 1)[0] if len(sel) >= 3 and p.std() > 0 else float('nan')
        corr = np.corrcoef(p, x)[0, 1] if len(sel) >= 3 and p.std() > 0 and x.std() > 0 else float('nan')
        return dev.mean(), (dev <= LN3).mean(), slope, corr

    for agent, layers in (('L3', (1, 2, 3)), ('L2', (1, 2)), ('L1', (1,))):
        P(f'=== {agent}')
        for layer in layers:
            all_ = [r for r in rows if r['agent'] == agent and r['layer'] == layer]
            det = [r for r in all_ if r['determined']]
            P(f'  τ{layer}: ячеек {len(all_)}, определённых {len(det)}, «плато» {len(all_) - len(det)}')
            for sub, name in ((det, 'все частоты'),) + tuple(([r for r in det if r['freq'] == f], f) for f in ('rare', 'mid', 'often')):
                if len(sub) == 0:
                    P(f'     {name:11s}: нет определённых'); continue
                P(f'     {name:11s} (n {len(sub):2d}): ' + ' | '.join(
                    f"{h}: ср |ln| {m:.2f}, в ln3 {s * 100:.0f} %, наклон {sl:.2f}, r {c:.2f}" for h in ('world', 'geo', 'info') for m, s, sl, c in [stats(sub, h)]))
    det2 = [r for r in rows if r['agent'] == 'L3' and r['layer'] == 2 and r['determined']]
    P('\nПРАВИЛО П. 4 (τ2, L3, определённые ячейки):')
    if len(det2) < 8:
        P(f'  определённых ячеек {len(det2)} < 8 — проверка неинформативна; решение за Егором')
    else:
        mw = stats(det2, 'world')[0]; ok_any = False
        for h in ('geo', 'info'):
            m, s, sl, c = stats(det2, h)
            a, b, v = s >= 0.75, m < mw, 0.5 <= sl <= 1.5
            ok_any |= a and b and v
            P(f"  {h}: а) в ln3 {s * 100:.0f} % {'+' if a else '−'}; б) ср |ln| {m:.2f} против {mw:.2f} у world {'+' if b else '−'}; в) наклон {sl:.2f} {'+' if v else '−'}"
              f" → {'ВЫПОЛНЕНО' if a and b and v else 'нет'}")
        P(f"  ИТОГ: {'версия 2 имеет смысл' if ok_any else 'не выполнено — линия оси 1 закрывается по правилу п. 0 без версии 2'}")
    P('\nЯЧЕЙКИ L3 (τ_fit [80 % интервал] | предсказания world / geo / info | T, Δ, I):')
    for r in rows:
        if r['agent'] == 'L3':
            P(f"  {r['cfg']:22s} {r['freq']:5s} τ{r['layer']}({r['var']}) {r['tau']:8.1f} [{r['lo']:.1f}, {r['hi']:.1f}]{'' if r['determined'] else ' ПЛАТО'} | "
              f"{r['pred']['world']:.0f} / {r['pred']['geo']:.1f} / {r['pred']['info']:.1f} | T {r['T']:.0f}, Δ {r['Delta']:.1f}, I {r['I']:.2f}")
    open(os.path.join(D, 'tau.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('\n'.join(out[:60]))


if __name__ == '__main__':
    if not os.path.exists(os.path.join(D, 'tau.json')):
        main()
    report()
