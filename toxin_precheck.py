"""Звено 6: предварительные проверки среды «скрытый токсин» (env_toxin.py) эвристиками, без ES и без LM, CPU.
Пререгистрация (критерии записаны до расчёта) — docs/RESULTS_2026-10-04_toxin.md.

Все политики: «богатую, если E < θ_E и <условие>», иначе — к ближайшей обычной; выбор вида → курс на ближайший предмет.
Семьи (условие):
  реактивные: ord (никогда), rich (всегда), any (к ближайшему любому), energy (только θ_E),
              hint — подсказка по последней богатой: её наблюдаемый прирост (тренд энергии в такт съедения) ≥ θ_g · G_RICH,
              либо E < θ_low (выход из тупика «больше никогда»);
  модель себя: time — тактов с последней богатой ≥ θ_t (грубейшая, одно число);
               counter — X̂ < θ_X, X̂ = Σ по своим богатым (100 / τ̂) e^{−Δt / τ̂} (затухание с оценкой τ̂, доза — средняя);
               win32 — тот же счётчик только по последним 32 тактам (окно LM), описательно;
  оракул — истинная X < θ_X.
Подбор порогов — на обучающих посевах, все сравнения и z — на отложенных (общие посевы, попарно).

python toxin_precheck.py [--lives 200]
"""
import argparse, json, math, os, time
import numpy as np

import env_flip as ef
import env_toxin as et
from stand_gpu import git_info

TAUS = [20, 50, 100, 200]
TH_E = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
TH_G = [0.25, 0.5, 0.75]
TH_LOW = [0.0, 0.1, 0.2]
TH_T = [10, 20, 40, 80, 160, 320]
TH_XF = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]          # доли X_C
TAU_F = [0.25, 0.5, 1, 2, 4]                                       # τ̂ / τ
COARSE_XF = [0.2, 0.5, 0.85]
COARSE_E = [0.4, 0.7, 1.01]
W = 32
REACTIVE = ('ord', 'rich', 'any', 'energy', 'hint')
SEED_TRAIN, SEED_TEST = 70000, 71000


def grid(tau):
    P = [dict(fam='ord'), dict(fam='rich'), dict(fam='any')]
    P += [dict(fam='energy', e=e) for e in TH_E]
    P += [dict(fam='hint', e=e, g=g, low=lo) for e in TH_E for g in TH_G for lo in TH_LOW]
    P += [dict(fam='time', e=e, tt=tt) for e in TH_E for tt in TH_T]
    for fam in ('counter', 'win32'):
        P += [dict(fam=fam, e=e, x=xf * et.X_C, xf=xf, tf=tf, th=tau * tf) for tf in TAU_F for xf in TH_XF for e in TH_E]
    P += [dict(fam='oracle', e=e, x=xf * et.X_C, xf=xf) for xf in TH_XF for e in TH_E]
    return P


def simulate(P, tau, seeds, record=False):
    """F (len(P), n) — средняя энергия за жизнь; record — ряды E, X, датчиков, X̂ для диагностики."""
    n, NP = len(seeds), len(P)
    b = et.ToxinBodies(np.tile(seeds, NP), tau)
    rep = lambda key, dflt: np.repeat(np.array([p.get(key, dflt) for p in P], float), n)
    fam = np.repeat(np.array([p['fam'] for p in P]), n)
    th_e, th_g, th_low, th_t = rep('e', 1.01), rep('g', 0.0), rep('low', 0.0), rep('tt', 0.0)
    th_x, th_hat = rep('x', np.inf), rep('th', tau)
    dec = np.exp(-1.0 / th_hat); dose = 100.0 / th_hat
    is_ = {f: fam == f for f in set(fam)}
    zero = np.zeros(b.B, bool)
    g_last = np.full(b.B, np.inf); t_last = np.full(b.B, -10 ** 6); xh = np.zeros(b.B)
    ring = np.zeros((b.B, W)); xw = np.zeros(b.B); decW = dec ** W
    rec = {'E': [], 'X': [], 'S': [], 'xh': [], 'xw': [], 'gl': [], 'tl': []} if record else None
    for t in range(b.L):
        s = b.sense()
        E = s[:, 6].astype(float); ar = s[:, 11] > 0
        g_last = np.where(ar, (s[:, 7] / 10 + ef.DRIFT) / et.G_RICH, g_last)
        t_last = np.where(ar, t, t_last)
        xh = xh * dec + ar * dose
        xw = np.maximum(xw * dec + ar * dose - ring[:, t % W] * dose * decW, 0.0)   # окно: событие 32 такта назад выпадает
        ring[:, t % W] = ar
        if record:
            rec['E'].append(E.copy()); rec['X'].append(b.X.copy()); rec['S'].append(s.copy()); rec['xh'].append(xh.copy())
            rec['xw'].append(xw.copy()); rec['gl'].append(np.minimum(g_last, 2.0)); rec['tl'].append(np.minimum(t - t_last, 2000))
        hungry = E < th_e
        cond = np.zeros(b.B, bool)
        cond |= is_.get('rich', zero)
        cond |= is_.get('energy', zero)
        cond |= is_.get('hint', zero) & ((g_last >= th_g) | (E < th_low))
        cond |= is_.get('time', zero) & ((t - t_last) >= th_t)
        cond |= is_.get('counter', zero) & (xh < th_x)
        cond |= is_.get('win32', zero) & (xw < th_x)
        cond |= is_.get('oracle', zero) & (b.X < th_x)
        rich = cond & (hungry | is_.get('rich', zero))
        anyk = is_.get('any', zero)
        rich = np.where(anyk, s[:, 5] < s[:, 2], rich)
        b.step(et.steer(s, rich))
    F = b.fitness().reshape(NP, n)
    extra = {'rich_per100': (b.n_rich_eat / b.L * 100).reshape(NP, n).mean(1), 'harm_per100': (b.n_harm / b.L * 100).reshape(NP, n).mean(1),
             'floor': (b.t_floor / b.L).reshape(NP, n).mean(1)}
    if record:
        rec = {k_: np.stack(v) for k_, v in rec.items()}                   # (L, B, ...)
    return F, extra, rec


def zp(a, b):
    d = a - b; return float(d.mean()), float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12))


def snr4(a, b):
    d = a - b; return float(d.mean() / (d.std(ddof=1) / 2 + 1e-12))


def r2(X, y):
    A = np.c_[X, np.ones(len(X))]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - ((y - A @ coef) ** 2).mean() / y.var())


def label(p):
    return p['fam'] + ''.join(f' {k}={v:g}' for k, v in p.items() if k not in ('fam', 'xf', 'th'))


def best(P, Ftr, sel):
    idx = [i for i, p in enumerate(P) if sel(p)]
    return idx[int(np.argmax([Ftr[i].mean() for i in idx]))]


def per_tau(tau, lives):
    t0 = time.time()
    P = grid(tau)
    tr = np.arange(SEED_TRAIN, SEED_TRAIN + lives); te = np.arange(SEED_TEST, SEED_TEST + lives)
    Ftr, _, _ = simulate(P, tau, tr)
    Fte, ex, _ = simulate(P, tau, te)
    i_R = best(P, Ftr, lambda p: p['fam'] in REACTIVE)
    i_RE = best(P, Ftr, lambda p: p['fam'] == 'energy')
    i_C = best(P, Ftr, lambda p: p['fam'] == 'counter')
    i_O = best(P, Ftr, lambda p: p['fam'] == 'oracle')
    i_T = best(P, Ftr, lambda p: p['fam'] == 'time')
    i_H = best(P, Ftr, lambda p: p['fam'] == 'hint')
    i_W = best(P, Ftr, lambda p: p['fam'] == 'win32')
    i_Cfix = {tf: best(P, Ftr, lambda p, tf=tf: p['fam'] == 'counter' and p['tf'] == tf) for tf in TAU_F}
    i_Wfix = {tf: best(P, Ftr, lambda p, tf=tf: p['fam'] == 'win32' and p['tf'] == tf) for tf in TAU_F}
    i_coarse = {tf: best(P, Ftr, lambda p, tf=tf: p['fam'] == 'counter' and p['tf'] == tf and p['xf'] in COARSE_XF and p['e'] in COARSE_E)
                for tf in TAU_F}
    R, RE, C, O = Fte[i_R], Fte[i_RE], Fte[i_C], Fte[i_O]
    gap = O.mean() - R.mean()
    out = {'tau': tau, 'sel': {k: label(P[i]) for k, i in (('R', i_R), ('R_E', i_RE), ('C', i_C), ('O', i_O), ('time', i_T), ('hint', i_H), ('win32', i_W))},
           'test': {k: float(Fte[i].mean()) for k, i in (('R', i_R), ('R_E', i_RE), ('C', i_C), ('O', i_O), ('time', i_T), ('hint', i_H), ('win32', i_W))},
           'ord': float(Fte[0].mean()), 'rich': float(Fte[1].mean()), 'any': float(Fte[2].mean()),
           'rich_per100': {k: float(ex['rich_per100'][i]) for k, i in (('R', i_R), ('C', i_C), ('O', i_O))},
           'harm_per100': {k: float(ex['harm_per100'][i]) for k, i in (('R', i_R), ('C', i_C), ('O', i_O))},
           'floor': {k: float(ex['floor'][i]) for k, i in (('R', i_R), ('C', i_C), ('O', i_O))}}
    d1, z1 = zp(O, R); d2, z2 = zp(C, R); dce, zce = zp(C, RE)
    closure = (C.mean() - R.mean()) / gap if gap > 0 else float('nan')
    out['c1'] = {'O_minus_R': d1, 'z': z1, 'pass': bool(d1 > 0 and z1 >= 3)}
    out['c2'] = {'C_minus_R': d2, 'z': z2, 'closure': closure, 'pass': bool(z2 >= 3 and closure >= 0.5)}
    # критерий 3 — диагностический проход на отложенных посевах для C и R
    Pd = [P[i_C], P[i_R]]
    _, _, rec = simulate(Pd, tau, te, record=True)
    n = lives
    diag = {}
    for j, nm in enumerate(('C', 'R')):
        sl = slice(j * n, (j + 1) * n)
        Ev, Xv = rec['E'][:, sl].ravel(), rec['X'][:, sl].ravel()
        Sv = rec['S'][:, sl].reshape(-1, rec['S'].shape[-1]).astype(float)
        r_pool = float(np.corrcoef(Ev, Xv)[0, 1])
        r_in = [np.corrcoef(rec['E'][:, sl][:, l], rec['X'][:, sl][:, l])[0, 1] for l in range(n)]
        diag[nm] = {'r_pool': r_pool, 'r_within_mean': float(np.nanmean(r_in)),
                    'R2_sensors': r2(np.c_[Sv, Sv[:, 6] ** 2], Xv),
                    'R2_xhat': r2(rec['xh'][:, sl].reshape(-1, 1), Xv), 'R2_xw32': r2(rec['xw'][:, sl].reshape(-1, 1), Xv),
                    'R2_hint': r2(np.c_[rec['gl'][:, sl].ravel(), np.log1p(rec['tl'][:, sl].ravel()), rec['gl'][:, sl].ravel() * np.log1p(rec['tl'][:, sl].ravel()), Ev], Xv),
                    'X_mean': float(Xv.mean()), 'X_p90': float(np.percentile(Xv, 90))}
    out['c3'] = {**diag, 'C_minus_RE': dce, 'z_C_RE': zce,
                 'fail': bool(abs(diag['C']['r_pool']) >= 0.5 and zce < 3)}
    # критерий 4 — путь
    coarse = {}
    for tf in (0.5, 2):
        d, z = zp(Fte[i_coarse[tf]], R)
        coarse[str(tf)] = {'label': label(P[i_coarse[tf]]), 'F': float(Fte[i_coarse[tf]].mean()), 'minus_R': d, 'z': z, 'snr4': snr4(Fte[i_coarse[tf]], R)}
    steps = []
    for far, near in ((0.25, 0.5), (0.5, 1), (4, 2), (2, 1)):
        d, z = zp(Fte[i_Cfix[near]], Fte[i_Cfix[far]])
        steps.append({'from': far, 'to': near, 'd': d, 'z': z, 'ok': bool(d >= 0 or z > -2)})
    out['c4'] = {'coarse': coarse, 'steps': steps, 'F_by_tauhat': {str(tf): float(Fte[i_Cfix[tf]].mean()) for tf in TAU_F},
                 'pass': bool(all(c['minus_R'] > 0 and c['z'] >= 2 for c in coarse.values()) and all(s_['ok'] for s_ in steps))}
    # описательно 5
    out['c5'] = {'W32_by_tauhat': {str(tf): float(Fte[i_Wfix[tf]].mean()) for tf in TAU_F},
                 'C_minus_W32': zp(C, Fte[i_W]), 'W32_minus_R': zp(Fte[i_W], R),
                 'time_minus_R': zp(Fte[i_T], R), 'hint_minus_RE': zp(Fte[i_H], RE),
                 'snr4': {'R_to_coarse2': snr4(Fte[i_coarse[2]], R), 'coarse2_to_C': snr4(C, Fte[i_coarse[2]]),
                          'R_to_C': snr4(C, R), 'any_to_R': snr4(R, Fte[2]), 'ord_to_R': snr4(R, Fte[0])}}
    # θ_X-соседи вокруг C (тот же τ̂ и θ_E) — шаги для будущего обучения
    pc = P[i_C]
    row = [i for i, p in enumerate(P) if p['fam'] == 'counter' and p['tf'] == pc['tf'] and p['e'] == pc['e']]
    out['c5']['theta_x_row'] = {'xf': [P[i]['xf'] for i in row], 'F': [float(Fte[i].mean()) for i in row],
                                'snr4_steps': [snr4(Fte[row[j + 1]], Fte[row[j]]) for j in range(len(row) - 1)]}
    # таблицы (отложенные посевы)
    out['tables'] = {
        'energy': {f'{p["e"]:g}': float(Fte[i].mean()) for i, p in enumerate(P) if p['fam'] == 'energy'},
        'counter_tauhat_x': {str(tf): {f'{xf:g}': max(float(Fte[i].mean()) for i, p in enumerate(P) if p['fam'] == 'counter' and p['tf'] == tf and p['xf'] == xf)
                                       for xf in TH_XF} for tf in TAU_F},
        'win32_tauhat_x': {str(tf): {f'{xf:g}': max(float(Fte[i].mean()) for i, p in enumerate(P) if p['fam'] == 'win32' and p['tf'] == tf and p['xf'] == xf)
                                     for xf in TH_XF} for tf in TAU_F},
        'oracle_x': {f'{xf:g}': max(float(Fte[i].mean()) for i, p in enumerate(P) if p['fam'] == 'oracle' and p['xf'] == xf) for xf in TH_XF},
        'time_t': {str(tt): max(float(Fte[i].mean()) for i, p in enumerate(P) if p['fam'] == 'time' and p['tt'] == tt) for tt in TH_T},
        'hint_g_low': {f'{g:g}/{lo:g}': max(float(Fte[i].mean()) for i, p in enumerate(P) if p['fam'] == 'hint' and p['g'] == g and p['low'] == lo)
                       for g in TH_G for lo in TH_LOW}}
    out['passes'] = bool(out['c1']['pass'] and out['c2']['pass'] and not out['c3']['fail'] and out['c4']['pass'])
    out['sec'] = round(time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lives', type=int, default=200)
    ap.add_argument('--taus', default=','.join(map(str, TAUS)))
    ap.add_argument('--out', default='results_toxin')
    a = ap.parse_args()
    commit, dirty = git_info()
    res = {'git_commit': commit, 'git_dirty': dirty, 'lives': a.lives, 'env': {'G_ORD': et.G_ORD, 'G_RICH': et.G_RICH, 'X_C': et.X_C, 'L': et.L,
           'dose': [et.DOSE_LO, et.DOSE_HI]}, 'taus': {}}
    os.makedirs(a.out, exist_ok=True)
    for tau in map(int, a.taus.split(',')):
        o = per_tau(tau, a.lives)
        res['taus'][str(tau)] = o
        t = o['test']
        print(f"τ = {tau}: R* {t['R']:.4f} ({o['sel']['R']}) | R_E {t['R_E']:.4f} | C {t['C']:.4f} ({o['sel']['C']}) | O {t['O']:.4f} | "
              f"1: z {o['c1']['z']:.1f} {o['c1']['pass']} | 2: доля {o['c2']['closure']:.2f}, z {o['c2']['z']:.1f} {o['c2']['pass']} | "
              f"3: r {o['c3']['C']['r_pool']:+.2f}, R² датч {o['c3']['C']['R2_sensors']:.2f}, провал {o['c3']['fail']} | 4: {o['c4']['pass']} | "
              f"проходит {o['passes']} | {o['sec']} с", flush=True)
        json.dump(res, open(os.path.join(a.out, 'precheck.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
