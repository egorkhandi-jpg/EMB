"""Звено 6, токсин-2 (медленное всасывание, env_toxin.ToxinBodies(tau_abs=...)): эвристики, без ES и без LM, CPU.
Пререгистрация и правило остановки — docs/RESULTS_2026-10-04_toxin.md, раздел «Токсин-2».

Семьи — как в toxin_precheck.py (токсин-1), плюс:
  time   — таймер рефрактерности, θ_t ∈ {1, 3, 5, 7, 10, 15, 20, 40, 80}; timew — он же по окну 32 (описательно);
  counter1 / win1 — счётчик одного отсека (токсин-1), полный / по окну 32;
  counter2 / win2 — модель себя с двумя отсеками: Ĝ, X̂ с оценками τ̂_abs, τ̂_tox (то же устройство, что у тела, доза средняя),
           богатую — если X̂ + w Ĝ < θ_X; окно 32 — вклад только событий последних 32 тактов (точно: s_t − A³² s_{t−32});
  oracle — истинные X + w G < θ_X.
Подбор — на обучающих посевах, сравнения — на отложенных.

python toxin2_precheck.py [--lives 200] [--pairs 20:50,40:50,...]
"""
import argparse, json, math, os, time
import numpy as np

import env_flip as ef
import env_toxin as et
from stand_gpu import git_info
from toxin_precheck import zp, snr4, r2, label, best, SEED_TRAIN, SEED_TEST, TH_E, TH_G, TH_LOW, TH_XF, COARSE_XF

PAIRS = [(20, 50), (40, 50), (80, 50), (20, 100), (40, 100), (80, 100)]
TH_T = [1, 3, 5, 7, 10, 15, 20, 40, 80]
TH_E_C = [0.7, 0.8, 0.9, 1.01]                       # θ_E у счётчиков (в токсине-1 счётчики выбирали 0,9 / 1,01)
TAU1_F = [0.25, 0.5, 1, 2, 4]                        # τ̂ / τ_tox у счётчика одного отсека
TAU2_F = [0.5, 1, 2]                                 # τ̂_abs / τ_abs и τ̂_tox / τ_tox у счётчика двух отсеков
WS = [0.0, 0.5, 1.0]
COARSE_E2 = [0.7, 1.01]
W = 32
REACTIVE = ('ord', 'rich', 'any', 'energy', 'hint')
CHUNK = 600


def grid(ta, tt):
    P = [dict(fam='ord'), dict(fam='rich'), dict(fam='any')]
    P += [dict(fam='energy', e=e) for e in TH_E]
    P += [dict(fam='hint', e=e, g=g, low=lo) for e in TH_E for g in TH_G for lo in TH_LOW]
    for fam in ('time', 'timew'):
        P += [dict(fam=fam, e=e, tt=t_) for e in TH_E for t_ in TH_T]
    for fam in ('counter1', 'win1'):
        P += [dict(fam=fam, e=e, x=xf * et.X_C, xf=xf, tf=tf, tht=tt * tf) for tf in TAU1_F for xf in TH_XF for e in TH_E_C]
    for fam in ('counter2', 'win2'):
        P += [dict(fam=fam, e=e, x=xf * et.X_C, xf=xf, w=w, af=af, tf=tf, tha=ta * af, tht=tt * tf)
              for af in TAU2_F for tf in TAU2_F for w in WS for xf in TH_XF for e in TH_E_C]
    P += [dict(fam='oracle', e=e, x=xf * et.X_C, xf=xf, w=w) for w in WS for xf in TH_XF for e in TH_E]
    return P


def simulate(P, ta, tt, seeds, record=False):
    n, NP = len(seeds), len(P)
    b = et.ToxinBodies(np.tile(seeds, NP), tt, tau_abs=ta)
    rep = lambda key, dflt: np.repeat(np.array([p.get(key, dflt) for p in P], float), n)
    fam = np.repeat(np.array([p['fam'] for p in P]), n)
    th_e, th_g, th_low, th_t = rep('e', 1.01), rep('g', 0.0), rep('low', 0.0), rep('tt', 0.0)
    th_x, w = rep('x', np.inf), rep('w', 1.0)
    tht, tha = rep('tht', tt), rep('tha', ta)
    d = np.exp(-1.0 / tht); a_ = np.exp(-1.0 / tha); dose = 100.0 / tht
    # A^W для окна: G ← a^W G; X ← d^W X + (1 − a) (a^W − d^W) / (a − d) G
    aW, dW = a_ ** W, d ** W
    dd = np.where(np.abs(a_ - d) > 1e-9, a_ - d, 1e-9)
    cW = (1 - a_) * (aW - dW) / dd
    is_ = {f: fam == f for f in set(fam)}
    zero = np.zeros(b.B, bool)
    g_last = np.full(b.B, np.inf); t_last = np.full(b.B, -10 ** 6)
    x1 = np.zeros(b.B); x1w = np.zeros(b.B); ring1 = np.zeros((b.B, W))
    G2 = np.zeros(b.B); X2 = np.zeros(b.B); ring2 = np.zeros((b.B, W, 2))
    rec = {'E': [], 'X': [], 'G': [], 'S': []} if record else None
    for t in range(b.L):
        s = b.sense()
        E = s[:, 6].astype(float); ar = s[:, 11] > 0
        g_last = np.where(ar, (s[:, 7] / 10 + ef.DRIFT) / et.G_RICH, g_last)
        t_last = np.where(ar, t, t_last)
        # один отсек (как токсин-1)
        x1 = x1 * d + ar * dose
        x1w = np.maximum(x1w * d + ar * dose - ring1[:, t % W] * dose * dW, 0.0)
        ring1[:, t % W] = ar
        # два отсека: то же устройство, что у тела (X распадается, перетекает доля G, затем доза в G)
        X2 = X2 * d; fl = G2 * (1 - a_); G2 = G2 - fl; X2 = X2 + fl; G2 = G2 + ar * dose
        oldG, oldX = ring2[:, t % W, 0], ring2[:, t % W, 1]
        G2w = np.maximum(G2 - aW * oldG, 0.0); X2w = np.maximum(X2 - (dW * oldX + cW * oldG), 0.0)
        ring2[:, t % W, 0] = G2; ring2[:, t % W, 1] = X2
        if record:
            rec['E'].append(E.copy()); rec['X'].append(b.X.copy()); rec['G'].append(b.G.copy()); rec['S'].append(s.copy())
        since = t - t_last
        since_w = np.where(since < W, since, np.inf)
        cond = np.zeros(b.B, bool)
        cond |= is_.get('rich', zero) | is_.get('energy', zero)
        cond |= is_.get('hint', zero) & ((g_last >= th_g) | (E < th_low))
        cond |= is_.get('time', zero) & (since >= th_t)
        cond |= is_.get('timew', zero) & (since_w >= th_t)
        cond |= is_.get('counter1', zero) & (x1 < th_x)
        cond |= is_.get('win1', zero) & (x1w < th_x)
        cond |= is_.get('counter2', zero) & (X2 + w * G2 < th_x)
        cond |= is_.get('win2', zero) & (X2w + w * G2w < th_x)
        cond |= is_.get('oracle', zero) & (b.X + w * b.G < th_x)
        rich = cond & ((E < th_e) | is_.get('rich', zero))
        rich = np.where(is_.get('any', zero), s[:, 5] < s[:, 2], rich)
        b.step(et.steer(s, rich))
    F = b.fitness().reshape(NP, n)
    extra = {'rich_per100': (b.n_rich_eat / b.L * 100).reshape(NP, n).mean(1), 'harm_per100': (b.n_harm / b.L * 100).reshape(NP, n).mean(1)}
    if record:
        rec = {k: np.stack(v) for k, v in rec.items()}
    return F, extra, rec


def sim_chunked(P, ta, tt, seeds):
    Fs, ex = [], {'rich_per100': [], 'harm_per100': []}
    for i in range(0, len(P), CHUNK):
        F, e, _ = simulate(P[i:i + CHUNK], ta, tt, seeds)
        Fs.append(F); [ex[k].append(e[k]) for k in ex]
    return np.concatenate(Fs), {k: np.concatenate(v) for k, v in ex.items()}


def per_pair(ta, tt, lives):
    t0 = time.time()
    P = grid(ta, tt)
    tr = np.arange(SEED_TRAIN, SEED_TRAIN + lives); te = np.arange(SEED_TEST, SEED_TEST + lives)
    Ftr, _ = sim_chunked(P, ta, tt, tr)
    Fte, ex = sim_chunked(P, ta, tt, te)
    fam_is = lambda *fs: (lambda p: p['fam'] in fs)
    I = {'R': best(P, Ftr, fam_is(*REACTIVE)), 'R_E': best(P, Ftr, fam_is('energy')), 'C': best(P, Ftr, fam_is('counter1', 'counter2')),
         'C1': best(P, Ftr, fam_is('counter1')), 'C2': best(P, Ftr, fam_is('counter2')), 'O': best(P, Ftr, fam_is('oracle')),
         'T': best(P, Ftr, fam_is('time')), 'TW': best(P, Ftr, fam_is('timew')), 'W': best(P, Ftr, fam_is('win1', 'win2')),
         'W1': best(P, Ftr, fam_is('win1')), 'W2': best(P, Ftr, fam_is('win2')), 'hint': best(P, Ftr, fam_is('hint'))}
    F = {k: Fte[i] for k, i in I.items()}
    m = {k: float(v.mean()) for k, v in F.items()}
    ordF = Fte[0]
    out = {'tau_abs': ta, 'tau_tox': tt, 'sel': {k: label(P[i]) for k, i in I.items()}, 'test': m,
           'ord': float(ordF.mean()), 'rich': float(Fte[1].mean()), 'any': float(Fte[2].mean()),
           'rich_per100': {k: float(ex['rich_per100'][I[k]]) for k in ('R', 'C', 'O', 'T')},
           'harm_per100': {k: float(ex['harm_per100'][I[k]]) for k in ('R', 'C', 'O', 'T')}}
    gap = m['O'] - m['R']
    d1, z1 = zp(F['O'], F['R']); d2, z2 = zp(F['C'], F['R']); dce, zce = zp(F['C'], F['R_E'])
    closure = (m['C'] - m['R']) / gap if gap > 0 else float('nan')
    out['c1'] = {'O_minus_R': d1, 'z': z1, 'pass': bool(d1 > 0 and z1 >= 3)}
    out['c2'] = {'C_minus_R': d2, 'z': z2, 'closure': closure, 'pass': bool(z2 >= 3 and closure >= 0.5)}
    # 3 — диагностический проход для C и R*
    _, _, rec = simulate([P[I['C']], P[I['R']]], ta, tt, te, record=True)
    n = lives; diag = {}
    for j, nm in enumerate(('C', 'R')):
        sl = slice(j * n, (j + 1) * n)
        Ev, Xv, Gv = rec['E'][:, sl].ravel(), rec['X'][:, sl].ravel(), rec['G'][:, sl].ravel()
        Sv = rec['S'][:, sl].reshape(-1, rec['S'].shape[-1]).astype(float)
        diag[nm] = {'r_pool': float(np.corrcoef(Ev, Xv)[0, 1]),
                    'r_within_mean': float(np.nanmean([np.corrcoef(rec['E'][:, sl][:, l], rec['X'][:, sl][:, l])[0, 1] for l in range(n)])),
                    'R2_sensors_X': r2(np.c_[Sv, Sv[:, 6] ** 2], Xv), 'R2_sensors_XG': r2(np.c_[Sv, Sv[:, 6] ** 2], Xv + Gv),
                    'X_mean': float(Xv.mean()), 'G_mean': float(Gv.mean()), 'X_p90': float(np.percentile(Xv, 90))}
    out['c3'] = {**diag, 'C_minus_RE': dce, 'z_C_RE': zce, 'fail': bool(abs(diag['C']['r_pool']) >= 0.5 and zce < 3)}
    # 4 — путь по модели двух отсеков
    def b2(af, tf, coarse=False):
        return best(P, Ftr, lambda p: p['fam'] == 'counter2' and p['af'] == af and p['tf'] == tf and
                    (not coarse or (p['xf'] in COARSE_XF and p['e'] in COARSE_E2 and p['w'] == 1.0)))
    coarse = {}
    for f_ in (0.5, 2):
        i = b2(f_, f_, True); dd_, z_ = zp(Fte[i], F['R'])
        coarse[str(f_)] = {'label': label(P[i]), 'F': float(Fte[i].mean()), 'minus_R': dd_, 'z': z_, 'snr4': snr4(Fte[i], F['R'])}
    steps = []
    for (fa, ft), (na, nt) in ((((0.5, 1), (1, 1))), (((2, 1), (1, 1))), (((1, 0.5), (1, 1))), (((1, 2), (1, 1)))):
        dd_, z_ = zp(Fte[b2(na, nt)], Fte[b2(fa, ft)])
        steps.append({'from': [fa, ft], 'to': [na, nt], 'd': dd_, 'z': z_, 'ok': bool(dd_ >= 0 or z_ > -2)})
    out['c4'] = {'coarse': coarse, 'steps': steps,
                 'F_by_tauhat2': {f'{af}/{tf}': float(Fte[b2(af, tf)].mean()) for af in TAU2_F for tf in TAU2_F},
                 'pass': bool(all(c['minus_R'] > 0 and c['z'] >= 2 for c in coarse.values()) and all(s_['ok'] for s_ in steps))}
    # 5 — вырождения нет (обязательный)
    t_closure = (m['T'] - m['R']) / gap if gap > 0 else float('nan')
    dcw, zcw = zp(F['C'], F['W'])
    re_closure = (m['R_E'] - out['ord']) / (m['O'] - out['ord'])
    out['c5'] = {'a_timer_closure': t_closure, 'a_pass': bool(gap > 0 and t_closure < 0.5),
                 'b_C_minus_W': dcw, 'b_z': zcw, 'b_pass': bool(zcw >= 3),
                 'c_RE_closure_from_ord': re_closure, 'c_pass': bool(re_closure < 0.5)}
    out['c5']['pass'] = out['c5']['a_pass'] and out['c5']['b_pass'] and out['c5']['c_pass']
    # описательно
    out['desc'] = {'C2_minus_C1': zp(F['C2'], F['C1']), 'C1_minus_W1': zp(F['C1'], F['W1']), 'C2_minus_W2': zp(F['C2'], F['W2']),
                   'T_minus_TW': zp(F['T'], F['TW']), 'O_minus_T': zp(F['O'], F['T']),
                   'snr4': {'ord_to_R': snr4(F['R'], ordF), 'R_to_T': snr4(F['T'], F['R']), 'T_to_C': snr4(F['C'], F['T']),
                            'R_to_C': snr4(F['C'], F['R']), 'R_to_coarse2': snr4(Fte[b2(2, 2, True)], F['R']),
                            'coarse2_to_C': snr4(F['C'], Fte[b2(2, 2, True)])},
                   'timer': {str(t_): float(max(Fte[i].mean() for i, p in enumerate(P) if p['fam'] == 'time' and p['tt'] == t_)) for t_ in TH_T},
                   'timer_w': {str(t_): float(max(Fte[i].mean() for i, p in enumerate(P) if p['fam'] == 'timew' and p['tt'] == t_)) for t_ in TH_T},
                   'energy': {f'{p["e"]:g}': float(Fte[i].mean()) for i, p in enumerate(P) if p['fam'] == 'energy'},
                   'counter1_by_tauhat': {str(tf): float(Fte[best(P, Ftr, lambda p, tf=tf: p['fam'] == 'counter1' and p['tf'] == tf)].mean()) for tf in TAU1_F},
                   'win1_by_tauhat': {str(tf): float(Fte[best(P, Ftr, lambda p, tf=tf: p['fam'] == 'win1' and p['tf'] == tf)].mean()) for tf in TAU1_F},
                   'oracle_by_w': {f'{w_:g}': float(Fte[best(P, Ftr, lambda p, w_=w_: p['fam'] == 'oracle' and p['w'] == w_)].mean()) for w_ in WS}}
    out['passes'] = bool(out['c1']['pass'] and out['c2']['pass'] and not out['c3']['fail'] and out['c4']['pass'] and out['c5']['pass'])
    out['sec'] = round(time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lives', type=int, default=200)
    ap.add_argument('--pairs', default=','.join(f'{a}:{b}' for a, b in PAIRS))
    ap.add_argument('--out', default='results_toxin')
    a = ap.parse_args()
    commit, dirty = git_info()
    res = {'git_commit': commit, 'git_dirty': dirty, 'lives': a.lives, 'env': {'G_ORD': et.G_ORD, 'G_RICH': et.G_RICH, 'X_C': et.X_C, 'L': et.L},
           'pairs': {}}
    os.makedirs(a.out, exist_ok=True)
    for pr in a.pairs.split(','):
        ta, tt = map(int, pr.split(':'))
        o = per_pair(ta, tt, a.lives)
        res['pairs'][pr] = o
        t = o['test']
        print(f"τ_abs {ta}, τ_tox {tt}: ord {o['ord']:.4f} R* {t['R']:.4f} ({o['sel']['R']}) | T {t['T']:.4f} | W {t['W']:.4f} | C {t['C']:.4f} ({o['sel']['C']}) | "
              f"O {t['O']:.4f} | 1 {o['c1']['pass']} (z {o['c1']['z']:.1f}) | 2 {o['c2']['pass']} ({o['c2']['closure']:.2f}) | 3 провал {o['c3']['fail']} | "
              f"4 {o['c4']['pass']} | 5а {o['c5']['a_pass']} ({o['c5']['a_timer_closure']:.2f}) 5б {o['c5']['b_pass']} (z {o['c5']['b_z']:.1f}) "
              f"5в {o['c5']['c_pass']} ({o['c5']['c_RE_closure_from_ord']:.2f}) | проходит {o['passes']} | {o['sec']} с", flush=True)
        json.dump(res, open(os.path.join(a.out, 'precheck2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
