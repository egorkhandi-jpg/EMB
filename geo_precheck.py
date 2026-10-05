"""Тезис 5, этап 1: предварительные проверки среды «я / мир» (env_geo.py) эвристиками, без ES и без LM, CPU.
Пререгистрация — docs/RESULTS_2026-10-05_geometry.md.

Общий скелет политик: у каждого предмета счёт «живой»; цель — предмет с наибольшим счётом, если он выше порога θ_s,
иначе ближайший; курс на цель (поворот, если |b| > TURN_G / 2; вперёд, если |b| < π/2). Появление предмета заново
(перестановка) обнуляет его счёт. Семьи:
  R0     — к ближайшему, без памяти;
  O      — оракул: к живому;
  FZ     — «замри и смотри»: стоять k тактов (без поворота), затем к предмету с наибольшим смещением за стоянку;
           повтор после P тактов движения и после любого появления предмета;
  NOEFF  — история событий без своих действий: затухающий счёт событий сдвига направления |Δb| > θ_ev; цель — наибольший
           счёт, если он больше второго на m, иначе ближайший;
  COMMON — опора на мир, без знания своих действий: общий сдвиг всех предметов считается своим; варианты
           med (медиана сдвига направления и дистанции), rigid (жёсткое преобразование по всем предметам, МНК),
           robust (то же, затем без предмета с наибольшим остатком, если остаётся ≥ 2); остаток копится как «живой»;
  EFF    — история + свои действия: предсказанное положение неподвижного предмета после своего действия
           (поворот на −turn·TURN_G, сдвиг на −move·VMAX вдоль оси), остаток копится как «живой»;
  EFFC   — то же с грубой моделью кинематики: поворот × {0,7; 1,3}, шаг × {0,5; 1,5}.
Накопление остатка: S ← λ R S + r (R — поворот кадра по своей модели: известный у EFF, оценённый у COMMON), счёт = |S|.
Подбор параметров — на обучающих посевах, сравнения — на отложенных.

python geo_precheck.py [--lives 200] [--configs K:U:Lfood,...]
"""
import argparse, json, math, os, time
import numpy as np

import env_geo as eg
from stand_gpu import git_info

KS = [2, 3, 4]
US = [0.002, 0.004, 0.008]
LFS = [50, 100, 200]
LAMS = [0.9, 0.97, 1.0]
THS = [0.01, 0.02, 0.04, 0.08]
SEED_TRAIN, SEED_TEST = 80000, 81000
FAMS_BASE = ('FZ', 'NOEFF', 'COMMON')


def grid():
    P = [dict(fam='R0'), dict(fam='O')]
    P += [dict(fam='FZ', k=k, per=per) for k in (2, 4, 8, 16) for per in (10, 20, 10 ** 9)]
    P += [dict(fam='NOEFF', tev=tev, lam=lam, m=m) for tev in (0.01, 0.02, 0.04) for lam in LAMS for m in (0.5, 1.0, 2.0)]
    P += [dict(fam='COMMON', var=v, lam=lam, th=th) for v in ('med', 'rigid', 'robust') for lam in LAMS for th in THS]
    P += [dict(fam='EFF', lam=lam, th=th, gs=1.0, vs=1.0) for lam in LAMS for th in THS]
    P += [dict(fam='EFFC', lam=lam, th=th, gs=gs, vs=vs) for gs in (0.7, 1.3) for vs in (0.5, 1.5) for lam in LAMS for th in THS]
    return P


def rot(v, ang):
    """Поворот векторов v (..., 2) на угол ang (широковещается по первым осям)."""
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def kabsch(a, b, w):
    """Жёсткое преобразование b ≈ R(θ) a + t по предметам с весами w (B, K); возвращает остатки (B, K, 2), θ (B,)."""
    ws = np.maximum(w.sum(1, keepdims=True), 1e-9)
    ca = (a * w[..., None]).sum(1) / ws; cb = (b * w[..., None]).sum(1) / ws
    a0, b0 = a - ca[:, None], b - cb[:, None]
    cross = (w * (a0[..., 0] * b0[..., 1] - a0[..., 1] * b0[..., 0])).sum(1)
    dot = (w * (a0[..., 0] * b0[..., 0] + a0[..., 1] * b0[..., 1])).sum(1)
    th = np.arctan2(cross, dot)
    t = cb - rot(ca, th)
    return b - (rot(a, th[:, None]) + t[:, None]), th


def steer(b_t):
    turn = np.where(b_t > eg.TURN_G / 2, 1, np.where(b_t < -eg.TURN_G / 2, -1, 0))
    move = (np.abs(b_t) < np.pi / 2).astype(int)
    return turn, move


def simulate(P, cfg, seeds, record_auc=None):
    """F (len(P), n) — съедено за жизнь; E (len(P), n) — средняя энергия. record_auc='R0'/'RND' — вернуть (|Δb|, живой)."""
    K, U, LF = cfg
    if record_auc:
        P = [dict(fam=record_auc)]
    n, NP = len(seeds), len(P)
    env = eg.GeoBodies(np.tile(seeds, NP), K, U, LF)
    B = env.B
    rep = lambda key, dflt: np.repeat(np.array([p.get(key, dflt) for p in P], float), n)
    fam = np.repeat(np.array([p['fam'] for p in P]), n)
    var = np.repeat(np.array([p.get('var', '') for p in P]), n)
    lam, th, tev, mm = rep('lam', 1.0), rep('th', 0.0), rep('tev', 0.02), rep('m', 1.0)
    gs, vs, fzk, per = rep('gs', 1.0), rep('vs', 1.0), rep('k', 4).astype(int), rep('per', 10 ** 9)
    is_ = lambda f: fam == f
    S = np.zeros((B, K, 2)); C = np.zeros((B, K))
    prev = None; prev_b = None; turn = np.zeros(B, int); move = np.zeros(B, int)
    fz_left = np.where(is_('FZ'), fzk, 0); fz_ref = np.zeros((B, K, 2)); fz_target = np.zeros(B, int); fz_go = np.zeros(B, int)
    rnd = np.random.RandomState(12345 + int(seeds[0]))
    auc_x, auc_y = [], []
    ar = np.arange(B)
    for t in range(env.L):
        b, d, rsp = env.sense()
        rel = np.stack([d * np.cos(b), d * np.sin(b)], -1)
        if prev is not None:
            valid = ~rsp
            db = eg.wrap(b - prev_b)
            if record_auc:
                lab = np.zeros((B, K), bool); lab[ar, env.alive_i] = True
                auc_x.append(np.abs(db)[valid]); auc_y.append(lab[valid])
            # EFF / EFFC: предсказание по своему действию
            ang = -turn * eg.TURN_G * gs
            pred = rot(prev, ang[:, None]) - np.stack([move * eg.VMAX * vs, np.zeros(B)], -1)[:, None, :]
            r_eff = rel - pred
            # COMMON
            w = valid.astype(float)
            r_rig, th_rig = kabsch(prev, rel, w)
            err = np.where(valid, np.linalg.norm(r_rig, axis=-1), -1)
            w2 = w.copy(); w2[ar, err.argmax(1)] = 0
            ok2 = w2.sum(1) >= 2
            r_rob2, th_rob2 = kabsch(prev, rel, np.where(ok2[:, None], w2, w))
            dd = d - np.linalg.norm(prev, axis=-1)
            with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
                __import__('warnings').simplefilter('ignore')
                mb = np.nan_to_num(np.nanmedian(np.where(valid, db, np.nan), 1))
                md = np.nan_to_num(np.nanmedian(np.where(valid, dd, np.nan), 1))
            r_med = np.stack([d * eg.wrap(db - mb[:, None]), dd - md[:, None]], -1)
            r = np.where((fam == 'EFF')[:, None, None] | (fam == 'EFFC')[:, None, None], r_eff,
                np.where((var == 'rigid')[:, None, None], r_rig, np.where((var == 'robust')[:, None, None], r_rob2, r_med)))
            rang = np.where((fam == 'EFF') | (fam == 'EFFC'), ang, np.where(var == 'rigid', th_rig, np.where(var == 'robust', th_rob2, mb)))
            S = lam[:, None, None] * rot(S, rang[:, None]) + r
            C = lam[:, None] * C + (np.abs(db) > tev[:, None])
        S[rsp] = 0.0; C[rsp] = 0.0
        score = np.linalg.norm(S, axis=-1)
        nearest = d.argmin(1)
        # выбор цели
        tgt = nearest.copy()
        top = score.argmax(1); smax = score.max(1)
        sel = np.isin(fam, ('COMMON', 'EFF', 'EFFC')) & (smax > th)
        tgt = np.where(sel, top, tgt)
        cs = np.sort(C, 1); ctop = C.argmax(1)
        sel_n = is_('NOEFF') & (cs[:, -1] - cs[:, -2] >= mm)
        tgt = np.where(sel_n, ctop, tgt)
        tgt = np.where(is_('O'), env.alive_i, tgt)
        # FZ: стоянка и решение
        if is_('FZ').any():
            restart = is_('FZ') & rsp.any(1) & (fz_left == 0)
            restart |= is_('FZ') & (fz_left == 0) & (fz_go >= per)
            fz_left = np.where(restart, fzk, fz_left)
            fz_left = np.where(is_('FZ') & rsp.any(1) & (fz_left > 0), fzk, fz_left)       # появление во время стоянки — заново
            start = is_('FZ') & (fz_left == fzk)
            fz_ref = np.where(start[:, None, None], rel, fz_ref)
            done = is_('FZ') & (fz_left == 1)
            moved = np.linalg.norm(rel - fz_ref, axis=-1)
            fz_target = np.where(done, moved.argmax(1), fz_target)
            fz_go = np.where(done, 0, fz_go + 1)
            tgt = np.where(is_('FZ'), fz_target, tgt)
        turn, move = steer(b[ar, tgt])
        if is_('FZ').any():
            freezing = is_('FZ') & (fz_left > 0)
            turn = np.where(freezing, 0, turn); move = np.where(freezing, 0, move)
            fz_left = np.where(freezing, fz_left - 1, fz_left)
        if record_auc == 'RND':
            u = rnd.rand(B, 2)
            turn = np.where(u[:, 0] < 0.25, 1, np.where(u[:, 0] < 0.5, -1, 0)); move = (u[:, 1] < 0.7).astype(int)
        prev, prev_b = rel, b
        env.step(turn, move)
    if record_auc:
        return np.concatenate(auc_x), np.concatenate(auc_y)
    return env.fitness().reshape(NP, n), env.mean_energy().reshape(NP, n)


def zp(a, b):
    d = a - b; return float(d.mean()), float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12))


def auc(x, y):
    from scipy.stats import rankdata
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def label(p):
    return p['fam'] + ''.join(f' {k}={v}' for k, v in p.items() if k != 'fam')


def best(P, Ftr, fams):
    idx = [i for i, p in enumerate(P) if p['fam'] in fams]
    return idx[int(np.argmax([Ftr[i].mean() for i in idx]))]


def per_config(cfg, lives):
    t0 = time.time()
    P = grid()
    tr = np.arange(SEED_TRAIN, SEED_TRAIN + lives); te = np.arange(SEED_TEST, SEED_TEST + lives)
    Ftr, Etr = simulate(P, cfg, tr)
    Fte, Ete = simulate(P, cfg, te)
    I = {f: best(P, Ftr, (f,)) for f in ('R0', 'O', 'FZ', 'NOEFF', 'COMMON', 'EFF', 'EFFC')}
    for v in ('med', 'rigid', 'robust'):
        idx = [i for i, p in enumerate(P) if p['fam'] == 'COMMON' and p['var'] == v]
        I[f'COMMON_{v}'] = idx[int(np.argmax([Ftr[i].mean() for i in idx]))]
    F = {k: Fte[i] for k, i in I.items()}; m = {k: float(v.mean()) for k, v in F.items()}
    base_te = max(FAMS_BASE, key=lambda f: m[f])                          # наибольшая из трёх на отложенных (консервативно)
    base_tr = max(float(Ftr[I[f]].mean()) for f in FAMS_BASE)
    out = {'cfg': list(cfg), 'sel': {k: label(P[i]) for k, i in I.items()}, 'test': m,
           'test_energy': {k: float(Ete[i].mean()) for k, i in I.items()}, 'base': base_te}
    d1, z1 = zp(F['O'], F['R0'])
    d2, z2 = zp(F['EFF'], F[base_te]); gap2 = m['O'] - m[base_te]
    clos2 = d2 / gap2 if gap2 > 0 else float('nan')
    g = m['O'] - m['R0']
    out['c1'] = {'O_minus_R0': d1, 'z': z1, 'pass': bool(d1 > 0 and z1 >= 3)}
    out['c2'] = {'EFF_minus_base': d2, 'z': z2, 'closure': clos2, 'pass': bool(d2 > 0 and z2 >= 3 and clos2 >= 0.5)}
    fz_c = (m['FZ'] - m['R0']) / g if g > 0 else float('nan'); co_c = (m['COMMON'] - m['R0']) / g if g > 0 else float('nan')
    x, y = simulate(None, cfg, te, record_auc='R0'); a_r0 = auc(x, y)
    x, y = simulate(None, cfg, te, record_auc='RND'); a_rnd = auc(x, y)
    out['c3'] = {'FZ_closure': fz_c, 'a_pass': bool(fz_c < 0.5), 'COMMON_closure': co_c, 'b_pass': bool(co_c < 0.5),
                 'AUC_R0': a_r0, 'AUC_RND': a_rnd, 'c_pass': bool(max(a_r0, a_rnd) <= 0.6)}
    out['c3']['pass'] = out['c3']['a_pass'] and out['c3']['b_pass'] and out['c3']['c_pass']
    d4, z4 = zp(F['EFFC'], F[base_te])
    out['c4'] = {'EFFC_minus_base': d4, 'z': z4, 'pass': bool(d4 > 0 and z4 >= 2)}
    out['passes'] = bool(out['c1']['pass'] and out['c2']['pass'] and out['c3']['pass'] and out['c4']['pass'])
    out['margin_train'] = float(Ftr[I['EFF']].mean()) - base_tr
    out['desc'] = {k: zp(F['EFF'], F[k]) for k in ('FZ', 'NOEFF', 'COMMON', 'COMMON_med', 'COMMON_rigid', 'COMMON_robust', 'EFFC', 'R0')}
    out['desc']['O_minus_EFF'] = zp(F['O'], F['EFF'])
    out['sec'] = round(time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lives', type=int, default=200)
    ap.add_argument('--configs', default=','.join(f'{k}:{u}:{lf}' for k in KS for u in US for lf in LFS))
    ap.add_argument('--out', default='results_geo')
    a = ap.parse_args()
    commit, dirty = git_info()
    res = {'git_commit': commit, 'git_dirty': dirty, 'lives': a.lives, 'configs': {}}
    os.makedirs(a.out, exist_ok=True)
    for c in a.configs.split(','):
        k, u, lf = c.split(':'); cfg = (int(k), float(u), int(lf))
        o = per_config(cfg, a.lives)
        res['configs'][c] = o
        t = o['test']
        print(f"K {cfg[0]} U {cfg[1]} Lf {cfg[2]}: R0 {t['R0']:.2f} FZ {t['FZ']:.2f} NOEFF {t['NOEFF']:.2f} COMMON {t['COMMON']:.2f} "
              f"EFF {t['EFF']:.2f} EFFC {t['EFFC']:.2f} O {t['O']:.2f} | 1 {o['c1']['pass']} | 2 {o['c2']['pass']} "
              f"({o['c2']['closure']:.2f}, z {o['c2']['z']:.1f}, база {o['base']}) | 3а {o['c3']['FZ_closure']:.2f} 3б {o['c3']['COMMON_closure']:.2f} "
              f"3в {o['c3']['AUC_R0']:.2f}/{o['c3']['AUC_RND']:.2f} | 4 {o['c4']['pass']} (z {o['c4']['z']:.1f}) | проходит {o['passes']} | {o['sec']} с", flush=True)
        json.dump(res, open(os.path.join(a.out, 'precheck.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
