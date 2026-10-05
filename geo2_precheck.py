"""Тезис 5, этап 1 по дополнениям 1–2 (2026-10-05): эвристики в env_geo.GeoBodies2, без ES и без LM, CPU.
Пререгистрация — docs/RESULTS_2026-10-05_geometry.md, разделы «Дополнения» и «Уточнения».

Кандидаты (цели) — 2: живой и дрожащая пустышка; ориентиры — n_lm неподвижных, не цели. Семьи, как в geo_precheck.py:
  R0, O, FZ, NOEFF — по кандидатам;
  COMMON — общий сдвиг считается своим: med (медианы по всем предметам), rigid / robust (жёсткое преобразование МНК по всем
           предметам; robust — затем без худшего), lm (жёсткое по ориентирам, если их ≥ 2, иначе по всем);
  EFF / EFFC — предсказание по своему действию (команда поворота, исполненный шаг); EFFC — грубая кинематика;
  EFFA — EFF с онлайн-переоценкой связи «команда поворота → поворот» по общему повороту мира (только в опыте с поворотом).
Опыт с поворотом связи (описательно): руль у всех политик знает фактическую связь (моторная часть вынесена за скобки),
различается только восприятие.

python geo2_precheck.py [--lives 200] [--part grid|rot]
"""
import argparse, json, math, os, time, warnings
import numpy as np

import env_geo as eg
from geo_precheck import rot, kabsch, zp, auc, label, best
from stand_gpu import git_info

US = [0.002, 0.004, 0.008]
NLM = [0, 1, 2, 3]
NOISE = [1.0, 2.0, 4.0]
LFOOD = 100
LAMS = [0.9, 0.97, 1.0]
THS = [0.01, 0.02, 0.04, 0.08]
SEED_TRAIN, SEED_TEST = 82000, 83000
BASES = ('FZ', 'NOEFF', 'COMMON')


def grid():
    P = [dict(fam='R0'), dict(fam='O')]
    P += [dict(fam='FZ', k=k, per=per) for k in (2, 4, 8, 16) for per in (10, 20, 10 ** 9)]
    P += [dict(fam='NOEFF', tev=tev, lam=lam, m=m) for tev in (0.01, 0.02, 0.04) for lam in LAMS for m in (0.5, 1.0, 2.0)]
    P += [dict(fam='COMMON', var=v, lam=lam, th=th) for v in ('med', 'rigid', 'robust', 'lm') for lam in LAMS for th in THS]
    P += [dict(fam='EFF', lam=lam, th=th, gs=1.0, vs=1.0) for lam in LAMS for th in THS]
    P += [dict(fam='EFFC', lam=lam, th=th, gs=gs, vs=vs) for gs in (0.7, 1.3) for vs in (0.5, 1.5) for lam in LAMS for th in THS]
    return P


def configs():
    C = []
    for u in US:
        C.append((u, LFOOD, 0, 1.0))
        C += [(u, LFOOD, n, s) for n in NLM[1:] for s in NOISE]
    return C


def simulate(P, cfg, seeds, record_auc=None, rot_g=None, acc_rec=False):
    U, LF, n_lm, lmn = cfg
    if record_auc:
        P = [dict(fam=record_auc)]
    n, NP = len(seeds), len(P)
    env = eg.GeoBodies2(np.tile(seeds, NP), U, LF, n_lm=n_lm, lm_noise=lmn, rot_g=rot_g)
    B, K, Kc = env.B, env.K, env.Kc
    rep = lambda key, dflt: np.repeat(np.array([p.get(key, dflt) for p in P], float), n)
    fam = np.repeat(np.array([p['fam'] for p in P]), n)
    var = np.repeat(np.array([p.get('var', '') for p in P]), n)
    lam, th, tev, mm = rep('lam', 1.0), rep('th', 0.0), rep('tev', 0.02), rep('m', 1.0)
    gs, vs, fzk, per = rep('gs', 1.0), rep('vs', 1.0), rep('k', 4).astype(int), rep('per', 10 ** 9)
    is_ = lambda f: fam == f
    effm = is_('EFF') | is_('EFFC') | is_('EFFA')
    S = np.zeros((B, Kc, 2)); C = np.zeros((B, Kc)); ghat = np.ones(B)
    prev = None; prev_b = None; cmd = np.zeros(B, int); mv = np.zeros(B, int)
    fz_left = np.where(is_('FZ'), fzk, 0); fz_ref = np.zeros((B, Kc, 2)); fz_target = np.zeros(B, int); fz_go = np.zeros(B, int)
    rnd = np.random.RandomState(12345 + int(seeds[0]))
    auc_x, auc_y = [], []
    acc_hist = np.zeros((2, 401)) if acc_rec else None                                 # верно / всего по тактам от поворота −200…+200
    ar = np.arange(B)
    lm_ok = n_lm >= 2
    for t in range(env.L):
        b, d, rsp = env.sense()
        rel = np.stack([d * np.cos(b), d * np.sin(b)], -1)
        if prev is not None:
            valid = ~rsp
            db = eg.wrap(b - prev_b)
            if record_auc:
                lab = np.zeros((B, Kc), bool); lab[ar, env.alive_i] = True
                v = valid[:, :Kc]; auc_x.append(np.abs(db[:, :Kc])[v]); auc_y.append(lab[v])
            w = valid.astype(float)
            r_rig, th_rig = kabsch(prev, rel, w)
            err = np.where(valid, np.linalg.norm(r_rig, axis=-1), -1)
            w2 = w.copy(); w2[ar, err.argmax(1)] = 0
            ok2 = w2.sum(1) >= 2
            r_rob, th_rob = kabsch(prev, rel, np.where(ok2[:, None], w2, w))
            if lm_ok:
                wl = w.copy(); wl[:, :Kc] = 0
                r_lm, th_lm = kabsch(prev, rel, np.where((wl.sum(1) >= 2)[:, None], wl, w))
            else:
                r_lm, th_lm = r_rig, th_rig
            dd = d - np.linalg.norm(prev, axis=-1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mb = np.nan_to_num(np.nanmedian(np.where(valid, db, np.nan), 1))
                md = np.nan_to_num(np.nanmedian(np.where(valid, dd, np.nan), 1))
            r_med = np.stack([d * eg.wrap(db - mb[:, None]), dd - md[:, None]], -1)
            # EFFA: онлайн-оценка связи команда → поворот по общему повороту мира (жёсткое по всем предметам)
            turned = cmd != 0
            g_obs = np.where(turned, -th_rig / np.where(turned, cmd * eg.TURN_G, 1), ghat)
            ghat = np.where(is_('EFFA') & turned, ghat + 0.2 * (g_obs - ghat), ghat)
            gm = np.where(is_('EFFA'), ghat, gs)
            ang = -cmd * eg.TURN_G * gm
            pred = rot(prev, ang[:, None]) - np.stack([mv * eg.VMAX * vs, np.zeros(B)], -1)[:, None, :]
            r_eff = rel - pred
            pick = lambda a, bb, c, dd_, e: np.where(effm[:, None, None], a, np.where((var == 'rigid')[:, None, None], bb,
                                       np.where((var == 'robust')[:, None, None], c, np.where((var == 'lm')[:, None, None], dd_, e))))
            r = pick(r_eff, r_rig, r_rob, r_lm, r_med)[:, :Kc]
            rang = np.where(effm, ang, np.where(var == 'rigid', th_rig, np.where(var == 'robust', th_rob, np.where(var == 'lm', th_lm, mb))))
            S = lam[:, None, None] * rot(S, rang[:, None]) + r
            C = lam[:, None] * C + (np.abs(db[:, :Kc]) > tev[:, None])
        S[rsp[:, :Kc]] = 0.0; C[rsp[:, :Kc]] = 0.0
        score = np.linalg.norm(S, axis=-1)
        if acc_rec:
            rel_t = np.clip(t - env.t_switch + 200, 0, 400)
            ok = (score.max(1) > 0) & ~rsp[:, :Kc].any(1)
            np.add.at(acc_hist[0], rel_t[ok], (score.argmax(1) == env.alive_i)[ok])
            np.add.at(acc_hist[1], rel_t[ok], 1)
        nearest = d[:, :Kc].argmin(1)
        tgt = nearest.copy()
        sel = np.isin(fam, ('COMMON', 'EFF', 'EFFC', 'EFFA')) & (score.max(1) > th)
        tgt = np.where(sel, score.argmax(1), tgt)
        cs = np.sort(C, 1)
        tgt = np.where(is_('NOEFF') & (cs[:, -1] - cs[:, -2] >= mm), C.argmax(1), tgt)
        tgt = np.where(is_('O'), env.alive_i, tgt)
        if is_('FZ').any():
            restart = is_('FZ') & (fz_left == 0) & (rsp[:, :Kc].any(1) | (fz_go >= per))
            fz_left = np.where(restart, fzk, fz_left)
            fz_left = np.where(is_('FZ') & rsp[:, :Kc].any(1) & (fz_left > 0), fzk, fz_left)
            fz_ref = np.where((is_('FZ') & (fz_left == fzk))[:, None, None], rel[:, :Kc], fz_ref)
            done = is_('FZ') & (fz_left == 1)
            fz_target = np.where(done, np.linalg.norm(rel[:, :Kc] - fz_ref, axis=-1).argmax(1), fz_target)
            fz_go = np.where(done, 0, fz_go + 1)
            tgt = np.where(is_('FZ'), fz_target, tgt)
        # руль: знает фактическую связь (моторная адаптация вынесена за скобки)
        g = env.gain() if rot_g is not None else np.ones(B)
        bt = b[ar, tgt]
        thr = np.abs(g) * eg.TURN_G / 2
        dirn = np.where(bt > thr, 1, np.where(bt < -thr, -1, 0))
        cmd = (dirn * np.sign(g)).astype(int)
        move = (np.abs(bt) < np.pi / 2).astype(int)
        if is_('FZ').any():
            frz = is_('FZ') & (fz_left > 0)
            cmd = np.where(frz, 0, cmd); move = np.where(frz, 0, move)
            fz_left = np.where(frz, fz_left - 1, fz_left)
        if record_auc == 'RND':
            u = rnd.rand(B, 2)
            cmd = np.where(u[:, 0] < 0.25, 1, np.where(u[:, 0] < 0.5, -1, 0)); move = (u[:, 1] < 0.7).astype(int)
        prev, prev_b = rel, b
        env.step(cmd, move)
        mv = env.exec_move.copy()
    if record_auc:
        return np.concatenate(auc_x), np.concatenate(auc_y)
    out = (env.fitness().reshape(NP, n), env.mean_energy().reshape(NP, n))
    if rot_g is not None:
        out = out + (env.eaten_after.reshape(NP, n), (env.L - env.t_switch).reshape(NP, n))
    if acc_rec:
        out = out + (acc_hist,)
    return out


def run_grid(lives, out_dir, res):
    P = grid()
    tr = np.arange(SEED_TRAIN, SEED_TRAIN + lives); te = np.arange(SEED_TEST, SEED_TEST + lives)
    per_cfg = {}
    for cfg in configs():
        t0 = time.time()
        Ftr, _ = simulate(P, cfg, tr)
        Fte, Ete = simulate(P, cfg, te)
        I = {f: best(P, Ftr, (f,)) for f in ('R0', 'O', 'FZ', 'NOEFF', 'COMMON', 'EFF', 'EFFC')}
        for v in ('med', 'rigid', 'robust', 'lm'):
            idx = [i for i, p in enumerate(P) if p['fam'] == 'COMMON' and p['var'] == v]
            I[f'COMMON_{v}'] = idx[int(np.argmax([Ftr[i].mean() for i in idx]))]
        m = {k: float(Fte[i].mean()) for k, i in I.items()}
        base = max(BASES, key=lambda f: m[f])
        base_tr = max(float(Ftr[I[f]].mean()) for f in BASES)
        x, y = simulate(None, cfg, te, record_auc='R0'); a0 = auc(x, y)
        x, y = simulate(None, cfg, te, record_auc='RND'); a1 = auc(x, y)
        key = f'{cfg[0]}:{cfg[2]}:{cfg[3]:g}'
        per_cfg[key] = {'cfg': list(cfg), 'I': I, 'F': {k: Fte[i] for k, i in I.items()}, 'base': base,
                        'D_train': float(Ftr[I['EFF']].mean()) - base_tr, 'auc': (a0, a1)}
        res['configs'][key] = {'cfg': list(cfg), 'sel': {k: label(P[i]) for k, i in I.items()}, 'test': m,
                               'test_energy': {k: float(Ete[i].mean()) for k, i in I.items()}, 'base': base,
                               'D_test': m['EFF'] - m[base], 'D_train': per_cfg[key]['D_train'], 'AUC_R0': a0, 'AUC_RND': a1,
                               'sec': round(time.time() - t0)}
        print(f"U {cfg[0]} ориентиров {cfg[2]} шум ×{cfg[3]:g}: R0 {m['R0']:.2f} FZ {m['FZ']:.2f} NOEFF {m['NOEFF']:.2f} COMMON {m['COMMON']:.2f} "
              f"EFF {m['EFF']:.2f} EFFC {m['EFFC']:.2f} O {m['O']:.2f} | D {m['EFF'] - m[base]:+.2f} (база {base}) | AUC {a0:.2f}/{a1:.2f} | "
              f"{time.time() - t0:.0f} с", flush=True)
        json.dump(res, open(os.path.join(out_dir, 'precheck2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    # ячейки (U, шум): тренд D по числу ориентиров и критерии при слабой опоре
    cells = {}
    for u in US:
        for s in NOISE:
            k = lambda nl: f'{u}:{nl}:{(1.0 if nl == 0 else s):g}'
            Dl = {nl: per_cfg[k(nl)]['F']['EFF'] - per_cfg[k(nl)]['F'][per_cfg[k(nl)]['base']] for nl in NLM}
            trend = {f'{a}-3': zp(Dl[a], Dl[3]) for a in (0, 1)}
            steps = [(a, a + 1) + zp(Dl[a], Dl[a + 1]) for a in (0, 1, 2)]
            mono = all(dd >= 0 or z > -2 for _, _, dd, z in steps)
            trend_pass = mono and any(z >= 3 and dd > 0 for dd, z in trend.values())
            wk = max((0, 1), key=lambda nl: per_cfg[k(nl)]['D_train'])
            c = per_cfg[k(wk)]; F = c['F']; mt = {kk: float(v.mean()) for kk, v in F.items()}; base = c['base']
            d1, z1 = zp(F['O'], F['R0']); d2, z2 = zp(F['EFF'], F[base])
            clos = d2 / (mt['O'] - mt[base]) if mt['O'] > mt[base] else float('nan')
            fzc = (mt['FZ'] - mt['R0']) / (mt['O'] - mt['R0']) if mt['O'] > mt['R0'] else float('nan')
            coc = (mt['COMMON'] - mt['R0']) / (mt['O'] - mt['R0']) if mt['O'] > mt['R0'] else float('nan')
            d4, z4 = zp(F['EFFC'], F[base])
            cr = {'weak_level': wk, 'trend': {kk: list(v) for kk, v in trend.items()}, 'steps': [list(x) for x in steps],
                  'D_mean_by_nlm': {nl: float(Dl[nl].mean()) for nl in NLM}, 'c2_trend_pass': bool(trend_pass),
                  'c1': [d1, z1], 'c1_pass': bool(d1 > 0 and z1 >= 3),
                  'c2_weak': [d2, z2, clos], 'c2_weak_pass': bool(d2 > 0 and z2 >= 3 and clos >= 0.5),
                  'c3a_FZ_closure': fzc, 'c3a_pass': bool(fzc < 0.5), 'c3b_COMMON_closure_desc': coc,
                  'c3c_auc': list(c['auc']), 'c3c_pass': bool(max(c['auc']) <= 0.6), 'c4': [d4, z4], 'c4_pass': bool(d4 > 0 and z4 >= 2),
                  'D_train_weak': c['D_train']}
            cr['passes'] = bool(cr['c1_pass'] and cr['c2_trend_pass'] and cr['c2_weak_pass'] and cr['c3a_pass'] and cr['c3c_pass'] and cr['c4_pass'])
            cells[f'{u}:{s:g}'] = cr
            print(f"ячейка U {u} шум ×{s:g}: D по ориентирам 0..3 " + ' '.join(f'{cr["D_mean_by_nlm"][nl]:+.2f}' for nl in NLM)
                  + f" | тренд {trend_pass} (0−3 z {trend['0-3'][1]:.1f}, 1−3 z {trend['1-3'][1]:.1f}) | слабая {wk}: 1 {cr['c1_pass']} 2 {cr['c2_weak_pass']} "
                  f"(z {z2:.1f}, доля {clos:.2f}) 3а {fzc:.2f} 3в {max(c['auc']):.2f} 4 {cr['c4_pass']} (z {z4:.1f}) | проходит {cr['passes']}", flush=True)
    res['cells'] = cells
    passing = [k for k, v in cells.items() if v['passes']]
    res['stage2_cell'] = max(passing, key=lambda k: cells[k]['D_train_weak']) if passing else None
    json.dump(res, open(os.path.join(out_dir, 'precheck2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('ячейка для этапа 2:', res['stage2_cell'], flush=True)


def run_rot(lives, out_dir, cell):
    """Описательно: поворот связи (зеркало, ×2) при слабой и богатой опоре выбранной ячейки; параметры семей — из сетки."""
    res = json.load(open(os.path.join(out_dir, 'precheck2.json'), encoding='utf-8'))
    u, s = cell.split(':'); u = float(u); s = float(s)
    weak = res['cells'][cell]['weak_level']
    te = np.arange(SEED_TEST, SEED_TEST + lives)
    P0 = grid(); out = {}
    for nl in (weak, 3):
        key = f'{u}:{nl}:{(1.0 if nl == 0 else s):g}'
        sel = res['configs'][key]['sel']
        pick = {f: next(p for p in P0 if label(p) == sel[f]) for f in ('R0', 'O', 'FZ', 'COMMON', 'EFF')}
        pick['EFFA'] = dict(pick['EFF'], fam='EFFA')
        P = list(pick.values())
        cfg = (u, LFOOD, nl, 1.0 if nl == 0 else s)
        for g in (-1.0, 2.0):
            F, E, Fa, Ta = simulate(P, cfg, te, rot_g=g)
            rate_after = Fa / np.maximum(Ta, 1) * 100
            rate_before = (F - Fa) / np.maximum(eg.L - Ta, 1) * 100
            accs = {}
            for j, f in enumerate(pick):
                if f in ('COMMON', 'EFF', 'EFFA'):
                    _, _, _, _, acc = simulate([P[j]], cfg, te, rot_g=g, acc_rec=True)
                    a = acc[0] / np.maximum(acc[1], 1)
                    accs[f] = {'before_100': float(a[100:200].mean()), 'after_0_32': float(a[200:232].mean()),
                               'after_32_100': float(a[232:300].mean()), 'after_100_200': float(a[300:400].mean()),
                               'curve': [float(x) for x in a]}
            out[f'{nl}:{g:g}'] = {'families': list(pick), 'rate_before': {f: float(rate_before[j].mean()) for j, f in enumerate(pick)},
                                  'rate_after': {f: float(rate_after[j].mean()) for j, f in enumerate(pick)}, 'acc': accs}
            print(f"поворот ×{g:g}, ориентиров {nl}: темп до / после (на 100 тактов) "
                  + ' '.join(f"{f} {rate_before[j].mean():.2f}/{rate_after[j].mean():.2f}" for j, f in enumerate(pick))
                  + ' | точность до → 0–32 после: ' + ' '.join(f"{f} {v['before_100']:.2f}→{v['after_0_32']:.2f}" for f, v in accs.items()), flush=True)
    res['rotation'] = {'cell': cell, 'runs': out}
    json.dump(res, open(os.path.join(out_dir, 'precheck2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lives', type=int, default=200)
    ap.add_argument('--part', default='all', choices=['all', 'grid', 'rot'])
    ap.add_argument('--rot-cell', default=None, help='ячейка «U:шум» для опыта с поворотом; по умолчанию — выбранная для этапа 2')
    ap.add_argument('--out', default='results_geo')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.part in ('all', 'grid'):
        commit, dirty = git_info()
        res = {'git_commit': commit, 'git_dirty': dirty, 'lives': a.lives, 'configs': {}}
        run_grid(a.lives, a.out, res)
    if a.part in ('all', 'rot'):
        res = json.load(open(os.path.join(a.out, 'precheck2.json'), encoding='utf-8'))
        cell = a.rot_cell or res.get('stage2_cell') or '0.004:1'
        run_rot(a.lives, a.out, cell)


if __name__ == '__main__':
    main()
