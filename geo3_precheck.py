"""Тезис 5, Г3 «назови живого» (последняя версия линии): эвристики на записи наблюдений env_geo.GeoObs, без ES и LM, CPU.
Пререгистрация — docs/RESULTS_2026-10-05_geometry.md, раздел «Г3».

Наблюдатель движется по RND; все методы считаются на одной и той же записи раунда. В конце раунда (кадр W) метод называет
живого — кандидата с наибольшим счётом (ничьи — случайно, по таблице раунда). Методы:
  NOEFF — затухающий счёт событий |Δb| > θ_ev (без своих действий);
  COMMON — общий сдвиг считается своим: med, rigid, robust, lm (жёсткое по ориентирам, если их ≥ 2, иначе по всем);
  EFF — предсказание неподвижного по своему действию (команда поворота, исполненный шаг), остаток копится: S ← λ R S + r;
  EFFC — то же с грубой кинематикой: поворот × (1 ± e), шаг × (1 ± e), e ∈ {0,1; 0,2; 0,3};
  EFFA — EFF с онлайн-переоценкой связи «команда → поворот» по общему повороту мира (шаг 0,2).
Подбор параметров — на обучающих раундах, сравнения — на отложенных (попарно по раундам).

python geo3_precheck.py [--n 2000] [--part grid|rot|all]
"""
import argparse, json, math, os, time, warnings
import numpy as np

import env_geo as eg
from geo_precheck import rot, kabsch, auc
from stand_gpu import git_info

US = [0.002, 0.004, 0.008]
NLM = [0, 1, 2, 3]
NOISE = [1.0, 2.0, 4.0]
LAMS = [0.8, 0.9, 0.97, 1.0]
TEVS = [0.01, 0.02, 0.04]
ERRS = [0.1, 0.2, 0.3]
W_MAIN, W_SHORT = 32, 16
SEED_TRAIN, SEED_TEST = 840000, 850000


def methods():
    M = [dict(fam='NOEFF', tev=tev, lam=lam) for tev in TEVS for lam in LAMS]
    M += [dict(fam='COMMON', var=v, lam=lam) for v in ('med', 'rigid', 'robust', 'lm') for lam in LAMS]
    M += [dict(fam='EFF', lam=lam, gs=1.0, vs=1.0) for lam in LAMS]
    M += [dict(fam=f'EFFC{e:g}', lam=lam, gs=1 + sg * e, vs=1 + sv * e) for e in ERRS for sg in (-1, 1) for sv in (-1, 1) for lam in LAMS]
    M += [dict(fam='EFFA', lam=lam) for lam in LAMS]
    return M


def label(m):
    return m['fam'] + ''.join(f' {k}={v:g}' if isinstance(v, float) else f' {k}={v}' for k, v in m.items() if k != 'fam')


def scores(o, m):
    """Счёт кандидатов после каждого кадра: (N, W+1, K)."""
    N, W, K = o.N, o.W, o.K
    rel, b, d = o.rel, o.b, o.d
    S = np.zeros((N, K, 2)); C = np.zeros((N, K)); ghat = np.ones(N)
    out = np.zeros((N, W + 1, K))
    fam, lam = m['fam'], m['lam']
    for t in range(1, W + 1):
        prev, cur = rel[:, t - 1], rel[:, t]
        if fam == 'NOEFF':
            C = lam * C + (np.abs(eg.wrap(b[:, t, :K] - b[:, t - 1, :K])) > m['tev'])
            out[:, t] = C; continue
        w = np.ones((N, o.Kt))
        if fam == 'COMMON' or fam == 'EFFA':
            r_rig, th_rig = kabsch(prev, cur, w)
        if fam == 'COMMON':
            v = m['var']
            if v == 'rigid':
                r, ang = r_rig, th_rig
            elif v == 'robust':
                err = np.linalg.norm(r_rig, axis=-1); w2 = w.copy(); w2[np.arange(N), err.argmax(1)] = 0
                r, ang = kabsch(prev, cur, w2 if o.Kt >= 3 else w)
            elif v == 'lm':
                if o.n_lm >= 2:
                    wl = w.copy(); wl[:, :K] = 0; r, ang = kabsch(prev, cur, wl)
                else:
                    r, ang = r_rig, th_rig
            else:
                db = eg.wrap(b[:, t] - b[:, t - 1]); dd = d[:, t] - d[:, t - 1]
                mb, md = np.median(db, 1), np.median(dd, 1)
                r = np.stack([d[:, t] * eg.wrap(db - mb[:, None]), dd - md[:, None]], -1); ang = mb
        else:
            cmd, mv = o.cmd[:, t - 1], o.exec_move[:, t - 1]
            if fam == 'EFFA':
                turned = cmd != 0
                g_obs = np.where(turned, -th_rig / np.where(turned, cmd * eg.TURN_G, 1), ghat)
                ghat = np.where(turned, ghat + 0.2 * (g_obs - ghat), ghat)
                gs, vs = ghat, 1.0
            else:
                gs, vs = m['gs'], m['vs']
            ang = -cmd * eg.TURN_G * gs
            pred = rot(prev, np.asarray(ang)[:, None]) - np.stack([mv * eg.VMAX * vs, np.zeros(N)], -1)[:, None, :]
            r = cur - pred
        S = lam * rot(S, np.asarray(ang)[:, None]) + r[:, :K]
        out[:, t] = np.linalg.norm(S, axis=-1)
    return out


def correct_curve(o, sc):
    """Верно ли назван живой после каждого кадра: (N, W+1) bool; ничьи — по таблице раунда."""
    tb = np.random.RandomState(7).rand(o.N, o.K) * 1e-12
    return (sc + tb[:, None, :]).argmax(2) == o.alive_i[:, None]


def zp(a, b):
    d = a.astype(float) - b.astype(float)
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12))


def evaluate(o_tr, o_te, M, W):
    """Подбор по обучающим (точность в кадре W), возвращает выбранное и бинарные векторы на отложенных."""
    sel, vec, curves = {}, {}, {}
    fams = sorted(set(m['fam'] for m in M))
    acc_tr = [correct_curve(o_tr, scores(o_tr, m))[:, W].mean() for m in M]
    for f in fams:
        idx = [i for i, m in enumerate(M) if m['fam'] == f]
        i = idx[int(np.argmax([acc_tr[j] for j in idx]))]
        cc = correct_curve(o_te, scores(o_te, M[i]))
        sel[f] = {'label': label(M[i]), 'acc_train': float(acc_tr[i])}
        vec[f] = cc[:, W]; curves[f] = cc.mean(0)
    for v in ('med', 'rigid', 'robust', 'lm'):
        idx = [i for i, m in enumerate(M) if m['fam'] == 'COMMON' and m['var'] == v]
        i = idx[int(np.argmax([acc_tr[j] for j in idx]))]
        sel[f'COMMON_{v}'] = {'label': label(M[i]), 'acc_train': float(acc_tr[i])}
        vec[f'COMMON_{v}'] = correct_curve(o_te, scores(o_te, M[i]))[:, W]
    return sel, vec, curves


def run_grid(n, out_dir, res):
    M = methods()
    per = {}
    for u in US:
        for nl in NLM:
            for s in ([1.0] if nl == 0 else NOISE):
                t0 = time.time()
                key = f'{u}:{nl}:{s:g}'
                o_tr = eg.GeoObs(np.arange(SEED_TRAIN, SEED_TRAIN + n), u, W_MAIN, n_lm=nl, lm_noise=s)
                o_te = eg.GeoObs(np.arange(SEED_TEST, SEED_TEST + n), u, W_MAIN, n_lm=nl, lm_noise=s)
                sel, vec, curves = evaluate(o_tr, o_te, M, W_MAIN)
                sel16, vec16, _ = evaluate(o_tr, o_te, M, W_SHORT)
                acc = {f: float(v.mean()) for f, v in vec.items()}
                base = max(('NOEFF', 'COMMON'), key=lambda f: acc[f])
                base_tr = max(sel[f]['acc_train'] for f in ('NOEFF', 'COMMON'))
                db = np.abs(eg.wrap(o_te.b[:, 1:, :o_te.K] - o_te.b[:, :-1, :o_te.K]))
                lab = np.zeros((o_te.N, o_te.K), bool); lab[np.arange(o_te.N), o_te.alive_i] = True
                a_auc = auc(db.reshape(-1), np.repeat(lab[:, None, :], W_MAIN, 1).reshape(-1))
                per[key] = {'vec': vec, 'base': base, 'D_train': sel['EFF']['acc_train'] - base_tr}
                res['configs'][key] = {'U': u, 'n_lm': nl, 'noise': s, 'sel': sel, 'acc': acc, 'base': base,
                                       'D_test': acc['EFF'] - acc[base], 'D_train': per[key]['D_train'], 'AUC_single_frame': a_auc,
                                       'curves': {f: [float(x) for x in c] for f, c in curves.items()},
                                       'W16': {'sel': sel16, 'acc': {f: float(v.mean()) for f, v in vec16.items()}}}
                print(f"U {u} ориентиров {nl} шум ×{s:g}: NOEFF {acc['NOEFF']:.3f} COMMON {acc['COMMON']:.3f} EFF {acc['EFF']:.3f} "
                      f"EFFA {acc['EFFA']:.3f} EFFC .1/.2/.3 {acc['EFFC0.1']:.3f}/{acc['EFFC0.2']:.3f}/{acc['EFFC0.3']:.3f} | D {acc['EFF'] - acc[base]:+.3f} "
                      f"(база {base}) | AUC {a_auc:.2f} | W16 EFF {res['configs'][key]['W16']['acc']['EFF']:.3f} | {time.time() - t0:.0f} с", flush=True)
                json.dump(res, open(os.path.join(out_dir, 'precheck3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    cells = {}
    for u in US:
        for s in NOISE:
            k = lambda nl: f'{u}:{nl}:{(1.0 if nl == 0 else s):g}'
            Dv = {nl: per[k(nl)]['vec']['EFF'].astype(float) - per[k(nl)]['vec'][per[k(nl)]['base']].astype(float) for nl in NLM}
            wk = max((0, 1), key=lambda nl: per[k(nl)]['D_train'])
            cw = res['configs'][k(wk)]
            Bw = cw['acc'][cw['base']]
            d2, z2 = zp(per[k(wk)]['vec']['EFF'], per[k(wk)]['vec'][cw['base']])
            trend = {f'{a}-3': zp(Dv[a], Dv[3]) for a in (0, 1)}
            steps = [[a, a + 1, *zp(Dv[a], Dv[a + 1])] for a in (0, 1, 2)]
            mono = all(x[2] >= 0 or x[3] > -2 for x in steps)
            cr = {'weak_level': wk, 'B_weak': Bw, 'c1_pass': bool(Bw <= 0.80), 'c2': [d2, z2], 'c2_pass': bool(d2 > 0 and z2 >= 3),
                  'D_by_nlm': {nl: float(Dv[nl].mean()) for nl in NLM}, 'trend': {kk: list(v) for kk, v in trend.items()}, 'steps': steps,
                  'c3_pass': bool(mono and any(dd > 0 and z >= 3 for dd, z in trend.values())),
                  'AUC': cw['AUC_single_frame'], 'c4_pass': bool(cw['AUC_single_frame'] <= 0.6),
                  'c5_desc_EFFC': {e: cw['acc'][f'EFFC{e:g}'] for e in ERRS}, 'D_train_weak': per[k(wk)]['D_train']}
            cr['passes'] = bool(cr['c1_pass'] and cr['c2_pass'] and cr['c3_pass'] and cr['c4_pass'])
            cells[f'{u}:{s:g}'] = cr
            print(f"ячейка U {u} шум ×{s:g}: D по ориентирам 0..3 " + ' '.join(f"{cr['D_by_nlm'][nl]:+.3f}" for nl in NLM)
                  + f" | слабая {wk}: 1 B* {Bw:.3f} {cr['c1_pass']} | 2 D {d2:+.3f} z {z2:.1f} {cr['c2_pass']} | 3 тренд {cr['c3_pass']} "
                  f"(0−3 z {trend['0-3'][1]:.1f}, 1−3 z {trend['1-3'][1]:.1f}) | 4 AUC {cr['AUC']:.2f} | проходит {cr['passes']}", flush=True)
    res['cells'] = cells
    passing = [kk for kk, v in cells.items() if v['passes']]
    res['stage2_cell'] = max(passing, key=lambda kk: cells[kk]['D_train_weak']) if passing else None
    json.dump(res, open(os.path.join(out_dir, 'precheck3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('ячейка для этапа 2:', res['stage2_cell'], flush=True)


def run_k3(n, out_dir):
    """Описательно: K = 3 (один живой) — для каждой конфигурации, те же правила подбора."""
    res = json.load(open(os.path.join(out_dir, 'precheck3.json'), encoding='utf-8'))
    M = methods(); out = {}
    for key, c in res['configs'].items():
        u, nl, s = c['U'], c['n_lm'], c['noise']
        o_tr = eg.GeoObs(np.arange(SEED_TRAIN, SEED_TRAIN + n), u, W_MAIN, n_lm=nl, lm_noise=s, K=3)
        o_te = eg.GeoObs(np.arange(SEED_TEST, SEED_TEST + n), u, W_MAIN, n_lm=nl, lm_noise=s, K=3)
        sel, vec, _ = evaluate(o_tr, o_te, M, W_MAIN)
        out[key] = {f: float(v.mean()) for f, v in vec.items()}
        print(f"K 3, {key}: NOEFF {out[key]['NOEFF']:.3f} COMMON {out[key]['COMMON']:.3f} EFF {out[key]['EFF']:.3f}", flush=True)
    res['K3'] = out
    json.dump(res, open(os.path.join(out_dir, 'precheck3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def run_rot(n, out_dir):
    """Описательно: поворот связи (зеркало, ×2) в раундах из 96 кадров, поворот в кадре 32–64; точность по кадрам от поворота."""
    res = json.load(open(os.path.join(out_dir, 'precheck3.json'), encoding='utf-8'))
    cell = res.get('stage2_cell') or '0.004:1'
    u, s = cell.split(':'); u, s = float(u), float(s)
    weak = res['cells'][cell]['weak_level']; M0 = methods(); out = {}
    for nl in (weak, 3):
        key = f'{u}:{nl}:{(1.0 if nl == 0 else s):g}'; sel = res['configs'][key]['sel']
        pick = {f: next(m for m in M0 if label(m) == sel[f]['label']) for f in ('NOEFF', 'COMMON', 'EFF', 'EFFA')}
        pick['EFF λ=0.9'] = dict(fam='EFF', lam=0.9, gs=1.0, vs=1.0); pick['EFFA λ=0.9'] = dict(fam='EFFA', lam=0.9)
        for g in (-1.0, 2.0):
            o = eg.GeoObs(np.arange(SEED_TEST, SEED_TEST + n), u, 96, n_lm=nl, lm_noise=1.0 if nl == 0 else s, rot_g=g)
            rows = {}
            for f, m in pick.items():
                cc = correct_curve(o, scores(o, m)).astype(float)
                rel = np.arange(97)[None, :] - o.t_switch[:, None]
                win = lambda a, b_: float(cc[(rel >= a) & (rel < b_)].mean())
                rows[f] = {'before_-16_0': win(-16, 0), 'after_0_8': win(0, 8), 'after_8_16': win(8, 16), 'after_16_32': win(16, 32)}
            out[f'{nl}:{g:g}'] = rows
            print(f"поворот ×{g:g}, ориентиров {nl}: " + ' | '.join(f"{f} {v['before_-16_0']:.2f}→{v['after_0_8']:.2f}→{v['after_8_16']:.2f}→{v['after_16_32']:.2f}"
                                                                for f, v in rows.items()), flush=True)
    res['rotation'] = {'cell': cell, 'runs': out}
    json.dump(res, open(os.path.join(out_dir, 'precheck3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--part', default='all', choices=['all', 'grid', 'k3', 'rot'])
    ap.add_argument('--out', default='results_geo3')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.part in ('all', 'grid'):
        commit, dirty = git_info()
        run_grid(a.n, a.out, {'git_commit': commit, 'git_dirty': dirty, 'n': a.n, 'configs': {}})
    if a.part in ('all', 'k3'):
        run_k3(a.n, a.out)
    if a.part in ('all', 'rot'):
        run_rot(a.n, a.out)


if __name__ == '__main__':
    main()
