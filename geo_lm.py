"""Тезис 5, Г3, этап 2: анализ LM без обучения на раундах «назови живого» (env_geo.GeoObs). Пререгистрация —
docs/RESULTS_2026-10-05_geometry.md, «Г3» и «Г3, этап 2 — дополнение».

Азбука событий кадра t (переход t−1 → t), словами + BOS:
  [поворот: left / right / straight] [шаг: walk / stay] и для каждого предмета (A, B — кандидаты; X, Y, Z — ориентиры)
  [метка] [сдвиг направления: rises (Δb > θ_b, влево в поле зрения) / falls (Δb < −θ_b) / holds] [дистанция: nearer / farther / level по θ_d].
Абляции: neutral — оба символа действия заменены на « me»; shuffle — кадры переставлены внутри последовательности.
Азбука состояний (описательно): вместо сдвига и дистанции — сектор направления (8, « 0»…« 7») и полоса дистанции (near / mid / far).
Меры: M1 — средняя неожиданность символа сдвига кандидата (−log p, нормировано по трём символам сдвига) → AUC «живой против
не живого»; M2 — AUC неожиданности: события, вызванные миром, против событий, вызванных собой (разметка по контрфактическому
кадру); M3 — логистический зонд «живой» по скрытому состоянию на метке кандидата в последнем кадре (слой 6 — основной, 12 —
описательно), перекрёстная проверка «один набор — проверка». Q-R1 — поворот связи, доля раундов, где не живой выглядит
неожиданнее живого (скользящие 8 кадров), рост за N = 32 кадра после поворота относительно 16 кадров до.

python geo_lm.py thresholds            — правило выбора порогов по обучающим раундам этапа 1 (до LM)
python geo_lm.py time                  — замер (правило 5) на посевах 990000+
python geo_lm.py run [--sets 10 --rounds 400]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

import env_geo as eg
from stand_gpu import git_info

MODEL = 'EleutherAI/pythia-160m'
U, NL_WEAK, NL_RICH, NOISE = 0.008, 0, 3, 1.0
W = 32
BOS = 0
TURN = {1: 1669, -1: 987, 0: 4951}          # « left» (+ — влево), « right», « straight»
MOVE = {1: 2940, 0: 3297}                   # « walk», « stay»
LABEL = [329, 378, 1594, 714, 1503]         # « A», « B», « X», « Y», « Z»
SHIFT = [22844, 11521, 6556]                # « rises», « falls», « holds»
DIST = [37335, 21816, 1268]                 # « nearer», « farther», « level»
NEUTRAL = 479                               # « me»
SECTOR = [470, 337, 374, 495, 577, 608, 721, 818]
DBIN = [2822, 4260, 2080]                   # « near», « mid», « far»
TH_B_GRID, TH_D_GRID = [0.02, 0.03, 0.05, 0.08], [0.005, 0.01, 0.02]
SET_SEED = lambda i: 900000 + 10000 * i
ROT_SEED = lambda i: 960000 + 10000 * i


def ev_codes(o, th_b, th_d):
    """Коды событий (N, W, Kt): сдвиг 0 rises / 1 falls / 2 holds; дистанция 0 nearer / 1 farther / 2 level."""
    db = eg.wrap(o.b[:, 1:] - o.b[:, :-1]); dd = o.d[:, 1:] - o.d[:, :-1]
    sh = np.where(db > th_b, 0, np.where(db < -th_b, 1, 2))
    di = np.where(dd < -th_d, 0, np.where(dd > th_d, 1, 2))
    return sh, di, db


def ev_eff_accuracy(o, th_b, th_d):
    """Событийный EFF (ориентир уровня азбуки): на кадрах «стою, не поворачиваю» событие неподвижного — holds / level;
    счёт кандидата = |Σ знаков сдвига| + |Σ знаков дистанции| по таким кадрам (стойкость сноса). Ничьи — случайно."""
    sh, di, _ = ev_codes(o, th_b, th_d)
    still = (o.cmd == 0) & (o.exec_move == 0)
    sgn = lambda c: np.where(c == 0, 1, np.where(c == 1, -1, 0))
    sc = np.abs((sgn(sh[:, :, :o.K]) * still[..., None]).sum(1)) + np.abs((sgn(di[:, :, :o.K]) * still[..., None]).sum(1))
    tb = np.random.RandomState(3).rand(*sc.shape) * 1e-6
    return float(((sc + tb).argmax(1) == o.alive_i).mean())


def ev_noeff_accuracy(o, th_b, th_d):
    """Событийный NOEFF: та же стойкость, но по всем кадрам, без знания своих действий."""
    sh, di, _ = ev_codes(o, th_b, th_d)
    sgn = lambda c: np.where(c == 0, 1, np.where(c == 1, -1, 0))
    sc = np.abs(sgn(sh[:, :, :o.K]).sum(1)) + np.abs(sgn(di[:, :, :o.K]).sum(1))
    tb = np.random.RandomState(3).rand(*sc.shape) * 1e-6
    return float(((sc + tb).argmax(1) == o.alive_i).mean())


def build(o, th_b, th_d, mode='act', alphabet='events', perm_seed=None):
    """Последовательности (N, 1 + W·L), позиции символов сдвига (N, W, K), позиции меток кандидатов в последнем кадре (N, K)."""
    N, Kt, K = o.N, o.Kt, o.K
    sh, di, _ = ev_codes(o, th_b, th_d)
    if alphabet == 'states':
        sec = ((o.b[:, 1:] + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        dbn = np.digitize(o.d[:, 1:], [0.3, 0.6])
        tok_a, tok_b = np.array(SECTOR)[sec], np.array(DBIN)[dbn]
    else:
        tok_a, tok_b = np.array(SHIFT)[sh], np.array(DIST)[di]
    turn = np.vectorize(TURN.get)(o.cmd); move = np.vectorize(MOVE.get)(o.exec_move)
    if mode == 'neutral':
        turn = np.full_like(turn, NEUTRAL); move = np.full_like(move, NEUTRAL)
    L = 2 + 3 * Kt
    fr = np.zeros((N, W, L), int)
    fr[:, :, 0], fr[:, :, 1] = turn, move
    for k in range(Kt):
        fr[:, :, 2 + 3 * k] = LABEL[k]; fr[:, :, 3 + 3 * k] = tok_a[:, :, k]; fr[:, :, 4 + 3 * k] = tok_b[:, :, k]
    order = np.tile(np.arange(W), (N, 1))
    if mode == 'shuffle':
        rs = np.random.RandomState(perm_seed)
        order = np.stack([rs.permutation(W) for _ in range(N)])
        fr = fr[np.arange(N)[:, None], order]
    ids = np.concatenate([np.full((N, 1), BOS), fr.reshape(N, W * L)], 1)
    # позиция символа сдвига кандидата k в кадре с исходным номером t
    slot = np.argsort(order, 1)                                           # где стоит исходный кадр t
    pos_shift = 1 + slot[:, :, None] * L + 3 + 3 * np.arange(K)[None, None, :]
    pos_label_last = 1 + (W - 1) * L + 2 + 3 * np.arange(K)[None, :] + np.zeros((N, 1), int)
    return ids, pos_shift, pos_label_last


@torch.no_grad()
def run_model(m, ids, pos_shift, pos_label, cand_ids, dev, layers=(6, 12), batch=16):
    """Неожиданность символа на pos_shift (нормирована по cand_ids) и скрытые состояния на pos_label по слоям."""
    N = len(ids); out_s = np.zeros(pos_shift.shape); out_h = {l: [] for l in layers}
    cid = torch.as_tensor(cand_ids, device=dev)
    for i in range(0, N, batch):
        x = torch.as_tensor(ids[i:i + batch], device=dev)
        o = m.gpt_neox(input_ids=x, output_hidden_states=bool(layers))               # логиты только по символам-кандидатам
        lp = torch.log_softmax(o.last_hidden_state @ m.get_output_embeddings().weight[cid].T, -1)   # (b, T, |cid|)
        ps = torch.as_tensor(pos_shift[i:i + batch], device=dev)                     # (b, W, K)
        b_ = ps.shape[0]
        tgt = x.gather(1, ps.reshape(b_, -1))                                         # символ, пришедший на позиции
        pred = lp.gather(1, (ps - 1).reshape(b_, -1, 1).expand(-1, -1, len(cand_ids)))   # предсказание с прошлой позиции
        k = (tgt[..., None] == cid).float().argmax(-1)
        out_s[i:i + b_] = (-pred.gather(2, k[..., None])[..., 0]).reshape(ps.shape).cpu().numpy()
        if layers:
            pl = torch.as_tensor(pos_label[i:i + batch], device=dev)
            for l in layers:
                h = o.hidden_states[l]
                out_h[l].append(h.gather(1, pl[..., None].expand(-1, -1, h.shape[-1])).cpu().numpy())
    return out_s, {l: np.concatenate(v) for l, v in out_h.items()}


def auc(score, lab):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def m1(o, surpr):
    """AUC «живой против не живого» по средней неожиданности символа сдвига; и доля раундов, где живой неожиданнее."""
    s = surpr.mean(1)                                                            # (N, K)
    lab = np.zeros_like(s, bool); lab[np.arange(o.N), o.alive_i] = True
    return auc(s.reshape(-1), lab.reshape(-1)), float((s.argmax(1) == o.alive_i).mean())


def cause_labels(o, th_b):
    """Причина события сдвига кандидата: мир — наблюдаемое событие отличается от контрфактического (предмет остался на прошлом
    месте, своё движение и шум датчика те же); себя — совпадает и событие не holds. (N, W, K): 1 мир, 0 себя, −1 вне разметки."""
    N, K = o.N, o.K
    prev_items = o.items[:, :-1, :K]                                             # мировое положение в кадре t−1
    w = prev_items - o.pos[:, 1:, None, :]
    c, s = np.cos(o.head[:, 1:])[..., None], np.sin(o.head[:, 1:])[..., None]
    rel = np.stack([c * w[..., 0] + s * w[..., 1], -s * w[..., 0] + c * w[..., 1]], -1)
    items_now = o.items[:, 1:, :K] - o.pos[:, 1:, None, :]
    rel_now = np.stack([c * items_now[..., 0] + s * items_now[..., 1], -s * items_now[..., 0] + c * items_now[..., 1]], -1)
    noise = eg.wrap(o.b[:, 1:, :K] - np.arctan2(rel_now[..., 1], rel_now[..., 0]))  # шум датчика этого кадра
    b_cf = eg.wrap(np.arctan2(rel[..., 1], rel[..., 0]) + noise)
    db_obs = eg.wrap(o.b[:, 1:, :K] - o.b[:, :-1, :K]); db_cf = eg.wrap(b_cf - o.b[:, :-1, :K])
    code = lambda x: np.where(x > th_b, 0, np.where(x < -th_b, 1, 2))
    e_obs, e_cf = code(db_obs), code(db_cf)
    return np.where(e_obs != e_cf, 1, np.where(e_obs != 2, 0, -1))


def probe_cv(H, y, groups):
    """Логистический зонд со стандартизацией, «один набор — проверка»: AUC по каждому набору."""
    from sklearn.linear_model import LogisticRegression
    out = {}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=2000).fit((H[tr] - mu) / sd, y[tr])
        out[int(g)] = auc(clf.decision_function((H[te] - mu) / sd), y[te])
    return out


def false_alive_curve(o, surpr):
    """Доля раундов, где не живой неожиданнее живого (скользящие 8 кадров), по кадрам от поворота −16…+31."""
    N, Wn, K = surpr.shape
    cs = np.concatenate([np.zeros((N, 1, K)), np.cumsum(surpr, 1)], 1)
    run = (cs[:, 8:] - cs[:, :-8]) / 8                                           # кадры 8..Wn
    ar = np.arange(N); dummy = 1 - o.alive_i
    fa = run[ar, :, dummy] > run[ar, :, o.alive_i]                               # (N, Wn−7)
    rel = np.arange(8, Wn + 1)[None, :] - 1 - o.t_switch[:, None]
    curve = {int(r): float(fa[rel == r].mean()) for r in range(-16, 32) if (rel == r).any()}
    before = float(fa[(rel >= -16) & (rel < 0)].mean()); after = float(fa[(rel >= 0) & (rel < 32)].mean())
    return curve, before, after


def thresholds():
    o = eg.GeoObs(np.arange(840000, 844000), U, W, n_lm=NL_WEAK, lm_noise=NOISE)
    res = {f'{tb}:{td}': (ev_eff_accuracy(o, tb, td), ev_noeff_accuracy(o, tb, td)) for tb in TH_B_GRID for td in TH_D_GRID}
    best = max(res, key=lambda k: res[k][0])
    for k, v in res.items():
        print(f'θ_b:θ_d {k}: событийный EFF {v[0]:.3f}, событийный NOEFF {v[1]:.3f}')
    print('выбрано по событийному EFF на обучающих раундах:', best)
    return best, res


def load(weights, seed, dev):
    from report_body import load_lm
    return load_lm(MODEL, weights, seed, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['thresholds', 'time', 'run'])
    ap.add_argument('--sets', type=int, default=10)
    ap.add_argument('--rounds', type=int, default=400)
    ap.add_argument('--th', default=None, help='θ_b:θ_d (из правила)')
    ap.add_argument('--out', default='results_geo3')
    a = ap.parse_args()
    if a.cmd == 'thresholds':
        best, res = thresholds()
        os.makedirs(a.out, exist_ok=True)
        json.dump({'best': best, 'grid': res}, open(os.path.join(a.out, 'thresholds.json'), 'w', encoding='utf-8'), indent=1)
        return
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    th_b, th_d = map(float, a.th.split(':'))
    if a.cmd == 'time':
        m = load('pretrained', 0, dev)
        for nl, Wn in ((NL_WEAK, W), (NL_RICH, W), (NL_WEAK, 72)):
            o = eg.GeoObs(np.arange(990000, 990400), U, Wn, n_lm=nl, lm_noise=NOISE, rot_g=-1.0 if Wn > W else None, sw_lo=24, sw_hi=40)
            ids, ps, pl = build_any(o, th_b, th_d, 'act', 'events')
            torch.cuda.synchronize(); t0 = time.time(); run_model(m, ids, ps, pl, SHIFT, dev); torch.cuda.synchronize()
            print(f'ориентиров {nl}, кадров {Wn}: 400 последовательностей по {ids.shape[1]} токенов — {time.time() - t0:.1f} с', flush=True)
        return
    run_all(a, th_b, th_d, dev)


def build_any(o, th_b, th_d, mode, alphabet, perm_seed=None):
    """build для любого числа кадров (поворот — 72 кадра)."""
    global W
    W0 = W; W = o.W
    try:
        return build(o, th_b, th_d, mode, alphabet, perm_seed)
    finally:
        W = W0


def sign_flip_p(d):
    """Точный односторонний перестановочный p (знаки) для среднего парных разностей d > 0."""
    d = np.asarray(d); n = len(d)
    signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
    return float(((signs * np.abs(d)).mean(1) >= d.mean() - 1e-12).mean())


def holm(ps):
    k = sorted(ps, key=ps.get); out = {}; run = 0.0
    for i, key in enumerate(k):
        run = max(run, min(1.0, ps[key] * (len(k) - i))); out[key] = run
    return out


def run_all(a, th_b, th_d, dev):
    commit, dirty = git_info(); t0 = time.time()
    res = {'git_commit': commit, 'git_dirty': dirty, 'th_b': th_b, 'th_d': th_d, 'sets': a.sets, 'rounds': a.rounds, 'per_set': {}}
    pre = load('pretrained', 0, dev)
    H = {k: [] for k in ('pre_act_6', 'pre_neu_6', 'pre_act_12', 'pre_neu_12', 'rnd_act_6', 'rnd_neu_6')}; Y, G = [], []
    for i in range(a.sets):
        rnd = load('random', 1000 + i, dev)
        seeds = np.arange(SET_SEED(i), SET_SEED(i) + a.rounds)
        r = {}
        ow = eg.GeoObs(seeds, U, W, n_lm=NL_WEAK, lm_noise=NOISE)
        orr = eg.GeoObs(seeds, U, W, n_lm=NL_RICH, lm_noise=NOISE)
        causes = cause_labels(ow, th_b)
        for mname, mdl in (('pre', pre), ('rnd', rnd)):
            for mode in ('act', 'neutral', 'shuffle'):
                if mode == 'shuffle' and mname == 'rnd':
                    continue
                ids, ps, pl = build(ow, th_b, th_d, mode, 'events', perm_seed=77 + i)
                s, h = run_model(mdl, ids, ps, pl, SHIFT, dev, layers=(6, 12) if mode != 'shuffle' else ())
                a1, acc1 = m1(ow, s)
                world, selfc = s[causes == 1], s[causes == 0]
                a2 = auc(np.r_[world, selfc], np.r_[np.ones(len(world), bool), np.zeros(len(selfc), bool)]) if len(world) and len(selfc) else float('nan')
                r[f'{mname}_{mode}'] = {'M1_auc': a1, 'M1_acc': acc1, 'M2_auc': a2, 'n_world': int(len(world)), 'n_self': int(len(selfc)),
                                        'surprise_mean': float(s.mean())}
                if mode != 'shuffle':
                    md = 'act' if mode == 'act' else 'neu'
                    for l in ((6, 12) if mname == 'pre' else (6,)):
                        H[f'{mname}_{md}_{l}'].append(h[l].reshape(-1, h[l].shape[-1]))
            # богатая опора — описательно
            for mode in ('act', 'neutral'):
                ids, ps, pl = build(orr, th_b, th_d, mode, 'events')
                s, _ = run_model(mdl, ids, ps, pl, SHIFT, dev, layers=())
                r[f'{mname}_{mode}_rich'] = {'M1_auc': m1(orr, s)[0], 'M1_acc': m1(orr, s)[1]}
            # азбука состояний — описательно (неожиданность символа сектора)
            ids, ps, pl = build(ow, th_b, th_d, 'act', 'states')
            s, _ = run_model(mdl, ids, ps, pl, SECTOR, dev, layers=())
            r[f'{mname}_states'] = {'M1_auc': m1(ow, s)[0], 'M1_acc': m1(ow, s)[1]}
            # поворот связи: 72 кадра, поворот в кадре 24–40
            for g in (-1.0, 2.0):
                orot = eg.GeoObs(np.arange(ROT_SEED(i), ROT_SEED(i) + a.rounds), U, 72, n_lm=NL_WEAK, lm_noise=NOISE, rot_g=g, sw_lo=24, sw_hi=40)
                for mode in ('act', 'neutral'):
                    ids, ps, pl = build_any(orot, th_b, th_d, mode, 'events')
                    s, _ = run_model(mdl, ids, ps, pl, SHIFT, dev, layers=())
                    curve, bef, aft = false_alive_curve(orot, s)
                    r[f'{mname}_{mode}_rot{g:g}'] = {'before': bef, 'after': aft, 'rise': aft - bef, 'curve': curve}
        lab = np.zeros((a.rounds, 2), bool); lab[np.arange(a.rounds), ow.alive_i] = True
        Y.append(lab.reshape(-1)); G.append(np.full(2 * a.rounds, i))
        res['per_set'][i] = r
        del rnd; torch.cuda.empty_cache()
        print(f"набор {i}: M1 пред / без действий / случ — {r['pre_act']['M1_auc']:.3f} / {r['pre_neutral']['M1_auc']:.3f} / {r['rnd_act']['M1_auc']:.3f}; "
              f"перемешано {r['pre_shuffle']['M1_auc']:.3f}; рост ложных (пред, действия / без) "
              f"{np.mean([r[f'pre_act_rot{g:g}']['rise'] for g in (-1.0, 2.0)]):+.3f} / {np.mean([r[f'pre_neutral_rot{g:g}']['rise'] for g in (-1.0, 2.0)]):+.3f} | "
              f"{time.time() - t0:.0f} с", flush=True)
        json.dump(res, open(os.path.join(a.out, 'lm_stage2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    Y = np.concatenate(Y); G = np.concatenate(G)
    M3 = {k: probe_cv(np.concatenate(v), Y, G) for k, v in H.items()}
    res['M3'] = M3
    S = lambda key, f='M1_auc': np.array([res['per_set'][i][key][f] for i in range(a.sets)])
    rise = lambda mn, md: np.array([np.mean([res['per_set'][i][f'{mn}_{md}_rot{g:g}']['rise'] for g in (-1.0, 2.0)]) for i in range(a.sets)])
    m3 = lambda k: np.array([M3[k][i] for i in range(a.sets)])
    diffs = {'G1': S('pre_act') - S('pre_neutral'), 'G2': S('pre_act') - S('rnd_act'), 'G3': m3('pre_act_6') - m3('pre_neu_6'),
             'Q-R1': rise('pre', 'act') - rise('pre', 'neutral')}
    ps = {k: sign_flip_p(v) for k, v in diffs.items()}
    res['family'] = {k: {'mean_diff': float(v.mean()), 'n_pos': int((v > 0).sum()), 'p': ps[k]} for k, v in diffs.items()}
    hp = holm(ps)
    for k in res['family']:
        res['family'][k]['p_holm'] = hp[k]
    res['sec'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(a.out, 'lm_stage2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k, v in res['family'].items():
        print(f"{k}: разность {v['mean_diff']:+.4f}, {v['n_pos']}/{a.sets} наборов, p {v['p']:.4f}, Холм {v['p_holm']:.4f}", flush=True)
    print('M3 (AUC по наборам, среднее):', {k: round(float(np.mean(list(v.values()))), 3) for k, v in M3.items()}, flush=True)


if __name__ == '__main__':
    main()
