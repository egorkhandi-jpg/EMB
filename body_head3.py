"""Вне контура, только быстрая часть, метки или правило, проигрывание опыта (решение Егора 2026-10-03 после v3-nlms).
flip-2 с разворотом, рефлекс (голова выбирает вид, скорость полная), Pythia-160M, окно 32.
Пререгистрация — docs/RESULTS_2026-10-01_flip.md, раздел «Вне контура, быстрая часть, метки и проигрывание».

Режимы:
  repro    — повтор жизней A11 (те же посевы, η и группировка, что в body_head2.py; веса и пары в A11 не
             сохранялись), сверка с записями A11; затем
             (1а) замороженная голова вне контура: траектории слепой к знаку политики (рефлекс к ближайшему
                  предмету любого вида) на 100 тестовых эпизодах; точность «A — пища» на информированных тактах;
             для dual — онлайн на тесте (медленные заморожены, быстрая учится) в контуре и вне контура;
             (2) только для меток all и считывания full: пакетная логистическая регрессия до сходимости на парах
                 (скрытое состояние, метка тела «последний исход») этой жизни — как зонд п. 15 (λ по CV с группами-
                 эпизодами), и та же регрессия на тех же парах с истинным знаком (оракул-метки); тест вне контура
                 (главное) и в контуре (описательно).
  fastonly — (1б) двухскоростная голова с медленными весами, замороженными в нуле: жизнь (онлайн), онлайн на тесте
             в контуре и вне контура.
  replay   — (3) медленные веса обновляются только в паузе после эпизода: проигрывание пар эпизода и буфера
             прошлых эпизодов (B эпизодов, включая текущий), K = 3 прохода, мини-пакеты по 32, логистическая
             потеря, нормированный шаг w += η Σ e z / Σ (|z|² + 1) — устойчив при 0 < η < 2 (правило 12);
             быстрая часть — как в dual (онлайн, шаг 0,3, сброс на эпизод). --tune — выбор (η, B) по замороженной
             голове на обучающих жизнях 101–103, оценка на 888000+k.

python body_head3.py repro --cond lm --weights pretrained --words c --labels all --readout full
python body_head3.py fastonly --cond lm --weights pretrained --words c
python body_head3.py replay --cond lm --weights pretrained --words c --tune
"""
import argparse, json, os, time
import numpy as np
import torch

from env_flip import FlipBodies, T
import stand_words as sw
import body_head as bh
import body_head2 as bh2
from readout import fit_probe
from probe_fast import _bacc
from stand_gpu import git_info

LIFE = bh2.LIFE
REPLAY_ETAS = (0.1, 0.3, 1.0)
REPLAY_BUFS = (1, 10, 50)
REPLAY_PASSES, REPLAY_MB = 3, 32
TUNE_EVAL = [888000 + k for k in range(100)]
life_seeds = lambda s: [500000 + 1000 * s + e for e in range(LIFE)]


def live_c(feat, seeds_per_life, head, labels_mode, device, collect=False, after_episode=None):
    """Повтор body_head2.live (поведение то же) со сбором пар (z, метка тела, истинный знак, информирован)
    на тактах обновления меток all и с вызовом after_episode(ep) в паузе после каждого эпизода; ep — словарь
    тензоров эпизода (L, T, …): z, bel (метка all), m_all, yA (исход события), m_ev (такт события, t ≥ 1), sign, inf."""
    L, n_ep = len(seeds_per_life), len(seeds_per_life[0])
    online = []; stats = {'fitness': [], 'food': [], 'poison': [], 'log': []}
    pairs = [[] for _ in range(L)]
    ev_count = np.zeros(L, int)
    for e in range(n_ep):
        head.reset_fast()
        bodies = FlipBodies(L, [seeds_per_life[i][e] for i in range(L)], True, 'flip-2')
        feat.reset(L)
        z_prev = None
        belief = np.zeros(L, bool); has_belief = np.zeros(L, bool)
        last_post = np.full(L, -1); rev_seen = np.zeros(L, bool); log = []
        EP = {k: [] for k in ('z', 'bel', 'm_all', 'yA', 'm_ev', 'sign', 'inf')}
        for t in range(T):
            s = bodies.sense()
            has_ev, yA = bh.labels(s)
            belief = np.where(has_ev, yA, belief); has_belief |= has_ev
            z = head.proj(bh.std(feat, feat.features(s)))
            m_all = has_belief & bodies.alive
            if labels_mode == 'event' and z_prev is not None:
                m = torch.as_tensor(has_ev & bodies.alive, dtype=torch.float32, device=device)
                if m.any():
                    head.update(z_prev, torch.as_tensor(yA, dtype=torch.float32, device=device), m)
            elif labels_mode == 'all':
                m = torch.as_tensor(m_all, dtype=torch.float32, device=device)
                if m.any():
                    head.update(z, torch.as_tensor(belief, dtype=torch.float32, device=device), m)
            after = bodies.t_rev < bodies.t
            newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1
            last_post[has_ev] = t
            informed = bodies.alive & (last_post >= 0)
            if collect:
                zc = z.cpu().numpy()
                for i in np.where(m_all)[0]:
                    pairs[i].append((zc[i], belief[i], bodies.good_A[i], informed[i], e, t))
            if after_episode is not None:
                f32 = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                EP['z'].append(z); EP['bel'].append(f32(belief)); EP['m_all'].append(f32(m_all)); EP['yA'].append(f32(yA))
                EP['m_ev'].append(f32(has_ev & bodies.alive & (t > 0))); EP['sign'].append(f32(bodies.good_A)); EP['inf'].append(f32(informed))
            chA = (head.logit(z) >= 0).cpu().numpy()
            correct = chA == bodies.good_A
            for i in np.where(informed)[0]:
                online.append((i, ev_count[i], bool(correct[i])))
            ev_count += has_ev & bodies.alive
            log.append(np.stack([chA, bodies.good_A.copy(), bodies.alive.copy(), has_ev, after], 1))
            act = sw.reflex(s, np.stack([np.where(chA, 1.0, -1.0), np.ones(L)], 1))
            act[~bodies.alive] = 0
            bodies.step(act)
            z_prev = z
        stats['fitness'] += bodies.fitness().tolist(); stats['food'] += bodies.n_good.tolist(); stats['poison'] += bodies.n_bad.tolist()
        stats['log'].append(np.stack(log, 1))
        if after_episode is not None:
            after_episode({k: torch.stack(v, 1) for k, v in EP.items()})      # (L, T, d) и (L, T)
    return online, stats, pairs


def blind_traj(feat, proj, seeds, device):
    """Траектории слепой к знаку политики (вне контура): Z (T, n, d) и маски (T, n)."""
    n = len(seeds)
    b = FlipBodies(n, seeds, True, 'flip-2'); feat.reset(n)
    belief = np.zeros(n, bool); has_belief = np.zeros(n, bool)
    last_post = np.full(n, -1); rev_seen = np.zeros(n, bool)
    Z, GA, INF, HB, BEL = [], [], [], [], []
    for t in range(T):
        s = b.sense()
        has_ev, yA = bh.labels(s)
        belief = np.where(has_ev, yA, belief); has_belief |= has_ev
        Z.append(proj(bh.std(feat, feat.features(s))))
        after = b.t_rev < b.t
        newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1
        last_post[has_ev] = t
        GA.append(b.good_A.copy()); INF.append(b.alive & (last_post >= 0)); HB.append(has_belief & b.alive); BEL.append(belief.copy())
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(n)], 1))
        act[~b.alive] = 0
        b.step(act)
    tt = lambda x: torch.as_tensor(np.stack(x), device=device)
    return {'Z': torch.stack(Z), 'GA': tt(GA), 'INF': tt(INF), 'HB': tt(HB), 'BEL': tt(BEL).float()}


def acc_out(pred, bt):
    """Точность и сбалансированная точность «A — пища» на информированных тактах вне контура."""
    m = bt['INF'].cpu().numpy(); p = pred.cpu().numpy()[m]; y = bt['GA'].cpu().numpy()[m]
    return {'acc': float((p == y).mean()), 'bacc': float(_bacc(p, y)), 'n': int(m.sum())}


def frozen_out(w, b, bt):
    return acc_out((bt['Z'] @ w + b) >= 0, bt)


def online_out(w, b, bt):
    """Вне контура: медленные (w, b) заморожены, быстрая добавка учится по меткам тела (шаг 0,3, сброс на эпизод);
    решение в такт t — после обновления по метке такта t, как в live."""
    Z = bt['Z']; n, d = Z.shape[1], Z.shape[2]
    wf = torch.zeros(n, d, device=Z.device); P = []
    for t in range(T):
        z = Z[t]
        err = (bt['BEL'][t] - torch.sigmoid(((w + wf) * z).sum(1) + b)) * bt['HB'][t].float()
        wf += (bh2.ETA_FAST * err / ((z * z).sum(1) + 1))[:, None] * z
        P.append((((w + wf) * z).sum(1) + b) >= 0)
    return acc_out(torch.stack(P), bt)


def head_eval(feat, w, b, readout, P, seeds, device, online):
    """В контуре: голова (w, b); online — быстрая добавка учится (медленные заморожены), иначе всё заморожено."""
    n = len(seeds)
    h1 = bh2.Head(n, 1, [0.0] * n, device, readout, P)
    h1.w = w.expand(n, -1).clone(); h1.b = b.expand(n).clone(); h1.wf = torch.zeros_like(h1.w)
    _, st = bh2.live(feat, [[s] for s in seeds], h1, 'all' if online else 'event', device, frozen=not online)
    return bh.summarize(st)


def short(m):
    return {'choice_informed': m['choice']['informed'], 'test': m['test'], 'food': m['food'], 'poison': m['poison'],
            'by_k': m['choice']['by_k']}


def build(a, net_seed, readout):
    f = bh.make_feat(a, net_seed, 32)
    P = bh2.pca_basis(f, a.device) if readout == 'pca' else None
    return f, P


def tag_of(a, suffix):
    m = '' if a.model == 'EleutherAI/pythia-160m' else a.model.split('-')[-1] + '_'
    return m + (f"{a.weights}_{a.words}" if a.cond == 'lm' else a.cond) + suffix


def is_shared(a):
    return not (a.cond == 'quad' or (a.cond == 'lm' and a.weights == 'random'))


def batch_fit(pairs, label_idx, bt, feat_args, device):
    """Пакетная логистическая регрессия на парах жизни (λ по CV с группами-эпизодами), тест вне контура."""
    X = np.stack([p[0] for p in pairs]); y = np.array([p[label_idx] for p in pairs], float); g = np.array([p[4] for p in pairs])
    mu, sd, w, cv, lam = fit_probe(X, y, g, device)
    w_eff = w[:-1] / sd; b_eff = w[-1] - (mu / sd * w[:-1]).sum()
    out = {'cv_bacc': cv, 'lambda': lam, 'n_pairs': int(len(y)), 'out_of_loop': frozen_out(w_eff, b_eff, bt)}
    feat, P = feat_args
    out['in_loop'] = short(head_eval(feat, w_eff, b_eff, 'full', P, sw.TEST_SEEDS, device, online=False))
    return out


def run_repro(a, commit, dirty):
    tag = tag_of(a, f'_{a.labels}_{a.readout}')
    src = json.load(open(os.path.join(a.src, f'result_{tag}.json')))
    eta = src['eta']; ref = {x['seed']: x for x in src['results']}
    path = os.path.join(a.out, f'repro_{tag}.json')
    res = json.load(open(path)) if os.path.exists(path) else {'tag': tag, 'eta': eta, 'src': a.src, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'seeds': a.seeds})
    todo = [s for s in a.seeds if s not in {x['seed'] for x in res['results']}]
    do_batch = a.labels == 'all' and a.readout == 'full'
    os.makedirs(os.path.join(a.out, 'heads'), exist_ok=True); os.makedirs(os.path.join(a.out, 'pairs'), exist_ok=True)
    t0 = time.time()
    for grp in ([todo] if is_shared(a) else [[s] for s in todo]):
        if not grp:
            continue
        feat, P = build(a, grp[0], a.readout)
        head = bh2.Head(len(grp), feat.mu.shape[0], [eta] * len(grp), a.device, a.readout, P)
        online, st, pairs = live_c(feat, [life_seeds(s) for s in grp], head, a.labels, a.device, collect=do_batch)
        bt = blind_traj(feat, head.proj, sw.TEST_SEEDS, a.device)
        for j, s in enumerate(grp):
            sub = {kk: (vv[j::len(grp)] if kk in ('fitness', 'food', 'poison') else vv) for kk, vv in st.items()}
            sub['log'] = [lg[j:j + 1] for lg in st['log']]
            lo = bh.summarize(sub)['choice']['informed']
            item = {'seed': s, 'repro': {'life_online_a11': ref[s]['life_online']['choice']['informed'], 'life_online_now': lo,
                                         'w_norm_a11': ref[s]['w_norm'], 'w_norm_now': float(head.w[j].norm())},
                    'frozen_test_a11': ref[s]['frozen_test']['choice']['informed'],
                    'frozen_out_of_loop': frozen_out(head.w[j], head.b[j], bt), 'git_commit': commit}
            r = item['repro']
            if abs(r['life_online_now'] - r['life_online_a11']) > 0.005 or abs(r['w_norm_now'] / r['w_norm_a11'] - 1) > 0.01:
                item['frozen_test_now'] = short(bh2.frozen_eval(feat, head, j, sw.TEST_SEEDS, a.device))   # повтор не точный
            if a.readout == 'dual':
                item['online_test_in_loop'] = short(head_eval(feat, head.w[j], head.b[j], 'dual', P, sw.TEST_SEEDS, a.device, online=True))
                item['online_test_out_of_loop'] = online_out(head.w[j], head.b[j], bt)
            if do_batch:
                pj = pairs[j]
                np.savez_compressed(os.path.join(a.out, 'pairs', f'pairs_{tag}_s{s}.npz'), z=np.stack([p[0] for p in pj]),
                                    body=np.array([p[1] for p in pj]), sign=np.array([p[2] for p in pj]),
                                    informed=np.array([p[3] for p in pj]), ep=np.array([p[4] for p in pj]), t=np.array([p[5] for p in pj]))
                inf = np.array([p[3] for p in pj]); agree = np.array([p[1] == p[2] for p in pj])
                item['labels'] = {'agree_all': float(agree.mean()), 'agree_informed': float(agree[inf].mean()),
                                  'agree_uninformed': float(agree[~inf].mean()) if (~inf).any() else None, 'frac_informed': float(inf.mean())}
                item['batch_body'] = batch_fit(pj, 1, bt, (feat, P), a.device)
                item['batch_oracle'] = batch_fit(pj, 2, bt, (feat, P), a.device)
            res['results'].append(item)
            msg = (f"{time.strftime('%H:%M:%S')} {tag} s{s}: повтор онлайн {lo:.4f} (A11 {item['repro']['life_online_a11']:.4f}), "
                   f"|w| {item['repro']['w_norm_now']:.3f} (A11 {item['repro']['w_norm_a11']:.3f}); вне контура {item['frozen_out_of_loop']['acc']:.3f}")
            if a.readout == 'dual':
                msg += f"; онлайн на тесте {item['online_test_in_loop']['choice_informed']:.3f} / вне {item['online_test_out_of_loop']['acc']:.3f}"
            if do_batch:
                msg += (f"; пакетная (тело) вне {item['batch_body']['out_of_loop']['acc']:.3f}, в контуре {item['batch_body']['in_loop']['choice_informed']:.3f}"
                        f"; оракул вне {item['batch_oracle']['out_of_loop']['acc']:.3f}")
            print(msg + f' | {time.time() - t0:.0f} с', flush=True)
        np.savez(os.path.join(a.out, 'heads', f'heads_{tag}_{"-".join(map(str, grp))}.npz'), w=head.w.cpu().numpy(), b=head.b.cpu().numpy(),
                 seeds=np.array(grp), P=(P.cpu().numpy() if P is not None else np.zeros(0)))
        json.dump(res, open(path, 'w'), ensure_ascii=False)
        del feat; torch.cuda.empty_cache() if a.device == 'cuda' else None
    json.dump(res, open(path, 'w'), ensure_ascii=False)


def run_fastonly(a, commit, dirty):
    tag = tag_of(a, '_fastonly')
    path = os.path.join(a.out, f'fastonly_{tag}.json')
    res = json.load(open(path)) if os.path.exists(path) else {'tag': tag, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'seeds': a.seeds})
    todo = [s for s in a.seeds if s not in {x['seed'] for x in res['results']}]
    t0 = time.time()
    for grp in ([todo] if is_shared(a) else [[s] for s in todo]):
        if not grp:
            continue
        feat, P = build(a, grp[0], 'dual')
        head = bh2.Head(len(grp), feat.mu.shape[0], [0.0] * len(grp), a.device, 'dual', P)
        online, st, _ = live_c(feat, [life_seeds(s) for s in grp], head, 'all', a.device)
        bt = blind_traj(feat, head.proj, sw.TEST_SEEDS, a.device)
        z0 = torch.zeros(feat.mu.shape[0], device=a.device); b0 = torch.zeros((), device=a.device)
        test_in = short(head_eval(feat, z0, b0, 'dual', P, sw.TEST_SEEDS, a.device, online=True))     # одна на все посевы
        test_out = online_out(z0, b0, bt)
        for j, s in enumerate(grp):
            sub = {kk: (vv[j::len(grp)] if kk in ('fitness', 'food', 'poison') else vv) for kk, vv in st.items()}
            sub['log'] = [lg[j:j + 1] for lg in st['log']]
            item = {'seed': s, 'life_online': short(bh.summarize(sub)), 'curve': bh.curve([o for o in online if o[0] == j]),
                    'online_test_in_loop': test_in, 'online_test_out_of_loop': test_out, 'git_commit': commit}
            res['results'].append(item)
            print(f"{time.strftime('%H:%M:%S')} {tag} s{s}: онлайн жизни {item['life_online']['choice_informed']:.3f}; "
                  f"на тесте {test_in['choice_informed']:.3f} / вне контура {test_out['acc']:.3f} | {time.time() - t0:.0f} с", flush=True)
        json.dump(res, open(path, 'w'), ensure_ascii=False)
        del feat; torch.cuda.empty_cache() if a.device == 'cuda' else None
    json.dump(res, open(path, 'w'), ensure_ascii=False)


def make_replayer(head, etas, B, gen, K=REPLAY_PASSES, rule='all', record=None):
    """Пауза после эпизода: проигрывание буфера последних B эпизодов (включая текущий), K проходов,
    мини-пакеты по 32, нормированный шаг по медленным весам (логистическая потеря, только медленный логит).
    rule: all — пары (z_t, «последний исход») на тактах с меткой; event — (z_{t−1}, исход) в такт события.
    record — список, куда складываются эпизоды на CPU (режим record)."""
    buf = []
    etas = torch.as_tensor(etas, dtype=torch.float32, device=head.w.device)

    def after_episode(ep):
        if record is not None:
            record.append({k: v.cpu().numpy() for k, v in ep.items()})
        if rule == 'all':
            Z, Y, M = ep['z'], ep['bel'], ep['m_all']
        else:
            Z = torch.cat([torch.zeros_like(ep['z'][:, :1]), ep['z'][:, :-1]], 1); Y, M = ep['yA'], ep['m_ev']
        buf.append((Z, Y, M, (Z * Z).sum(-1) + 1))
        del buf[:-B]
        Zb, Yb, Mb, Nb = (torch.cat([x[k] for x in buf], 1) for k in range(4))
        n = Zb.shape[1]
        for _ in range(K):
            perm = torch.randperm(n, generator=gen).to(Zb.device)
            for k in range(0, n, REPLAY_MB):
                idx = perm[k:k + REPLAY_MB]
                z, y, m = Zb[:, idx], Yb[:, idx], Mb[:, idx]
                if not m.any():
                    continue
                e = (y - torch.sigmoid((z * head.w[:, None]).sum(-1) + head.b[:, None])) * m
                norm = (m * Nb[:, idx]).sum(1).clamp_min(1.0)
                head.w += (etas / norm)[:, None] * (e[:, :, None] * z).sum(1)
                head.b += etas * e.sum(1) / norm
    return after_episode


def replay_life(a, feat, P, seeds, etas, B, gen_seed, K=REPLAY_PASSES, rule='all', n_ep=None, record=None):
    head = bh2.Head(len(seeds), feat.mu.shape[0], [0.0] * len(seeds), a.device, 'dual', P)   # онлайн — только быстрая часть
    gen = torch.Generator().manual_seed(gen_seed)
    n_ep = n_ep or LIFE
    online, st, _ = live_c(feat, [[500000 + 1000 * s + e for e in range(n_ep)] for s in seeds], head, 'all', a.device,
                           after_episode=make_replayer(head, etas, B, gen, K, rule, record))
    return head, online, st


def run_replay(a, commit, dirty):
    tag = tag_of(a, '_replay' + (f'_{a.name}' if a.name else ''))
    if a.tune:
        res = {'tag': tag, 'grid': {}, 'git_commit': commit, 'git_dirty': dirty}
        t0 = time.time()
        for grp in ([[101, 102, 103]] if is_shared(a) else [[101], [102], [103]]):
            feat, P = build(a, grp[0], 'dual')
            for B in REPLAY_BUFS:
                lives = [(s, e) for s in grp for e in REPLAY_ETAS]
                head, _, _ = replay_life(a, feat, P, [s for s, _ in lives], [e for _, e in lives], B, 7000 + grp[0] + B)
                for j, (s, e) in enumerate(lives):
                    m = bh2.frozen_eval(feat, head, j, TUNE_EVAL, a.device)['choice']['informed']
                    res['grid'].setdefault(f'{e}_{B}', []).append(m)
                print(f"{time.strftime('%H:%M:%S')} посевы {grp} B = {B}: " + ', '.join(f"η {e}: {np.round(res['grid'][f'{e}_{B}'], 3).tolist()}" for e in REPLAY_ETAS)
                      + f' | {time.time() - t0:.0f} с', flush=True)
            del feat; torch.cuda.empty_cache() if a.device == 'cuda' else None
        keys = [f'{e}_{B}' for B in REPLAY_BUFS for e in REPLAY_ETAS]
        best = max(keys, key=lambda k: (np.mean(res['grid'][k]), -keys.index(k)))
        res['best'] = {'eta': float(best.split('_')[0]), 'B': int(best.split('_')[1]), 'mean': float(np.mean(res['grid'][best]))}
        res['sec'] = round(time.time() - t0)
        json.dump(res, open(os.path.join(a.out, f'tune_{tag}.json'), 'w'), ensure_ascii=False)
        print('выбрано', res['best'], flush=True)
        return
    if a.params:
        best = json.load(open(a.params))[a.params_key]
    elif a.eta is not None:
        best = {'eta': a.eta, 'B': a.B, 'K': a.K, 'rule': a.rule}
    else:
        best = json.load(open(os.path.join(a.out, f'tune_{tag}.json')))['best']
    best = {'K': REPLAY_PASSES, 'rule': 'all', **best}
    path = os.path.join(a.out, f'replay_{tag}.json')
    res = json.load(open(path)) if os.path.exists(path) else {'tag': tag, 'eta': best['eta'], 'B': best['B'], 'K': best['K'],
                                                              'rule': best['rule'], 'model': a.model, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'seeds': a.seeds})
    todo = [s for s in a.seeds if s not in {x['seed'] for x in res['results']}]
    t0 = time.time()
    for grp in ([todo] if is_shared(a) else [[s] for s in todo]):
        if not grp:
            continue
        feat, P = build(a, grp[0], 'dual')
        head, online, st = replay_life(a, feat, P, grp, [best['eta']] * len(grp), best['B'], 9000 + grp[0], best['K'], best['rule'])
        bt = blind_traj(feat, head.proj, sw.TEST_SEEDS, a.device)
        for j, s in enumerate(grp):
            sub = {kk: (vv[j::len(grp)] if kk in ('fitness', 'food', 'poison') else vv) for kk, vv in st.items()}
            sub['log'] = [lg[j:j + 1] for lg in st['log']]
            item = {'seed': s, 'life_online': short(bh.summarize(sub)), 'curve': bh.curve([o for o in online if o[0] == j]),
                    'frozen_test': short(bh2.frozen_eval(feat, head, j, sw.TEST_SEEDS, a.device)),
                    'frozen_out_of_loop': frozen_out(head.w[j], head.b[j], bt),
                    'online_test_in_loop': short(head_eval(feat, head.w[j], head.b[j], 'dual', P, sw.TEST_SEEDS, a.device, online=True)),
                    'online_test_out_of_loop': online_out(head.w[j], head.b[j], bt), 'w_norm': float(head.w[j].norm()), 'git_commit': commit}
            res['results'].append(item)
            print(f"{time.strftime('%H:%M:%S')} {tag} s{s}: замороженная {item['frozen_test']['choice_informed']:.3f} (вне контура "
                  f"{item['frozen_out_of_loop']['acc']:.3f}); онлайн жизни {item['life_online']['choice_informed']:.3f}, на тесте "
                  f"{item['online_test_in_loop']['choice_informed']:.3f} / вне {item['online_test_out_of_loop']['acc']:.3f} | {time.time() - t0:.0f} с", flush=True)
        json.dump(res, open(path, 'w'), ensure_ascii=False)
        del feat; torch.cuda.empty_cache() if a.device == 'cuda' else None
    json.dump(res, open(path, 'w'), ensure_ascii=False)


def run_record(a, commit, dirty):
    """Запись опыта для эволюции правила (п. 1A): жизни агента с проигрыванием (параметры сетки A15), a.n_ep эпизодов
    на посев; по каждому посеву — все такты (L, T): z, метки, маски. Только для общих сетей (lm pretrained)."""
    assert is_shared(a)
    os.makedirs(os.path.join(a.out, 'record'), exist_ok=True)
    feat, P = build(a, a.seeds[0], 'dual')
    rec = []; t0 = time.time()
    replay_life(a, feat, P, a.seeds, [a.eta] * len(a.seeds), a.B, 9000 + a.seeds[0], a.K, a.rule, n_ep=a.n_ep, record=rec)
    for j, s in enumerate(a.seeds):
        np.savez(os.path.join(a.out, 'record', f'rec_{tag_of(a, "")}_s{s}.npz'),
                 **{k: np.stack([ep[k][j] for ep in rec]) for k in rec[0]})      # (эпизоды, T, …)
    json.dump({'seeds': a.seeds, 'n_ep': a.n_ep, 'eta': a.eta, 'B': a.B, 'K': a.K, 'rule': a.rule, 'git_commit': commit,
               'git_dirty': dirty, 'sec': round(time.time() - t0)}, open(os.path.join(a.out, 'record', f'rec_{tag_of(a, "")}.json'), 'w'))
    print(f'записано {len(a.seeds)} жизней × {a.n_ep} эпизодов за {time.time() - t0:.0f} с', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['repro', 'fastonly', 'replay', 'record'])
    ap.add_argument('--cond', choices=['lm', 'quad', 'raw'], required=True)
    ap.add_argument('--model', default='EleutherAI/pythia-160m')
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--words', choices=sorted(sw.WORDS), default='c')
    ap.add_argument('--labels', choices=['event', 'all'], default='all')
    ap.add_argument('--readout', choices=['full', 'pca', 'dual'], default='full')
    ap.add_argument('--tune', action='store_true')
    ap.add_argument('--eta', type=float, default=None, help='replay/record: явный шаг медленных')
    ap.add_argument('--B', type=int, default=50)
    ap.add_argument('--K', type=int, default=REPLAY_PASSES)
    ap.add_argument('--rule', choices=['all', 'event'], default='all')
    ap.add_argument('--params', default=None, help='replay: JSON с параметрами (eta, B, K, rule)')
    ap.add_argument('--params-key', default='es_final')
    ap.add_argument('--name', default='', help='replay: суффикс метки')
    ap.add_argument('--n-ep', type=int, default=100, help='record: эпизодов на посев')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--src', default='results_bodyhead2')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_bodyhead3')
    ap.add_argument('--smoke', action='store_true', help='замер времени: жизнь 3 эпизода, тест 20 эпизодов (в выводы не идёт)')
    a = ap.parse_args()
    if a.smoke:
        global LIFE, TUNE_EVAL
        LIFE = 3; TUNE_EVAL = TUNE_EVAL[:20]; sw.TEST_SEEDS = sw.TEST_SEEDS[:20]; a.out = a.out + '_smoke'
    os.makedirs(a.out, exist_ok=True)
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    {'repro': run_repro, 'fastonly': run_fastonly, 'replay': run_replay, 'record': run_record}[a.mode](a, commit, dirty)


if __name__ == '__main__':
    main()
