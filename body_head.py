"""Голова, обученная при жизни от тела (п. 2 плана 2026-10-03): без ES и без оракула.
flip-2 с разворотом, рефлекс (голова выбирает вид, скорость полная), Pythia-160M.
Пререгистрация — docs/RESULTS_2026-10-01_flip.md, раздел «Дрейф ES и голова от тела».

Учитель — тело. Кадр-событие (флаг «съел вид X» + тренд энергии) приходит в такт после поедания;
знак изменения энергии говорит, был ли X пищей в момент поедания: рост или «без изменения» (пища
при полной энергии) — пища, падение — яд. Метка y = «A сейчас пища». Признаки — стандартизованное
скрытое состояние такта поедания (до того как исход виден): это состояние, по которому принималось
решение. Голова — логистическая p(A пища) = σ(w·h̃ + b); на каждом событии одно обновление
дельта-правилом: w += η (y − p) h̃, b += η (y − p). Выбор вида — A, если p ≥ 0,5.

Режимы:
  i  «развитие»  — голова сохраняется между 50 эпизодами одной жизни; тест — на отложенных 100 эпизодах
                   (те же, что в пилоте D и п. 15) с замороженной после жизни головой; плюс онлайн-кривая
                   по числу событий жизни. Окно N = 32.
  ii «память в весах» — голова сбрасывается в начале каждого эпизода, окно N = 1 (только текущий такт):
                   знак может хранить лишь сама голова. Описательно; мера — онлайн-выбор в эпизодах жизни.
Признаки: lm (pretrained/random, слова c/c_arb1), quad (768) — как в stand_words; raw — one-hot окно событий.
Стандартизация — фиксированная, по случайному поведению (как в stand_words).
η — по сетке на обучающих жизнях (посевы 101–103), оценка на отдельных 100 эпизодах (888000+k);
главные жизни — посевы 1–10, эпизоды жизни 500000 + 1000·s + e.

python body_head.py --cond lm --weights pretrained --words c --tune     # выбор η
python body_head.py --cond lm --weights pretrained --words c --eta 0.03 --seeds 1 2 ... 10
"""
import argparse, json, math, os, time
import numpy as np
import torch

import env_flip as ef
from env_flip import FlipBodies, T
import stand_words as sw
from stand_gpu import git_info

ETAS = (0.003, 0.01, 0.03, 0.1, 0.3)
LIFE = 50


class RawWords(sw.Standardized):
    """One-hot окно событий (N тактов × 6) — линейное чтение сырых событий; величина билинейная."""

    def __init__(self, N, device):
        self.N, self.device, self.D = N, device, N * 6

    def reset(self, B):
        self.buf = torch.zeros(B, self.N, 6, device=self.device)

    def features(self, s):
        self.buf = torch.cat([self.buf[:, 1:], torch.as_tensor(sw.onehot(s), device=self.device)[:, None]], 1)
        return self.buf.reshape(len(s), -1)


def make_feat(a, net_seed, N):
    if a.cond == 'lm':
        f = sw.LMWords(a.model, a.weights, a.words, N, a.device, net_seed)
    elif a.cond == 'quad':
        f = sw.QuadWords(N, 768, a.device, net_seed)
    else:
        f = RawWords(N, a.device)
    f.fit_std(np.random.RandomState(0))
    return f


def std(feat, h):
    return (h - feat.mu) / feat.sd


def labels(s):
    """По кадру-событию: (есть событие, A — пища)."""
    ateA, ateB = s[:, 10] > 0, s[:, 11] > 0
    food = s[:, 7] > -0.5                                   # рост или «без изменения»
    return ateA | ateB, np.where(ateA, food, ~food)


def live(feat, seeds_per_life, eta, device, reset_each_episode=False, frozen=None):
    """Параллельные жизни (одна сеть на все). seeds_per_life — список по жизням списков посевов эпизодов.
    frozen — (w, b) на жизнь: без обучения (тест). Возвращает веса, онлайн-статистику, итоги эпизодов."""
    L = len(seeds_per_life); n_ep = len(seeds_per_life[0])
    D = feat.mu.shape[0]
    w = torch.zeros(L, D, device=device) if frozen is None else frozen[0].clone()
    b = torch.zeros(L, device=device) if frozen is None else frozen[1].clone()
    ev_count = np.zeros(L, int); online = []          # (жизнь, номер события жизни, верно ли, информирован)
    stats = {'correct': 0, 'n': 0, 'fitness': [], 'food': [], 'poison': [], 'log': []}
    for e in range(n_ep):
        if reset_each_episode and frozen is None:
            w.zero_(); b.zero_()
        bodies = FlipBodies(L, [seeds_per_life[i][e] for i in range(L)], True, 'flip-2')
        feat.reset(L)
        h_prev = None
        last_post = np.full(L, -1); rev_seen = np.zeros(L, bool); log = []
        for t in range(T):
            s = bodies.sense()
            has_ev, yA = labels(s)
            if frozen is None and h_prev is not None:
                upd = torch.as_tensor(has_ev & bodies.alive, device=device)
                if upd.any():
                    p = torch.sigmoid((w * h_prev).sum(1) + b)
                    err = (torch.as_tensor(yA, dtype=torch.float32, device=device) - p) * upd
                    w += eta * err[:, None] * h_prev; b += eta * err
            after = bodies.t_rev < bodies.t
            newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1
            last_post[has_ev] = t
            informed = bodies.alive & (last_post >= 0)
            h = std(feat, feat.features(s))
            chA = (((w * h).sum(1) + b) >= 0).cpu().numpy()
            correct = chA == bodies.good_A
            for i in np.where(informed)[0]:
                online.append((i, ev_count[i], bool(correct[i])))
            ev_count += has_ev & bodies.alive
            stats['correct'] += int(correct[informed].sum()); stats['n'] += int(informed.sum())
            log.append(np.stack([chA, bodies.good_A.copy(), bodies.alive.copy(), has_ev, after], 1))
            act = sw.reflex(s, np.stack([np.where(chA, 1.0, -1.0), np.ones(L)], 1))
            act[~bodies.alive] = 0
            bodies.step(act)
            h_prev = h
        stats['fitness'] += bodies.fitness().tolist(); stats['food'] += bodies.n_good.tolist(); stats['poison'] += bodies.n_bad.tolist()
        stats['log'].append(np.stack(log, 1))
    return w, b, online, stats


def frozen_eval(feat, w1, b1, seeds, device):
    """Замороженная голова одной жизни на наборе эпизодов — параллельно, по эпизоду на тело."""
    n = len(seeds)
    _, _, _, st = live(feat, [[s] for s in seeds], 0.0, device, frozen=(w1.expand(n, -1).clone(), b1.expand(n).clone()))
    return summarize(st)


def summarize(stats):
    logs = np.concatenate(stats['log'], 0)
    ch = sw.choice_stats(logs)
    f = np.array(stats['fitness'])
    return {'choice': ch, 'test': float(f.mean()), 'test_per_episode': [round(float(x), 4) for x in f],
            'food': float(np.mean(stats['food'])), 'poison': float(np.mean(stats['poison']))}


def curve(online, bins=(0, 5, 10, 20, 40, 80, 160, 10 ** 9)):
    o = np.array([(c, k) for _, c, k in online]) if online else np.zeros((0, 2))
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (o[:, 0] >= lo) & (o[:, 0] < hi) if len(o) else np.zeros(0, bool)
        out.append([lo, None if not m.any() else float(o[m, 1].mean()), int(m.sum())])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=['lm', 'quad', 'raw'], required=True)
    ap.add_argument('--model', default='EleutherAI/pythia-160m')
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--words', choices=sorted(sw.WORDS), default='c')
    ap.add_argument('--mode', choices=['i', 'ii'], default='i')
    ap.add_argument('--tune', action='store_true', help='выбор η по сетке на обучающих жизнях 101–103')
    ap.add_argument('--eta', type=float, default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_bodyhead')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = (f"{a.weights}_{a.words}" if a.cond == 'lm' else a.cond) + f'_mode{a.mode}'
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    N = 32 if a.mode == 'i' else 1
    t0 = time.time()
    shared = not (a.cond == 'quad' or (a.cond == 'lm' and a.weights == 'random'))   # одна сеть на все жизни?
    life_seeds = lambda s: [500000 + 1000 * s + e for e in range(LIFE)]
    if a.tune:
        res = {'tag': tag, 'mode': a.mode, 'etas': {}, 'git_commit': commit, 'git_dirty': dirty}
        for eta in ETAS:
            accs = []
            for s in (101, 102, 103):
                feat = make_feat(a, s, N)
                w, b, online, st = live(feat, [life_seeds(s)], eta, a.device, reset_each_episode=(a.mode == 'ii'))
                if a.mode == 'i':
                    accs.append(frozen_eval(feat, w[0], b[0], [888000 + k for k in range(100)], a.device)['choice']['informed'])
                else:
                    accs.append(summarize(st)['choice']['informed'])
                del feat; torch.cuda.empty_cache() if a.device == 'cuda' else None
            res['etas'][str(eta)] = accs
            print(f'η = {eta}: выбор (обучающие жизни) {np.round(accs, 3).tolist()}', flush=True)
        best = max(ETAS, key=lambda e: (np.mean(res['etas'][str(e)]), -ETAS.index(e)))
        res['best_eta'] = best; res['sec'] = round(time.time() - t0)
        json.dump(res, open(os.path.join(a.out, f'tune_{tag}.json'), 'w'), ensure_ascii=False)
        print('выбран η =', best, flush=True)
        return
    eta = a.eta if a.eta is not None else json.load(open(os.path.join(a.out, f'tune_{tag}.json')))['best_eta']
    path = os.path.join(a.out, f'result_{tag}.json')
    res = json.load(open(path)) if os.path.exists(path) else {'tag': tag, 'mode': a.mode, 'eta': eta, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'seeds': a.seeds, 'eta': eta})
    done = {x['seed'] for x in res['results']}
    todo = [s for s in a.seeds if s not in done]
    groups = [todo] if shared else [[s] for s in todo]
    for grp in groups:
        if not grp:
            continue
        feat = make_feat(a, grp[0], N)
        w, b, online, st = live(feat, [life_seeds(s) for s in grp], eta, a.device, reset_each_episode=(a.mode == 'ii'))
        for j, s in enumerate(grp):
            sub = {kk: (vv[j::len(grp)] if kk in ('fitness', 'food', 'poison') else vv) for kk, vv in st.items()}
            sub['log'] = [lg[j:j + 1] for lg in st['log']]
            item = {'seed': s, 'life_online': summarize(sub),
                    'curve': curve([o for o in online if o[0] == j]), 'git_commit': commit}
            if a.mode == 'i':
                item['frozen_test'] = frozen_eval(feat, w[j], b[j], sw.TEST_SEEDS, a.device)
                item['w_norm'] = float(w[j].norm()); item['b'] = float(b[j])
            res['results'].append(item)
            m = item.get('frozen_test', item['life_online'])
            print(f"{time.strftime('%H:%M:%S')} {tag} s{s}: выбор {m['choice']['informed']:.3f} тест {m['test']:.3f} "
                  f"пища/яд {m['food']:.1f}/{m['poison']:.1f} | {time.time() - t0:.0f} с", flush=True)
        json.dump(res, open(path, 'w'), ensure_ascii=False)
        del feat; torch.cuda.empty_cache() if a.device == 'cuda' else None
    json.dump(res, open(path, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
