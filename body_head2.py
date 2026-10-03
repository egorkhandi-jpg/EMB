"""Голова, обученная при жизни от тела — повтор режима i с нормированным шагом (решение Егора 2026-10-03).
flip-2 с разворотом, рефлекс (голова выбирает вид, скорость полная), Pythia-160M. Пререгистрация —
docs/RESULTS_2026-10-01_flip.md, раздел «Режим i с нормированным шагом и ES с простым шагом».

Зачем. v3-body (A10): дельта-правило w += η (y − p) h при |h|² ≈ 768 и сетке η ≥ 0,003 лежало вне области
устойчивости (η·|h|² < 2); голова перескакивала на каждом событии — режим i по существу не проверен.
Здесь — нормированное правило (устойчиво при 0 < η < 2, правило 12 CLAUDE.md):
    w += η (y − p) h̃ / (|h̃|² + 1),  b += η (y − p) / (|h̃|² + 1),  η ∈ {0,01; 0,03; 0,1; 0,3; 1}.

Варианты:
  --labels event — метка только в такт события: признаки — состояние такта поедания, метка — из кадра-события
                   (рост или «без изменения» — пища, падение — яд), как в v3-body;
  --labels all   — метка на каждом информированном такте: «последний исход по виду» держится до следующего
                   события (правило органа, не оракул); признаки — текущее состояние;
  --readout pca  — считывание по первым 32 главным компонентам стандартизованного скрытого состояния; базис
                   снимается один раз на траекториях развития (слепая к знаку политика, 50 эпизодов), не обучается;
  --readout dual — двухскоростная голова: медленные веса (шаг η) + быстрая добавка (шаг η_f = 0,3), сбрасываемая
                   на эпизод; решение — по сумме; на тесте медленные заморожены, быстрая обнулена.
Выбор η — по замороженной голове: обучающие жизни посевов 101–103, оценка на отдельных 100 эпизодах (888000+k).
Главные жизни — посевы 1–10 (50 эпизодов, 500000 + 1000·s + e), тест — те же 100 эпизодов, что в пилоте D.

python body_head2.py --cond lm --weights pretrained --words c --labels all --readout full --tune
python body_head2.py --cond lm --weights pretrained --words c --labels all --readout full
"""
import argparse, json, os, time
import numpy as np
import torch

from env_flip import FlipBodies, T
import stand_words as sw
import body_head as bh
from stand_gpu import git_info

ETAS = (0.01, 0.03, 0.1, 0.3, 1.0)
ETA_FAST = 0.3
LIFE = 50


class Head:
    """Параллельные головы (по одной на жизнь), нормированное дельта-правило, проекция признаков."""

    def __init__(self, L, D, etas, device, readout, P=None):
        self.etas = torch.as_tensor(etas, dtype=torch.float32, device=device)
        self.readout, self.P = readout, P
        d = P.shape[1] if readout == 'pca' else D
        self.w = torch.zeros(L, d, device=device); self.b = torch.zeros(L, device=device)
        self.wf = torch.zeros(L, d, device=device)

    def proj(self, h):
        return h @ self.P if self.readout == 'pca' else h

    def logit(self, z, fast=True):
        wf = self.wf if (self.readout == 'dual' and fast) else 0
        return ((self.w + wf) * z).sum(1) + self.b

    def update(self, z, y, mask):
        err = (y - torch.sigmoid(self.logit(z))) * mask
        norm = (z * z).sum(1) + 1
        g = err / norm
        self.w += (self.etas * g)[:, None] * z; self.b += self.etas * g
        if self.readout == 'dual':
            self.wf += (ETA_FAST * g)[:, None] * z

    def reset_fast(self):
        self.wf.zero_()


def live(feat, seeds_per_life, head, labels_mode, device, frozen=False):
    L, n_ep = len(seeds_per_life), len(seeds_per_life[0])
    online = []; stats = {'fitness': [], 'food': [], 'poison': [], 'log': []}
    ev_count = np.zeros(L, int)
    for e in range(n_ep):
        head.reset_fast()
        bodies = FlipBodies(L, [seeds_per_life[i][e] for i in range(L)], True, 'flip-2')
        feat.reset(L)
        z_prev = None
        belief = np.zeros(L, bool); has_belief = np.zeros(L, bool)
        last_post = np.full(L, -1); rev_seen = np.zeros(L, bool); log = []
        for t in range(T):
            s = bodies.sense()
            has_ev, yA = bh.labels(s)
            belief = np.where(has_ev, yA, belief); has_belief |= has_ev
            z = head.proj(bh.std(feat, feat.features(s)))
            if not frozen:
                if labels_mode == 'event' and z_prev is not None:
                    m = torch.as_tensor(has_ev & bodies.alive, dtype=torch.float32, device=device)
                    if m.any():
                        head.update(z_prev, torch.as_tensor(yA, dtype=torch.float32, device=device), m)
                elif labels_mode == 'all':
                    m = torch.as_tensor(has_belief & bodies.alive, dtype=torch.float32, device=device)
                    if m.any():
                        head.update(z, torch.as_tensor(belief, dtype=torch.float32, device=device), m)
            after = bodies.t_rev < bodies.t
            newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1
            last_post[has_ev] = t
            informed = bodies.alive & (last_post >= 0)
            chA = (head.logit(z, fast=not frozen) >= 0).cpu().numpy()
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
    return online, stats


def frozen_eval(feat, head, j, seeds, device):
    n = len(seeds)
    h1 = Head(n, 1, [0.0] * n, device, head.readout, head.P)
    h1.w = head.w[j].expand(n, -1).clone(); h1.b = head.b[j].expand(n).clone(); h1.wf = torch.zeros_like(h1.w)
    _, st = live(feat, [[s] for s in seeds], h1, 'event', device, frozen=True)
    return bh.summarize(st)


def pca_basis(feat, device, k=32, n_ep=50, seed0=650000):
    """Базис 32 главных компонент стандартизованного h на траекториях развития (слепая политика)."""
    b = FlipBodies(n_ep, [seed0 + i for i in range(n_ep)], True, 'flip-2'); feat.reset(n_ep); H = []
    for _ in range(T):
        s = b.sense()
        H.append(bh.std(feat, feat.features(s))[torch.as_tensor(b.alive, device=device)])
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(n_ep)], 1)); act[~b.alive] = 0
        b.step(act)
    H = torch.cat(H); H = H - H.mean(0)
    _, _, V = torch.linalg.svd(H, full_matrices=False)
    return V[:k].T.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=['lm', 'quad', 'raw'], required=True)
    ap.add_argument('--model', default='EleutherAI/pythia-160m')
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--words', choices=sorted(sw.WORDS), default='c')
    ap.add_argument('--labels', choices=['event', 'all'], required=True)
    ap.add_argument('--readout', choices=['full', 'pca', 'dual'], default='full')
    ap.add_argument('--tune', action='store_true')
    ap.add_argument('--eta', type=float, default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_bodyhead2')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = (f"{a.weights}_{a.words}" if a.cond == 'lm' else a.cond) + f'_{a.labels}_{a.readout}'
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    t0 = time.time()
    shared = not (a.cond == 'quad' or (a.cond == 'lm' and a.weights == 'random'))
    life_seeds = lambda s: [500000 + 1000 * s + e for e in range(LIFE)]

    def build(net_seed):
        f = bh.make_feat(a, net_seed, 32)
        P = pca_basis(f, a.device) if a.readout == 'pca' else None
        return f, P

    if a.tune:
        res = {'tag': tag, 'etas': {str(e): [] for e in ETAS}, 'git_commit': commit, 'git_dirty': dirty}
        groups = [[101, 102, 103]] if shared else [[101], [102], [103]]
        for grp in groups:
            feat, P = build(grp[0])
            lives = [(s, e) for s in grp for e in ETAS]
            head = Head(len(lives), feat.mu.shape[0], [e for _, e in lives], a.device, a.readout, P)
            live(feat, [life_seeds(s) for s, _ in lives], head, a.labels, a.device)
            for j, (s, e) in enumerate(lives):
                res['etas'][str(e)].append(frozen_eval(feat, head, j, [888000 + k for k in range(100)], a.device)['choice']['informed'])
            del feat; torch.cuda.empty_cache() if a.device == 'cuda' else None
        for e in ETAS:
            print(f'η = {e}: замороженная голова на обучающих жизнях {np.round(res["etas"][str(e)], 3).tolist()}', flush=True)
        res['best_eta'] = max(ETAS, key=lambda e: (np.mean(res['etas'][str(e)]), -ETAS.index(e)))
        res['best_mean'] = float(np.mean(res['etas'][str(res['best_eta'])])); res['sec'] = round(time.time() - t0)
        json.dump(res, open(os.path.join(a.out, f'tune_{tag}.json'), 'w'), ensure_ascii=False)
        print('выбран η =', res['best_eta'], flush=True)
        return
    eta = a.eta if a.eta is not None else json.load(open(os.path.join(a.out, f'tune_{tag}.json')))['best_eta']
    path = os.path.join(a.out, f'result_{tag}.json')
    res = json.load(open(path)) if os.path.exists(path) else {'tag': tag, 'eta': eta, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'seeds': a.seeds, 'eta': eta})
    todo = [s for s in a.seeds if s not in {x['seed'] for x in res['results']}]
    for grp in ([todo] if shared else [[s] for s in todo]):
        if not grp:
            continue
        feat, P = build(grp[0])
        head = Head(len(grp), feat.mu.shape[0], [eta] * len(grp), a.device, a.readout, P)
        online, st = live(feat, [life_seeds(s) for s in grp], head, a.labels, a.device)
        for j, s in enumerate(grp):
            sub = {kk: (vv[j::len(grp)] if kk in ('fitness', 'food', 'poison') else vv) for kk, vv in st.items()}
            sub['log'] = [lg[j:j + 1] for lg in st['log']]
            item = {'seed': s, 'life_online': bh.summarize(sub), 'curve': bh.curve([o for o in online if o[0] == j]),
                    'frozen_test': frozen_eval(feat, head, j, sw.TEST_SEEDS, a.device), 'w_norm': float(head.w[j].norm()),
                    'git_commit': commit}
            res['results'].append(item)
            m = item['frozen_test']
            print(f"{time.strftime('%H:%M:%S')} {tag} s{s}: выбор {m['choice']['informed']:.3f} тест {m['test']:.3f} "
                  f"пища/яд {m['food']:.1f}/{m['poison']:.1f} | {time.time() - t0:.0f} с", flush=True)
        json.dump(res, open(path, 'w'), ensure_ascii=False)
        del feat; torch.cuda.empty_cache() if a.device == 'cuda' else None
    json.dump(res, open(path, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
