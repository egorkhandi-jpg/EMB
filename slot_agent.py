"""Опыт со слотом, вариант F (решение Егора 2026-10-04). Пререгистрация — docs/RESULTS_2026-10-02_slot.md, дополнение 2.

Вопрос: может ли собственная запись агента в слот, которую читает замороженный аппарат LM, заменить быстрые веса как память
дальше окна — без обучения при жизни.

flip-2 с разворотом, рефлекс, Pythia-160M, интерфейс c + BOS, fp32, фиксированная стандартизация (по случайному поведению).
Окно N = 4 такта. Слот — одна позиция после BOS: символ из алфавита 8 однотокенных слов (' cat' ' table' ' lamp' ' cloud'
' violin' ' carpet' ' bottle' ' feather') или «пусто» (' stone') в начале эпизода.
  organ  — на каждом событии слот := код (вид съеденного, исход): A-пища ' cat', A-яд ' table', B-пища ' lamp', B-яд ' cloud'
           (исход — по тренду энергии, как метка тела: рост или «без изменения» — пища); правило не зависит от политики;
  random — на каждом событии слот := случайный из 8 символов (содержание знака не несёт).
Агенты: slot_organ, slot_random, noslot (окно 4), fast (noslot + быстрые веса: нормированное правило A11, η 0,3, сброс на эпизод),
win32 (окно 32, без слота), oracle_sign.
Замороженные головы — пакетная логистическая регрессия до сходимости по меткам тела «последний исход» (как A14): жизнь
развития посева s — 50 эпизодов слепой политики (500000 + 1000·s + e; траектории одни для всех агентов, отличается вход).
Тест: вне контура — слепая политика на 100 тестовых эпизодах (слот пишется тем же правилом); в контуре — рефлекс, те же эпизоды.
Мера — выбор вида на информированных тактах при k > 4 (k — тактов после последнего события) и при k ≤ 4.

python slot_agent.py --weights pretrained --seeds 1 ... 10
"""
import argparse, json, os, time
import numpy as np
import torch

import env_flip as ef
from env_flip import FlipBodies, T
import stand_words as sw
import body_head as bh
from readout import fit_probe
from stand_gpu import git_info

SLOT = [5798, 2829, 18067, 9005, 36608, 23656, 11996, 22676]
EMPTY = 8805
ORGAN = {(0, True): 0, (0, False): 1, (1, True): 2, (1, False): 3}       # (вид: 0 A / 1 B, пища?) -> индекс символа
AGENTS = ['slot_organ', 'slot_random', 'noslot', 'fast', 'win32', 'oracle_sign']
ETA_FAST = 0.3


class SlotLM(sw.Standardized):
    def __init__(self, weights, slot, N, device, net_seed, lm=None):
        if lm is None:
            base = sw.LMWords('EleutherAI/pythia-160m', weights, 'c', N, device, net_seed)
            lm = base.lm
        self.lm, self.E = lm, lm.get_input_embeddings().weight
        (tA, tB, tN), (tU, tD, tS) = sw.WORDS['c']
        self.kind_tok = torch.tensor([tA, tB, tN], device=device); self.en_tok = torch.tensor([tU, tD, tS], device=device)
        self.slot, self.N, self.device = slot, N, device
        self.slot_tok = torch.tensor(SLOT + [EMPTY], device=device)

    def reset(self, B, seeds=None):
        self.buf = []
        self.state = torch.full((B,), 8, dtype=torch.long, device=self.device)        # «пусто»
        self.rr = [np.random.RandomState((s if seeds is not None else i) + 4_000_000) for i, s in enumerate(seeds if seeds is not None else range(B))]

    def write(self, s):
        """Запись в слот по событию кадра (правило органа или случайная)."""
        a0, a1 = s[:, 10] > 0, s[:, 11] > 0
        ev = a0 | a1
        if not ev.any() or self.slot is None:
            return
        food = s[:, 7] > -0.5
        new = self.state.cpu().numpy().copy()
        for i in np.nonzero(ev)[0]:
            new[i] = ORGAN[(0 if a0[i] else 1, bool(food[i]))] if self.slot == 'organ' else self.rr[i].randint(8)
        self.state = torch.as_tensor(new, device=self.device)

    def features(self, s):
        self.write(s)
        kind, en = sw.events(s)
        tok = torch.stack([self.kind_tok[torch.as_tensor(kind, device=self.device)], self.en_tok[torch.as_tensor(en, device=self.device)]], 1)
        self.buf.append(tok); self.buf = self.buf[-self.N:]
        win = torch.stack(self.buf, 1).reshape(len(s), -1)
        parts = [torch.full((len(s), 1), sw.BOS, device=self.device, dtype=torch.long)]
        if self.slot is not None:
            parts.append(self.slot_tok[self.state][:, None])
        ids = torch.cat(parts + [win], 1)
        out = []
        for i in range(0, len(s), 200):
            with torch.no_grad():
                h = self.lm(inputs_embeds=self.E[ids[i:i + 200]]).last_hidden_state[:, -1]
            if not torch.isfinite(h).all():
                raise FloatingPointError('нечисловое скрытое состояние')
            out.append(h)
        return torch.cat(out)


def blind(feat, seeds, dev):
    """Слепая политика: стандартизованные признаки, метки тела, знак, информированность, k."""
    n = len(seeds)
    b = FlipBodies(n, seeds, True, 'flip-2'); feat.reset(n, seeds)
    belief = np.zeros(n, bool); hb = np.zeros(n, bool); last_post = np.full(n, -1); rev_seen = np.zeros(n, bool); last_ev = np.full(n, -10 ** 6)
    out = {k: [] for k in ('Z', 'BEL', 'HB', 'INF', 'GA', 'K', 'EP')}
    for t in range(T):
        s = b.sense()
        has_ev, yA = bh.labels(s)
        belief = np.where(has_ev, yA, belief); hb |= has_ev; last_ev = np.where(has_ev, t, last_ev)
        out['Z'].append(bh.std(feat, feat.features(s)))
        after = b.t_rev < b.t; newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1; last_post[has_ev] = t
        out['BEL'].append(belief.copy()); out['HB'].append(hb & b.alive); out['INF'].append(b.alive & (last_post >= 0))
        out['GA'].append(b.good_A.copy()); out['K'].append(t - last_ev); out['EP'].append(np.arange(n))
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(n)], 1)); act[~b.alive] = 0
        b.step(act)
    r = {'Z': torch.stack(out['Z'])}
    for k in ('BEL', 'HB', 'INF', 'GA', 'K', 'EP'):
        r[k] = np.stack(out[k])
    return r                                                                     # (T, n, …)


def fit_head(d, dev):
    m = d['HB']
    X = d['Z'][torch.as_tensor(m, device=dev)].cpu().numpy(); y = d['BEL'][m].astype(float); g = d['EP'][m]
    mu, sd, w, cv, lam = fit_probe(X, y, g, dev)
    return (w[:-1] / sd), (w[-1] - (mu / sd * w[:-1]).sum())


def acc_bins(pred, d):
    inf, k, ga = d['INF'], d['K'], d['GA']
    res = {}
    for nm, m in (('k_gt4', inf & (k > 4)), ('k_le4', inf & (k <= 4)), ('all', inf)):
        res[nm] = float((pred[m] == ga[m]).mean()) if m.any() else None
        res['n_' + nm] = int(m.sum())
    return res


def out_of_loop(d, w, b, fast):
    Z = d['Z']
    if not fast:
        return acc_bins(((Z @ w + b) >= 0).cpu().numpy(), d)
    n, D = Z.shape[1], Z.shape[2]
    wf = torch.zeros(n, D, device=Z.device); P = []
    for t in range(T):
        z = Z[t]
        hb = torch.as_tensor(d['HB'][t], dtype=torch.float32, device=Z.device); y = torch.as_tensor(d['BEL'][t], dtype=torch.float32, device=Z.device)
        err = (y - torch.sigmoid(((w + wf) * z).sum(1) + b)) * hb
        wf += (ETA_FAST * err / ((z * z).sum(1) + 1))[:, None] * z
        P.append(((((w + wf) * z).sum(1) + b) >= 0).cpu().numpy())
    return acc_bins(np.stack(P), d)


def in_loop(feat, w, b, seeds, dev, fast=False, oracle=False):
    n = len(seeds)
    bd = FlipBodies(n, seeds, True, 'flip-2'); feat.reset(n, seeds) if feat is not None else None
    belief = np.zeros(n, bool); hb = np.zeros(n, bool); last_post = np.full(n, -1); rev_seen = np.zeros(n, bool); last_ev = np.full(n, -10 ** 6)
    wf = torch.zeros(n, w.shape[0], device=dev) if w is not None else None
    P, INF, K, GA = [], [], [], []
    for t in range(T):
        s = bd.sense()
        has_ev, yA = bh.labels(s)
        belief = np.where(has_ev, yA, belief); hb |= has_ev; last_ev = np.where(has_ev, t, last_ev)
        after = bd.t_rev < bd.t; newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1; last_post[has_ev] = t
        if oracle:
            chA = bd.good_A.copy()
        else:
            z = bh.std(feat, feat.features(s))
            if fast:
                m = torch.as_tensor(hb & bd.alive, dtype=torch.float32, device=dev); y = torch.as_tensor(belief, dtype=torch.float32, device=dev)
                err = (y - torch.sigmoid(((w + wf) * z).sum(1) + b)) * m
                wf += (ETA_FAST * err / ((z * z).sum(1) + 1))[:, None] * z
            chA = ((((w + wf) * z).sum(1) + b) >= 0).cpu().numpy()
        P.append(chA); INF.append(bd.alive & (last_post >= 0)); K.append(t - last_ev); GA.append(bd.good_A.copy())
        act = sw.reflex(s, np.stack([np.where(chA, 1.0, -1.0), np.ones(n)], 1)); act[~bd.alive] = 0
        bd.step(act)
    r = acc_bins(np.stack(P), {'INF': np.stack(INF), 'K': np.stack(K), 'GA': np.stack(GA)})
    r.update({'test': float(bd.fitness().mean()), 'food': float(bd.n_good.mean()), 'poison': float(bd.n_bad.mean())})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--agents', nargs='+', default=AGENTS)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--dev-eps', type=int, default=50)
    ap.add_argument('--test-eps', type=int, default=100)
    ap.add_argument('--out', default='results_slot')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    dev = a.device; commit, dirty = git_info(); t0 = time.time()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'slot_{a.weights}.json')
    res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'weights': a.weights, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'seeds': a.seeds, 'agents': a.agents})
    done = {(x['seed'], x['agent']) for x in res['results']}
    tests = sw.TEST_SEEDS[:a.test_eps]
    spec = {'slot_organ': ('organ', 4), 'slot_random': ('random', 4), 'noslot': (None, 4), 'fast': (None, 4), 'win32': (None, 32)}
    shared = a.weights == 'pretrained'
    lm_cache = None
    for s in a.seeds:
        if not shared or lm_cache is None:
            lm_cache = sw.LMWords('EleutherAI/pythia-160m', a.weights, 'c', 4, dev, s if not shared else 1).lm
        feats = {}
        for ag in a.agents:
            if (s, ag) in done:
                continue
            if ag == 'oracle_sign':
                r = {'seed': s, 'agent': ag, 'out_of_loop': {'k_gt4': 1.0, 'k_le4': 1.0, 'all': 1.0}, 'in_loop': in_loop(None, None, None, tests, dev, oracle=True)}
            else:
                key = spec[ag]
                if key not in feats:
                    f = SlotLM(a.weights, key[0], key[1], dev, s, lm=lm_cache); f.fit_std(np.random.RandomState(0))
                    d_dev = blind(f, [500000 + 1000 * s + e for e in range(a.dev_eps)], dev)
                    w, b = fit_head(d_dev, dev)
                    d_test = blind(f, tests, dev)
                    feats[key] = (f, w, b, d_test)
                f, w, b, d_test = feats[key]
                fast = ag == 'fast'
                r = {'seed': s, 'agent': ag, 'out_of_loop': out_of_loop(d_test, w, b, fast), 'in_loop': in_loop(f, w, b, tests, dev, fast=fast)}
            r['git_commit'] = commit
            res['results'].append(r)
            json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
            print(f"{time.strftime('%H:%M:%S')} {a.weights} s{s} {ag:12s}: вне контура k>4 {r['out_of_loop']['k_gt4']:.3f} k≤4 {r['out_of_loop']['k_le4']:.3f} | "
                  f"в контуре k>4 {r['in_loop']['k_gt4']} k≤4 {r['in_loop']['k_le4']} тест {r['in_loop']['test']:.3f} | {time.time() - t0:.0f} с", flush=True)
        del feats; torch.cuda.empty_cache() if dev == 'cuda' else None


if __name__ == '__main__':
    main()
