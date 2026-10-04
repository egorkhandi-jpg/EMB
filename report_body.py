"""Звено 4 — отчёт и тело (вариант A, решение Егора 2026-10-04). Без обучения LM. Пререгистрация — docs/RESULTS_2026-10-04_report.md.

Вопрос: предсказывают ли логиты ответных символов замороженной LM состояние тела — выше копирования и выше случайной сети —
на переменной, которой нет в окне буквально (знак эпизода: связка вида и исхода)?

Данные: flip-2 с разворотом, слепая политика (рефлекс к ближайшему предмету любого вида), 10 наборов × 100 эпизодов
(800000 + 1000·i + e). Интерфейс c + BOS, окно 32 такта (как LMWords). Запросы — информированные такты t ∈ {12, 16, …, 96}
с меткой тела на t − 8 (для few-shot); на каждом запросе к окну дописываются символы вопроса и читаются логиты ответов.
Режимы (суффикс после окна):
  zs1 / zs2 / zs3 — zero-shot: ' Food:' / ' Safe to eat:' / ' Good kind:' → ' A' против ' B';
  fs   — few-shot: 4 прошлые пары «' Food:' ответ» (ответ — метка тела на t−8, t−6, t−4, t−2: ' A' или ' B'), затем ' Food:';
  fsx  — few-shot с произвольными символами: вопрос ' chair', ответы ' door' (A) / ' stone' (B) — описательно;
  zs1_sh, fs_sh — то же с перемешанным порядком кадров окна (кадры целиком; вопрос и пары — на месте);
  en   — описательно: ' Energy:' → ' up' против ' down', истина — последний исход (пища / яд).
Мера — AUC разности логитов (первый ответ − второй) против истинного знака (en — против последнего исхода).
Контроли: случайная LM (сеть от набора), копирование (вид последнего съеденного), «последний ответ примера» (для few-shot),
линейный зонд по скрытому состоянию на позиции вопроса (верхняя граница, AUC по отложенным эпизодам).

python report_body.py --model EleutherAI/pythia-160m --weights pretrained
"""
import argparse, json, os, time
import numpy as np
import torch

from env_flip import FlipBodies, T
import stand_words as sw
import body_head as bh
from readout import fit_probe
from probe_fast import _newton
from stand_gpu import git_info

A, B, NONE, UP, DOWN, SAME = 329, 378, 5293, 598, 1066, 1072
Q = {'zs1': [11431, 27], 'zs2': [28481, 281, 6008, 27], 'zs3': [7088, 2238, 27], 'en': [11669, 27]}
QX, AX, BX = [6951], 3369, 8805
N = 32
QT = list(range(12, 97, 4))


def trajectories(seeds):
    """Слепая политика: по тактам — токены событий, метка тела, информированность, знак, последний вид и исход."""
    n = len(seeds)
    b = FlipBodies(n, seeds, True, 'flip-2')
    belief = np.zeros(n, bool); hb = np.zeros(n, bool); last_post = np.full(n, -1); rev_seen = np.zeros(n, bool)
    last_kind = np.full(n, -1); last_food = np.full(n, -1)
    rec = {k: [] for k in ('kt', 'et', 'bel', 'hb', 'inf', 'ga', 'lk', 'lf')}
    for t in range(T):
        s = b.sense()
        kind, en = sw.events(s)
        has_ev, yA = bh.labels(s)
        belief = np.where(has_ev, yA, belief); hb |= has_ev
        last_kind = np.where(has_ev, kind, last_kind); last_food = np.where(has_ev, (s[:, 7] > -0.5).astype(int), last_food)
        after = b.t_rev < b.t; newrev = after & ~rev_seen; rev_seen |= after; last_post[newrev] = -1; last_post[has_ev] = t
        rec['kt'].append(np.array([A, B, NONE])[kind]); rec['et'].append(np.array([UP, DOWN, SAME])[en])
        rec['bel'].append(belief.copy()); rec['hb'].append(hb & b.alive); rec['inf'].append(b.alive & (last_post >= 0))
        rec['ga'].append(b.good_A.copy()); rec['lk'].append(last_kind.copy()); rec['lf'].append(last_food.copy())
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(n)], 1)); act[~b.alive] = 0
        b.step(act)
    return {k: np.stack(v, 1) for k, v in rec.items()}                       # (эпизоды, T)


def queries(tr):
    q = [(e, t) for e in range(tr['ga'].shape[0]) for t in QT if tr['inf'][e, t] and tr['hb'][e, t - 8]]
    return np.array(q)


def window(tr, e, t, rng=None):
    fr = []
    for u in range(t - N + 1, t + 1):
        fr.append([tr['kt'][e, u], tr['et'][e, u]] if u >= 0 else [NONE, SAME])
    if rng is not None:
        fr = [fr[i] for i in rng.permutation(N)]
    return [x for f in fr for x in f]


def build(tr, qs, mode, seed=0):
    rng = np.random.RandomState(seed) if mode.endswith('_sh') else None
    base = mode.replace('_sh', '')
    ids = []
    for e, t in qs:
        w = [sw.BOS] + window(tr, e, t, rng)
        if base in ('zs1', 'zs2', 'zs3', 'en'):
            ids.append(w + Q[base])
        elif base == 'fs':
            pairs = []
            for d in (8, 6, 4, 2):
                pairs += Q['zs1'] + [A if tr['bel'][e, t - d] else B]
            ids.append(w + pairs + Q['zs1'])
        elif base == 'fsx':
            pairs = []
            for d in (8, 6, 4, 2):
                pairs += QX + [AX if tr['bel'][e, t - d] else BX]
            ids.append(w + pairs + QX)
    return torch.tensor(ids)


def answers(mode):
    base = mode.replace('_sh', '')
    return (UP, DOWN) if base == 'en' else ((AX, BX) if base == 'fsx' else (A, B))


def auc(score, y):
    y = y.astype(bool); pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    # средние ранги при равенствах
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def probe_auc(H, y, g, dev):
    """Линейный зонд: λ по CV с группами (fit_probe), затем решения на отложенных эпизодах (5 частей) → AUC."""
    _, _, _, _, lam = fit_probe(H, y.astype(float), g, dev)
    ug = np.unique(g); r = np.random.RandomState(0); r.shuffle(ug); dec = np.zeros(len(y))
    Ht = torch.as_tensor(H, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    for f in range(5):
        te = np.isin(g, ug[f::5]); tr_ = torch.as_tensor(~te, device=dev)
        mu, sd = Ht[tr_].mean(0), Ht[tr_].std(0, unbiased=False) + 1e-6
        X1 = torch.cat([(Ht - mu) / sd, torch.ones(len(y), 1, device=dev)], 1)
        w = _newton(X1[tr_], yt[tr_], lam, torch.zeros(X1.shape[1], device=dev))
        dec[te] = (X1[torch.as_tensor(te, device=dev)] @ w).cpu().numpy()
    return auc(dec, y)


def load_lm(model, weights, seed, dev):
    from transformers import AutoModelForCausalLM, AutoConfig
    if weights == 'random':
        pre = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        pn = pre.get_input_embeddings().weight.norm(dim=1).mean().item(); del pre
        torch.manual_seed(seed); m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model)).to(torch.float32)
        with torch.no_grad():
            E = m.get_input_embeddings().weight; E.mul_(pn / E.norm(dim=1).mean())
    else:
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_lm(m, ids, ans, dev, chunk):
    """Последнее скрытое состояние (после финальной нормировки) и разность логитов ответов."""
    E = m.get_input_embeddings().weight; W = m.get_output_embeddings().weight
    H, D = [], []
    for i in range(0, len(ids), chunk):
        x = ids[i:i + chunk].to(dev)
        with torch.no_grad():
            h = m.gpt_neox(inputs_embeds=E[x]).last_hidden_state[:, -1]
            if not torch.isfinite(h).all():
                raise FloatingPointError('нечисловое скрытое состояние')
            lg = h @ W[list(ans)].T
        H.append(h.cpu().numpy()); D.append((lg[:, 0] - lg[:, 1]).cpu().numpy())
    return np.concatenate(H), np.concatenate(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='EleutherAI/pythia-160m')
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--sets', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--modes', nargs='+', default=['zs1', 'zs2', 'zs3', 'fs', 'fsx', 'zs1_sh', 'fs_sh', 'en'])
    ap.add_argument('--probe-modes', nargs='+', default=['zs1', 'fs'])
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--out', default='results_report')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    dev = a.device; commit, dirty = git_info(); t0 = time.time()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.model.split('-')[-1]}_{a.weights}"
    path = os.path.join(a.out, f'report_{tag}.json')
    res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'model': a.model, 'weights': a.weights, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'sets': a.sets})
    done = {x['set'] for x in res['results']}
    m = None
    for i in a.sets:
        if i in done:
            continue
        if m is None or a.weights == 'random':
            del m; torch.cuda.empty_cache() if dev == 'cuda' else None
            m = load_lm(a.model, a.weights, i, dev)
        tr = trajectories([800000 + 1000 * i + e for e in range(a.episodes)])
        qs = queries(tr)
        e_, t_ = qs[:, 0], qs[:, 1]
        y = tr['ga'][e_, t_]; ylast = tr['lf'][e_, t_] == 1
        copy = np.where(tr['lk'][e_, t_] == 0, 1.0, np.where(tr['lk'][e_, t_] == 1, -1.0, 0.0))
        lastans = np.where(tr['bel'][e_, t_ - 2], 1.0, -1.0)
        r = {'set': i, 'n_queries': int(len(qs)), 'auc_copy': auc(copy, y), 'auc_lastanswer': auc(lastans, y),
             'acc_copy': float(((copy > 0) == y).mean()), 'acc_lastanswer': float(((lastans > 0) == y).mean()), 'modes': {}}
        for mode in a.modes:
            ids = build(tr, qs, mode, seed=i)
            H, Dd = run_lm(m, ids, answers(mode), dev, a.chunk)
            yy = ylast if mode == 'en' else y
            mr = {'auc': auc(Dd, yy), 'acc': float(((Dd > 0) == yy).mean())}
            if mode in a.probe_modes:
                mr['probe_auc'] = probe_auc(H, y, e_, dev)
            r['modes'][mode] = mr
        r['git_commit'] = commit
        res['results'].append(r)
        json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f"{time.strftime('%H:%M:%S')} {tag} набор {i}: запросов {len(qs)} | " + ' '.join(f"{k} {v['auc']:.3f}" + (f"/зонд {v['probe_auc']:.3f}" if 'probe_auc' in v else '')
              for k, v in r['modes'].items()) + f" | копирование {r['auc_copy']:.3f} | {time.time() - t0:.0f} с", flush=True)


if __name__ == '__main__':
    main()
