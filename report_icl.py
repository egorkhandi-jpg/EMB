"""Звено 4, продолжение (A23): few-shot с примерами из ДРУГИХ эпизодов (решение Егора 2026-10-04).
Пререгистрация — docs/RESULTS_2026-10-04_report.md, раздел A23. Без обучения LM.

Данные, модели, интерфейс, слепая политика и запросы — как в A22 (report_body.py). Контекст:
BOS; k примеров, каждый = окно 32 такта другого эпизода того же набора на его информированном такте (из сетки запросов)
+ ' Food:' + ответ (метка тела на этом такте: ' A' / ' B'); затем текущее окно + ' Food:'. Ответы примеров
сбалансированы (k/2 A, k/2 B), эпизоды примеров различны и не совпадают с текущим, порядок случайный.
Режимы: c4 (основной, k = 4), c2, c8; c4_shcur — перемешанные кадры текущего окна; c4_shex — перемешанные кадры окон примеров
(ответы те же); c4x — произвольные символы (' chair' → ' door' / ' stone'); c4_en — последний исход (' Energy:' → ' up' / ' down',
примеры сбалансированы по исходу).
Контроли: случайная LM; «последний ответ примера» и «большинство ответов» (по построению ≈ 0,5); зонд на позиции вопроса (c4).

python report_icl.py --model EleutherAI/pythia-160m --weights pretrained
"""
import argparse, json, os, time
import numpy as np
import torch

import stand_words as sw
import report_body as rb
from stand_gpu import git_info

FOOD, EN = rb.Q['zs1'], rb.Q['en']
MODES = ['c4', 'c2', 'c8', 'c4_shcur', 'c4_shex', 'c4x', 'c4_en']


def pools(tr):
    """Кандидаты в примеры: информированные такты сетки запросов (эпизод, такт, метка тела, последний исход)."""
    P = [(e, t) for e in range(tr['ga'].shape[0]) for t in rb.QT if tr['inf'][e, t]]
    return np.array(P)


def pick(P, tr, e, t, k, by, rng):
    """k примеров из других эпизодов, сбалансированных по метке (by = 'bel' или 'lf'), эпизоды различны."""
    lab = tr['bel'][P[:, 0], P[:, 1]] if by == 'bel' else (tr['lf'][P[:, 0], P[:, 1]] == 1)
    chosen, used = [], {e}
    for want in [True] * (k // 2) + [False] * (k // 2):
        idx = np.nonzero((lab == want) & ~np.isin(P[:, 0], list(used)))[0]
        j = idx[rng.randint(len(idx))]
        chosen.append((P[j, 0], P[j, 1], bool(want))); used.add(P[j, 0])
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


def build(tr, qs, P, mode, set_id):
    k = int(mode[1]); by = 'lf' if mode.endswith('_en') else 'bel'
    q = EN if by == 'lf' else ([6951] if mode == 'c4x' else FOOD)
    ans = (rb.UP, rb.DOWN) if by == 'lf' else ((rb.AX, rb.BX) if mode == 'c4x' else (rb.A, rb.B))
    ids = []
    for n, (e, t) in enumerate(qs):
        rng = np.random.RandomState(set_id * 1_000_003 + n * 7 + (1 if by == 'lf' else 0))   # одни примеры для всех режимов по знаку
        ex = pick(P, tr, e, t, k, by, rng)
        rsh = np.random.RandomState(set_id * 1_000_003 + n * 7 + 99)
        seq = [sw.BOS]
        for (e2, t2, lab) in ex:
            seq += rb.window(tr, e2, t2, rsh if mode == 'c4_shex' else None) + q + [ans[0] if lab else ans[1]]
        seq += rb.window(tr, e, t, rsh if mode == 'c4_shcur' else None) + q
        ids.append(seq)
    return torch.tensor(ids), ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='EleutherAI/pythia-160m')
    ap.add_argument('--weights', choices=['pretrained', 'random'], default='pretrained')
    ap.add_argument('--sets', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--modes', nargs='+', default=MODES)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--out', default='results_report')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    dev = a.device; commit, dirty = git_info(); t0 = time.time()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.model.split('-')[-1]}_{a.weights}"
    path = os.path.join(a.out, f'icl_{tag}.json')
    res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'model': a.model, 'weights': a.weights, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'sets': a.sets})
    done = {x['set'] for x in res['results']}
    m = None
    for i in a.sets:
        if i in done:
            continue
        if m is None or a.weights == 'random':
            del m; torch.cuda.empty_cache() if dev == 'cuda' else None
            m = rb.load_lm(a.model, a.weights, i, dev)
        tr = rb.trajectories([800000 + 1000 * i + e for e in range(a.episodes)])
        qs = rb.queries(tr); P = pools(tr)
        e_, t_ = qs[:, 0], qs[:, 1]
        y = tr['ga'][e_, t_]; ylast = tr['lf'][e_, t_] == 1
        # контроли конструкции: последний ответ примера, большинство (по примерам режима c4)
        last, maj = [], []
        for n, (e, t) in enumerate(qs):
            ex = pick(P, tr, e, t, 4, 'bel', np.random.RandomState(i * 1_000_003 + n * 7))
            last.append(1.0 if ex[-1][2] else -1.0); maj.append(np.mean([1.0 if x[2] else -1.0 for x in ex]))
        r = {'set': i, 'n_queries': int(len(qs)), 'auc_lastanswer': rb.auc(np.array(last), y), 'auc_majority': rb.auc(np.array(maj), y),
             'auc_copy': rb.auc(np.where(tr['lk'][e_, t_] == 0, 1.0, np.where(tr['lk'][e_, t_] == 1, -1.0, 0.0)), y), 'modes': {}}
        for mode in a.modes:
            ids, ans = build(tr, qs, P, mode, i)
            ch = max(8, a.chunk * 335 // max(ids.shape[1], 1))
            H, Dd = rb.run_lm(m, ids, ans, dev, ch)
            yy = ylast if mode.endswith('_en') else y
            mr = {'auc': rb.auc(Dd, yy), 'acc': float(((Dd > 0) == yy).mean()), 'len': int(ids.shape[1])}
            if mode == 'c4':
                mr['probe_auc'] = rb.probe_auc(H, y, e_, dev)
            r['modes'][mode] = mr
        r['git_commit'] = commit
        res['results'].append(r)
        json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f"{time.strftime('%H:%M:%S')} {tag} набор {i}: запросов {len(qs)} | контроли: последний {r['auc_lastanswer']:.3f}, большинство {r['auc_majority']:.3f} | "
              + ' '.join(f"{k} {v['auc']:.3f}" + (f"/зонд {v['probe_auc']:.3f}" if 'probe_auc' in v else '') for k, v in r['modes'].items())
              + f' | {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
