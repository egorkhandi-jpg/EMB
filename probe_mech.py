"""Механизм преимущества предобученной модели (п. 2 решения Егора 2026-10-02), протокол П3 без обучения.
Pythia-160M, flip-2 с разворотом, те же 10 наборов по 200 эпизодов, N = 32, цель — знак эпизода.

1) Абляция индукционных голов (--part ablation).
   Оценка головы — внимание с позиции повтора на токен, следующий за прошлым вхождением того же
   токена (BOS + 50 случайных токенов, повторённые дважды; 20 последовательностей; eager-внимание).
   Обнуляются выходы верхних k = 2 / 4 / 8 голов по этой оценке (вход проекции dense, срез головы),
   те же индексы — у случайной LM; контроль специфичности — 8 случайных голов вне верхних 8
   (от посева). Условия c (слова, как в п. 4) и e1 (шесть случайных векторов нормы токена).
2) Азбука ощущений (--part alphabet).
   Непрерывные 12 признаков органа (как в условии a) квантуются в собственный алфавит:
   стандартизация по набору → случайная проекция в 8 измерений (от посева) → k-средних (K = 4 / 8 / 16 / 64)
   по всем кадрам набора (без меток) → каждому кластеру свой случайный вектор нормы токена (общий
   для обеих LM в посеве). Вход — BOS + один символ на такт. Вопрос: возвращает ли дискретизация
   ощущений преимущество предобученной без всяких слов.

python probe_mech.py --part ablation|alphabet [--first 1 --last 10]
"""
import argparse, json, math, os, time
import numpy as np
import torch

import probe_untrained as pu
import probe_words as pw
from probe_fast import probe, _bacc
from stand_gpu import git_info

MODEL = 'EleutherAI/pythia-160m'
N = 32
K_MAX = 10


def induction_scores(device, L=50, n=20, seed=0):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation='eager').to(device).eval()
    g = torch.Generator().manual_seed(seed)
    seqs = []
    for _ in range(n):
        r = torch.randperm(20000, generator=g)[:L] + 1000
        seqs.append(torch.cat([torch.tensor([0]), r, r]))
    x = torch.stack(seqs).to(device)
    with torch.no_grad():
        att = m(input_ids=x, output_attentions=True).attentions          # слой: (n, голов, T, T)
    j = torch.arange(L - 1, device=device)
    src, dst = L + 1 + j, 2 + j
    S = torch.stack([a[:, :, src, dst].mean((0, 2)) for a in att])       # (слоёв, голов)
    del m; torch.cuda.empty_cache()
    return S.cpu().numpy()


class Ablator:
    def __init__(self, lm):
        self.lm, self.hooks = lm, []
        self.hd = lm.config.hidden_size // lm.config.num_attention_heads

    def set(self, heads):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        by_layer = {}
        for l, h in heads:
            by_layer.setdefault(l, []).append(h)
        for l, hs in by_layer.items():
            def pre(mod, args, hs=hs):
                x = args[0].clone()
                for h in hs:
                    x[..., h * self.hd:(h + 1) * self.hd] = 0
                return (x,) + tuple(args[1:])
            self.hooks.append(self.lm.layers[l].attention.dense.register_forward_pre_hook(pre))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['ablation', 'alphabet'], required=True)
    ap.add_argument('--first', type=int, default=1)
    ap.add_argument('--last', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_mech')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'{a.part}.json')
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    res = json.load(open(path)) if os.path.exists(path) else {'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'first': a.first, 'last': a.last,
                        'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    done = {(x['seed'], x['weights'], x['cond'], x['variant'], x['layer']) for x in res['results']}
    if a.part == 'ablation' and 'scores' not in res:
        S = induction_scores(a.device)
        order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
        res['scores'] = S.round(4).tolist()
        res['top8'] = [[int(l), int(h)] for l, h in order[:8]]
        print('индукционные головы (слой, голова, оценка):', [(int(l), int(h), round(float(S[l, h]), 3)) for l, h in order[:8]], flush=True)
    pre = pw.LM(MODEL, 'pretrained', 0, a.device)
    t0 = time.time()
    for seed in range(a.first, a.last + 1):
        D_ = pu.collect('flip-2', True, a.episodes, seed0=31000 + 1000 * seed)
        idx = np.where(D_['valid']); groups, k = idx[0], D_['k'][idx]
        y = D_['sign'][idx].astype(float)
        W, L = pu.windows(D_['S'], idx, N)
        rnd = pw.LM(MODEL, 'random', seed, a.device, ref_norm=pre.tok_norm)

        def rec(wname, cond, variant, hl, hm, extra=None):
            for ly, X in (('last', hl), ('mid', hm)):
                if (seed, wname, cond, variant, ly) in done:
                    continue
                pred, acc = probe(X, y, groups, a.device)
                curve = [_bacc(pred[k == kk], y[k == kk]) for kk in range(K_MAX + 1)]
                res['results'].append({'seed': seed, 'weights': wname, 'cond': cond, 'variant': variant, 'layer': ly,
                                       'bacc': acc, 'curve_k': curve, 'git_commit': commit, **(extra or {})})
                done.add((seed, wname, cond, variant, ly))

        if a.part == 'ablation':
            top = [tuple(x) for x in res['top8']]
            rs = np.random.RandomState(8000 + seed)
            pool = [(l, h) for l in range(12) for h in range(12) if (l, h) not in top]
            rand8 = [pool[i] for i in rs.choice(len(pool), 8, replace=False)]
            g = torch.Generator(device='cpu').manual_seed(5000 + seed)          # e1 — как в probe_words
            E1 = pre.E0.clone()
            for t in pw.SIX:
                v = torch.randn(pre.D, generator=g).to(a.device)
                E1[t] = v / v.norm() * pre.E0[t].norm()
            variants = {'k0': [], 'k2': top[:2], 'k4': top[:4], 'k8': top[:8], 'rand8': rand8}
            plan = [('pretrained', pre, 'c', pre.E0, v) for v in variants] + [('pretrained', pre, 'e1', E1, v) for v in variants] + \
                   [('random', rnd, 'c', rnd.E0, v) for v in ('k0', 'k2', 'k4', 'k8', 'rand8')]
            abl = {'pretrained': Ablator(pre.lm), 'random': Ablator(rnd.lm)}
            for wname, lm, cond, E, var in plan:
                if all((seed, wname, cond, var, ly) in done for ly in ('last', 'mid')):
                    continue
                abl[wname].set(variants[var])
                hl, hm = lm(W, L, cond, E)
                rec(wname, cond, var, hl, hm, {'heads': [list(x) for x in variants[var]]})
            for ab in abl.values():
                ab.set([])
        else:
            from sklearn.cluster import KMeans
            S_all = D_['S'].reshape(-1, D_['S'].shape[2])
            mu, sd = S_all.mean(0), S_all.std(0) + 1e-6
            rp = np.random.RandomState(9000 + seed).randn(S_all.shape[1], 8).astype(np.float32) / math.sqrt(S_all.shape[1])
            Z_all = ((S_all - mu) / sd) @ rp
            for K in (4, 8, 16, 64):
                var = f'K{K}'
                if all((seed, w, 'alphabet', var, ly) in done for w in ('pretrained', 'random') for ly in ('last', 'mid')):
                    continue
                km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(Z_all)
                ga = torch.Generator(device='cpu').manual_seed(9500 + seed * 100 + K)
                alph = torch.randn(K, pre.D, generator=ga)
                alph = (alph / alph.norm(dim=1, keepdim=True) * pre.tok_norm).to(a.device)
                codes = km.predict(((W.reshape(-1, W.shape[2]) - mu) / sd) @ rp).reshape(W.shape[0], W.shape[1])
                # кластер «события» — доля кадров-событий, попавших в кластеры с >50 % событий (для отчёта)
                ev = (W[:, :, 10] + W[:, :, 11]).reshape(-1) > 0
                cl = codes.reshape(-1)
                ev_share = {int(c): float(ev[cl == c].mean()) for c in np.unique(cl)}
                for wname, lm in (('pretrained', pre), ('random', rnd)):
                    hl, hm = alphabet_hidden(lm, codes, L, alph)
                    rec(wname, 'alphabet', var, hl, hm, {'event_share_by_cluster': ev_share})
        del rnd; torch.cuda.empty_cache() if a.device == 'cuda' else None
        json.dump(res, open(path, 'w'), ensure_ascii=False)
        print(f'{time.strftime("%H:%M:%S")} {a.part} посев {seed}: {time.time() - t0:.0f} с', flush=True)
    json.dump(res, open(path, 'w'), ensure_ascii=False)


def alphabet_hidden(lm, codes, L, alph):
    """BOS + один символ алфавита на такт; окно без подкладки, группами одной длины."""
    n, Nw = codes.shape
    last = np.zeros((n, lm.D), np.float32); mid = np.zeros((n, lm.D), np.float32)
    for l in np.unique(L):
        sel = np.where(L == l)[0]
        for i in range(0, len(sel), lm.chunk):
            s = sel[i:i + lm.chunk]
            c = torch.as_tensor(codes[s, Nw - l:], device=lm.device)
            v = torch.cat([lm.E0[0].expand(len(s), 1, lm.D), alph[c]], 1)
            with torch.no_grad():
                o = lm.lm(inputs_embeds=v, output_hidden_states=True)
            hl, hm = o.last_hidden_state[:, -1], o.hidden_states[lm.mid][:, -1]
            if not (torch.isfinite(hl).all() and torch.isfinite(hm).all()):
                raise FloatingPointError('нечисловое скрытое состояние')
            last[s] = hl.cpu().numpy(); mid[s] = hm.cpu().numpy()
    return last, mid


if __name__ == '__main__':
    main()
