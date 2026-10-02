"""Контроль смысла слов (A) и округлённый проектор (C), протокол П3 без обучения.
Пререгистрация — docs/RESULTS_2026-10-01_flip.md, раздел «A–C».

Зачем. П. 4: с событиями такта настоящими словами + BOS предобученная Pythia-160M впервые лучше
случайной (0,950 против 0,813, Холм 0,004). Не установлено, что даёт преимущество: аппарат
трансформера на токенах из распределения (связывание, копирование, давность) или смысл слов
(' up'/' down', ' A'/' B').

Условия A (окно N = 2/8/32, BOS в начале, 2 токена на такт: вид, изменение энергии):
  c       — ' A'/' B'/' none', ' up'/' down'/' same' (повтор п. 4);
  c_arb1  — ' cat'/' table'/' stone', ' river'/' chair'/' door' — без связи с едой, энергией, буквами;
  c_arb2  — ' lamp'/' cloud'/' brick', ' pencil'/' garden'/' window';
  c_swap  — как c, но ' down' при росте энергии и ' up' при падении;
  e1      — предобученный трансформер, эмбеддинги шести токенов c заменены случайными той же нормы
            (каждый своей), BOS — свой; только для предобученной;
  e2      — случайный трансформер с предобученными эмбеддингами шести токенов c; только для случайной.
Условия C (только 160M, N = 32):
  b       — случайный проектор (2 псевдотокена на такт) + BOS (повтор п. 4, для парного сравнения);
  a_q     — тот же проектор, каждый выход заменён эмбеддингом ближайшего по косинусу токена словаря
            своей модели (у случайной — по её матрице), + BOS.
Случайная LM — те же номера токенов, своя случайная матрица, приведённая к средней норме
эмбеддингов предобученной; сеть от посева. Посевы — 10 наборов по 200 эпизодов (как в п. 4).
Цель — знак эпизода. Зонд — probe_fast.

python probe_words.py --model EleutherAI/pythia-160m [--first 1 --last 10] [--with-c]
"""
import argparse, json, math, os, time
import numpy as np
import torch

import probe_untrained as pu
from probe_fast import probe, _bacc
from env_flip import S_DIM
from stand_gpu import git_info

NS = (2, 8, 32)
K_MAX = 10
BOS = 0
WORDS = {   # (A, B, none), (up, down, same) — номера токенов GPT-NeoX, однотокенность проверена
    'c':      ((329, 378, 5293), (598, 1066, 1072)),       # ' A' ' B' ' none' | ' up' ' down' ' same'
    'c_arb1': ((5798, 2829, 8805), (8281, 6951, 3369)),    # ' cat' ' table' ' stone' | ' river' ' chair' ' door'
    'c_arb2': ((18067, 9005, 20098), (29514, 10329, 3497)),  # ' lamp' ' cloud' ' brick' | ' pencil' ' garden' ' window'
    'c_swap': ((329, 378, 5293), (1066, 598, 1072)),       # рост -> ' down', падение -> ' up'
}
SIX = (329, 378, 5293, 598, 1066, 1072)


class LM:
    def __init__(self, model, weights, net_seed, device, ref_norm=None):
        from transformers import AutoModel, AutoConfig
        if weights == 'pretrained':
            self.lm = AutoModel.from_pretrained(model, dtype=torch.float32)
        else:
            torch.manual_seed(net_seed)
            self.lm = AutoModel.from_config(AutoConfig.from_pretrained(model)).to(torch.float32)
            with torch.no_grad():
                E = self.lm.get_input_embeddings().weight
                E.mul_(ref_norm / E.norm(dim=1).mean())
        self.lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.device = device
        self.E0 = self.lm.get_input_embeddings().weight.detach().clone()   # исходная матрица
        self.D = self.lm.config.hidden_size
        self.mid = self.lm.config.num_hidden_layers // 2
        self.tok_norm = self.E0.norm(dim=1).mean().item()
        self.chunk = max(8, int(0.25e9 / (4 * 72 * self.D * (self.lm.config.num_hidden_layers + 1))))
        self.En = None

    def seq(self, W, cond, E, P):
        m, l = W.shape[0], W.shape[1]
        x = torch.as_tensor(W, device=self.device)
        if cond in ('b', 'a_q'):
            Wp, bp = P
            v = (x @ Wp + bp).view(m, l, 2, self.D)
            v = (v / (v.norm(dim=3, keepdim=True) + 1e-6) * self.tok_norm).reshape(m, 2 * l, self.D)
            if cond == 'a_q':                    # ближайший токен словаря по косинусу
                if self.En is None:
                    self.En = self.E0 / self.E0.norm(dim=1, keepdim=True)
                vn = v / v.norm(dim=2, keepdim=True)
                idx = torch.empty(m, 2 * l, dtype=torch.long, device=self.device)
                for i in range(0, m, 16):        # 16 × 65 × 50 тыс. косинусов ≈ 0,2 ГБ
                    idx[i:i + 16] = (vn[i:i + 16] @ self.En.T).argmax(2)
                v = self.E0[idx]
                self.last_snap = idx
        else:
            words = WORDS['c'] if cond in ('e1', 'e2') else WORDS[cond]
            (tA, tB, tN), (tU, tD, tS) = words
            ate = torch.full((m, l), tN, device=self.device, dtype=torch.long)
            ate[x[:, :, 10] > 0] = tA; ate[x[:, :, 11] > 0] = tB
            en = torch.full((m, l), tS, device=self.device, dtype=torch.long)
            en[x[:, :, 7] > 0.5] = tU; en[x[:, :, 7] < -0.5] = tD
            v = E[torch.stack([ate, en], 2).reshape(m, 2 * l)]
        return torch.cat([E[BOS].expand(m, 1, self.D), v], 1)

    def __call__(self, W, L, cond, E, P=None):
        n, N, _ = W.shape
        last = np.zeros((n, self.D), np.float32); mid = np.zeros((n, self.D), np.float32)
        for l in np.unique(L):
            sel = np.where(L == l)[0]
            for i in range(0, len(sel), self.chunk):
                s = sel[i:i + self.chunk]
                with torch.no_grad():
                    o = self.lm(inputs_embeds=self.seq(W[s, N - l:], cond, E, P), output_hidden_states=True)
                hl, hm = o.last_hidden_state[:, -1], o.hidden_states[self.mid][:, -1]
                if not (torch.isfinite(hl).all() and torch.isfinite(hm).all()):
                    raise FloatingPointError('нечисловое скрытое состояние')
                last[s] = hl.cpu().numpy(); mid[s] = hm.cpu().numpy()
                del o
        return last, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--first', type=int, default=1)
    ap.add_argument('--last', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--with-c', action='store_true', help='условия C (b, a_q) при N = 32')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_words')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    short = a.model.split('/')[-1]
    path = os.path.join(a.out, f'probe_{short}.json')
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    res = json.load(open(path)) if os.path.exists(path) else {'model': a.model, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'first': a.first, 'last': a.last,
                        'with_c': a.with_c, 'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    done = {(x['seed'], x['weights'], x['cond'], x['layer'], x['N']) for x in res['results']}
    pre = LM(a.model, 'pretrained', 0, a.device)
    t0 = time.time()
    for seed in range(a.first, a.last + 1):
        D_ = pu.collect('flip-2', True, a.episodes, seed0=31000 + 1000 * seed)
        idx = np.where(D_['valid']); groups, k = idx[0], D_['k'][idx]
        y = D_['sign'][idx].astype(float)
        Wc = {N: pu.windows(D_['S'], idx, N) for N in NS}
        r = np.random.RandomState(seed)
        Wp = torch.as_tensor(r.randn(S_DIM, 2 * pre.D).astype(np.float32) / math.sqrt(S_DIM), device=a.device)
        P = (Wp, torch.zeros(2 * pre.D, device=a.device))
        rnd = LM(a.model, 'random', seed, a.device, ref_norm=pre.tok_norm)
        # e1: предобученная матрица, шесть токенов — случайные векторы той же нормы (посев s)
        g = torch.Generator(device='cpu').manual_seed(5000 + seed)
        E1 = pre.E0.clone()
        for t in SIX:
            v = torch.randn(pre.D, generator=g).to(a.device)
            E1[t] = v / v.norm() * pre.E0[t].norm()
        # e2: матрица случайной LM, шесть токенов — предобученные эмбеддинги
        E2 = rnd.E0.clone()
        E2[list(SIX)] = pre.E0[list(SIX)]
        plan = [('pretrained', pre, c, pre.E0) for c in ('c', 'c_arb1', 'c_arb2', 'c_swap')] + [('pretrained', pre, 'e1', E1)] + \
               [('random', rnd, c, rnd.E0) for c in ('c', 'c_arb1', 'c_arb2', 'c_swap')] + [('random', rnd, 'e2', E2)]
        snap = {}
        for wname, lm, cond, E in plan + ([(w, l_, c, l_.E0) for w, l_ in (('pretrained', pre), ('random', rnd)) for c in ('b', 'a_q')] if a.with_c else []):
            for N in (NS if cond not in ('b', 'a_q') else (32,)):
                if all((seed, wname, cond, ly, N) in done for ly in ('last', 'mid')):
                    continue
                W, L = Wc[N]
                hl, hm = lm(W, L, cond, E, P)
                if cond == 'a_q':        # какие токены выбрал округлённый проектор (для отчёта)
                    ids = lm.last_snap.flatten().cpu().numpy()
                    u, c_ = np.unique(ids, return_counts=True)
                    snap[wname] = {'n_unique': int(len(u)), 'top': [[int(i), int(n)] for i, n in sorted(zip(u, c_), key=lambda z: -z[1])[:10]]}
                for ly, X in (('last', hl), ('mid', hm)):
                    pred, acc = probe(X, y, groups, a.device)
                    curve = [_bacc(pred[k == kk], y[k == kk]) for kk in range(K_MAX + 1)]
                    res['results'].append({'seed': seed, 'weights': wname, 'cond': cond, 'layer': ly, 'N': N,
                                           'bacc': acc, 'curve_k': curve, 'git_commit': commit,
                                           'snap': snap.get(wname) if cond == 'a_q' else None})
                    done.add((seed, wname, cond, ly, N))
        del rnd; torch.cuda.empty_cache() if a.device == 'cuda' else None
        json.dump(res, open(path, 'w'), ensure_ascii=False)
        print(f'{time.strftime("%H:%M:%S")} {short} посев {seed}: {time.time() - t0:.0f} с', flush=True)
    json.dump(res, open(path, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
