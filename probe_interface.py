"""Положительный контроль интерфейса (п. 4 программы 2026-10-02), протокол П3 без обучения.

Вопрос: не в интерфейсе ли причина «предобученная ≤ случайная» (шкала: у 160M отставание
−0,053, p Холма 0,02). Случайный проектор кладёт кадры вне распределения эмбеддингов; без BOS роль
стока внимания (Sun et al. 2024, arXiv 2402.17762), вероятно, достаётся первому псевдотокену.

Условия (окно N тактов, скрытое состояние последней позиции, последний и средний слой):
  a — исходный интерфейс: случайный проектор, 2 псевдотокена на такт (повтор для сверки);
  b — тот же интерфейс + настоящий BOS-токен (<|endoftext|>, id 0) в начале окна;
  c — события такта настоящими однотокенными словами, без проектора, + BOS:
      съеденный вид (' A' / ' B' / ' none'), изменение энергии (' up' / ' down' / ' same');
      порог «up/down» — |тренд энергии| > 0,5 (дрейф даёт −0,12, пища +3,4, яд −2,1);
  d — случайная LM в тех же условиях: те же номера токенов, своя случайная матрица эмбеддингов,
      приведённая к средней норме эмбеддингов предобученной модели (сеть — своя на посев).
Посевы: 10 разных наборов эпизодов (посевы 31000 + 1000·s + k, одинаковые для всех условий);
в a/b ещё и проектор от посева; у случайной LM — своя сеть от посева.
Цель зонда — знак эпизода (в c нет признаков навигации, цель «ближайший съедобен» неприменима).
Зонд — probe_fast (сверен с probe_untrained на 70M, см. RESULTS).

python probe_interface.py --model EleutherAI/pythia-160m [--first 1 --last 10] [--envs flip-2_rev]
"""
import argparse, json, math, os, time
import numpy as np
import torch

import probe_untrained as pu
from probe_fast import probe, _bacc
from env_flip import S_DIM
from stand_gpu import git_info

ENVS = {'flip-2_rev': ('flip-2', True), 'flip-1_const': ('flip-1', False)}
NS = (2, 8, 32)
TOK = {'A': 329, 'B': 378, 'none': 5293, 'up': 598, 'down': 1066, 'same': 1072, 'bos': 0}   # GPT-NeoX, проверено
K_MAX = 10


class LM:
    def __init__(self, model, weights, net_seed, device, ref_norm=None):
        from transformers import AutoModel, AutoConfig
        if weights == 'pretrained':
            self.lm = AutoModel.from_pretrained(model, dtype=torch.float32)
        else:
            torch.manual_seed(net_seed)
            self.lm = AutoModel.from_config(AutoConfig.from_pretrained(model)).to(torch.float32)
            E = self.lm.get_input_embeddings().weight
            with torch.no_grad():                       # к средней норме эмбеддингов предобученной
                E.mul_(ref_norm / E.norm(dim=1).mean())
        self.lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.device = device
        self.E = self.lm.get_input_embeddings().weight
        self.D = self.lm.config.hidden_size
        self.mid = self.lm.config.num_hidden_layers // 2
        self.tok_norm = self.E.norm(dim=1).mean().item()
        self.chunk = max(8, int(0.25e9 / (4 * 72 * self.D * (self.lm.config.num_hidden_layers + 1))))

    def seq(self, W, L, cond, P):
        """Последовательности эмбеддингов (m, позиций, D) для окон одной длины l."""
        m, l = W.shape[0], W.shape[1]
        x = torch.as_tensor(W, device=self.device)
        if cond in ('a', 'b'):
            Wp, bp = P
            v = (x @ Wp + bp).view(m, l, 2, self.D)
            v = (v / (v.norm(dim=3, keepdim=True) + 1e-6) * self.tok_norm).reshape(m, 2 * l, self.D)
        else:
            ate = torch.full((m, l), TOK['none'], device=self.device, dtype=torch.long)
            ate[x[:, :, 10] > 0] = TOK['A']; ate[x[:, :, 11] > 0] = TOK['B']
            en = torch.full((m, l), TOK['same'], device=self.device, dtype=torch.long)
            en[x[:, :, 7] > 0.5] = TOK['up']; en[x[:, :, 7] < -0.5] = TOK['down']
            v = self.E[torch.stack([ate, en], 2).reshape(m, 2 * l)]
        if cond in ('b', 'c'):
            v = torch.cat([self.E[TOK['bos']].expand(m, 1, self.D), v], 1)
        return v

    def __call__(self, W, L, cond, P=None):
        n, N, _ = W.shape
        last = np.zeros((n, self.D), np.float32); mid = np.zeros((n, self.D), np.float32)
        for l in np.unique(L):
            sel = np.where(L == l)[0]
            for i in range(0, len(sel), self.chunk):
                s = sel[i:i + self.chunk]
                with torch.no_grad():
                    o = self.lm(inputs_embeds=self.seq(W[s, N - l:], L[s], cond, P), output_hidden_states=True)
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
    ap.add_argument('--envs', default='flip-2_rev')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_interface')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    short = a.model.split('/')[-1]
    path = os.path.join(a.out, f'probe_{short}.json')
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    res = json.load(open(path)) if os.path.exists(path) else {'model': a.model, 'results': [], 'runs': []}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'first': a.first, 'last': a.last,
                        'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    done = {(x['env'], x['seed'], x['weights'], x['cond'], x['layer'], x['N']) for x in res['results']}
    t0 = time.time()
    pre = LM(a.model, 'pretrained', 0, a.device)
    ref_norm = pre.tok_norm
    for ename in a.envs.split(','):
        env, rev = ENVS[ename]
        for seed in range(a.first, a.last + 1):
            if all((ename, seed, w, c, ly, N) in done for w in ('pretrained', 'random') for c in 'abc'
                   for ly in ('last', 'mid') for N in NS):
                continue
            D_ = pu.collect(env, rev, a.episodes, seed0=31000 + 1000 * seed)
            idx = np.where(D_['valid']); groups, k = idx[0], D_['k'][idx]
            y = D_['sign'][idx]
            Wc = {N: pu.windows(D_['S'], idx, N) for N in NS}
            r = np.random.RandomState(seed)
            Wp = torch.as_tensor(r.randn(S_DIM, 2 * pre.D).astype(np.float32) / math.sqrt(S_DIM), device=a.device)
            P = (Wp, torch.zeros(2 * pre.D, device=a.device))
            meta = {'eatA': D_['eatA'], 'eatB': D_['eatB'], 'n': int(len(groups)), 'sign_mean': float(y.mean())}
            rnd = LM(a.model, 'random', seed, a.device, ref_norm=ref_norm)
            for wname, lm in (('pretrained', pre), ('random', rnd)):
                for cond in 'abc':
                    for N in NS:
                        if all((ename, seed, wname, cond, ly, N) in done for ly in ('last', 'mid')):
                            continue
                        W, L = Wc[N]
                        hl, hm = lm(W, L, cond, P)
                        for ly, X in (('last', hl), ('mid', hm)):
                            pred, acc = probe(X, y.astype(float), groups, a.device)
                            curve = [_bacc(pred[k == kk], y[k == kk]) for kk in range(K_MAX + 1)]
                            res['results'].append({'env': ename, 'seed': seed, 'weights': wname, 'cond': cond,
                                                   'layer': ly, 'N': N, 'bacc': acc, 'curve_k': curve,
                                                   'meta': meta, 'git_commit': commit})
                            done.add((ename, seed, wname, cond, ly, N))
            for N in NS:   # справочные: явная билинейная статистика и «съедено A − B»
                W = Wc[N][0]
                for nm, X in (('bilinear', pu.feat_bilinear(W)), ('count', (W[:, :, 10] - W[:, :, 11]).sum(1, keepdims=True))):
                    if (ename, seed, nm, '-', '-', N) in done:
                        continue
                    done.add((ename, seed, nm, '-', '-', N))
                    _, acc = probe(X, y.astype(float), groups, a.device)
                    res['results'].append({'env': ename, 'seed': seed, 'weights': nm, 'cond': '-', 'layer': '-',
                                           'N': N, 'bacc': acc, 'curve_k': None, 'meta': meta, 'git_commit': commit})
            del rnd; torch.cuda.empty_cache() if a.device == 'cuda' else None
            json.dump(res, open(path, 'w'), ensure_ascii=False)
            print(f'{time.strftime("%H:%M:%S")} {short} {ename} посев {seed}: {time.time() - t0:.0f} с', flush=True)
    json.dump(res, open(path, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
