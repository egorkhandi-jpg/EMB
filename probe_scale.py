"""П3 без обучения на шкале размеров (шаг 1 программы 2026-10-01).
Протокол — как probe_untrained.py для Pythia-70M (те же среды, траектории, окна, зонд, 10 посевов),
плюс новые контроли той же ширины D, что скрытое состояние модели:
  quad      — случайные признаки второго порядка: (A·x) ⊙ (B·x), x — окно, стандартизованное по
              признакам; проверка «внимание ли это или любая квадратичная форма»;
  mlp_std   — случайный MLP (tanh) на окне, стандартизованном по признакам: в probe_untrained MLP
              при N = 32 работал почти линейно (вход 1/√384), 0,646 против 0,774 при N = 8;
  count     — «съедено A − съедено B за окно»: канал отбора по выживанию (сырое окно при N = 32
              читало знак на 0,556 / 0,696, вероятно, через него); прирост LM мерится над ним.
Стандартизация входа контролей — по всем тактам без меток (не использует цель зонда).

Память: одна LM за раз (1B в fp32 — ~4 ГБ): сначала все посевы на предобученной, затем по
посеву случайная. Признаки зондируются сразу и не хранятся.
Термика: перед каждым посевом температура; выше 85 °C — ждать < 70 °C (правило 5).

python probe_scale.py --model EleutherAI/pythia-160m [--seeds 10] [--episodes 200] [--envs flip-2_rev,flip-1_const]
"""
import argparse, json, math, os, subprocess, time
import numpy as np
import torch

import probe_untrained as pu
from probe_untrained import collect, windows, probe, feat_bilinear, _bacc, NS, K_MAX
from env_flip import S_DIM
from stand_gpu import git_info

ENVS = {'flip-2_rev': ('flip-2', True), 'flip-1_const': ('flip-1', False)}


def temp():
    try:
        return int(subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.strip().splitlines()[0])
    except Exception:
        return None


def cool(log):
    t = temp()
    print(f'{time.strftime("%H:%M:%S")} GPU {t} °C', file=log, flush=True)
    if t is not None and t > 85:
        while (t := temp()) is not None and t >= 70:
            print(f'{time.strftime("%H:%M:%S")} остывание {t} °C', file=log, flush=True); time.sleep(30)


class LMFeat:
    def __init__(self, model, weights, net_seed, device):
        from transformers import AutoModel, AutoConfig
        if weights == 'pretrained':
            self.lm = AutoModel.from_pretrained(model, dtype=torch.float32)
        else:
            torch.manual_seed(net_seed)
            self.lm = AutoModel.from_config(AutoConfig.from_pretrained(model)).to(torch.float32)
        self.lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.device = device
        self.D = self.lm.config.hidden_size
        self.mid = self.lm.config.num_hidden_layers // 2
        self.tok_norm = self.lm.get_input_embeddings().weight.float().norm(dim=1).mean().item()
        # Пакет так, чтобы скрытые состояния всех слоёв на 64 позициях занимали ~0,25 ГБ: замер
        # 2026-10-01 — 1B в fp32 при пакете 64 × 64 позиции занимал 7,2 ГБ из 8, а рядом ещё зонд.
        self.chunk = max(8, int(0.25e9 / (4 * 64 * self.D * (self.lm.config.num_hidden_layers + 1))))

    def __call__(self, W, L, P):
        Wp, bp = P
        n, N, _ = W.shape
        last = np.zeros((n, self.D), np.float32); mid = np.zeros((n, self.D), np.float32)
        for l in np.unique(L):
            sel = np.where(L == l)[0]
            x = torch.as_tensor(W[sel, N - l:], device=self.device)
            v = (x @ Wp + bp).view(len(sel), l, 2, self.D)
            v = (v / (v.norm(dim=3, keepdim=True) + 1e-6) * self.tok_norm).reshape(len(sel), 2 * l, self.D)
            for i in range(0, len(sel), self.chunk):
                with torch.no_grad():
                    o = self.lm(inputs_embeds=v[i:i + self.chunk], output_hidden_states=True)
                hl, hm = o.last_hidden_state[:, -1], o.hidden_states[self.mid][:, -1]
                if not (torch.isfinite(hl).all() and torch.isfinite(hm).all()):
                    raise FloatingPointError('нечисловое скрытое состояние')
                last[sel[i:i + self.chunk]] = hl.cpu().numpy(); mid[sel[i:i + self.chunk]] = hm.cpu().numpy()
                del o
        return last, mid


def zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def feat_quad(W, seed, D):
    r = np.random.RandomState(2000 + seed)
    x = zscore(W.reshape(len(W), -1))
    A = r.randn(x.shape[1], D).astype(np.float32) / math.sqrt(x.shape[1])
    B = r.randn(x.shape[1], D).astype(np.float32) / math.sqrt(x.shape[1])
    return (x @ A) * (x @ B)


def feat_mlp_std(W, seed, D):
    r = np.random.RandomState(3000 + seed)
    x = zscore(W.reshape(len(W), -1))
    A = r.randn(x.shape[1], D).astype(np.float32) / math.sqrt(x.shape[1])
    return np.tanh(x @ A + r.randn(D).astype(np.float32) * 0.1)


def feat_count(W):
    return (W[:, :, 10] - W[:, :, 11]).sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seeds', type=int, default=10, help='последний посев')
    ap.add_argument('--first', type=int, default=1, help='первый посев (для частей по правилу 5)')
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--envs', default='flip-2_rev,flip-1_const')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_scale')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    short = a.model.split('/')[-1]
    path = os.path.join(a.out, f'probe_{short}.json')
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    res = json.load(open(path)) if os.path.exists(path) else {
        'model': a.model, 'git_commit': commit, 'git_dirty': dirty, 'episodes': a.episodes, 'seeds': a.seeds,
        'dtype': 'float32', 'results': []}
    done = {(x['env'], x['seed'], x['features'], x['N'], x['target']) for x in res['results']}
    log = open(os.path.join(a.out, f'log_{short}.txt'), 'a')
    t0 = time.time()

    def save():
        json.dump(res, open(path, 'w'), ensure_ascii=False)

    for ename in a.envs.split(','):
        env, rev = ENVS[ename]
        D_ = collect(env, rev, a.episodes)
        idx = np.where(D_['valid']); groups, k = idx[0], D_['k'][idx]
        targets = {'sign': D_['sign'][idx], 'edible': D_['edible'][idx]}
        res[ename] = {kk: D_[kk] for kk in ('eatA', 'eatB', 'frac_valid', 'alive')} | {'n_samples': int(len(groups))}
        Wc = {N: windows(D_['S'], idx, N) for N in NS}

        def record(seed, fset, N, X):
            for tn, y in targets.items():
                if (ename, seed, fset, N, tn) in done:
                    continue
                pred, acc = probe(X, y.astype(float), groups, a.device)
                curve = [_bacc(pred[k == kk], y[k == kk]) for kk in range(K_MAX + 1)]
                res['results'].append({'env': ename, 'seed': seed, 'features': fset, 'N': N, 'target': tn,
                                       'bacc': acc, 'curve_k': curve, 'dim': int(X.shape[1])})
                done.add((ename, seed, fset, N, tn))

        for N in NS:                                     # не зависят от модели и посева
            record(0, 'count', N, feat_count(Wc[N][0]))
            record(0, 'bilinear', N, feat_bilinear(Wc[N][0]))
        record(0, 'raw_current', 1, D_['S'][idx])
        save()
        for weights in ('pretrained', 'random'):
            lm = LMFeat(a.model, weights, 0, a.device) if weights == 'pretrained' else None
            for seed in range(a.first, a.seeds + 1):
                if all((ename, seed, f'lm_{weights}_{ly}', N, tn) in done for ly in ('last', 'mid') for N in NS for tn in targets):
                    continue
                cool(log)
                if weights == 'random':
                    lm = LMFeat(a.model, 'random', seed, a.device)
                D = lm.D
                r = np.random.RandomState(seed)
                Wp = torch.as_tensor(r.randn(S_DIM, 2 * D).astype(np.float32) / math.sqrt(S_DIM), device=a.device)
                P = (Wp, torch.zeros(2 * D, device=a.device))
                for N in NS:
                    W, L = Wc[N]
                    if weights == 'pretrained':              # контроли ширины D — один раз на посев
                        record(seed, 'quad', N, feat_quad(W, seed, D))
                        record(seed, 'mlp_std', N, feat_mlp_std(W, seed, D))
                        record(seed, 'random_mlp', N, pu.feat_mlp(W, seed, D))
                    hl, hm = lm(W, L, P)
                    record(seed, f'lm_{weights}_last', N, hl)
                    record(seed, f'lm_{weights}_mid', N, hm)
                    del hl, hm
                msg = f'{time.strftime("%H:%M:%S")} {short} {ename} {weights} посев {seed}: {time.time() - t0:.0f} с'
                print(msg, flush=True); print(msg, file=log, flush=True)
                if weights == 'random':
                    del lm; lm = None
                    torch.cuda.empty_cache() if a.device == 'cuda' else None
                save()
            del lm; torch.cuda.empty_cache() if a.device == 'cuda' else None
    res['sec_last_run'] = round(time.time() - t0)
    save()
    print('готово', short, res['sec_last_run'], 'с', flush=True)


if __name__ == '__main__':
    main()
