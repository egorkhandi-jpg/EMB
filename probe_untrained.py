"""П3 без обучения: читается ли знак эпизода линейно из скрытого состояния необученного
конвейера (случайный проектор -> замороженная LM), и лучше ли у предобученной LM, чем у
случайной, и у случайной LM, чем у случайных нелинейных признаков окна.
Пререгистрация — docs/RESULTS_2026-10-01_flip.md, раздел «П3 без обучения».

Зачем без обучения. Пилоты flip-1/flip-2: ES (16 пар × 4 эпизода × 40 поколений) не находит
связку «вид × исход» ни в одном из пяти вариантов среды. Здесь звено 2 проверяется на уровне
представления: если знания модели дают линейно читаемую связку, хватило бы линейной головы.

Почему зонд на сырых признаках должен провалиться. Знак выводится из совпадения в одном
кадре: «съел вид X» × «энергия выросла/упала» — билинейная величина (XOR по знакам), и
линейное чтение сырых признаков окна её не достаёт ни при каких весах. Тот же довод
объясняет провал linwin в пилотах.

Ловушка. Тела ведёт политика, слепая к знаку: к ближайшему предмету любого вида + шум курса
(ст. откл. 0,3 рад/такт). Эвристика с памятью шла бы к пище, и знак утёк бы в текущий кадр
(направление/дистанция к виду-пище) — зонд читал бы его без всякой памяти. Проверка утечки —
зонд по признакам текущего кадра (ожидается 0,5).

python probe_untrained.py [--seeds 10] [--episodes 200] [--out results_probe]
"""
import argparse, json, math, os, time
import numpy as np
import torch

import env_flip as ef
from env_flip import FlipBodies, S_DIM, T
from stand_gpu import git_info

MODEL = 'EleutherAI/pythia-70m'
NS = (2, 8, 32)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
K_MAX = 10


# ---------------- данные ----------------
def collect(env, reversal, n_ep, seed0=31000):
    """Траектории под политикой, слепой к знаку. Возвращает кадры и разметку по тактам."""
    b = FlipBodies(n_ep, [seed0 + k for k in range(n_ep)], reversal, env)
    rng = np.random.RandomState(7)
    S = np.zeros((n_ep, T, S_DIM), np.float32)
    sign = np.zeros((n_ep, T), bool); edible = np.zeros((n_ep, T), bool); alive = np.zeros((n_ep, T), bool)
    for t in range(T):
        s = b.sense()
        S[:, t] = s
        sign[:, t] = b.good_A
        nearest_A = s[:, 2] <= s[:, 5]                     # дистанции до ближайшего A и B
        edible[:, t] = nearest_A == b.good_A
        alive[:, t] = b.alive
        a = ef.greedy_any(s, None, b)
        a[:, 0] = np.clip(a[:, 0] + rng.randn(n_ep) * 0.3 / ef.TURN, -1, 1)
        a[~b.alive] = 0
        b.step(a)
    ate = S[:, :, 10:12].sum(2) > 0                        # кадр-событие: съел в предыдущем такте
    # Знак в кадре t определим, если после последнего разворота (кадры >= t_rev+1) был кадр-событие.
    valid = np.zeros((n_ep, T), bool); k_since = np.full((n_ep, T), 99)
    for i in range(n_ep):
        last = -1
        for t in range(T):
            if ate[i, t]:
                last = t
            start = b.t_rev[i] + 1 if (b.t_rev[i] >= 0 and t >= b.t_rev[i] + 1) else 0
            valid[i, t] = alive[i, t] and last >= start
            k_since[i, t] = t - last if last >= 0 else 99
    eatA = (S[:, :, 10] > 0).sum(); eatB = (S[:, :, 11] > 0).sum()
    return dict(S=S, sign=sign, edible=edible, valid=valid, k=k_since, eatA=int(eatA), eatB=int(eatB),
                frac_valid=float(valid.mean()), alive=float(alive.mean()))


def windows(S, idx, N):
    """Окна сырых кадров (n, N, S_DIM), слева нули (как flywin), и длина min(t+1, N)."""
    ep, t = idx
    W = np.zeros((len(ep), N, S_DIM), np.float32)
    for j in range(N):
        tt = t - (N - 1 - j)
        ok = tt >= 0
        W[ok, j] = S[ep[ok], tt[ok]]
    return W, np.minimum(t + 1, N)


# ---------------- признаки ----------------
def feat_bilinear(W):
    """Явная билинейная статистика по окну: Σ (съел A − съел B) × тренд энергии."""
    return ((W[:, :, 10] - W[:, :, 11]) * W[:, :, 7]).sum(1, keepdims=True)


def feat_mlp(W, seed, width=512):
    r = np.random.RandomState(1000 + seed)
    x = W.reshape(len(W), -1)
    A = r.randn(x.shape[1], width).astype(np.float32) / math.sqrt(x.shape[1])
    c = r.randn(width).astype(np.float32) * 0.1
    return np.tanh(x @ A + c)


class LMFeat:
    def __init__(self, weights, net_seed, device):
        from transformers import AutoModel, AutoConfig
        if weights == 'pretrained':
            self.lm = AutoModel.from_pretrained(MODEL, dtype=torch.float32)
        else:
            torch.manual_seed(net_seed)
            self.lm = AutoModel.from_config(AutoConfig.from_pretrained(MODEL)).to(torch.float32)
        self.lm.to(device).eval()
        self.device = device
        self.D = self.lm.config.hidden_size
        self.mid = self.lm.config.num_hidden_layers // 2
        self.tok_norm = self.lm.get_input_embeddings().weight.float().norm(dim=1).mean().item()

    def __call__(self, W, L, P, chunk=512):
        """W (n, N, S) окна, L — длина без нулевой подкладки, P — проектор (S -> 2D, + смещение).
        Окно короче N подаётся без подкладки (как в stand_flip), группами одной длины."""
        Wp, bp = P
        n, N, _ = W.shape
        last = np.zeros((n, self.D), np.float32); mid = np.zeros((n, self.D), np.float32)
        for l in np.unique(L):
            sel = np.where(L == l)[0]
            x = torch.as_tensor(W[sel, N - l:], device=self.device)                  # (m, l, S)
            v = (x @ Wp + bp).view(len(sel), l, 2, self.D)
            v = v / (v.norm(dim=3, keepdim=True) + 1e-6) * self.tok_norm
            v = v.reshape(len(sel), 2 * l, self.D)
            for i in range(0, len(sel), chunk):
                with torch.no_grad():
                    o = self.lm(inputs_embeds=v[i:i + chunk], output_hidden_states=True)
                hl, hm = o.last_hidden_state[:, -1], o.hidden_states[self.mid][:, -1]
                if not (torch.isfinite(hl).all() and torch.isfinite(hm).all()):
                    raise FloatingPointError('нечисловое скрытое состояние')
                last[sel[i:i + chunk]] = hl.cpu().numpy(); mid[sel[i:i + chunk]] = hm.cpu().numpy()
        return last, mid


# ---------------- зонд ----------------
# Все 5 внешних фолдов и все λ решаются одним вызовом LBFGS: задачи независимы (сумма
# потерь, раздельные веса), а по отдельности мелкие вызовы на GPU тонули в синхронизациях —
# смок на 40 эпизодах шёл 10 мин на одну среду при загрузке видеокарты 13 %.
def _fit_batched(Xs, masks, lams, device, iters=20):
    """Xs (F, n, d) — признаки, стандартизованные по-своему для каждой задачи; masks (F, n) —
    обучающие выборки; lams (F, L) — λ. Логистическая регрессия с L2: потеря задачи — среднее
    по её выборке + λ/2·||w||² (без свободного члена). Метод Ньютона (задача строго выпукла;
    гессиан (d+1)² дёшев при d ≤ 513). ИЗМЕНЕНО 2026-10-01: был пакетный LBFGS — ~40 с на зонд
    при 9 тыс. примеров, ~6,7 ч на весь план. Возвращает веса (F, d+1, L) и X с единицами."""
    F, n, d = Xs.shape
    X1 = torch.cat([Xs, torch.ones(F, n, 1, device=device)], 2)
    m = masks / masks.sum(1, keepdim=True)
    lam = torch.as_tensor(lams, dtype=torch.float32, device=device)          # (F, L)
    L = lam.shape[1]
    reg = torch.ones(d + 1, device=device); reg[-1] = 0
    y = YY[0, :, 0]
    w = torch.zeros(F, d + 1, L, device=device)
    eye = torch.eye(d + 1, device=device)
    for _ in range(iters):
        z = torch.bmm(X1, w)                                                 # (F, n, L)
        pz = torch.sigmoid(z)
        g = torch.bmm(X1.transpose(1, 2), (pz - y[None, :, None]) * m[:, :, None]) + w * reg[None, :, None] * lam[:, None, :]
        step = torch.empty_like(w)
        for l in range(L):
            sw = (pz[:, :, l] * (1 - pz[:, :, l]) * m)                        # (F, n)
            H = torch.bmm(X1.transpose(1, 2), X1 * sw[:, :, None])
            H = H + eye[None] * (reg * lam[:, l:l + 1] + 1e-6)[:, :, None]
            step[:, :, l] = torch.linalg.solve(H, g[:, :, l:l + 1])[:, :, 0]
        w = w - step
        if step.abs().max() < 1e-5:
            break
    return w, X1


def _bacc(pred, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return np.nan
    return 0.5 * ((pred[y]).mean() + (~pred[~y]).mean())


def _std(X, mask):
    mu = (X * mask[:, None]).sum(0) / mask.sum()
    sd = torch.sqrt((((X - mu) ** 2) * mask[:, None]).sum(0) / mask.sum()) + 1e-6
    return (X - mu) / sd


def probe(X, y, groups, device, folds=5):
    """Внешние 5 фолдов по эпизодам; λ — по внутреннему отложенному 20 % эпизодов обучающей
    части; стандартизация по координатам — только по обучающей (внутри — по внутренней
    обучающей) части. Возвращает внефолдовые предсказания и сбалансированную точность."""
    global YY
    ug = np.unique(groups); r = np.random.RandomState(0); r.shuffle(ug)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    YY = torch.as_tensor(y, dtype=torch.float32, device=device)[None, :, None]
    te_m, tr_m, it_m, iv_m = [], [], [], []
    for f in range(folds):
        te = np.isin(groups, ug[f::folds]); tr = ~te
        trg = np.unique(groups[tr]); r2 = np.random.RandomState(f); r2.shuffle(trg)
        iv = np.isin(groups, trg[:max(1, len(trg) // 5)]) & tr
        te_m.append(te); tr_m.append(tr); it_m.append(tr & ~iv); iv_m.append(iv)
    tm = lambda L: torch.as_tensor(np.stack(L), dtype=torch.float32, device=device)
    IT, TR = tm(it_m), tm(tr_m)
    # 1) выбор λ на внутреннем разбиении
    Xs = torch.stack([_std(Xt, IT[f]) for f in range(folds)])
    w, X1 = _fit_batched(Xs, IT, np.tile(LAMBDAS, (folds, 1)), device)
    z = torch.bmm(X1, w).cpu().numpy() > 0                                   # (F, n, L)
    best = []
    for f in range(folds):
        accs = [_bacc(z[f, iv_m[f], l], y[iv_m[f]]) for l in range(len(LAMBDAS))]
        best.append(LAMBDAS[int(np.nanargmax(accs))])
    # 2) переобучение на всей обучающей части с выбранным λ
    Xs = torch.stack([_std(Xt, TR[f]) for f in range(folds)])
    w, X1 = _fit_batched(Xs, TR, np.array(best)[:, None], device)
    z = torch.bmm(X1, w).cpu().numpy()[:, :, 0] > 0
    pred = np.zeros(len(y), bool)
    for f in range(folds):
        pred[te_m[f]] = z[f, te_m[f]]
    return pred, _bacc(pred, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_probe')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    t0 = time.time()
    lm_pre = LMFeat('pretrained', 0, a.device)
    envs = {'flip-2_rev': ('flip-2', True), 'flip-1_const': ('flip-1', False)}
    res = {'git_commit': commit, 'git_dirty': dirty, 'episodes': a.episodes, 'seeds': a.seeds, 'results': []}
    for ename, (env, rev) in envs.items():
        D = collect(env, rev, a.episodes)
        idx = np.where(D['valid'])
        groups, k = idx[0], D['k'][idx]
        targets = {'sign': D['sign'][idx], 'edible': D['edible'][idx]}
        res[ename] = {'eatA': D['eatA'], 'eatB': D['eatB'], 'frac_valid': D['frac_valid'], 'alive': D['alive'],
                      'n_samples': int(len(groups)), 'sign_mean': float(targets['sign'].mean()),
                      'edible_mean': float(targets['edible'].mean())}
        print(ename, res[ename], flush=True)
        cur = D['S'][idx]
        def record(seed, fset, N, X):
            for tn, y in targets.items():
                pred, acc = probe(X, y.astype(float), groups, a.device)
                curve = [_bacc(pred[k == kk], y[k == kk]) for kk in range(K_MAX + 1)]
                res['results'].append({'env': ename, 'seed': seed, 'features': fset, 'N': N, 'target': tn,
                                       'bacc': acc, 'curve_k': curve, 'dim': int(X.shape[1])})
        record(0, 'raw_current', 1, cur)                               # не зависит от посева
        Wcache = {N: windows(D['S'], idx, N) for N in NS}
        for N in NS:
            record(0, 'raw_window', N, Wcache[N][0].reshape(len(groups), -1))
            record(0, 'bilinear', N, feat_bilinear(Wcache[N][0]))
        for seed in range(1, a.seeds + 1):
            lm_rnd = LMFeat('random', seed, a.device)
            r = np.random.RandomState(seed)
            # общий для обеих LM проектор (как в stand_flip.LMWindow.init), K = 2
            Wp = torch.as_tensor(r.randn(S_DIM, 2 * lm_pre.D).astype(np.float32) / math.sqrt(S_DIM), device=a.device)
            P = (Wp, torch.zeros(2 * lm_pre.D, device=a.device))
            for N in NS:
                W, L = Wcache[N]
                record(seed, 'random_mlp', N, feat_mlp(W, seed))
                for nm, lm in (('lm_random', lm_rnd), ('lm_pretrained', lm_pre)):
                    hl, hm = lm(W, L, P)
                    record(seed, nm + '_last', N, hl)
                    record(seed, nm + '_mid', N, hm)
                print(f'{ename} посев {seed} N={N} готов, {time.time() - t0:.0f} с', flush=True)
            del lm_rnd; torch.cuda.empty_cache() if a.device == 'cuda' else None
            json.dump(res, open(os.path.join(a.out, 'probe.json'), 'w'), ensure_ascii=False)
    res['sec'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(a.out, 'probe.json'), 'w'), ensure_ascii=False)
    print('готово за', res['sec'], 'с')


if __name__ == '__main__':
    main()
