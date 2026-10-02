"""Быстрый зонд (ДОБАВЛЕНО 2026-10-02): та же логистическая регрессия с L2 и тот же протокол, что
probe_untrained.probe — внешние 5 фолдов по эпизодам, λ по внутреннему отложенному 20 % эпизодов,
стандартизация только по обучающей части, сбалансированная точность, — но решатель другой.

Зачем. В probe_untrained все фолды и все λ решались одним пакетным Ньютоном через
torch.linalg.solve на пачке гессианов (F·L штук). При d = 1024–2048 это медленный пакетный путь
(предупреждения MAGMA «batched routines are designed for small sizes» — 15 тыс. за прогон шкалы),
и шкала 410M заняла ~7 ч вместо ~1 ч по оценке. Здесь: по фолду, гессиан положительно определён
→ разложение Холецкого одной матрицы; λ проходятся от большего к меньшему с прогревом от
предыдущего решения (обычно 2–4 шага Ньютона на λ вместо 20).

Использовать только после сверки с probe_untrained.probe на 70M (совпадение до 3-го знака) —
см. check_probe_fast() и docs/RESULTS_2026-10-01_flip.md.

ИСПРАВЛЕНО 2026-10-02 (первая сверка: макс. |Δ| 0,192 — только на lm_last предобученной).
Переобучение на всей обучающей части прогревалось решением внутреннего разбиения, стандартизованным
по другой статистике; при λ = 10⁻³ Ньютон без шага с проверкой от такой точки разлетался (|w| ~10⁵,
точность 0,500 в 3 фолдах из 5). Теперь: переобучение с нуля и шаг с откатом (половиним, пока
потеря не перестанет расти). Сам Ньютон от нуля совпадал с sklearn до 5-го знака при всех λ.
"""
import numpy as np
import torch

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _bacc(pred, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return np.nan
    return 0.5 * ((pred[y]).mean() + (~pred[~y]).mean())


def _std(X, mask):
    Xm = X[mask]
    mu = Xm.mean(0); sd = Xm.std(0, unbiased=False) + 1e-6
    return (X - mu) / sd


def _newton(X1, y, lam, w0, iters=30, tol=1e-5):
    """X1 (n, d+1) с единичным столбцом, y (n,), λ — число; потеря: среднее BCE + λ/2·||w без смещения||²."""
    n, d1 = X1.shape
    reg = torch.ones(d1, device=X1.device); reg[-1] = 0
    w = w0.clone()
    def loss(v):
        return (torch.nn.functional.binary_cross_entropy_with_logits(X1 @ v, y) + lam / 2 * ((v * reg) ** 2).sum()).item()
    cur = loss(w)
    for _ in range(iters):
        p = torch.sigmoid(X1 @ w)
        g = X1.T @ (p - y) / n + lam * reg * w
        H = (X1 * (p * (1 - p) / n)[:, None]).T @ X1
        H.diagonal().add_(lam * reg + 1e-6)
        L, info = torch.linalg.cholesky_ex(H)
        if info.item() != 0:                       # не положительно определена — подстраховка
            step = torch.linalg.solve(H, g)
        else:
            step = torch.cholesky_solve(g[:, None], L)[:, 0]
        t = 1.0
        while True:                                 # шаг с откатом: потеря не должна расти
            w_new = w - t * step
            new = loss(w_new)
            if new <= cur + 1e-7 or t < 1e-4:
                break
            t /= 2
        w, cur = w_new, new
        if (t * step).abs().max().item() < tol:
            break
    return w


def probe(X, y, groups, device, folds=5):
    ug = np.unique(groups); r = np.random.RandomState(0); r.shuffle(ug)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    ones = torch.ones(len(y), 1, device=device)
    pred = np.zeros(len(y), bool)
    for f in range(folds):
        te = np.isin(groups, ug[f::folds]); tr = ~te
        trg = np.unique(groups[tr]); r2 = np.random.RandomState(f); r2.shuffle(trg)
        iv = np.isin(groups, trg[:max(1, len(trg) // 5)]) & tr; it = tr & ~iv
        it_t, tr_t = torch.as_tensor(it, device=device), torch.as_tensor(tr, device=device)
        # 1) путь по λ от большего к меньшему на внутренней обучающей части
        X1 = torch.cat([_std(Xt, it_t), ones], 1)
        w = torch.zeros(X1.shape[1], device=device); accs = {}; ws = {}
        for lam in sorted(LAMBDAS, reverse=True):
            w = _newton(X1[it_t], yt[it_t], lam, w)
            ws[lam] = w
            accs[lam] = _bacc((X1[torch.as_tensor(iv, device=device)] @ w).cpu().numpy() > 0, y[iv])
        best = max(LAMBDAS, key=lambda l: (-np.inf if np.isnan(accs[l]) else accs[l], -LAMBDAS.index(l)))
        # при равенстве точности — меньший λ, как np.nanargmax по возрастающей сетке в probe_untrained
        # 2) переобучение на всей обучающей части
        X1 = torch.cat([_std(Xt, tr_t), ones], 1)
        w = _newton(X1[tr_t], yt[tr_t], best, torch.zeros(X1.shape[1], device=device))
        pred[te] = (X1[torch.as_tensor(te, device=device)] @ w).cpu().numpy() > 0
    return pred, _bacc(pred, y)


def check_probe_fast(n_ep=200, seeds=(1, 2, 3), device='cuda'):
    """Сверка с probe_untrained.probe на признаках Pythia-70M (П3-протокол, flip-2 с разворотом)."""
    import math, time
    import probe_untrained as pu
    from env_flip import S_DIM
    D = pu.collect('flip-2', True, n_ep)
    idx = np.where(D['valid']); g = idx[0]; y = D['sign'][idx].astype(float)
    lm = pu.LMFeat('pretrained', 0, device)
    rows = []
    for s in seeds:
        r = np.random.RandomState(s)
        Wp = torch.as_tensor(r.randn(S_DIM, 2 * lm.D).astype(np.float32) / math.sqrt(S_DIM), device=device)
        P = (Wp, torch.zeros(2 * lm.D, device=device))
        for N in (2, 8, 32):
            W, L = pu.windows(D['S'], idx, N)
            hl, hm = lm(W, L, P)
            for nm, X in (('raw_window', W.reshape(len(g), -1)), ('mlp', pu.feat_mlp(W, s)), ('lm_last', hl), ('lm_mid', hm)):
                t = time.time(); _, a_old = pu.probe(X, y, g, device); t_old = time.time() - t
                t = time.time(); _, a_new = probe(X, y, g, device); t_new = time.time() - t
                rows.append((s, N, nm, a_old, a_new, t_old, t_new))
                print(f's{s} N={N:2d} {nm:10s} старый {a_old:.4f} ({t_old:.1f} с) | новый {a_new:.4f} ({t_new:.1f} с) | Δ {a_new - a_old:+.4f}', flush=True)
    d = np.array([abs(r[4] - r[3]) for r in rows])
    print(f'макс. |Δ| = {d.max():.4f}, средн. |Δ| = {d.mean():.4f}; время: старый {sum(r[5] for r in rows):.0f} с, новый {sum(r[6] for r in rows):.0f} с')
    return rows


if __name__ == '__main__':
    check_probe_fast()
