"""Ось 1, этап 1 (2026-10-06): «спектр памяти агента повторяет спектр временных масштабов мира». Только эвристики, CPU.
Пререгистрация — docs/RESULTS_2026-10-06_spectrum.md. Старые среды не меняются.

Мир (основа flip-2: два вида, знак «какой вид — пища», пища +0,35, яд −0,20; навигация — рефлекс к выбранному виду).
Скрытые переменные (1 — «в пользу A»):
  g ∈ {0, 1} — медленная: переключается (меняет значение) с вероятностью h_g за такт; в варианте «const» постоянна за жизнь;
  m ∈ {0, 1} — средняя: с вероятностью h_m за такт пересэмплируется, P(m = 1) = p_hi при g = 1, p_lo при g = 0;
  s ∈ {0, 1} — быстрая, знак («A — пища»): с вероятностью h_f за такт пересэмплируется, P(s = 1) = p_hi при m = 1, p_lo при m = 0.
Встреча (свидетельство): с вероятностью q за такт агент встречает предметы обоих видов; рефлекс приводит его к выбранному виду,
он съедает его и по исходу (рост / падение энергии) узнаёт знак s точно — какой бы вид ни выбрал. Поток свидетельств поэтому
не зависит от политики: агенты различаются только решениями, миры у всех общие.
Мягкое выживание: смерти нет, энергия — утечка к уровню покоя без границ: E ← E − λ (E − E0) + r, λ = 0,01, E0 = 0,5.
Средняя энергия за жизнь = E0 + Σ_k r_k w(t_k), w(t) = (1 − (1 − λ)^(T − t)) / (λ T) — аффинна по выгодам решений.
Пространственная навигация flip-2 сведена к встрече с вероятностью q и рефлексу к выбранному виду: в flip-2 рефлекс надёжно доводит
до выбранного вида, пространство добавляет только шум и связь частоты встреч с выбором (см. пререгистрацию).

Агенты (решение на встрече: A, если P(A — пища) > 0,5):
  каскад следов L1 / L2 / L3: след i — взвешенное по времени среднее наблюдённых знаков с постоянной τ_i (в тактах):
    S_i ← S_i e^(−Δ/τ_i), N_i ← N_i e^(−Δ/τ_i); при наблюдении y: S_i += y, N_i += 1;
    оценка слоя сжимается к слою выше с псевдосчётом κ = 1: x_L = (S_L + κ·0,5) / (N_L + κ), x_i = (S_i + κ x_(i+1)) / (N_i + κ);
    P(A) = x_1. Считается через d_i = x_i − 0,5 (иначе при редких свидетельствах разность уходит ниже машинной точности
    и решение систематически становится «B»); точная ничья (свидетельств нет) — общая для всех агентов монетка мира. L1 — один след (τ1), L2 — два (τ1 ≤ τ2), L3 — три (τ1 ≤ τ2 ≤ τ3).
  L3_mis — подобранные постоянные L3, изменённые: все ×10, все ÷10, обратный порядок; берётся лучший на обучающих (консервативно).
  BAYES — точный фильтр по (g, m, s) с известными частотами (потолок). Реактивный — без памяти (монетка), нижняя граница.

python env_spectrum.py --check          # проверка конструкции по траекториям
python env_spectrum.py --time           # замер по правилу 5
python env_spectrum.py --run            # вся сетка → results_spectrum/spectrum.json
"""
import argparse, itertools, json, os, time
import numpy as np

T, LAM, E0, FOOD, POISON, KAPPA = 5000, 0.01, 0.5, 0.35, 0.20, 1.0
N_TRAIN, N_TEST = 200, 200
TAU_FINE = 10 ** (0.5 + 0.25 * np.arange(19))            # 3,2 … 100 000
TAU_COARSE = 10 ** (0.5 + 0.5 * np.arange(10))
FREQ = {'rare': 0.5, 'mid': 2.0, 'often': 8.0}            # свидетельств на эпоху знака: q = c · h_f


def configs():
    out = []
    for hf, r, gv, (ph, pl) in itertools.product((1 / 20, 1 / 50), (1 / 10, 1 / 30), ('switch', 'const'), ((0.8, 0.2), (0.9, 0.1))):
        out.append({'name': f"hf{round(1 / hf)}_r{round(1 / r)}_{gv}_p{int(ph * 10)}", 'h_f': hf, 'h_m': hf * r,
                    'h_g': hf * r * r if gv == 'switch' else 0.0, 'p_hi': ph, 'p_lo': pl, 'g': gv})
    return out


# ---------------- мир ----------------
def world(cfg, q, n, seed):
    """n жизней: времена встреч, знак, m, g на встречах; флаг «знак сменился после прошлого наблюдения»; ряды m, g, встреч для проверок."""
    rng = np.random.RandomState(seed)
    ph, pl = cfg['p_hi'], cfg['p_lo']
    g = (rng.rand(n) < 0.5).astype(np.int8)
    m = (rng.rand(n) < np.where(g == 1, ph, pl)).astype(np.int8)
    s = (rng.rand(n) < np.where(m == 1, ph, pl)).astype(np.int8)
    G, M, S, ENC = (np.zeros((n, T), np.int8) for _ in range(4))
    for t in range(T):
        g ^= (rng.rand(n) < cfg['h_g']).astype(np.int8)
        rm = rng.rand(n) < cfg['h_m']; m = np.where(rm, (rng.rand(n) < np.where(g == 1, ph, pl)), m).astype(np.int8)
        rs = rng.rand(n) < cfg['h_f']; s = np.where(rs, (rng.rand(n) < np.where(m == 1, ph, pl)), s).astype(np.int8)
        G[:, t], M[:, t], S[:, t] = g, m, s
        ENC[:, t] = rng.rand(n) < q
    K = ENC.sum(1); Kmax = int(K.max())
    tt = np.full((n, Kmax), -1); y = np.zeros((n, Kmax), np.int8); mm = np.zeros((n, Kmax), np.int8); gg = np.zeros((n, Kmax), np.int8)
    for i in range(n):
        idx = np.nonzero(ENC[i])[0]; tt[i, :len(idx)] = idx; y[i, :len(idx)] = S[i, idx]; mm[i, :len(idx)] = M[i, idx]; gg[i, :len(idx)] = G[i, idx]
    valid = tt >= 0
    coin = np.random.RandomState(seed + 777).rand(n, Kmax) < 0.5                  # разбивка ничьих, общая для всех агентов
    prev = np.concatenate([np.full((n, 1), -1), y[:, :-1].astype(int)], 1)
    flipped = valid & (prev >= 0) & (prev != y)                                    # знак другой, чем при прошлом наблюдении
    dt = np.where(valid, tt - np.concatenate([np.zeros((n, 1), int), np.where(tt[:, :-1] >= 0, tt[:, :-1], 0)], 1), 0)
    dt[:, 0] = np.where(valid[:, 0], tt[:, 0] + 1, 0)
    w = np.where(valid, (1 - (1 - LAM) ** (T - np.maximum(tt, 0))) / (LAM * T), 0.0)
    return {'t': tt, 'y': y, 'm': mm, 'g': gg, 'valid': valid, 'flipped': flipped, 'dt': dt, 'w': w, 'n_enc': K, 'coin': coin,
            'G': G, 'M': M, 'ENC': ENC}


def score(W, choiceA):
    """choiceA (…, n, Kmax) bool → средняя энергия за жизнь, выгода на встречу, точность после невидимой смены знака."""
    right = choiceA == (W['y'] == 1)
    r = np.where(right, FOOD, -POISON) * W['valid']
    E = E0 + (r * W['w']).sum(-1)
    pay = r.sum(-1) / np.maximum(W['valid'].sum(-1), 1)
    fl = W['flipped']; acc_fl = (right & fl).sum(-1) / np.maximum(fl.sum(-1), 1)
    return E, pay, acc_fl


def cascade(W, taus):
    """taus (C, L) — постоянные C агентов с L слоями. Возвращает решения (C, n, Kmax)."""
    taus = np.asarray(taus, float); C, L = taus.shape
    n, Kmax = W['y'].shape
    S = np.zeros((C, L, n)); N = np.zeros((C, L, n)); ch = np.zeros((C, n, Kmax), bool)
    for k in range(Kmax):
        v = W['valid'][:, k]
        if not v.any():
            break
        dec = np.exp(-W['dt'][:, k][None, None, :] / taus[:, :, None])
        S *= dec; N *= dec
        d = np.zeros((C, n))                                                       # d_i = x_i − 0,5 без вычитания близких чисел
        for i in range(L - 1, -1, -1):
            d = (S[:, i] - 0.5 * N[:, i] + KAPPA * d) / (N[:, i] + KAPPA)
        ch[:, :, k] = np.where(d == 0, W['coin'][:, k][None], d > 0)            # ничья — общая монетка мира
        yk = W['y'][:, k] * v
        S += yk[None, None, :]; N += v[None, None, :]
    return ch


def trans(cfg):
    """Матрица перехода за такт по состояниям (g, m, s), индекс 4g + 2m + s."""
    ph, pl = cfg['p_hi'], cfg['p_lo']; P = np.zeros((8, 8))
    for g, m, s in itertools.product((0, 1), repeat=3):
        for g2 in (0, 1):
            pg = cfg['h_g'] if g2 != g else 1 - cfg['h_g']
            pm1 = ph if g2 == 1 else pl
            for m2 in (0, 1):
                pmm = cfg['h_m'] * (pm1 if m2 == 1 else 1 - pm1) + (1 - cfg['h_m']) * (m2 == m)
                ps1 = ph if m2 == 1 else pl
                for s2 in (0, 1):
                    pss = cfg['h_f'] * (ps1 if s2 == 1 else 1 - ps1) + (1 - cfg['h_f']) * (s2 == s)
                    P[4 * g + 2 * m + s, 4 * g2 + 2 * m2 + s2] += pg * pmm * pss
    return P


def bayes(W, cfg):
    P = trans(cfg); n, Kmax = W['y'].shape
    maxd = int(W['dt'].max()) + 1
    pw = np.zeros((maxd + 1, 8, 8)); pw[0] = np.eye(8)
    for d in range(1, maxd + 1):
        pw[d] = pw[d - 1] @ P
    ph, pl = cfg['p_hi'], cfg['p_lo']
    b = np.zeros((n, 8))
    pr = lambda parent, child: (ph if parent == 1 else pl) if child == 1 else (1 - ph if parent == 1 else 1 - pl)
    for g, m, s in itertools.product((0, 1), repeat=3):                         # начальное распределение — как в world()
        b[:, 4 * g + 2 * m + s] = 0.5 * pr(g, m) * pr(m, s)
    b /= b.sum(1, keepdims=True)
    sidx = np.array([st % 2 for st in range(8)])
    ch = np.zeros((n, Kmax), bool)
    for k in range(Kmax):
        v = W['valid'][:, k]
        if not v.any():
            break
        d = np.where(v, W['dt'][:, k], 0)                                         # число переходов с прошлого наблюдения (первое — t + 1)
        b = np.einsum('ni,nij->nj', b, pw[d])
        pa = b[:, sidx == 1].sum(1); pb = b[:, sidx == 0].sum(1)
        ch[:, k] = np.where(pa == pb, W['coin'][:, k], pa > pb)
        like = (sidx[None, :] == W['y'][:, k][:, None]).astype(float)
        nb = b * like; nb /= np.maximum(nb.sum(1, keepdims=True), 1e-300)
        b = np.where(v[:, None], nb, b)
    return ch


def reactive(W, seed):
    return np.random.RandomState(seed).rand(*W['y'].shape) < 0.5


# ---------------- подбор и оценка ----------------
def best(W, combos):
    E = score(W, cascade(W, combos))[0].mean(1)
    i = int(E.argmax()); return list(map(float, combos[i])), float(E[i])


def monotone(grid, L):
    return np.array([c for c in itertools.product(grid, repeat=L) if all(c[j] <= c[j + 1] for j in range(L - 1))])


def fit(Wtr):
    t1, _ = best(Wtr, TAU_FINE[:, None])
    t2, _ = best(Wtr, monotone(TAU_FINE, 2))
    t3c, _ = best(Wtr, monotone(TAU_COARSE, 3))
    ref = [np.log10(t3c[j]) + d for j, d in zip(range(3), (0, 0, 0))]
    near = np.array([c for c in itertools.product(*[[10 ** (r + d) for d in (-0.25, 0, 0.25)] for r in ref])
                     if c[0] <= c[1] <= c[2]])
    t3, _ = best(Wtr, near)
    mis = {'x10': [t * 10 for t in t3], 'div10': [t / 10 for t in t3], 'reversed': t3[::-1]}
    Em = {k: score(Wtr, cascade(Wtr, [v]))[0].mean() for k, v in mis.items()}
    kmis = max(Em, key=Em.get)
    return {'L1': t1, 'L2': t2, 'L3': t3, 'L3_mis': mis[kmis], 'L3_mis_kind': kmis}


def evaluate(W, cfg, taus, seed):
    out = {}
    for k in ('L1', 'L2', 'L3', 'L3_mis'):
        ch = cascade(W, [taus[k]])[0]
        out[k] = score(W, ch)
    out['BAYES'] = score(W, bayes(W, cfg)); out['REACT'] = score(W, reactive(W, seed))
    return {k: {'E': v[0].tolist(), 'pay': v[1].tolist(), 'acc_flip': v[2].tolist()} for k, v in out.items()}


def seeds(ci, fi, part):
    return 5_000_000 + 100_000 * ci + 10_000 * fi + (0 if part == 'train' else 5_000)


# ---------------- проверка конструкции ----------------
def check():
    """По траекториям (обучающие посевы): что, кроме истории знаков, могло бы нести информацию о m и g."""
    print('конфигурация, частота: доля встреч при m=1 / m=0 | при g=1 / g=0 | доля «A — пища» при m=1 / m=0 | при g=1 / g=0 | '
          'AUC последнего знака → m | эпох s, m, g за жизнь')
    for ci, cfg in enumerate(configs()):
        for fi, (fn, c) in enumerate(FREQ.items()):
            W = world(cfg, c * cfg['h_f'], 100, seeds(ci, fi, 'train'))
            ENC, M, G = W['ENC'], W['M'], W['G']
            em1, em0 = ENC[M == 1].mean(), ENC[M == 0].mean(); eg1, eg0 = ENC[G == 1].mean(), ENC[G == 0].mean()
            y, mm, gg, v = W['y'], W['m'], W['g'], W['valid']
            a1, a0 = y[v & (mm == 1)].mean(), y[v & (mm == 0)].mean(); b1, b0 = y[v & (gg == 1)].mean(), y[v & (gg == 0)].mean()
            auc = a1 * (1 - a0) + 0.5 * (a1 * a0 + (1 - a1) * (1 - a0))              # AUC бинарной оценки «последний знак» для m
            print(f"  {cfg['name']:22s} {fn:5s}: {em1:.4f} / {em0:.4f} | {eg1:.4f} / {eg0:.4f} | {a1:.2f} / {a0:.2f} | {b1:.2f} / {b0:.2f} | "
                  f"{auc:.2f} | {T * cfg['h_f']:.0f}, {T * cfg['h_m']:.1f}, {T * cfg['h_g']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true'); ap.add_argument('--time', action='store_true'); ap.add_argument('--run', action='store_true')
    ap.add_argument('--out', default='results_spectrum')
    a = ap.parse_args()
    if a.check:
        check(); return
    from stand_gpu import git_info
    commit, dirty = git_info()
    cfgs = configs()
    if a.time:                                                                  # замер по правилу 5 — на конфигурации вне сетки, только время
        cfg = {'name': 'timing', 'h_f': 1 / 30, 'h_m': 1 / 300, 'h_g': 1 / 3000, 'p_hi': 0.8, 'p_lo': 0.2, 'g': 'switch'}
        for fn, c in FREQ.items():
            ts = time.time(); Wtr = world(cfg, c * cfg['h_f'], N_TRAIN, 99); Wte = world(cfg, c * cfg['h_f'], N_TEST, 98)
            evaluate(Wte, cfg, fit(Wtr), 1); print(f'{fn}: {time.time() - ts:.0f} с', flush=True)
        return
    todo = list(enumerate(cfgs))
    os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, 'spectrum.json')
    res = json.load(open(path, encoding='utf-8')) if (os.path.exists(path) and not a.time) else {'runs': [], 'results': {}}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty})
    t0 = time.time()
    for ci, cfg in todo:
        for fi, (fn, c) in enumerate(FREQ.items()):
            key = f"{cfg['name']}/{fn}"
            if key in res['results']:
                continue
            ts = time.time(); q = c * cfg['h_f']
            Wtr = world(cfg, q, N_TRAIN, seeds(ci, fi, 'train')); Wte = world(cfg, q, N_TEST, seeds(ci, fi, 'test'))
            tw = time.time() - ts
            taus = fit(Wtr); tf = time.time() - ts - tw
            ev = evaluate(Wte, cfg, taus, seeds(ci, fi, 'test') + 1)
            tr = {k: float(np.mean(score(Wtr, cascade(Wtr, [taus[k]])[0])[0])) for k in ('L1', 'L2', 'L3')}
            res['results'][key] = {'cfg': cfg, 'freq': fn, 'q': q, 'taus': taus, 'train_E': tr, 'test': ev, 'git_commit': commit, 'git_dirty': dirty,
                                   'sec': round(time.time() - ts)}
            print(f"{time.strftime('%H:%M:%S')} {key}: мир {tw:.0f} с, подбор {tf:.0f} с, всего {time.time() - ts:.0f} с | τ L3 {[round(t) for t in taus['L3']]}", flush=True)
            if not a.time:
                json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'всего {time.time() - t0:.0f} с')


if __name__ == '__main__':
    main()
