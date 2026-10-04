"""Предварительные проверки S2 (env_sleep2.py) эвристиками, без ES и без LM. Пререгистрация — docs/RESULTS_2026-10-04_sleep.md, S2.

Модель агента (как M2 из S1, по символам):
  быстрая память — какое гнездо сейчас пища по последнему исходу, держится τ тактов;
  медленное считывание — правило по символам: качество для вида X q_X = 1 − e^{−C_X / C₀ᵥ}, C_X — консолидированные
     во снах события с видом X; C₀ᵥ = C₀ / 2 — калибровка (results_sleep2/calib_c0.json): правило A17 доходит до 0,85 за
     ≈ 198 событий двух видов → C₀ = 198 / ln(1/0,3) ≈ 165, на вид — половина. Переноса между видами нет (допущение);
  когда быстрая память погасла и в окне 32 такта есть событие с ещё присутствующим видом X — с вероятностью q_X верно
     определяет, какое гнездо пища, иначе — случайный выбор; событий в окне нет — к ближайшему предмету;
  конечный буфер: при консолидации учитываются только события не старше B_max тактов; старше — затираются;
  консолидация в конце сна длиной k — с вероятностью 1 − e^{−k/κ}; «бесплатный сон» — мгновенно и полностью.
Политики: never, every N, pressure θ, free θ, oracle.

python sleep2_precheck.py
"""
import argparse, json, math, os, time
import numpy as np

import env_flip as ef
from env_sleep2 import Sleep2Bodies
from stand_gpu import git_info

THETAS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, None]
NS = [25, 50, 100, 200, 400, None]
KS = [5, 10, 20, 40]
W = 32
CAL = json.load(open('results_sleep2/calib_c0.json', encoding='utf-8'))
C0_V = CAL['A17']['C0'] / 2


def steer(s, ch0):
    a = ef._steer(s, 0); b = ef._steer(s, 1); a[~ch0] = b[~ch0]; return a


def run(policy, par, lives=200, L=1500, seed0=50000, k=10, tau=10, kappa=10, bmax=200, c0v=C0_V, metab=0.3):
    B = lives
    b = Sleep2Bodies(B, [seed0 + i for i in range(B)], L, sleep_metab=metab)
    rc = np.random.RandomState(seed0 + 321)
    Cx = [dict() for _ in range(B)]
    buf = [[] for _ in range(B)]                                       # (такт, вид) неконсолидированных событий
    fast0 = np.zeros(B, bool); fast_t = np.full(B, -10 ** 9)
    ev_t = np.full(B, -10 ** 9); ev_sym = np.full(B, -1); ev_good0 = np.zeros(B, bool)
    press = np.zeros(B, int); since_wake = np.zeros(B, int); pend = np.zeros(B, bool)
    correct = np.zeros(B); awake_n = np.zeros(B)

    def consolidate(rows, t):
        for i in rows:
            for (te, x) in buf[i]:
                if te >= t - bmax:
                    Cx[i][x] = Cx[i].get(x, 0) + 1
            buf[i] = []

    for t in range(L):
        s = b.sense()
        a0, a1 = s[:, 10] > 0, s[:, 11] > 0
        ev = a0 | a1
        food = s[:, 7] > -0.5
        g0 = np.where(a0, food, ~food)                                 # вывод «гнездо 0 — пища»
        sym_e = np.where(a0, b.sym[:, 0], b.sym[:, 1])
        fast0 = np.where(ev, g0, fast0); fast_t = np.where(ev, t, fast_t)
        ev_t = np.where(ev, t, ev_t); ev_sym = np.where(ev, sym_e, ev_sym); ev_good0 = np.where(ev, g0, ev_good0)
        for i in np.where(ev)[0]:
            buf[i].append((t, int(sym_e[i])))
        press += ev
        awake = b.sleep_left == 0
        since_wake += awake
        if policy == 'every':
            trig = awake & (since_wake >= par) if par else np.zeros(B, bool)
        elif policy in ('pressure', 'free'):
            trig = awake & (press >= par) if par else np.zeros(B, bool)
        else:
            trig = np.zeros(B, bool)
        if policy == 'free':
            consolidate(np.where(trig)[0], t); press = np.where(trig, 0, press)
        elif trig.any():
            st = b.start_sleep(trig, k); pend |= st
            press = np.where(st, 0, press); since_wake = np.where(st, 0, since_wake)
        if policy == 'oracle':
            ch0 = b.good0.copy()
        else:
            blind = s[:, 2] <= s[:, 5]
            fast_ok = (t - fast_t) <= tau
            present = (ev_sym == b.sym[:, 0]) | (ev_sym == b.sym[:, 1])
            in_win = ((t - ev_t) <= W) & present
            q = np.array([1 - math.exp(-Cx[i].get(int(ev_sym[i]), 0) / c0v) for i in range(B)])
            rd = rc.rand(B) < q
            guess = np.where(rd, ev_good0, rc.rand(B) < 0.5)           # верное считывание — вывод из события окна
            ch0 = np.where(fast_ok, fast0, np.where(in_win, guess, blind))
        aw = b.sleep_left == 0
        correct += (ch0 == b.good0) & aw; awake_n += aw
        waking = b.sleep_left == 1
        b.step(steer(s, ch0))
        if waking.any():
            ok = waking & pend & (rc.rand(B) < 1 - math.exp(-k / kappa))
            consolidate(np.where(ok)[0], t + 1); pend &= ~waking
    f = b.fitness()
    return {'f': f, 'food': float(b.n_good.mean()), 'poison': float(b.n_bad.mean()), 'floor_frac': float((b.t_floor / L).mean()),
            'sleep_frac': float((b.t_sleep / L).mean()), 'n_sleeps': float(b.n_sleeps.mean()), 'acc_awake': float((correct / np.maximum(awake_n, 1)).mean()),
            'seasons': float(b.n_season.mean()), 'q_mean': float(np.mean([np.mean([1 - math.exp(-v / c0v) for v in c.values()]) if c else 0 for c in Cx]))}


def z_paired(a, b):
    d = a - b; return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12))


def snr4(a, b):
    d = a - b; return float(d.mean() / (d.std(ddof=1) / 2 + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lives', type=int, default=200)
    ap.add_argument('--out', default='results_sleep2')
    a = ap.parse_args()
    commit, dirty = git_info(); t0 = time.time(); R = {}
    go = lambda key, *x, **kw: R.setdefault(key, run(*x, lives=a.lives, **kw))
    nv, orc = go('never', 'never', None), go('oracle', 'oracle', None)
    print(f"C₀ на вид {C0_V:.1f} | не спать {nv['f'].mean():.3f} (на нуле {nv['floor_frac']:.2f}) | оракул {orc['f'].mean():.3f} | {time.time() - t0:.0f} с", flush=True)
    for k in KS:
        for th in THETAS[:-1]:
            go(f'pressure_{th}_k{k}', 'pressure', th, k=k)
        for N in NS[:-1]:
            go(f'every_{N}_k{k}', 'every', N, k=k)
        print(f'k = {k} готово | {time.time() - t0:.0f} с', flush=True)
    for th in THETAS[:-1]:
        go(f'free_{th}', 'free', th)
    curve = lambda pol, grid, k: [R[f'{pol}_{g}_k{k}']['f'] if g is not None else nv['f'] for g in grid]
    out = {'git_commit': commit, 'git_dirty': dirty, 'C0_v': C0_V, 'calibration': CAL['source'], 'never': float(nv['f'].mean()), 'oracle': float(orc['f'].mean())}
    rep = []
    for pol, grid, nm in (('pressure', THETAS, 'θ'), ('every', NS, 'N')):
        bk = max(KS, key=lambda k: max(f.mean() for f in curve(pol, grid, k)))
        cv = curve(pol, grid, bk); m = [float(f.mean()) for f in cv]; i = int(np.argmax(m))
        res = {'best_k': bk, 'grid': [str(g) for g in grid], 'mean': m, 'argmax': str(grid[i]), 'interior': 0 < i < len(grid) - 1,
               'z_low': z_paired(cv[i], cv[0]), 'z_high': z_paired(cv[i], cv[-1]),
               'mono_from_never': all(m[j] >= m[j + 1] for j in range(i, len(m) - 1)), 'mono_from_low': all(m[j] <= m[j + 1] for j in range(i)),
               'snr4_from_never': [snr4(cv[j], cv[j + 1]) for j in range(i, len(m) - 1)],
               'table_k': {str(k): [float(f.mean()) for f in curve(pol, grid, k)] for k in KS}}
        out[pol] = res
        rep.append(f"{pol}: k = {bk}; максимум {nm} = {grid[i]} ({m[i]:.4f}), внутри: {res['interior']}; z к краям {res['z_low']:.1f} / {res['z_high']:.1f}; "
                   f"монотонно от «никогда» {res['mono_from_never']}, от нижнего края {res['mono_from_low']}; SNR шагов (4 жизни) от «никогда» "
                   + ' '.join(f'{x:+.2f}' for x in res['snr4_from_never']))
    free = [R[f'free_{th}']['f'] for th in THETAS[:-1]]; fi = int(np.argmax([f.mean() for f in free]))
    pc = curve('pressure', THETAS, out['pressure']['best_k']); pi = int(np.argmax([f.mean() for f in pc]))
    out['free'] = {'mean': [float(f.mean()) for f in free], 'best_theta': THETAS[fi], 'z_free_vs_paid': z_paired(free[fi], pc[pi])}
    rep.append(f"бесплатный сон: лучшая θ = {THETAS[fi]} ({free[fi].mean():.4f}) против платного {pc[pi].mean():.4f}, z {out['free']['z_free_vs_paid']:.1f}")
    sens = {}
    for nm, kw in (('bmax100', {'bmax': 100}), ('bmax400', {'bmax': 400}), ('metab0.1', {'metab': 0.1}), ('metab0.6', {'metab': 0.6}),
                   ('tau5', {'tau': 5}), ('tau20', {'tau': 20}), ('c0x0.5', {'c0v': C0_V / 2}), ('c0x2', {'c0v': C0_V * 2})):
        base = run('never', None, lives=a.lives, **{k_: v for k_, v in kw.items() if k_ in ('tau', 'metab')})['f'].mean()
        row = {str(th): float(run('pressure', th, lives=a.lives, k=10, **kw)['f'].mean()) for th in (2, 4, 8, 16, 32)}
        sens[nm] = {'never': float(base), **row}
    out['sens'] = sens
    out['details'] = {k_: {kk: vv for kk, vv in v.items() if kk != 'f'} | {'f_mean': float(v['f'].mean())} for k_, v in R.items()}
    out['sec'] = round(time.time() - t0)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'precheck.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print('\n'.join(rep), flush=True)
    print(f'готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
