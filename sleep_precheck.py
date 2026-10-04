"""Предварительные проверки среды со сном (env_sleep.py) эвристиками, без ES и без LM.
Пререгистрация и критерии — docs/RESULTS_2026-10-04_sleep.md.

Модель агента: выбор вида — быстрая память (знак последнего исхода, держится τ тактов) → медленная память → к ближайшему.
  M1 (по программе): медленная память — «последний исход», обновляется только во сне: в конце сна длиной k с вероятностью
     1 − e^{−k/κ} принимает знак последнего исхода, известного к началу сна.
  M2 (описательно): медленное считывание — правило, качество q = 1 − e^{−C/C₀}; C — события, консолидированные во снах;
     когда быстрая память погасла, а в окне 32 такта есть событие, даёт верный знак с вероятностью q, иначе вид случайный.
Политики сна: never; every N (от пробуждения); pressure θ (съедено с прошлого сна ≥ θ); free θ — как pressure, но
консолидация мгновенная и полная, без сна; oracle — знает знак.

python sleep_precheck.py [--lives 200] [--out results_sleep]
"""
import argparse, json, math, os, time
import numpy as np

import env_flip as ef
from env_sleep import SleepBodies
from stand_gpu import git_info

THETAS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, None]          # None = никогда
NS = [25, 50, 100, 200, 400, None]
KS = [5, 10, 20, 40]
W = 32


def steer(s, chA):
    a = ef._steer(s, 0); b = ef._steer(s, 1); a[~chA] = b[~chA]; return a


def run(policy, par, lives, L, seed0=40000, k=10, tau=10, kappa=10, model='M1', C0=100):
    B = lives
    seeds = [seed0 + i for i in range(B)]
    b = SleepBodies(B, seeds, L)
    rc = np.random.RandomState(seed0 + 123)                       # случайность консолидации и M2 (общая у точек)
    fast_sign = np.zeros(B, bool); fast_t = np.full(B, -10 ** 9)
    last_out = np.full(B, -1); last_ev_t = np.full(B, -10 ** 9)
    slow = np.full(B, -1); C = np.zeros(B); press = np.zeros(B, int); since_wake = np.zeros(B, int)
    pend_out = np.full(B, -1); pend_C = np.zeros(B)
    correct = np.zeros(B); awake_n = np.zeros(B)
    for t in range(L):
        s = b.sense()
        ateA, ateB = s[:, 10] > 0, s[:, 11] > 0
        ev = ateA | ateB
        food = s[:, 7] > -0.5
        yA = np.where(ateA, food, ~food)
        fast_sign = np.where(ev, yA, fast_sign); fast_t = np.where(ev, t, fast_t)
        last_out = np.where(ev, yA.astype(int), last_out); last_ev_t = np.where(ev, t, last_ev_t)
        press += ev
        awake = (b.sleep_left == 0) & b.alive
        since_wake += awake
        # решение о сне
        if policy == 'every':
            trig = awake & (since_wake >= par) if par else np.zeros(B, bool)
        elif policy in ('pressure', 'free'):
            trig = awake & (press >= par) if par else np.zeros(B, bool)
        else:
            trig = np.zeros(B, bool)
        if policy == 'free':
            if model == 'M1':
                slow = np.where(trig & (last_out >= 0), last_out, slow)
            else:
                C = C + np.where(trig, press, 0)
            press = np.where(trig, 0, press)
        elif trig.any():
            st = b.start_sleep(trig, k)
            pend_out = np.where(st, last_out, pend_out); pend_C = np.where(st, press, pend_C)
            press = np.where(st, 0, press); since_wake = np.where(st, 0, since_wake)
        # выбор вида
        if policy == 'oracle':
            chA = b.good_A.copy()
        else:
            blind = s[:, 2] <= s[:, 5]
            fast_ok = (t - fast_t) <= tau
            if model == 'M1':
                chA = np.where(fast_ok, fast_sign, np.where(slow >= 0, slow == 1, blind))
            else:
                q = 1 - np.exp(-C / C0)
                in_win = (t - last_ev_t) <= W
                rd = rc.rand(B) < q
                guess = np.where(rd, last_out == 1, rc.rand(B) < 0.5)
                chA = np.where(fast_ok, fast_sign, np.where(in_win & (last_out >= 0), guess, blind))
        aw = (b.sleep_left == 0) & b.alive
        correct += (chA == b.good_A) & aw; awake_n += aw
        act = steer(s, chA); act[~b.alive] = 0
        waking = (b.sleep_left == 1) & b.alive
        b.step(act)
        if waking.any():                                            # консолидация в конце сна
            ok = waking & (rc.rand(B) < 1 - math.exp(-k / kappa))
            if model == 'M1':
                slow = np.where(ok & (pend_out >= 0), pend_out, slow)
            else:
                C = C + np.where(ok, pend_C, 0)
    f = b.fitness()
    return {'f': f, 'alive_end': float(b.alive.mean()), 'food': float(b.n_good.mean()), 'poison': float(b.n_bad.mean()),
            'sleep_frac': float((b.t_sleep / np.maximum(b.t_alive, 1)).mean()), 'n_sleeps': float(b.n_sleeps.mean()),
            'acc_awake': float((correct / np.maximum(awake_n, 1)).mean()), 't_alive': float(b.t_alive.mean())}


def z_paired(a, b):
    d = a - b
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12))


def snr4(a, b):
    d = a - b
    return float(d.mean() / (d.std(ddof=1) / 2 + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lives', type=int, default=200)
    ap.add_argument('--L', type=int, default=1000)
    ap.add_argument('--out', default='results_sleep')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    commit, dirty = git_info()
    t0 = time.time()
    R = {}

    def go(key, *args, **kw):
        r = run(*args, lives=a.lives, L=a.L, **kw)
        R[key] = r
        return r

    ref = {'never': go('never', 'never', None), 'oracle': go('oracle', 'oracle', None)}
    print(f"не спать {ref['never']['f'].mean():.3f} | оракул {ref['oracle']['f'].mean():.3f} | {time.time() - t0:.0f} с", flush=True)
    # основная конфигурация M1, τ = 10, κ = 10
    for k in KS:
        for th in THETAS[:-1]:
            go(f'M1_t10_c10_pressure_{th}_k{k}', 'pressure', th, k=k)
        for N in NS[:-1]:
            go(f'M1_t10_c10_every_{N}_k{k}', 'every', N, k=k)
    for th in THETAS[:-1]:
        go(f'M1_t10_free_{th}', 'free', th)
    print(f'основная сетка готова | {time.time() - t0:.0f} с', flush=True)

    def curve(prefix, grid, k):
        return [R[f'{prefix}_{g}_k{k}']['f'] if g is not None else ref['never']['f'] for g in grid]

    out = {'args': vars(a), 'git_commit': commit, 'git_dirty': dirty, 'never': ref['never']['f'].mean(), 'oracle': ref['oracle']['f'].mean()}
    rep = []
    for pol, grid, edges in (('pressure', THETAS, 'θ'), ('every', NS, 'N')):
        best_k = max(KS, key=lambda k: max(f.mean() for f in curve(f'M1_t10_c10_{pol}', grid, k)))
        cv = curve(f'M1_t10_c10_{pol}', grid, best_k)
        m = [f.mean() for f in cv]; i = int(np.argmax(m))
        interior = 0 < i < len(grid) - 1
        z_lo, z_hi = z_paired(cv[i], cv[0]), z_paired(cv[i], cv[-1])
        mono_from_never = all(m[j] >= m[j + 1] for j in range(i, len(m) - 1))
        mono_from_low = all(m[j] <= m[j + 1] for j in range(0, i))
        snr = [snr4(cv[j], cv[j + 1]) for j in range(i, len(m) - 1)]          # шаги от «никогда» к оптимуму
        out[pol] = {'best_k': best_k, 'grid': [str(g) for g in grid], 'mean': m, 'se': [float(f.std(ddof=1) / math.sqrt(len(f))) for f in cv],
                    'argmax': str(grid[i]), 'interior': interior, 'z_vs_low_edge': z_lo, 'z_vs_high_edge': z_hi,
                    'mono_from_never': mono_from_never, 'mono_from_low': mono_from_low, 'snr4_path_from_never': snr,
                    'table_k': {k: [float(f.mean()) for f in curve(f'M1_t10_c10_{pol}', grid, k)] for k in KS}}
        rep.append(f"{pol}: лучшая k = {best_k}; максимум при {edges} = {grid[i]} ({m[i]:.4f}), внутри сетки: {interior}; "
                   f"z против краёв {z_lo:.1f} / {z_hi:.1f}; монотонно от «никогда»: {mono_from_never}, от нижнего края: {mono_from_low}")
    free = [R[f'M1_t10_free_{th}']['f'] for th in THETAS[:-1]]
    fi = int(np.argmax([f.mean() for f in free]))
    pk = out['pressure']['best_k']; pc = curve('M1_t10_c10_pressure', THETAS, pk); pi = int(np.argmax([f.mean() for f in pc]))
    out['free'] = {'mean': [float(f.mean()) for f in free], 'best_theta': THETAS[fi], 'best': float(free[fi].mean()),
                   'z_free_vs_paid': z_paired(free[fi], pc[pi])}
    rep.append(f"бесплатный сон: лучшая θ = {THETAS[fi]} ({free[fi].mean():.4f}) против платного {pc[pi].mean():.4f}, z = {out['free']['z_free_vs_paid']:.1f}")
    # чувствительность: τ, κ (M1) и M2
    sens = {}
    for tau in (5, 10, 20, 40, 10 ** 6):
        for kappa in (5, 10, 20):
            best = None
            for k in (10, 20):
                for th in (2, 4, 8, 16):
                    r = run('pressure', th, lives=a.lives, L=a.L, k=k, tau=tau, kappa=kappa)
                    if best is None or r['f'].mean() > best[0]:
                        best = (float(r['f'].mean()), th, k)
            nv = run('never', None, lives=a.lives, L=a.L, tau=tau)['f'].mean()
            sens[f'tau{tau}_kappa{kappa}'] = {'never': float(nv), 'best_sleep': best[0], 'theta': best[1], 'k': best[2], 'gain': best[0] - float(nv)}
    out['sens_M1'] = sens
    m2 = {}
    for C0 in (30, 100, 300):
        row = {'never': float(run('never', None, lives=a.lives, L=a.L, model='M2', C0=C0)['f'].mean())}
        for th in (1, 2, 4, 8, 16, 32):
            row[str(th)] = float(run('pressure', th, lives=a.lives, L=a.L, k=10, model='M2', C0=C0)['f'].mean())
        row['free_best'] = max(float(run('free', th, lives=a.lives, L=a.L, model='M2', C0=C0)['f'].mean()) for th in (1, 4, 16))
        m2[str(C0)] = row
    out['M2'] = m2
    out['details'] = {k: {kk: vv for kk, vv in v.items() if kk != 'f'} | {'f_mean': float(v['f'].mean())} for k, v in R.items()}
    out['sec'] = round(time.time() - t0)
    json.dump(out, open(os.path.join(a.out, 'precheck.json'), 'w'), ensure_ascii=False, default=float)
    print('\n'.join(rep), flush=True)
    print(f'готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
