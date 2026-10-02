"""Среда «много видов» — предварительные проверки эвристиками (только CPU, решение Егора 2026-10-02).

Зачем. В flip-2 шаги памяти для ES имеют SNR ≤ 2 (против 18,5 у первого шага исходной арены):
выгода памяти в эпизоде зависит от немногих случайных встреч. Здесь K видов, в каждом эпизоде
половина видов — пища, половина — яд (случайно), разворота нет. Вопрос — даёт ли рост числа видов
лестницу частичных решений (правило 6) с SNR выше, чем в flip-2.

Мир как в env_flip (T = 100, пища +0,35, яд −0,20 как в flip-2, дрейф 0,012, E0 = 0,6, радиус
поедания 0,07), но видов K, по N_PER предметов каждого. Агент — в духе рефлекса: эвристика выбирает
вид, к ближайшему предмету выбранного вида ведёт зашитая навигация (курс по углу, полная скорость).

Эвристики (200 эпизодов, посевы 999000+k): стоять; без памяти (к ближайшему любому); «последний исход
по виду» с окном 4 / 8 / 32 / весь эпизод (к ближайшему из видов с последним исходом «рост»; если
таких нет — к ближайшему из видов без исхода «падение»); байесовский фильтр — независимая оценка по
каждому виду (при детерминированных исходах и без разворота совпадает с «последним исходом, весь
эпизод»; считается отдельно как проверка); оракул.
Лестница правила 6: оракул знает m из K связок (случайный набор на эпизод), остальные — без памяти:
к ближайшему из видов «известная пища ∪ неизвестные».
SNR шага — разность двух политик попарно по эпизоду: среднее / (ст. откл. / √4) — при 4 эпизодах на особь.

python env_multi.py  (понижает себе приоритет процесса)
"""
import numpy as np

T, EAT_R, FOOD_GAIN, POISON, DRIFT, E0, VMAX, TURN = 100, 0.07, 0.35, 0.20, 0.012, 0.6, 0.03, 0.6


class MultiBodies:
    def __init__(self, B, seeds, K, n_per=2):
        self.B, self.K, self.n_per = B, K, n_per
        self.rs = [np.random.RandomState(s) for s in seeds]
        self.pos = np.stack([r.rand(2) for r in self.rs])
        self.head = np.array([r.rand() * 2 * np.pi for r in self.rs])
        self.items = np.stack([r.rand(K * n_per, 2) for r in self.rs])          # вид предмета j — j // n_per
        self.kind_of = np.arange(K * n_per) // n_per
        self.food = np.zeros((B, K), bool)
        for b, r in enumerate(self.rs):
            self.food[b, r.permutation(K)[:K // 2]] = True
        self.E = np.full(B, E0); self.alive = np.ones(B, bool); self.t_alive = np.zeros(B)
        self.ate_kind = np.full(B, -1); self.ate_good = np.zeros(B, bool)
        self.n_good = np.zeros(B); self.n_bad = np.zeros(B); self.t = 0

    def nearest_of(self, allowed):
        """allowed (B, K) bool -> вектор к ближайшему предмету допустимых видов (B, 2) и наличие."""
        d = np.linalg.norm(self.items - self.pos[:, None, :], axis=2)
        ok = allowed[:, self.kind_of]
        d = np.where(ok, d, np.inf)
        i = d.argmin(1)
        vec = self.items[np.arange(self.B), i] - self.pos
        return vec, np.isfinite(d[np.arange(self.B), i])

    def steer(self, allowed, move=None):
        vec, has = self.nearest_of(allowed)
        ang = np.arctan2(vec[:, 1], vec[:, 0]) - self.head
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        a = np.stack([np.clip(ang / TURN, -1, 1), np.ones(self.B)], 1)
        if move is not None:
            a[~move, 1] = -1
        a[~has] = [0, -1]
        return a

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.head = self.head + a[:, 0] * TURN
        v = (a[:, 1] + 1) / 2 * VMAX
        self.pos = np.clip(self.pos + v[:, None] * np.stack([np.cos(self.head), np.sin(self.head)], 1), 0, 1)
        self.E = self.E - DRIFT
        d = np.linalg.norm(self.items - self.pos[:, None, :], axis=2)
        i = d.argmin(1)
        eat = (d[np.arange(self.B), i] < EAT_R) & self.alive
        k = self.kind_of[i]
        good = self.food[np.arange(self.B), k]
        self.E = np.where(eat, np.minimum(self.E + np.where(good, FOOD_GAIN, -POISON), 1.0), self.E)
        self.ate_kind = np.where(eat, k, -1); self.ate_good = good & eat
        self.n_good += eat & good; self.n_bad += eat & ~good
        for b in np.where(eat)[0]:
            self.items[b, i[b]] = self.rs[b].rand(2)
        self.alive &= self.E > 0
        self.t_alive += self.alive; self.t += 1

    def fitness(self):
        return self.t_alive / T + 0.5 * np.where(self.alive, self.E, 0)


def run(policy, K, n_ep=200, base=999000, n_per=2):
    b = MultiBodies(n_ep, [base + i for i in range(n_ep)], K, n_per)
    hist = []                     # история событий: (такт, вид, хорошо) на тело
    ev = [[] for _ in range(n_ep)]
    rng = np.random.RandomState(base + 7)
    known_m = None
    for t in range(T):
        a = policy(b, ev, t, rng)
        a[~b.alive] = 0
        b.step(a)
        for i in np.where(b.ate_kind >= 0)[0]:
            ev[i].append((t, int(b.ate_kind[i]), bool(b.ate_good[i])))
    return b.fitness(), b


def p_stand(b, ev, t, rng):
    return np.tile([0.0, -1.0], (b.B, 1))


def p_memless(b, ev, t, rng):
    return b.steer(np.ones((b.B, b.K), bool))


def p_last(W):
    def pol(b, ev, t, rng):
        allowed = np.ones((b.B, b.K), bool)
        for i in range(b.B):
            last = {}
            for (tt, k, g) in ev[i]:
                if W is None or tt > t - W:
                    last[k] = g
            good = [k for k, g in last.items() if g]
            bad = [k for k, g in last.items() if not g]
            if good:
                allowed[i] = False; allowed[i, good] = True
            elif bad and len(bad) < b.K:
                allowed[i, bad] = False
        return b.steer(allowed)
    return pol


def p_bayes(b, ev, t, rng):
    """Независимый фильтр по каждому виду: P(пища) = 0,5 до наблюдений; исходы детерминированы."""
    allowed = np.ones((b.B, b.K), bool)
    for i in range(b.B):
        p = np.full(b.K, 0.5)
        for (tt, k, g) in ev[i]:
            p[k] = 1.0 if g else 0.0
        best = p.max()
        allowed[i] = p >= best - 1e-9
    return b.steer(allowed)


def p_oracle(b, ev, t, rng):
    return b.steer(b.food.copy())


def p_ladder(m):
    def pol(b, ev, t, rng):
        if not hasattr(b, 'known'):
            r = np.random.RandomState(12345 + m)
            b.known = np.zeros((b.B, b.K), bool)
            for i in range(b.B):
                b.known[i, r.permutation(b.K)[:m]] = True
        allowed = ~(b.known & ~b.food)              # исключить известный яд
        known_food = b.known & b.food
        has_kf = known_food.any(1)
        allowed[has_kf] = known_food[has_kf]         # если известна пища — только к ней
        return b.steer(allowed)
    return pol


def snr(fa, fb, eps=4):
    d = fa - fb
    return d.mean(), d.mean() / (d.std() / np.sqrt(eps) + 1e-12)


if __name__ == '__main__':
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception as e:
        print('приоритет не понижен:', e)
    out = {}
    for K in (2, 4, 6):
        F = {}
        rows = [('стоять', p_stand), ('без памяти', p_memless)] + [(f'последний, окно {W or "всё"}', p_last(W)) for W in (4, 8, 32, None)] + \
               [('байесовский (по виду)', p_bayes), ('оракул', p_oracle)]
        print(f'\n=== K = {K} видов (по 2 предмета), яд {POISON}')
        for nm, p in rows:
            f, b = run(p, K)
            F[nm] = f
            print(f'  {nm:24s} {f.mean():.3f} ± {f.std():.3f}  съедено: пищи {b.n_good.mean():5.2f}, яда {b.n_bad.mean():5.2f}, доля < 0,7: {(f < 0.7).mean():.2f}')
        chain = [('без памяти', 'стоять'), ('последний, окно 4', 'без памяти'), ('последний, окно 8', 'последний, окно 4'),
                 ('последний, окно 32', 'последний, окно 8'), ('последний, окно всё', 'последний, окно 32'), ('оракул', 'последний, окно всё')]
        print('  цепочка памяти, Δ / SNR (4 эпизода):')
        for x, y in chain:
            d, s = snr(F[x], F[y]); print(f'    {x} − {y}: {d:+.3f} / {s:.1f}')
        print('  лестница правила 6 (оракул знает m из K связок):')
        L = []
        for m in range(K + 1):
            f, _ = run(p_ladder(m), K); L.append(f)
        for m in range(K + 1):
            line = f'    m = {m}: {L[m].mean():.3f}'
            if m:
                d, s = snr(L[m], L[m - 1]); line += f'   шаг {m - 1}→{m}: {d:+.3f} / SNR {s:.1f}'
            print(line)
        mono = all(L[m].mean() > L[m - 1].mean() for m in range(1, K + 1))
        print(f'  монотонность лестницы: {"да" if mono else "НЕТ"}')
        out[K] = {'means': {k: float(v.mean()) for k, v in F.items()}, 'ladder': [float(x.mean()) for x in L]}
    print('\nсправочно: flip-2 (разворот) — SNR шагов памяти ≤ 2; исходная арена — первый шаг 18,5.')
