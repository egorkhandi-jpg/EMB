"""Среда этапа 3, вариант S2: непрерывная жизнь со сном, сезоны, мягкое выживание (решение Егора 2026-10-04 после S1).
Пререгистрация — docs/RESULTS_2026-10-04_sleep.md, раздел S2. S2 — последний вариант среды со сном.

Против S1 (env_sleep.py):
  - сезоны: в случайные такты (вероятность season_rate за такт, по умолчанию 1/300) один из двух присутствующих видов
    исчезает и появляется новый вид со своим символом (номер вида — в признаках). Роль нового вида (пища или яд) —
    монетка; знак по-прежнему разворачивается внутри сезона (rev_rate, по умолчанию 1/100);
  - обмен во сне ниже: дрейф энергии во сне — sleep_metab · DRIFT (по умолчанию 0,3);
  - мягче выживание: смерти нет, энергия не ниже 0; приспособленность — средняя энергия за жизнь Ē (время на нуле —
    штраф через саму среднюю).
Конечный буфер — свойство агента (эвристика / голова), не среды.

Признаки (B, 16): 12 столбцов как во flip (вид 0 / вид 1 — «гнёзда» с текущими видами), 12 — давление сна (съедено с
последнего сна), 13 — «сплю», 14, 15 — номера видов в гнёздах 0 и 1 (символы для азбуки LM).
"""
import numpy as np

import env_flip as ef

N_PER = 2


class Sleep2Bodies:
    def __init__(self, B, seeds, L=1500, rev_rate=0.01, season_rate=1 / 300, sleep_metab=0.3,
                 poison=ef.ENVS['flip-2']['poison']):
        self.B, self.L, self.rev_rate, self.season_rate, self.metab, self.poison = B, L, rev_rate, season_rate, sleep_metab, poison
        self.rs = [np.random.RandomState(s) for s in seeds]
        self.rw = [np.random.RandomState(s + 7_000_000) for s in seeds]       # развороты и сезоны — общий поток у политик
        self.pos = np.stack([r.rand(2) for r in self.rs])
        self.head = np.array([r.rand() * 2 * np.pi for r in self.rs])
        self.items = np.stack([r.rand(2 * N_PER, 2) for r in self.rs])        # первые N_PER — гнездо 0
        self.good0 = np.array([r.rand() < 0.5 for r in self.rw])
        self.sym = np.tile([0, 1], (B, 1)); self.next_sym = np.full(B, 2)
        self.E = np.full(B, ef.E0); self.prev_E = self.E.copy(); self.t = 0; self.E_sum = np.zeros(B)
        self.ate = np.zeros((B, 2)); self.n_good = np.zeros(B); self.n_bad = np.zeros(B); self.t_floor = np.zeros(B)
        self.sleep_left = np.zeros(B, int); self.pressure = np.zeros(B, int)
        self.n_sleeps = np.zeros(B, int); self.t_sleep = np.zeros(B, int); self.n_rev = np.zeros(B, int); self.n_season = np.zeros(B, int)
        self.alive = np.ones(B, bool)                                          # смерти нет; для совместимости

    def _nearest(self, sl):
        it = self.items[:, sl]
        d = np.linalg.norm(it - self.pos[:, None, :], axis=2)
        i = d.argmin(1)
        return it[np.arange(self.B), i] - self.pos, d[np.arange(self.B), i]

    def sense(self):
        feats = []
        for sl in (slice(0, N_PER), slice(N_PER, 2 * N_PER)):
            vec, d = self._nearest(sl)
            ang = np.arctan2(vec[:, 1], vec[:, 0]) - self.head
            feats += [np.cos(ang), np.sin(ang), d * 2]
        etrend = (self.E - self.prev_E) * 10
        hx, hy = np.cos(self.head), np.sin(self.head)
        tx = np.where(hx > 0, (1 - self.pos[:, 0]) / np.maximum(hx, 1e-6), self.pos[:, 0] / np.maximum(-hx, 1e-6))
        ty = np.where(hy > 0, (1 - self.pos[:, 1]) / np.maximum(hy, 1e-6), self.pos[:, 1] / np.maximum(-hy, 1e-6))
        wall = np.clip(np.minimum(tx, ty), 0, 0.2) / 0.2
        s = np.stack(feats + [self.E, etrend, (self.E < 0.3).astype(float), 1 - wall, self.ate[:, 0], self.ate[:, 1],
                              self.pressure.astype(float), (self.sleep_left > 0).astype(float),
                              self.sym[:, 0].astype(float), self.sym[:, 1].astype(float)], 1)
        self.prev_E = self.E.copy()
        return s.astype(np.float32)

    def start_sleep(self, mask, k):
        m = mask & (self.sleep_left == 0)
        self.sleep_left = np.where(m, k, self.sleep_left); self.n_sleeps += m
        return m

    def step(self, act):
        a = np.clip(act, -1, 1).copy()
        asleep = self.sleep_left > 0
        a[asleep] = [0.0, -1.0]
        u = np.array([r.rand(3) for r in self.rw])                            # разворот, сезон, какое гнездо / роль
        flip = u[:, 0] < self.rev_rate
        self.good0 ^= flip; self.n_rev += flip
        season = u[:, 1] < self.season_rate
        slot = (u[:, 2] < 0.5).astype(int)
        for b in np.where(season)[0]:
            self.sym[b, slot[b]] = self.next_sym[b]; self.next_sym[b] += 1
            self.good0[b] = self.rw[b].rand() < 0.5                            # роль нового вида — монетка
            sl = slice(0, N_PER) if slot[b] == 0 else slice(N_PER, 2 * N_PER)
            self.items[b, sl] = self.rs[b].rand(N_PER, 2)                      # предметы нового вида — на новых местах
        self.n_season += season
        self.t += 1
        self.head = self.head + a[:, 0] * ef.TURN
        v = (a[:, 1] + 1) / 2 * ef.VMAX
        self.pos = np.clip(self.pos + v[:, None] * np.stack([np.cos(self.head), np.sin(self.head)], 1), 0, 1)
        self.E = self.E - ef.DRIFT * np.where(asleep, self.metab, 1.0)
        d = np.linalg.norm(self.items - self.pos[:, None, :], axis=2)
        i = d.argmin(1)
        eat = (d[np.arange(self.B), i] < ef.EAT_R) & ~asleep
        is0 = i < N_PER
        good = is0 == self.good0
        self.E = np.where(eat, np.minimum(self.E + np.where(good, ef.FOOD_GAIN, -self.poison), 1.0), self.E)
        self.E = np.maximum(self.E, 0.0)
        self.ate[:] = 0
        self.ate[eat & is0, 0] = 1; self.ate[eat & ~is0, 1] = 1
        self.n_good += eat & good; self.n_bad += eat & ~good
        for b in np.where(eat)[0]:
            self.items[b, i[b]] = self.rs[b].rand(2)
        self.t_sleep += asleep
        woke = asleep & (self.sleep_left == 1)
        self.sleep_left = np.maximum(self.sleep_left - 1, 0)
        self.pressure = np.where(woke, 0, self.pressure + eat)
        self.E_sum += self.E; self.t_floor += self.E <= 0

    def fitness(self):
        return self.E_sum / self.L
