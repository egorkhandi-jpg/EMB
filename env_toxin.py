"""Звено 6, среда «скрытый токсин» (только эвристики; пререгистрация — docs/RESULTS_2026-10-04_toxin.md).

Вопрос: есть ли среда, где для удержания себя нужно знать о себе то, чего не показывает ни один датчик, и это знание
приходится выводить из собственной истории («вынужденная модель себя»).

Мир. Два вида предметов, оба известны агенту (знака и разворотов нет), по N_PER каждого:
  вид 0 — обычная пища: +G_ORD, токсина нет;
  вид 1 — богатая пища: прирост G_RICH · (1 − X / X_C), обрезанный снизу −G_RICH, и доза токсина.
Токсин X — скрытая переменная тела: X ← X · e^{−1/τ} каждый такт; за каждую богатую X += dose · 100 / τ, dose ~ U[0,5; 1,5]
(у каждого предмета своя, не видна). Нормировка 100 / τ: при ровном темпе r богатых на 100 тактов X ≈ r независимо от τ —
τ меняет только длину памяти тела, не экономику. При X > X_C богатая вредит (энергия падает).
Датчики (12, как во flip): направление и дистанция до ближайшего предмета каждого вида, энергия, её тренд, «мало энергии»,
стена, «съел вид 0 / вид 1». Токсина в датчиках нет; о нём говорит только тренд энергии в такт съедения богатой.
Мягкое выживание (как S2): смерти нет, энергия в [0, 1]; приспособленность — средняя энергия за жизнь L = 1000.

Посевы: раскладка, место после съедения и дозы задаются таблицами от посева, а не потоком генератора, — у всех политик
с одним посевом один и тот же мир (k-я переставленная вещь в гнезде, k-я доза), разности считаются попарно.
"""
import numpy as np

import env_flip as ef

N_PER = 2
L = 1000
G_ORD = 0.066         # экономика — docs/RESULTS_2026-10-04_toxin.md, «Экономика» (калибровка при выключенном токсине, до критериев)
G_RICH = 0.60
X_C = 8.0
DOSE_LO, DOSE_HI = 0.5, 1.5
MAX_RESPAWN = 400     # на гнездо (предмет) за жизнь; с запасом
MAX_RICH = 600


class ToxinBodies:
    def __init__(self, seeds, tau, L=L, g_ord=G_ORD, g_rich=G_RICH, x_c=X_C, toxin=True):
        seeds = np.asarray(seeds)
        self.B, self.L, self.tau, self.g_ord, self.g_rich, self.x_c, self.toxin = len(seeds), L, tau, g_ord, g_rich, x_c, toxin
        uniq, self.sidx = np.unique(seeds, return_inverse=True)
        tabs = [np.random.RandomState(int(s)) for s in uniq]
        init = np.stack([r.rand(2 * N_PER + 2) for r in tabs])                       # 4 предмета + положение; угол — отдельно
        head = np.stack([r.rand() for r in tabs]) * 2 * np.pi
        self.resp = np.stack([r.rand(2 * N_PER, MAX_RESPAWN, 2) for r in tabs]).astype(np.float64)
        self.dose_tab = np.stack([r.uniform(DOSE_LO, DOSE_HI, MAX_RICH) for r in tabs])
        it0 = np.stack([np.random.RandomState(int(s) + 11).rand(2 * N_PER, 2) for s in uniq])
        self.items = it0[self.sidx].copy()
        self.pos = init[self.sidx, :2].copy(); self.head = head[self.sidx].copy()
        self.n_resp = np.zeros((self.B, 2 * N_PER), int); self.n_rich = np.zeros(self.B, int)
        self.E = np.full(self.B, ef.E0); self.prev_E = self.E.copy(); self.X = np.zeros(self.B)
        self.t = 0; self.E_sum = np.zeros(self.B); self.t_floor = np.zeros(self.B)
        self.ate = np.zeros((self.B, 2)); self.n_ord = np.zeros(self.B); self.n_rich_eat = np.zeros(self.B)
        self.n_harm = np.zeros(self.B); self.last_gain = np.full(self.B, np.nan)
        self.alive = np.ones(self.B, bool)                                            # смерти нет; для совместимости
        self._ar = np.arange(self.B)

    def _nearest(self, sl):
        it = self.items[:, sl]
        d = np.linalg.norm(it - self.pos[:, None, :], axis=2)
        i = d.argmin(1)
        return it[self._ar, i] - self.pos, d[self._ar, i]

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
        s = np.stack(feats + [self.E, etrend, (self.E < 0.3).astype(float), 1 - wall, self.ate[:, 0], self.ate[:, 1]], 1)
        self.prev_E = self.E.copy()
        return s.astype(np.float32)

    def step(self, act):
        a = np.clip(act, -1, 1)
        self.t += 1
        self.head = self.head + a[:, 0] * ef.TURN
        v = (a[:, 1] + 1) / 2 * ef.VMAX
        self.pos = np.clip(self.pos + v[:, None] * np.stack([np.cos(self.head), np.sin(self.head)], 1), 0, 1)
        self.X *= np.exp(-1.0 / self.tau)
        E_before = self.E - ef.DRIFT
        d = np.linalg.norm(self.items - self.pos[:, None, :], axis=2)
        i = d.argmin(1)
        eat = d[self._ar, i] < ef.EAT_R
        rich = eat & (i >= N_PER); ordn = eat & (i < N_PER)
        g_r = np.clip(self.g_rich * (1 - self.X / self.x_c), -self.g_rich, self.g_rich) if self.toxin else np.full(self.B, self.g_rich)
        gain = np.where(rich, g_r, np.where(ordn, self.g_ord, 0.0))
        self.E = np.clip(E_before + gain, 0.0, 1.0)
        dose = self.dose_tab[self.sidx, np.minimum(self.n_rich, MAX_RICH - 1)]
        self.X = np.where(rich, self.X + dose * 100.0 / self.tau, self.X)
        self.n_rich += rich
        self.last_gain = np.where(rich, self.E - E_before, self.last_gain)          # наблюдаемый прирост (с обрезкой 0 / 1)
        self.n_harm += rich & (g_r < 0)
        self.ate[:] = 0; self.ate[ordn, 0] = 1; self.ate[rich, 1] = 1
        self.n_ord += ordn; self.n_rich_eat += rich
        b = np.where(eat)[0]
        if len(b):
            j = i[b]; k = np.minimum(self.n_resp[b, j], MAX_RESPAWN - 1)
            self.items[b, j] = self.resp[self.sidx[b], j, k]
            self.n_resp[b, j] += 1
        self.E_sum += self.E; self.t_floor += self.E <= 0

    def fitness(self):
        return self.E_sum / self.L


def steer(s, rich):
    """Курс на ближайший предмет выбранного вида (rich: bool (B,))."""
    a = ef._steer(s, 0); b = ef._steer(s, 1); a[rich] = b[rich]; return a
