"""Среда этапа 3: непрерывная жизнь со сном — «время как ресурс» (решение Егора 2026-10-04 после v4-copies).
Пререгистрация предварительных проверок — docs/RESULTS_2026-10-04_sleep.md. env_flip.py не трогается.

Зачем. В нынешней постановке проигрывание опыта идёт в паузе вне времени. Консолидация до сходимости бесплатна,
поэтому эволюция находит угол области («учиться как можно больше»), а копии сходятся к одному решению. Здесь пауза
на проигрывание — действие «спать», которое отнимает время у еды. Проверяется:
  (1) настраивает ли эволюция правило обучения при настоящем компромиссе;
  (2) звено 6 — выбор между двумя потребностями (еда и переработка опыта) по собственному состоянию;
  (3) звено 1 в сильной форме — время можно потратить и потерять.

Мир — физика flip-2 (два вида по 2 предмета, пища +0,35, яд −0,20, дрейф энергии 0,012 за такт, E0 0,6), но:
  - жизнь непрерывна: L тактов без сброса эпизодов, тело и энергия переносятся, смерть заканчивает жизнь;
  - знак (какой вид — пища) переворачивается в случайные такты: вероятность разворота за такт rev_rate
    (по умолчанию 1/100 — в среднем раз в 100 тактов, как один разворот на эпизод flip-2);
  - действие «спать k тактов» (start_sleep): агент стоит и не ест, энергия падает с обычной скоростью; сон нельзя
    прервать; во время сна агент проигрывает буфер (это делает агент, не среда);
  - давление сна — число съеденных предметов с конца последнего сна (столбец 12 признаков); для азбуки событий LM —
    символ по порогам pressure_bin; столбец 13 — «сплю».
Приспособленность жизни: t_alive / L + 0,5 · Ē, где Ē — средняя энергия по L тактам (после смерти — 0).

Интерфейс: sense() -> (B, 14), step(act) с act (B, 2) как во flip (поворот, скорость), start_sleep(mask, k), fitness().
"""
import numpy as np

import env_flip as ef

N_PER = 2
S_DIM = 14
PRESSURE_BINS = (1, 3, 6)          # символы давления сна: 0 | 1–2 | 3–5 | ≥ 6 (для азбуки событий LM)


def pressure_bin(p):
    return np.searchsorted(np.array(PRESSURE_BINS), p, side='right')


class SleepBodies:
    def __init__(self, B, seeds, L=1000, rev_rate=0.01, poison=ef.ENVS['flip-2']['poison']):
        self.B, self.L, self.rev_rate, self.poison = B, L, rev_rate, poison
        self.rs = [np.random.RandomState(s) for s in seeds]
        self.rrev = [np.random.RandomState(s + 7_000_000) for s in seeds]     # развороты — свой поток (общие у политик)
        self.pos = np.stack([r.rand(2) for r in self.rs])
        self.head = np.array([r.rand() * 2 * np.pi for r in self.rs])
        self.items = np.stack([r.rand(2 * N_PER, 2) for r in self.rs])      # первые N_PER — A
        self.good_A = np.array([r.rand() < 0.5 for r in self.rs])
        self.E = np.full(B, ef.E0); self.prev_E = self.E.copy()
        self.alive = np.ones(B, bool); self.t = 0; self.t_alive = np.zeros(B); self.E_sum = np.zeros(B)
        self.ate = np.zeros((B, 2)); self.n_good = np.zeros(B); self.n_bad = np.zeros(B)
        self.sleep_left = np.zeros(B, int); self.pressure = np.zeros(B, int)
        self.n_sleeps = np.zeros(B, int); self.t_sleep = np.zeros(B, int); self.n_rev = np.zeros(B, int)

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
                              self.pressure.astype(float), (self.sleep_left > 0).astype(float)], 1)
        self.prev_E = self.E.copy()
        return s.astype(np.float32)

    def start_sleep(self, mask, k):
        """Начать сон длиной k тактов (k — число или массив) у живых бодрствующих особей mask."""
        m = mask & self.alive & (self.sleep_left == 0)
        self.sleep_left = np.where(m, k, self.sleep_left)
        self.n_sleeps += m
        return m

    def step(self, act):
        a = np.clip(act, -1, 1).copy()
        asleep = self.sleep_left > 0
        a[asleep] = [0.0, -1.0]                                              # стоит
        flip = np.array([r.rand() < self.rev_rate for r in self.rrev])
        self.good_A ^= flip; self.n_rev += flip & self.alive
        self.t += 1
        self.head = self.head + a[:, 0] * ef.TURN
        v = (a[:, 1] + 1) / 2 * ef.VMAX
        self.pos = np.clip(self.pos + v[:, None] * np.stack([np.cos(self.head), np.sin(self.head)], 1), 0, 1)
        self.E = self.E - ef.DRIFT
        d = np.linalg.norm(self.items - self.pos[:, None, :], axis=2)
        i = d.argmin(1)
        eat = (d[np.arange(self.B), i] < ef.EAT_R) & self.alive & ~asleep
        is_A = i < N_PER
        good = is_A == self.good_A
        self.E = np.where(eat, np.minimum(self.E + np.where(good, ef.FOOD_GAIN, -self.poison), 1.0), self.E)
        self.ate[:] = 0
        self.ate[eat & is_A, 0] = 1; self.ate[eat & ~is_A, 1] = 1
        self.n_good += eat & good; self.n_bad += eat & ~good
        for b in np.where(eat)[0]:
            self.items[b, i[b]] = self.rs[b].rand(2)
        self.t_sleep += asleep & self.alive
        woke = asleep & (self.sleep_left == 1)
        self.sleep_left = np.maximum(self.sleep_left - 1, 0)
        self.pressure = np.where(woke, 0, self.pressure + eat)
        self.alive &= self.E > 0
        self.t_alive += self.alive
        self.E_sum += np.where(self.alive, self.E, 0)

    def fitness(self):
        return self.t_alive / self.L + 0.5 * self.E_sum / self.L
