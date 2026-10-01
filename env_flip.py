"""Среда этапа 2 (набросок 2026-10-01, не используется в сериях): «смена знака».

Зачем. В исходной арене линейная политика на 8 признаках органа даёт 1,33–1,44 при жадной
1,46 (5 посевов, CPU) — задача не требует ни памяти, ни знаний, и звено 2 на ней
непроверяемо. Здесь у агента появляется своё время: без памяти задача не решается.

Мир. Предметы двух видов, A и B (по 2 каждого), орган видит их раздельно. В каждом
эпизоде один вид — пища (+FOOD_GAIN), другой — яд (−POISON); какой — решает монетка
эпизода и ниоткуда не виден, кроме как через последствия съеденного. Кадр, в котором
агент что-то съел, несёт флаг вида (ateA/ateB) и тренд энергии со знаком исхода.
Связь «вид → исход» — медленная переменная эпизода: её надо удержать в контексте.

Ожидание (спекуляция): это ассоциативное извлечение из истории — то, что у
предобученных трансформеров делают индукционные головы. Контроль со случайными весами
здесь сильный (Zhong & Andreas 2024, по памяти: случайный трансформер с обучаемыми
входом/выходом решает часть алгоритмических задач).

Разворот (reversal=True, ДОБАВЛЕНО 2026-10-01): знак меняется один раз в случайный такт
из [REV_LO, REV_HI). Без разворота любой кадр-событие отвечает одинаково, и решение
не зависит от порядка кадров — тогда контроль «LM с перемешанным окном» не
диагностичен: он не упадёт и у модели, честно читающей последовательность.

Интерфейс как у stand_gpu.Bodies: sense() -> (B, S_DIM), step(act), fitness(), alive.
python env_flip.py  — предварительные проверки эвристиками (барьер, выгода, окно памяти).
"""
import numpy as np

T = 100
N_PER = 2          # предметов каждого вида
EAT_R = 0.07
FOOD_GAIN = 0.35
POISON = 0.30
DRIFT = 0.012
E0 = 0.6
VMAX = 0.03
TURN = 0.6
S_DIM = 12
REV_LO, REV_HI = 30, 70


class FlipBodies:
    def __init__(self, B, seeds, reversal=False):
        self.B = B
        self.t = 0
        self.rs = [np.random.RandomState(s) for s in seeds]
        self.pos = np.stack([r.rand(2) for r in self.rs])
        self.head = np.array([r.rand() * 2 * np.pi for r in self.rs])
        self.items = np.stack([r.rand(2 * N_PER, 2) for r in self.rs])   # первые N_PER — A
        self.good_A = np.array([r.rand() < 0.5 for r in self.rs])
        self.t_rev = np.array([r.randint(REV_LO, REV_HI) for r in self.rs]) if reversal else np.full(B, -1)
        self.E = np.full(B, E0)
        self.alive = np.ones(B, bool)
        self.t_alive = np.zeros(B)
        self.prev_E = self.E.copy()
        self.ate = np.zeros((B, 2))
        self.n_good = np.zeros(B)
        self.n_bad = np.zeros(B)

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
        s = np.stack(feats + [self.E, etrend, (self.E < 0.3).astype(float), 1 - wall,
                              self.ate[:, 0], self.ate[:, 1]], 1)
        self.prev_E = self.E.copy()
        return s.astype(np.float32)

    def step(self, act):
        a = np.clip(act, -1, 1)
        self.good_A ^= self.t_rev == self.t
        self.t += 1
        self.head = self.head + a[:, 0] * TURN
        v = (a[:, 1] + 1) / 2 * VMAX
        self.pos = np.clip(self.pos + v[:, None] * np.stack([np.cos(self.head), np.sin(self.head)], 1), 0, 1)
        self.E = self.E - DRIFT
        d = np.linalg.norm(self.items - self.pos[:, None, :], axis=2)
        i = d.argmin(1)
        eat = (d[np.arange(self.B), i] < EAT_R) & self.alive
        is_A = i < N_PER
        good = is_A == self.good_A
        self.E = np.where(eat, np.minimum(self.E + np.where(good, FOOD_GAIN, -POISON), 1.0), self.E)
        self.ate[:] = 0
        self.ate[eat & is_A, 0] = 1
        self.ate[eat & ~is_A, 1] = 1
        self.n_good += eat & good
        self.n_bad += eat & ~good
        for b in np.where(eat)[0]:
            self.items[b, i[b]] = self.rs[b].rand(2)
        self.alive &= self.E > 0
        self.t_alive += self.alive

    def fitness(self):
        return self.t_alive / T + 0.5 * np.where(self.alive, self.E, 0)


# ---------------- предварительные проверки ----------------
def _steer(s, kind):
    """Курс на ближайший предмет вида kind (0 = A, 1 = B) по признакам органа."""
    c, sn = s[np.arange(len(s)), 3 * kind], s[np.arange(len(s)), 3 * kind + 1]
    return np.stack([np.clip(np.arctan2(sn, c) / TURN, -1, 1), np.ones(len(s))], 1)


def run(policy, n_ep=200, base=999000, reversal=False):
    b = FlipBodies(n_ep, [base + k for k in range(n_ep)], reversal)
    hist = []
    for _ in range(T):
        s = b.sense(); hist.append(s)
        a = policy(s, hist, b); a[~b.alive] = 0; b.step(a)
    f = b.fitness()
    return f.mean(), f.std(), b.n_good.mean(), b.n_bad.mean()


def greedy_any(s, hist, b):
    kind = (s[:, 5] < s[:, 2]).astype(int)          # ближайший из двух видов
    a = _steer(s, 0); a[kind == 1] = _steer(s, 1)[kind == 1]; return a


def oracle(s, hist, b):
    kind = (~b.good_A).astype(int)
    a = _steer(s, 0); a[kind == 1] = _steer(s, 1)[kind == 1]; return a


def bag_learner(N):
    """Без порядка: каждое событие в окне голосует за «какой вид пища», большинство решает.
    Это потолок для политики, видящей окно как мешок кадров (перемешанный порядок)."""
    def pol(s, hist, b):
        H = np.stack(hist[-N:] if N else hist, 1)
        a = greedy_any(s, hist, b)
        for k in range(len(s)):
            ev = H[k, H[k, :, 10:12].sum(1) > 0]
            if len(ev):
                vote = np.where((ev[:, 10] > 0) == (ev[:, 7] > 0), 1, -1).sum()   # +1 = «A пища»
                if vote:
                    a[k] = _steer(s[k:k + 1], 0 if vote > 0 else 1)[0]
        return a
    return pol


def learner(N):
    """Окно из N последних кадров: найти последний кадр с флагом «съел» и по знаку тренда
    энергии решить, какой вид пища. Нет такого кадра в окне — жадно к ближайшему."""
    def pol(s, hist, b):
        H = np.stack(hist[-N:] if N else hist, 1)            # (B, n, S_DIM)
        a = greedy_any(s, hist, b)
        for k in range(len(s)):
            ev = np.where(H[k, :, 10:12].sum(1) > 0)[0]
            if len(ev):
                fr = H[k, ev[-1]]
                eaten_A = fr[10] > 0
                good_is_A = eaten_A if fr[7] > 0 else not eaten_A   # 7 = тренд энергии
                a[k] = _steer(s[k:k + 1], 0 if good_is_A else 1)[0]
        return a
    return pol


if __name__ == '__main__':
    import sys
    REV = '--reversal' in sys.argv
    print('вариант:', 'с разворотом знака' if REV else 'знак постоянен')
    rng = np.random.RandomState(0)
    rows = {'стоять': lambda s, h, b: np.tile([0, -1.0], (len(s), 1)),
            'случайно': lambda s, h, b: rng.uniform(-1, 1, (len(s), 2)),
            'жадно к любому (без памяти)': greedy_any}
    for N in (1, 2, 4, 8, 16, 32, 0):
        rows[f'память, окно {N if N else "всё"}'] = learner(N)
    for N in (8, 16, 32, 0):
        rows[f'мешок, окно {N if N else "всё"}'] = bag_learner(N)
    rows['оракул (знает знак)'] = oracle
    res = {}
    for k, p in rows.items():
        res[k] = run(p, reversal=REV)
        m, sd, g, bd = res[k]
        print(f'{k:30s} {m:.3f} ± {sd:.3f}   съедено: пищи {g:5.2f}, яда {bd:5.2f}')
    ml, orc = res['жадно к любому (без памяти)'][0], res['оракул (знает знак)'][0]
    print(f'выгода памяти (оракул − без памяти): {orc - ml:.3f}; '
          f'различимость: {(orc - ml) / (res["жадно к любому (без памяти)"][1] + 1e-9):.1f} ст.откл.')
