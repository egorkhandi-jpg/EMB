"""Тезис 5 (геометрия): среда «я / мир» — разделение изменений мира и изменений, вызванных собой.
Пререгистрация — docs/RESULTS_2026-10-05_geometry.md.

Мир. Арена [0, 1]², тело с курсом. Действие кадра: поворот (−1 влево / 0 / +1 вправо) × TURN_G и движение (0 стоять /
1 вперёд на VMAX). K предметов, ровно один живой: медленно смещается сам (случайное блуждание с инерцией ρ, типичный шаг U),
остальные неподвижны. Живой — пища, неподвижные — пустышки.
Раунд: все K предметов ставятся заново, живой — случайный номер. Раунд кончается, когда живой съеден (дистанция < EAT_R)
или прожил L_food тактов (пища потеряна). Касание пустышки: она переставляется в новое место (появилась заново).
Энергия дрейфует (DRIFT), пища +FOOD_GAIN, мягкое выживание (как S2); главная мера — число съеденных за жизнь L.

Датчики — строго в собственных координатах: для каждого предмета направление b (рад, 0 — прямо, + — влево) и дистанция d,
с шумом (σ_b, σ_d); плюс флаг «предмет появился заново» (перестановка видна как событие). Ничего в мировых координатах.
Шум нужен по построению: без него неподвижный предмет при стоянии не сдвигается вовсе, и «замри и смотри» за один кадр
так же точен, как вычитание своего действия.

Посевы: расстановки, номер живого, шум блуждания и шум датчиков — таблицы от посева, общие у всех политик (разности попарны).
"""
import numpy as np

ARENA_LO, ARENA_HI = 0.1, 0.9
TURN_G = 0.3
VMAX = 0.03
EAT_R = 0.05
RHO = 0.95
SIG_B = 0.02
SIG_D = 0.01
DRIFT = 0.012
FOOD_GAIN = 0.35
E0 = 0.6
L = 1000
MAXP = 1200          # расстановок предметов на посев (с запасом)
MAXR = 400           # раундов на посев


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeoBodies:
    def __init__(self, seeds, K, U, L_food, L=L, sig_b=SIG_B, sig_d=SIG_D, wall_block=False):
        seeds = np.asarray(seeds)
        self.B, self.K, self.U, self.L_food, self.L = len(seeds), K, U, L_food, L
        uniq, self.sidx = np.unique(seeds, return_inverse=True)
        rs = [np.random.RandomState(int(s)) for s in uniq]
        self.place_tab = np.stack([r.uniform(ARENA_LO, ARENA_HI, (MAXP, 2)) for r in rs])
        self.alive_tab = np.stack([r.randint(0, K, MAXR) for r in rs])
        self.walk_tab = np.stack([r.randn(L + 1, 2) for r in rs]).astype(np.float32)
        self.nb_tab = np.stack([r.randn(L + 1, K) * sig_b for r in rs]).astype(np.float32)
        self.nd_tab = np.stack([r.randn(L + 1, K) * sig_d for r in rs]).astype(np.float32)
        init = np.stack([np.r_[r.uniform(0.2, 0.8, 2), r.rand() * 2 * np.pi] for r in rs])
        self.pos = init[self.sidx, :2].copy(); self.head = init[self.sidx, 2].copy()
        self._ar = np.arange(self.B)
        self.n_place = np.zeros(self.B, int); self.n_round = np.zeros(self.B, int)
        self.items = np.zeros((self.B, K, 2)); self.vel = np.zeros((self.B, 2))
        self.alive_i = np.zeros(self.B, int); self.age = np.zeros(self.B, int)
        self.respawn = np.ones((self.B, K), bool)
        self.E = np.full(self.B, E0); self.E_sum = np.zeros(self.B)
        self.eaten = np.zeros(self.B); self.expired = np.zeros(self.B); self.dummy_hits = np.zeros(self.B)
        self.t = 0; self.wall_block = wall_block; self.exec_move = np.zeros(self.B, int)
        self._new_round(np.ones(self.B, bool))

    def _place(self, b, k):
        """Новые места для предметов (b[j], k[j]) из таблицы посева; повторы одного тела в вызове — подряд по таблице."""
        o = np.argsort(b, kind='stable'); b, k = b[o], k[o]
        first = np.r_[True, b[1:] != b[:-1]]
        grp = np.cumsum(first) - 1
        rank = np.arange(len(b)) - np.flatnonzero(first)[grp]
        n = self.n_place[b] + rank
        self.items[b, k] = self.place_tab[self.sidx[b], np.minimum(n, MAXP - 1)]
        np.add.at(self.n_place, b, 1)

    def _new_round(self, m):
        b = np.where(m)[0]
        if not len(b):
            return
        for k in range(self.K):
            self._place(b, np.full(len(b), k))
        self.alive_i[b] = self.alive_tab[self.sidx[b], np.minimum(self.n_round[b], MAXR - 1)]
        self.n_round[b] += 1
        self.vel[b] = 0.0; self.age[b] = 0
        self.respawn[b] = True

    def true_rel(self):
        """Истинное положение предметов в собственных координатах (x — вперёд, y — влево): (B, K, 2)."""
        w = self.items - self.pos[:, None, :]
        c, s = np.cos(self.head)[:, None], np.sin(self.head)[:, None]
        return np.stack([c * w[..., 0] + s * w[..., 1], -s * w[..., 0] + c * w[..., 1]], -1)

    def sense(self):
        """b (B, K), d (B, K) с шумом; respawn (B, K) — предмет появился заново с прошлого кадра."""
        rel = self.true_rel()
        b = wrap(np.arctan2(rel[..., 1], rel[..., 0]) + self.nb_tab[self.sidx, self.t])
        d = np.maximum(np.linalg.norm(rel, axis=-1) + self.nd_tab[self.sidx, self.t], 1e-3)
        rsp = self.respawn.copy(); self.respawn[:] = False
        return b, d, rsp

    def step(self, turn, move):
        """turn ∈ {−1, 0, 1} (+ — влево, против часовой), move ∈ {0, 1}."""
        self.t += 1
        self.head = self.head + turn * TURN_G
        new = self.pos + (move * VMAX)[:, None] * np.stack([np.cos(self.head), np.sin(self.head)], 1)
        if self.wall_block:                     # шаг за стену не выполняется, тело знает это (exec_move — исполненное движение)
            inside = ((new >= 0) & (new <= 1)).all(1)
            self.exec_move = np.where(inside, move, 0); self.pos = np.where(inside[:, None], new, self.pos)
        else:
            self.exec_move = np.asarray(move); self.pos = np.clip(new, 0, 1)
        # живой: блуждание с инерцией, типичный шаг U, отражение от стен
        self.vel = RHO * self.vel + np.sqrt(1 - RHO ** 2) * self.U * self.walk_tab[self.sidx, self.t] / np.sqrt(2)
        ai = self.alive_i
        p = self.items[self._ar, ai] + self.vel
        out = (p < ARENA_LO) | (p > ARENA_HI)
        self.vel = np.where(out, -self.vel, self.vel)
        self.items[self._ar, ai] = np.clip(p, ARENA_LO, ARENA_HI)
        self.age += 1
        self.E -= DRIFT
        dist = np.linalg.norm(self.items - self.pos[:, None, :], axis=-1)
        touch = dist < EAT_R
        ate = touch[self._ar, ai]
        hit = touch.copy(); hit[self._ar, ai] = False
        hit[ate] = False
        bb, kk = np.where(hit)
        if len(bb):
            self._place(bb, kk); self.respawn[bb, kk] = True
        self.dummy_hits += hit.sum(1)
        self.E = np.where(ate, self.E + FOOD_GAIN, self.E)
        self.E = np.clip(self.E, 0.0, 1.0)
        self.eaten += ate
        exp = ~ate & (self.age >= self.L_food)
        self.expired += exp
        self._new_round(ate | exp)
        self.E_sum += self.E

    def fitness(self):
        return self.eaten.copy()

    def mean_energy(self):
        return self.E_sum / self.L


class GeoBodies2:
    """Версия по дополнениям 1–2 (2026-10-05). Кандидаты: Kc = 2 — живой (снос с инерцией) и пустышка, дрожащая на месте
    без сноса (покадровый шаг того же распределения, что у живого: o_t ~ N(0, (U/2)²) на ось, отсчёт от «дома»), так что
    величина сдвига в одном кадре их не различает. Плюс n_lm неподвижных ориентиров известного непищевого вида
    (стоят всю жизнь, не цели, сквозь них можно пройти) с шумом датчиков × lm_noise — опора на мир; при n_lm = 0 опоры нет.
    Шаг за стену не выполняется, и тело это знает (exec_move). Поворот связи: с такта t_switch (из таблицы посева,
    равномерно 300–700) фактический поворот тела = rot_g × команда (−1 — зеркало, 2 — двойной); rot_g = None — без поворота.
    Столбцы датчиков: 0..Kc−1 — кандидаты, далее ориентиры. Таблицы кандидатов не зависят от n_lm (те же миры)."""

    def __init__(self, seeds, U, L_food, n_lm=0, lm_noise=1.0, rot_g=None, L=L, Kc=2):
        seeds = np.asarray(seeds)
        self.B, self.K, self.Kc, self.n_lm, self.U, self.L_food, self.L, self.rot_g = len(seeds), Kc + n_lm, Kc, n_lm, U, L_food, L, rot_g
        uniq, self.sidx = np.unique(seeds, return_inverse=True)
        rs = [np.random.RandomState(int(s)) for s in uniq]
        self.place_tab = np.stack([r.uniform(ARENA_LO, ARENA_HI, (MAXP, 2)) for r in rs])
        self.alive_tab = np.stack([r.randint(0, Kc, MAXR) for r in rs])
        self.walk_tab = np.stack([r.randn(L + 1, 2) for r in rs]).astype(np.float32)
        self.jit_tab = np.stack([r.randn(L + 1, Kc, 2) for r in rs]).astype(np.float32)
        nbc = np.stack([r.randn(L + 1, Kc) * SIG_B for r in rs]); ndc = np.stack([r.randn(L + 1, Kc) * SIG_D for r in rs])
        init = np.stack([np.r_[r.uniform(0.2, 0.8, 2), r.rand() * 2 * np.pi] for r in rs])
        self.t_switch_tab = np.array([r.randint(300, 701) for r in rs])
        rl = [np.random.RandomState(int(s) + 5_000_000) for s in uniq]          # ориентиры — отдельный поток
        lm = np.stack([r.uniform(ARENA_LO, ARENA_HI, (3, 2)) for r in rl])[:, :n_lm]
        nbl = np.stack([r.randn(L + 1, 3) for r in rl])[:, :, :n_lm] * SIG_B * lm_noise
        ndl = np.stack([r.randn(L + 1, 3) for r in rl])[:, :, :n_lm] * SIG_D * lm_noise
        self.nb_tab = np.concatenate([nbc, nbl], 2).astype(np.float32); self.nd_tab = np.concatenate([ndc, ndl], 2).astype(np.float32)
        self.pos = init[self.sidx, :2].copy(); self.head = init[self.sidx, 2].copy()
        self.t_switch = self.t_switch_tab[self.sidx]
        self._ar = np.arange(self.B)
        self.n_place = np.zeros(self.B, int); self.n_round = np.zeros(self.B, int)
        self.home = np.zeros((self.B, Kc, 2)); self.items = np.zeros((self.B, self.K, 2))
        self.items[:, Kc:] = lm[self.sidx]
        self.vel = np.zeros((self.B, 2))
        self.alive_i = np.zeros(self.B, int); self.age = np.zeros(self.B, int)
        self.respawn = np.ones((self.B, self.K), bool)
        self.E = np.full(self.B, E0); self.E_sum = np.zeros(self.B)
        self.eaten = np.zeros(self.B); self.expired = np.zeros(self.B); self.dummy_hits = np.zeros(self.B)
        self.eaten_after = np.zeros(self.B)
        self.t = 0; self.exec_move = np.zeros(self.B, int)
        self._new_round(np.ones(self.B, bool))

    _place = GeoBodies._place

    def _new_round(self, m):
        b = np.where(m)[0]
        if not len(b):
            return
        for k in range(self.Kc):
            self._place(b, np.full(len(b), k))
        self.home[b] = self.items[b, :self.Kc]
        self.alive_i[b] = self.alive_tab[self.sidx[b], np.minimum(self.n_round[b], MAXR - 1)]
        self.n_round[b] += 1
        self.vel[b] = 0.0; self.age[b] = 0
        self.respawn[b, :self.Kc] = True

    true_rel = GeoBodies.true_rel
    sense = GeoBodies.sense

    def gain(self):
        """Фактическое отображение команды поворота в поворот тела на текущем такте."""
        g = np.ones(self.B)
        if self.rot_g is not None:
            g = np.where(self.t >= self.t_switch, self.rot_g, 1.0)
        return g

    def step(self, turn, move):
        self.t += 1
        self.head = self.head + turn * TURN_G * self.gain()
        new = self.pos + (move * VMAX)[:, None] * np.stack([np.cos(self.head), np.sin(self.head)], 1)
        inside = ((new >= 0) & (new <= 1)).all(1)
        self.exec_move = np.where(inside, move, 0); self.pos = np.where(inside[:, None], new, self.pos)
        self.vel = RHO * self.vel + np.sqrt(1 - RHO ** 2) * self.U * self.walk_tab[self.sidx, self.t] / np.sqrt(2)
        ai = self.alive_i
        p = self.items[self._ar, ai] + self.vel
        out = (p < ARENA_LO) | (p > ARENA_HI)
        self.vel = np.where(out, -self.vel, self.vel)
        dum = self.home + self.U / 2 * self.jit_tab[self.sidx, self.t]                   # дрожь на месте, без сноса
        alive_new = np.clip(p, ARENA_LO, ARENA_HI)
        self.items[:, :self.Kc] = dum
        self.items[self._ar, ai] = alive_new
        self.home[self._ar, ai] = alive_new                                              # у живого «дом» не используется
        self.age += 1
        self.E -= DRIFT
        dist = np.linalg.norm(self.items[:, :self.Kc] - self.pos[:, None, :], axis=-1)
        touch = dist < EAT_R
        ate = touch[self._ar, ai]
        hit = touch.copy(); hit[self._ar, ai] = False; hit[ate] = False
        bb, kk = np.where(hit)
        if len(bb):
            self._place(bb, kk); self.home[bb, kk] = self.items[bb, kk]; self.respawn[bb, kk] = True
        self.dummy_hits += hit.sum(1)
        self.E = np.clip(np.where(ate, self.E + FOOD_GAIN, self.E), 0.0, 1.0)
        self.eaten += ate
        if self.rot_g is not None:
            self.eaten_after += ate & (self.t >= self.t_switch)
        exp = ~ate & (self.age >= self.L_food)
        self.expired += exp
        self._new_round(ate | exp)
        self.E_sum += self.E

    def fitness(self):
        return self.eaten.copy()

    def mean_energy(self):
        return self.E_sum / self.L


class GeoObs:
    """Г3 «назови живого» (2026-10-05): только восприятие, без касаний, еды и перестановок. Каждый посев — независимый раунд
    из W кадров. Наблюдатель движется по политике RND (поворот влево / нет / вправо — 0,25 / 0,5 / 0,25, движение — 0,7),
    слепой к тому, кто живой; шаг за стену не выполняется, и тело это знает (exec_move). Кандидаты: K (основное 2) —
    один живой (снос с инерцией ρ, скорость с самого начала из стационарного распределения) и K − 1 пустышек, дрожащих
    на месте с тем же покадровым распределением шага (как GeoBodies2). Ориентиры: n_lm неподвижных, шум × lm_noise.
    Поворот связи: с кадра t_switch (из таблицы посева, равномерно в [sw_lo, sw_hi]) фактический поворот = rot_g × команда.
    Траектория наблюдателя не зависит от метода, поэтому все методы считаются на одной записи.
    Таблицы кандидатов, действий и их шума не зависят от n_lm (те же миры при любом числе ориентиров).
    """

    def __init__(self, seeds, U, W, n_lm=0, lm_noise=1.0, K=2, rot_g=None, sw_lo=32, sw_hi=64):
        seeds = np.asarray(seeds); N = len(seeds)
        self.N, self.U, self.W, self.n_lm, self.K, self.Kt = N, U, W, n_lm, K, K + n_lm
        rs = [np.random.RandomState(int(s)) for s in seeds]
        init = np.stack([np.r_[r.uniform(0.2, 0.8, 2), r.rand() * 2 * np.pi] for r in rs])
        cand = np.stack([r.uniform(ARENA_LO, ARENA_HI, (3, 2)) for r in rs])[:, :K]
        alive = np.array([r.randint(0, K) for r in rs])
        vel0 = np.stack([r.randn(2) for r in rs]) * U / np.sqrt(2)
        walk = np.stack([r.randn(W + 1, 2) for r in rs]); jit = np.stack([r.randn(W + 1, 3, 2) for r in rs])[:, :, :K]
        nbc = np.stack([r.randn(W + 1, 3) for r in rs])[:, :, :K] * SIG_B; ndc = np.stack([r.randn(W + 1, 3) for r in rs])[:, :, :K] * SIG_D
        act = np.stack([r.rand(W, 2) for r in rs])
        t_sw = np.array([r.randint(sw_lo, sw_hi + 1) for r in rs])
        rl = [np.random.RandomState(int(s) + 5_000_000) for s in seeds]
        lm = np.stack([r.uniform(ARENA_LO, ARENA_HI, (3, 2)) for r in rl])[:, :n_lm]
        nbl = np.stack([r.randn(W + 1, 3) for r in rl])[:, :, :n_lm] * SIG_B * lm_noise
        ndl = np.stack([r.randn(W + 1, 3) for r in rl])[:, :, :n_lm] * SIG_D * lm_noise
        self.alive_i = alive; self.t_switch = t_sw
        cmd = np.where(act[..., 0] < 0.25, 1, np.where(act[..., 0] < 0.5, -1, 0)); move = (act[..., 1] < 0.7).astype(int)
        ar = np.arange(N)
        pos, head, vel = init[:, :2].copy(), init[:, 2].copy(), vel0.copy()
        home = cand.copy(); items = np.concatenate([cand, lm], 1)
        b_all, d_all, pos_all, head_all, items_all, mv_all, g_all = [], [], [], [], [], [], []
        for t in range(W + 1):
            if t > 0:
                g = np.where(t - 1 >= t_sw, rot_g, 1.0) if rot_g is not None else np.ones(N)
                head = head + cmd[:, t - 1] * TURN_G * g
                new = pos + (move[:, t - 1] * VMAX)[:, None] * np.stack([np.cos(head), np.sin(head)], 1)
                inside = ((new >= 0) & (new <= 1)).all(1)
                mv_all.append(np.where(inside, move[:, t - 1], 0)); g_all.append(g)
                pos = np.where(inside[:, None], new, pos)
                vel = RHO * vel + np.sqrt(1 - RHO ** 2) * U * walk[:, t] / np.sqrt(2)
                p = items[ar, alive] + vel
                out = (p < ARENA_LO) | (p > ARENA_HI)
                vel = np.where(out, -vel, vel)
                items[:, :K] = home + U / 2 * jit[:, t]
                items[ar, alive] = np.clip(p, ARENA_LO, ARENA_HI)
                home[ar, alive] = items[ar, alive]
            w = items - pos[:, None, :]
            c, s = np.cos(head)[:, None], np.sin(head)[:, None]
            rel = np.stack([c * w[..., 0] + s * w[..., 1], -s * w[..., 0] + c * w[..., 1]], -1)
            nb = np.concatenate([nbc[:, t], nbl[:, t]], 1); nd = np.concatenate([ndc[:, t], ndl[:, t]], 1)
            b_all.append(wrap(np.arctan2(rel[..., 1], rel[..., 0]) + nb))
            d_all.append(np.maximum(np.linalg.norm(rel, axis=-1) + nd, 1e-3))
            pos_all.append(pos.copy()); head_all.append(head.copy()); items_all.append(items.copy())
        self.b = np.stack(b_all, 1); self.d = np.stack(d_all, 1)                       # (N, W+1, Kt)
        self.rel = np.stack([self.d * np.cos(self.b), self.d * np.sin(self.b)], -1)     # (N, W+1, Kt, 2)
        self.cmd = cmd; self.exec_move = np.stack(mv_all, 1); self.gain = np.stack(g_all, 1)   # (N, W)
        self.pos = np.stack(pos_all, 1); self.head = np.stack(head_all, 1); self.items = np.stack(items_all, 1)
