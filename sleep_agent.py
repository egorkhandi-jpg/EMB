"""Настоящий агент в среде со сном S2 (env_sleep2.py) — сетка порогов сна без эволюции (решение Егора 2026-10-04).
Пререгистрация — docs/RESULTS_2026-10-04_sleep.md, раздел «Настоящий агент в S2».

Агент:
  LM — Pythia-160M (предобученная), fp32, вход: BOS + окно 32 такта × (символ съеденного вида | ' none', символ энергии
       ' up' / ' down' / ' same') + символ давления сна (по порогам 1 / 3 / 6) + символы двух присутствующих видов
       (гнездо 0, гнездо 1). Символы видов — однотокенные слова из пула 24, у каждой жизни своя перестановка пула;
       пул не пересекается с символами событий и давления.
  Быстрая память — только окно LM (основная конфигурация); вариант — быстрые веса с затуханием τ.
  Медленная голова — линейная по стандартизованному последнему скрытому состоянию: logit > 0 — гнездо 0, < 0 — гнездо 1,
       = 0 (не обучена) — к ближайшему предмету любого вида. Учится ТОЛЬКО во сне проигрыванием буфера: нормированный
       шаг по мини-пакетам 32, η = 1,9996 (правило A17; устойчиво при 0 < η < 2, правило 12), число проходов —
       min(k, 10): один проход на такт сна. Метки — тело: «гнездо 0 — пища» по последнему исходу.
  Буфер — пары (z, метка) бодрствующих тактов с меткой, не старше B_max = 200 тактов; проигранные во сне удаляются.
  Сон — правило: давление (съедено с последнего сна) ≥ θ → спать k тактов. «Бесплатный сон» — то же проигрывание
       (10 проходов) мгновенно и без сна. Оракул — знает знак, LM не нужна.

python sleep_agent.py --theta 6 --k 5 [--lives 200] [--tau 0] [--policy pressure|never|free|oracle|online]
«онлайн без сна» (дополнение 2026-10-04): сна нет, медленная голова учится на каждом бодрствующем такте по метке тела
нормированным правилом A11 (η = 0,3 из подбора A11; устойчиво при 0 < η < 2).
"""
import argparse, json, math, os, time
import numpy as np
import torch

import env_flip as ef
from env_sleep2 import Sleep2Bodies
import stand_words as sw
from stand_gpu import git_info

POOL = [28725, 16583, 11281, 11472, 18553, 9729, 11996, 22676, 36608, 23656, 5798, 2829,
        18067, 9005, 8281, 6951, 3369, 8805, 29514, 10329, 3497, 20098, 23465, 11129]   # candle … mountain (однотокенные)
NONE, UP, DOWN, SAME, BOS = 5293, 598, 1066, 1072, sw.BOS
PRESS = [6146, 9268, 6420, 8935]                     # ' north' ' east' ' south' ' west' — давление 0 | 1–2 | 3–5 | ≥ 6 (проверено токенизатором)
ETA, MB, BMAX, WIN = 1.999609452367605, 32, 200, 32
ETA_ONLINE = 0.3          # «онлайн без сна»: η A11 для pretrained c, all/full (results_bodyhead2/tune_pretrained_c_all_full.json, замороженная голова на посевах 101–103)


class SleepLM:
    def __init__(self, weights, device, net_seed=1, chunk=200):
        from transformers import AutoModel, AutoConfig
        name = 'EleutherAI/pythia-160m'
        if weights == 'random':
            pre = AutoModel.from_pretrained(name, dtype=torch.float32); pn = pre.get_input_embeddings().weight.norm(dim=1).mean().item(); del pre
            torch.manual_seed(net_seed); self.lm = AutoModel.from_config(AutoConfig.from_pretrained(name)).to(torch.float32)
            with torch.no_grad():
                E = self.lm.get_input_embeddings().weight; E.mul_(pn / E.norm(dim=1).mean())
        else:
            self.lm = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.E = self.lm.get_input_embeddings().weight; self.device = device; self.chunk = chunk
        self.pool = torch.tensor(POOL, device=device)

    def reset(self, seeds):
        B = len(seeds)
        self.perm = torch.as_tensor(np.stack([np.random.RandomState(s + 3_000_000).permutation(len(POOL)) for s in seeds]), device=self.device)
        self.win = torch.full((B, WIN, 2), NONE, device=self.device, dtype=torch.long); self.win[:, :, 1] = SAME

    def symtok(self, sym):
        sym = torch.as_tensor(sym, device=self.device)
        return self.pool[self.perm.gather(1, (sym % len(POOL)).long())]

    def features(self, s, sym):
        B = len(s)
        a0, a1 = s[:, 10] > 0, s[:, 11] > 0
        st = self.symtok(sym)                                                   # (B, 2) токены видов гнёзд
        kt = torch.full((B,), NONE, device=self.device, dtype=torch.long)
        kt = torch.where(torch.as_tensor(a0, device=self.device), st[:, 0], kt)
        kt = torch.where(torch.as_tensor(a1, device=self.device), st[:, 1], kt)
        et = torch.full((B,), SAME, device=self.device, dtype=torch.long)
        et[torch.as_tensor(s[:, 7] > 0.5, device=self.device)] = UP; et[torch.as_tensor(s[:, 7] < -0.5, device=self.device)] = DOWN
        self.win = torch.cat([self.win[:, 1:], torch.stack([kt, et], 1)[:, None]], 1)
        pb = torch.as_tensor(np.searchsorted(np.array([1, 3, 6]), s[:, 12], side='right'), device=self.device)
        ids = torch.cat([torch.full((B, 1), BOS, device=self.device, dtype=torch.long), self.win.reshape(B, -1),
                         torch.tensor(PRESS, device=self.device)[pb][:, None], st], 1)
        out = []
        for i in range(0, B, self.chunk):
            with torch.no_grad():
                h = self.lm(inputs_embeds=self.E[ids[i:i + self.chunk]]).last_hidden_state[:, -1]
            if not torch.isfinite(h).all():
                raise FloatingPointError('нечисловое скрытое состояние')
            out.append(h)
        return torch.cat(out)


def fit_std(feat, device, n=40, T=200):
    b = Sleep2Bodies(n, [555000 + k for k in range(n)], T); feat.reset([555000 + k for k in range(n)]); H = []
    rng = np.random.RandomState(0)
    for _ in range(T):
        H.append(feat.features(b.sense(), b.sym)); b.step(rng.uniform(-1, 1, (n, 2)))
    H = torch.cat(H); return H.mean(0), H.std(0) + 1e-6


def steer(s, ch0):
    a = ef._steer(s, 0); b = ef._steer(s, 1); a[~ch0] = b[~ch0]; return a


def run(feat, mu, sd, policy, theta, k, lives, L, device, tau=0, seed0=50000, metab=0.3):
    B = lives; seeds = [seed0 + i for i in range(B)]
    bd = Sleep2Bodies(B, seeds, L, sleep_metab=metab)
    use_lm = policy != 'oracle'
    D = mu.shape[0] if use_lm else 1
    w = torch.zeros(B, D, device=device); bb = torch.zeros(B, device=device); wf = torch.zeros(B, D, device=device)
    bufz = torch.zeros(B, BMAX, D, device=device); bufy = torch.zeros(B, BMAX, device=device)
    buft = torch.full((B, BMAX), -10 ** 9, device=device); bufv = torch.zeros(B, BMAX, dtype=torch.bool, device=device)
    belief0 = np.zeros(B, bool); has_b = np.zeros(B, bool); press = np.zeros(B, int)
    corr = np.zeros(B); cnt = np.zeros(B); corr_slow = np.zeros(B)
    season_acc = {}                                                             # (сезон) -> [верно, всего] медленной головы
    gen = torch.Generator(device='cpu').manual_seed(seed0 + 99)
    decay = math.exp(-1 / tau) if tau else 0.0
    if use_lm:
        feat.reset(seeds)

    def replay(rows, passes, t):
        if len(rows) == 0 or passes == 0:
            return
        r = torch.as_tensor(rows, device=device)
        Z, Y = bufz[r], bufy[r]; M = (bufv[r] & ((t - buft[r]) < BMAX)).float()
        N2 = (Z * Z).sum(-1) + 1
        for _ in range(passes):
            perm = torch.randperm(BMAX, generator=gen).to(device)
            for j in range(0, BMAX, MB):
                ix = perm[j:j + MB]; m = M[:, ix]
                if not m.any():
                    continue
                z = Z[:, ix]; e = (Y[:, ix] - torch.sigmoid((z * w[r][:, None]).sum(-1) + bb[r][:, None])) * m
                nrm = (m * N2[:, ix]).sum(1).clamp_min(1.0)
                w[r] += (ETA / nrm)[:, None] * (e[:, :, None] * z).sum(1); bb[r] += ETA * e.sum(1) / nrm
        bufv[r] = False                                                          # проигранное удаляется

    for t in range(L):
        s = bd.sense()
        a0, a1 = s[:, 10] > 0, s[:, 11] > 0; ev = a0 | a1
        food = s[:, 7] > -0.5
        g0 = np.where(a0, food, ~food)
        belief0 = np.where(ev, g0, belief0); has_b |= ev; press += ev
        awake = bd.sleep_left == 0
        if use_lm:
            z = (feat.features(s, bd.sym) - mu) / sd
            lab = torch.as_tensor(has_b & awake, device=device)
            yb = torch.as_tensor(belief0, dtype=torch.float32, device=device)
            slot = t % BMAX
            bufz[:, slot] = torch.where(lab[:, None], z, bufz[:, slot]); bufy[:, slot] = torch.where(lab, yb, bufy[:, slot])
            buft[:, slot] = torch.where(lab, torch.tensor(float(t), device=device), torch.tensor(-1e9, device=device))
            bufv[:, slot] = lab
            if policy == 'online':                                            # медленная голова учится на каждом бодрствующем такте (A11)
                err = (yb - torch.sigmoid((w * z).sum(1) + bb)) * lab.float()
                g = ETA_ONLINE * err / ((z * z).sum(1) + 1)
                w += g[:, None] * z; bb += g
            if tau:
                wf.mul_(decay)
                err = (yb - torch.sigmoid(((w + wf) * z).sum(1) + bb)) * lab.float()
                wf += (0.3 * err / ((z * z).sum(1) + 1))[:, None] * z
            lg = ((w + wf) * z).sum(1) + bb
            lg_slow = ((w * z).sum(1) + bb).cpu().numpy()
            lgn = lg.cpu().numpy()
            blind = s[:, 2] <= s[:, 5]
            ch0 = np.where(lgn > 0, True, np.where(lgn < 0, False, blind))
            sl0 = np.where(lg_slow > 0, True, np.where(lg_slow < 0, False, blind))
        else:
            ch0 = bd.good0.copy(); sl0 = ch0
        # сон
        if policy in ('pressure', 'free') and theta is not None:
            trig = awake & (press >= theta)
            if trig.any():
                rows = np.nonzero(trig)[0]
                if policy == 'free':
                    replay(rows, 10, t)
                else:
                    bd.start_sleep(trig, k); replay(rows, min(k, 10), t)
                press[rows] = 0
        aw = (bd.sleep_left == 0) & has_b
        corr += (ch0 == bd.good0) & aw; corr_slow += (sl0 == bd.good0) & aw; cnt += aw
        sn = bd.n_season
        for sv in np.unique(sn):
            m = aw & (sn == sv)
            if m.any():
                c = season_acc.setdefault(int(sv), [0, 0]); c[0] += int((sl0 == bd.good0)[m].sum()); c[1] += int(m.sum())
        bd.step(steer(s, ch0))
    f = bd.fitness()
    return {'f': f, 'food': float(bd.n_good.mean()), 'poison': float(bd.n_bad.mean()), 'floor_frac': float((bd.t_floor / L).mean()),
            'sleep_frac': float((bd.t_sleep / L).mean()), 'n_sleeps': float(bd.n_sleeps.mean()),
            'acc_awake': float((corr / np.maximum(cnt, 1)).mean()), 'acc_slow': float((corr_slow / np.maximum(cnt, 1)).mean()),
            'acc_slow_by_season': {k_: v[0] / max(v[1], 1) for k_, v in sorted(season_acc.items())}, 'seasons': float(bd.n_season.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['pressure', 'never', 'free', 'oracle', 'online'], default='pressure')
    ap.add_argument('--theta', type=int, default=None)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--tau', type=float, default=0)
    ap.add_argument('--lives', type=int, default=200)
    ap.add_argument('--L', type=int, default=1500)
    ap.add_argument('--weights', default='pretrained')
    ap.add_argument('--out', default='results_sleep_agent')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    commit, dirty = git_info(); t0 = time.time()
    feat = SleepLM(a.weights, a.device) if a.policy != 'oracle' else None
    mu, sd = fit_std(feat, a.device) if feat else (torch.zeros(1), torch.ones(1))
    r = run(feat, mu, sd, a.policy, a.theta, a.k, a.lives, a.L, a.device, a.tau)
    tag = f"{a.policy}_th{a.theta}_k{a.k}" + (f'_tau{int(a.tau)}' if a.tau else '') + (f'_{a.weights}' if a.weights != 'pretrained' else '')
    os.makedirs(a.out, exist_ok=True)
    json.dump({'tag': tag, 'args': vars(a), 'f': r['f'].tolist(), **{k_: v for k_, v in r.items() if k_ != 'f'}, 'f_mean': float(r['f'].mean()),
               'git_commit': commit, 'git_dirty': dirty, 'sec': round(time.time() - t0)},
              open(os.path.join(a.out, f'{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"{tag}: F {r['f'].mean():.4f} | сон {r['sleep_frac']:.3f} | выбор {r['acc_awake']:.3f} (медленная {r['acc_slow']:.3f}) | пища/яд {r['food']:.1f}/{r['poison']:.1f} | "
          f"на нуле {r['floor_frac']:.3f} | {time.time() - t0:.0f} с", flush=True)


if __name__ == '__main__':
    main()
