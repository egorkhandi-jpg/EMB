"""Узкий вопрос C (2026-10-06): связывает ли замороженная модель смысл слова с опытом тела. Без обучения; бюджет среды не расходуется.
Пререгистрация — docs/RESULTS_2026-10-06_bind.md; слова — meaning_words.json (пулы A24), вопросы и предложения — bind_words.json (до расчёта).

Траектории: flip-2 без разворота (знак постоянен в эпизоде), слепая политика (рефлекс к ближайшему предмету любого вида), 10 наборов ×
90 эпизодов (1100000 + 1000·i + e). Условие эпизода — e mod 3: 0 правдивые (вид-еда — слово пула «еда», вид-яд — пула «вред»),
1 лживые (наоборот), 2 нейтральные (оба вида — разные нейтральные слова). Запросы — такты QT, где тело живо; окно 32 такта
(до начала эпизода — пустые кадры). n — число информативных событий в окне (съедено и исход «рост» или «падение»).
Интерфейсы: c — BOS + (слово вида | ' none', ' up' | ' down' | ' same') на такт; s — предложения на такт (bind_words.json).
Варианты: q1 / q2 / q3 — вопросы после окна; q1_sh — q1 на окне с независимо перемешанными слотами вида и исхода по кадрам
(связка «съел → исход» разрушена, счёт видов и исходов сохранён).
Чтение: логиты слов видов A и B эпизода и 11 прямых слов на последней позиции. Одинаковые входы (окна с n = 0) считаются один раз.
Случайная модель — та же архитектура, сеть от посева набора, эмбеддинги приведены к средней норме предобученных.
Меры по набору — measures(); сводка и семья — summarize_bind.py.

python meaning_bind.py --model models/pythia-1b --bs 16
"""
import argparse, json, os, time
import numpy as np
import torch

from env_flip import FlipBodies, T
import stand_words as sw
from stand_gpu import git_info

HERE = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(HERE, 'meaning_words.json'), encoding='utf-8'))
BW = json.load(open(os.path.join(HERE, 'bind_words.json'), encoding='utf-8'))
NONE, UP, DOWN, SAME, BOS = 5293, 598, 1066, 1072, 0
N = 32
QT = [3, 7, 11, 15, 23, 31, 47, 63, 79, 95]
EPISODES = 90
VARIANTS = ['q1', 'q2', 'q3', 'q1_sh']
IFACES = ['c', 's']
NSTRATA = [2, 3, 4, 5]                     # 5 — «5 и больше»


def tok1(tk, w):
    ids = tk.encode(' ' + w)
    assert len(ids) == 1, f'слово {w!r} не однотокенное: {ids}'
    return ids[0]


def trajectories(i):
    """Набор i: по эпизодам — вид съеденного (0 A, 1 B, 2 нет), исход (0 рост, 1 падение, 2 нет), жив ли, какой вид еда."""
    seeds = [1100000 + 1000 * i + e for e in range(EPISODES)]
    b = FlipBodies(EPISODES, seeds, False, 'flip-2')
    K, E, AL = [], [], []
    for t in range(T):
        s = b.sense()
        kind, en = sw.events(s)
        K.append(kind); E.append(en); AL.append(b.alive.copy())
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(EPISODES)], 1)); act[~b.alive] = 0
        b.step(act)
    return np.stack(K, 1), np.stack(E, 1), np.stack(AL, 1), b.good_A.copy()


def episode_words(i, good_A):
    """Слова видов A и B по эпизодам и условие (0 правдивые, 1 лживые, 2 нейтральные)."""
    rng = np.random.RandomState(7000 + i)
    cond = np.arange(EPISODES) % 3
    wA, wB = [], []
    for e in range(EPISODES):
        f, h = W['food'][rng.randint(30)], W['harm'][rng.randint(30)]
        n1, n2 = rng.choice(30, 2, replace=False); n1, n2 = W['neutral'][n1], W['neutral'][n2]
        if cond[e] == 2:
            a, b_ = n1, n2
        else:
            food_word_on_food_kind = cond[e] == 0
            wfk, wpk = (f, h) if food_word_on_food_kind else (h, f)     # слово вида-еды, слово вида-яда
            a, b_ = (wfk, wpk) if good_A[e] else (wpk, wfk)
        wA.append(a); wB.append(b_)
    return wA, wB, cond


def frames(K, E, e, t, rng=None):
    """Кадры окна (вид, исход) с дополнением до начала эпизода; rng — независимая перестановка слотов вида и исхода."""
    fr = [(K[e, u], E[e, u]) if u >= 0 else (2, 2) for u in range(t - N + 1, t + 1)]
    if rng is not None:
        ks, es = [f[0] for f in fr], [f[1] for f in fr]
        ks = [ks[j] for j in rng.permutation(N)]; es = [es[j] for j in rng.permutation(N)]
        fr = list(zip(ks, es))
    return fr


def seq_c(tk, fr, tA, tB, q):
    ids = [BOS]
    for k, en in fr:
        ids += [(tA, tB, NONE)[k], (UP, DOWN, SAME)[en]]
    return ids + tk.encode(q)


def seq_s(tk, fr, wA, wB, q):
    S = BW['s_sentences']; parts = []
    for k, en in fr:
        if k < 2:
            parts.append(S['eat'].replace('{w}', ' ' + (wA, wB)[k]))
            parts.append((S['up'], S['down'], S['same_after_eat'])[en])
        else:
            parts.append((S['up'], S['down'], S['nothing'])[en])
    return [BOS] + tk.encode(' '.join(parts) + q)


def build(tk, i, K, E, AL, good_A, wA, wB, cond):
    meta, seqs = [], {v: {f: [] for f in IFACES} for v in VARIANTS}
    for e in range(EPISODES):
        tA, tB = tok1(tk, wA[e]), tok1(tk, wB[e])
        for t in QT:
            if not AL[e, t]:
                continue
            fr = frames(K, E, e, t)
            inf = [(k, en) for k, en in fr if k < 2 and en < 2]
            nA = sum(1 for k, _ in fr if k == 0); nB = sum(1 for k, _ in fr if k == 1)
            evid = sum(1 if (k == 0) == (en == 0) else -1 for k, en in inf)          # байесовский ориентир: + за «A — еда»
            meta.append((e, t, int(cond[e]), len(inf), nA, nB, evid, int(good_A[e]), tA, tB))
            rng = np.random.RandomState(100000 * i + 1000 * e + t)
            fsh = frames(K, E, e, t, rng)
            for v in VARIANTS:
                q = BW['questions'][v.replace('_sh', '')]; f_ = fsh if v.endswith('_sh') else fr
                seqs[v]['c'].append(seq_c(tk, f_, tA, tB, q))
                seqs[v]['s'].append(seq_s(tk, f_, wA[e], wB[e], q))
    return np.array(meta), seqs


def load(model, weights, dev, seed=None, pre_norm=None):
    from transformers import AutoModelForCausalLM, AutoConfig
    if weights == 'pretrained':
        # сразу на устройство (без полной копии в ОЗУ: у экземпляров бывает 30 ГБ при 28 ГБ весов 6.9B в fp32); fp16 → fp32 точно
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, device_map=dev)
    else:
        torch.manual_seed(seed)
        with torch.device(dev):
            m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model), dtype=torch.float32)
        with torch.no_grad():
            Em = m.get_input_embeddings().weight; Em.mul_(pre_norm / Em.norm(dim=1).mean())
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def read_logits(m, seqs, cols, dev, bs):
    """Логиты на последней позиции: cols — (число входов, k) номеров токенов на вход. Одинаковые входы — один раз."""
    uniq, inv = {}, []
    for s in seqs:
        inv.append(uniq.setdefault(tuple(s), len(uniq)))
    U = list(uniq); order = sorted(range(len(U)), key=lambda j: len(U[j]))
    H = torch.zeros(len(U), m.config.hidden_size, device=dev)
    for a in range(0, len(order), bs):
        idx = order[a:a + bs]; L = max(len(U[j]) for j in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        last = torch.tensor([len(U[j]) - 1 for j in idx])
        for r, j in enumerate(idx):
            x[r, :len(U[j])] = torch.tensor(U[j])                               # дополнение справа: причинное внимание его не видит
        with torch.no_grad():
            h = m.gpt_neox(input_ids=x.to(dev)).last_hidden_state[torch.arange(len(idx)), last.to(dev)]
        if not torch.isfinite(h).all():
            raise FloatingPointError('нечисловое скрытое состояние')
        H[idx] = h
    Wout = m.get_output_embeddings().weight
    inv = torch.as_tensor(inv, device=dev); C = torch.as_tensor(cols, device=dev)
    lg = torch.einsum('nd,nkd->nk', H[inv], Wout[C])
    return lg.cpu().numpy(), len(U)


# ---------------- меры по набору ----------------
def auc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float); p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0:
        return float('nan')
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def measures(meta, la, lb, ldir, food_ids):
    """meta — build(); la, lb — логиты слов видов A и B; ldir — логиты прямых слов (5 пищевых, затем 6 вредных)."""
    e, t, cond, n, nA, nB, evid, gA = (meta[:, j] for j in range(8))
    tA = meta[:, 8]
    a_food_word = np.isin(tA, food_ids)                                          # слово вида A — из пула «еда»
    a = np.where(a_food_word, la - lb, lb - la)                                  # «слово еды» − «слово вреда» (для правдивых и лживых)
    d = np.where(gA == 1, la - lb, lb - la)                                      # к истинной еде
    r = {}
    m0 = (cond < 2) & (n == 0)
    r['M'] = float((a[m0] > 0).mean()) if m0.any() else float('nan'); r['n_M'] = int(m0.sum())
    pairs = ldir[m0][:, :5, None] - ldir[m0][:, None, 5:]
    r['M_direct'] = float((pairs > 0).mean()) if m0.any() else float('nan')
    ne = (cond == 2) & (n >= 1)
    r['E'] = auc((la - lb)[ne], gA[ne]); r['F'] = auc((nA - nB)[ne], gA[ne]); r['bayes'] = auc(evid[ne], gA[ne]); r['n_E'] = int(ne.sum())
    num_raw = num_auc = den_raw = den_auc = 0.0
    for k in NSTRATA:
        sel = (n == k) if k < NSTRATA[-1] else (n >= k)
        tr, li = a[sel & (cond == 0)], a[sel & (cond == 1)]
        if len(tr) == 0 or len(li) == 0:
            continue
        wr = len(tr) * len(li) / (len(tr) + len(li)); wa = len(tr) * len(li)
        num_raw += wr * (tr.mean() - li.mean()); den_raw += wr
        num_auc += wa * auc(np.concatenate([tr, li]), np.r_[np.ones(len(tr)), np.zeros(len(li))]); den_auc += wa
    r['B'] = float(num_raw / den_raw) if den_raw else float('nan')
    r['B_auc'] = float(num_auc / den_auc) if den_auc else float('nan')
    r['n_B'] = int(((cond < 2) & (n >= 2)).sum())
    r['B_by_n'], r['outweigh_by_n'], r['M_by_cond'] = {}, {}, {}
    for k, nm in ((1, '1'), (2, '2'), (3, '3'), (4, '4+')):
        sel = (n == k) if k < 4 else (n >= 4)
        tr, li = a[sel & (cond == 0)], a[sel & (cond == 1)]
        r['B_by_n'][nm] = float(tr.mean() - li.mean()) if len(tr) and len(li) else None
        r['outweigh_by_n'][nm] = float((d[sel & (cond == 1)] > 0).mean()) if len(li) else None   # в лживых ответ — к опыту
    for c_, nm in ((0, 'truthful'), (1, 'lying')):
        s_ = (cond == c_) & (n == 0)
        r['M_by_cond'][nm] = float((a[s_] > 0).mean()) if s_.any() else None
    r['n_hist'] = {str(k): int((n == k).sum()) for k in range(4)} | {'3+': int((n >= 3).sum())}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/pythia-1b')
    ap.add_argument('--sets', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--weights', nargs='+', default=['pretrained', 'random'])
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--out', default='results_bind')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    dev = a.device; commit, dirty = git_info(); t0 = time.time()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    os.makedirs(a.out, exist_ok=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    for k in ('food', 'harm', 'neutral', 'direct_food', 'direct_harm'):
        for w in W[k]:
            tok1(tk, w)                                                          # однотокенность для токенизатора этой модели
    dir_ids = [tok1(tk, w) for w in W['direct_food'] + W['direct_harm']]
    food_ids = [tok1(tk, w) for w in W['food']]
    tag = os.path.basename(a.model.rstrip('/')).split('-')[-1]
    path = os.path.join(a.out, f'bind_{tag}.json')
    res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'model': a.model, 'runs': [], 'sets': {}}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'sets': a.sets, 'weights': a.weights})
    data = {i: (lambda tr: (tr, episode_words(i, tr[3])))(trajectories(i)) for i in a.sets}
    built = {i: build(tk, i, *data[i][0], *data[i][1]) for i in a.sets}
    pre_norm = None
    for wts in a.weights:
        m = None
        for i in a.sets:
            key = f'{wts}/{i}'
            if key in res['sets']:
                continue
            if wts == 'pretrained' and m is None:
                m = load(a.model, 'pretrained', dev)
            if wts == 'random':
                if pre_norm is None:
                    pre_norm = res.get('pre_norm')
                if pre_norm is None:
                    mp = load(a.model, 'pretrained', 'cpu'); pre_norm = mp.get_input_embeddings().weight.norm(dim=1).mean().item(); del mp
                del m; torch.cuda.empty_cache() if dev == 'cuda' else None
                m = load(a.model, 'random', dev, seed=i, pre_norm=pre_norm)
            if wts == 'pretrained':
                res['pre_norm'] = m.get_input_embeddings().weight.norm(dim=1).mean().item()
            meta, seqs = built[i]; ts = time.time()
            cols = np.concatenate([meta[:, 8:10], np.tile(dir_ids, (len(meta), 1))], 1)
            out, raw, nu = {}, {}, 0
            for v in VARIANTS:
                out[v] = {}
                for f in IFACES:
                    lg, u = read_logits(m, seqs[v][f], cols, dev, a.bs); nu += u
                    out[v][f] = measures(meta, lg[:, 0], lg[:, 1], lg[:, 2:], food_ids)
                    raw[f'{v}_{f}'] = lg.astype(np.float32)
            res['sets'][key] = {'measures': out, 'n_windows': int(len(meta)), 'n_unique_inputs': nu, 'sec': round(time.time() - ts),
                                'git_commit': commit, 'git_dirty': dirty}
            np.savez_compressed(os.path.join(a.out, f'raw_{tag}_{wts}_{i}.npz'), meta=meta, **raw)
            json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
            print(f"{time.strftime('%H:%M:%S')} {tag} {wts} набор {i}: окон {len(meta)}, входов {nu}, {time.time() - ts:.0f} с (всего {time.time() - t0:.0f} с)", flush=True)
        del m; torch.cuda.empty_cache() if dev == 'cuda' else None


if __name__ == '__main__':
    main()
