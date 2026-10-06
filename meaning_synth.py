"""Последняя проверка для варианта C (A27, 2026-10-06): связывает ли замороженная модель смысл слова с опытом тела —
на синтетических окнах без частотного канала (без симуляции тел, без отбора выживших). Без обучения; бюджет среды не расходуется.
Пререгистрация — docs/RESULTS_2026-10-06_bind.md, раздел A27; пулы слов — meaning_words.json (A24), предложения и вопросы — bind_words.json.

Окно — 32 такта. В окне n ∈ {2, 4, 6} информативных событий на случайных разных тактах, ровно n/2 каждого вида (поровну упоминаний).
Вид F — еда (после него «up»), вид P — яд («down»). Порядок видов среди событий случаен, вид последнего и вид первого события — ровно поровну F и P
в каждой ячейке (условие × n). Условия: 0 правдивые (слово «еды» на F, «вреда» на P), 1 лживые (наоборот), 2 нейтральные (два разных
нейтральных слова). На набор — 3 условия × 3 n × 72 окна = 648 (72 = 8 × 9 — точный баланс сочетаний «вид последнего × опорный × вид первого события»;
при n = 2 первый вид противоположен последнему и сбалансирован через него).
Переставленные исходы (_perm): те же окна (слова, позиции, порядок видов), исходы перераспределены между событиями при прежнем числе
«up» и «down» так, что связка вид → исход неинформативна: n = 4 — у каждого вида один «up» и один «down»; n = 2 и n = 6 — перевес
в одно событие, его знак (к F или к P) ровно поровну в каждой ячейке.
Опорный вид окна назначен крест-накрест с видами последнего и первого события (по 9 окон на сочетание в ячейке), не по порядку; оценка — l(слово опорного) − l(слово другого), метка — «опорный — еда»;
d — к истинной еде: l(слово F) − l(слово P).

python meaning_synth.py --check                         # проверка конструкции по самим окнам (без модели)
python meaning_synth.py --model models/pythia-6.9b --ifaces s c --bs 16
"""
import argparse, json, os, time
import numpy as np
import torch

from meaning_bind import W, BW, BOS, NONE, UP, DOWN, SAME, tok1, load, auc
from stand_gpu import git_info

N, NS, PER = 32, (2, 4, 6), 72
CONDS = ('truthful', 'lying', 'neutral')


def windows(i):
    """Набор i: список окон — словари с кадрами (вид 0 F / 1 P / 2 нет, исход 0 up / 1 down / 2 нет) в двух версиях и словами."""
    rng = np.random.RandomState(31000 + i)
    out = []
    for c in range(3):
        for n in NS:
            cell = np.arange(PER) % 8; rng.shuffle(cell)                       # 8 × 9: последний × опорный × первый вид
            lastF, refF, firstF = cell % 2 == 0, (cell // 2) % 2 == 0, cell // 4 == 0
            tilt = np.r_[np.ones(PER // 2, bool), np.zeros(PER // 2, bool)]; rng.shuffle(tilt)   # знак перевеса в _perm (n = 2, 6)
            for k in range(PER):
                kinds = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
                while True:
                    rng.shuffle(kinds)
                    if (kinds[-1] == 0) == lastF[k] and (n == 2 or (kinds[0] == 0) == firstF[k]):   # при n = 2 первый — не последний
                        break
                pos = np.sort(rng.choice(N, n, replace=False))
                outc = kinds.copy()                                               # F → up (0), P → down (1)
                # переставленные исходы: n/2 «up» и n/2 «down», у вида F — m «up»
                m = (n // 2) // 2 + ((n // 2) % 2) * int(tilt[k]) if n != 4 else 1
                if n == 2:
                    m = int(tilt[k])                                              # 1 — как было (F «up»), 0 — наоборот
                iF, iP = np.where(kinds == 0)[0], np.where(kinds == 1)[0]
                po = np.empty(n, int)
                uF = rng.permutation(len(iF)) < m; po[iF] = np.where(uF, 0, 1)
                uP = rng.permutation(len(iP)) < (n // 2 - m); po[iP] = np.where(uP, 0, 1)
                f, h = W['food'][rng.randint(30)], W['harm'][rng.randint(30)]
                a_, b_ = rng.choice(30, 2, replace=False)
                wF, wP = {0: (f, h), 1: (h, f), 2: (W['neutral'][a_], W['neutral'][b_])}[c]
                fr = [(2, 2)] * N; frp = [(2, 2)] * N
                for j, t in enumerate(pos):
                    fr[t] = (kinds[j], outc[j]); frp[t] = (kinds[j], po[j])
                out.append({'cond': c, 'n': n, 'fr': fr, 'frp': frp, 'wF': wF, 'wP': wP, 'ref': 0 if refF[k] else 1, 'first': int(kinds[0]), 'last': int(kinds[-1]),
                            'mentF': int((kinds == 0).sum()), 'mentP': int((kinds == 1).sum()),
                            'evid': int(sum(1 if (k_ == 0) == (o == 0) else -1 for k_, o in zip(kinds, outc))),
                            'evid_p': int(sum(1 if (k_ == 0) == (o == 0) else -1 for k_, o in zip(kinds, po)))})
    return out


def check(sets):
    """Проверка конструкции (п. 2 программы): ориентиры по самим окнам, по набору; метка — «опорный вид — еда»."""
    rows = []
    for i in sets:
        ws = windows(i)
        y = np.array([w['ref'] == 0 for w in ws])
        sgn = np.where(y, 1, -1)                                                   # перевод «к F» в «к опорному»
        freq = np.array([(w['mentF'] - w['mentP']) for w in ws]) * sgn
        last = np.array([1.0 if w['last'] == w['ref'] else -1.0 for w in ws])
        first = np.array([1.0 if w['first'] == w['ref'] else -1.0 for w in ws])
        ev = np.array([w['evid'] for w in ws]) * sgn; evp = np.array([w['evid_p'] for w in ws]) * sgn
        ups_same = all(sum(o == 0 for _, o in w['fr'] if o < 2) == sum(o == 0 for _, o in w['frp'] if o < 2) for w in ws)
        same_words = all([k for k, _ in w['fr']] == [k for k, _ in w['frp']] for w in ws)
        lastF = {(c, n): np.mean([w['last'] == 0 for w in ws if w['cond'] == c and w['n'] == n]) for c in range(3) for n in NS}
        rows.append((auc(freq, y), auc(last, y), auc(ev, y), auc(evp, y), np.mean(y), ups_same, same_words, min(lastF.values()), max(lastF.values()), len(ws), auc(first, y)))
    R = np.array([r[:5] + r[7:] for r in rows], float)   # столбцы: 0–4, мин, макс, окон, первый
    print('набор: частотный | последний упомянутый | байес по связке | байес на переставленных | доля «опорный — еда» | число up прежнее | виды и позиции прежние | доля последнего F по ячейкам (мин–макс) | окон | первый упомянутый')
    for i, r in zip(sets, rows):
        print(f'  {i:2d}: {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]} | {r[6]} | {r[7]:.2f}–{r[8]:.2f} | {r[9]} | {r[10]:.3f}')
    print(f'  все: частотный {R[:, 0].mean():.3f}, последний {R[:, 1].min():.3f}–{R[:, 1].max():.3f}, байес {R[:, 2].mean():.3f}, '
          f'байес перест. {R[:, 3].min():.3f}–{R[:, 3].max():.3f}, опорный — еда {R[:, 4].min():.3f}–{R[:, 4].max():.3f}, первый {R[:, 8].min():.3f}–{R[:, 8].max():.3f}')
    return rows


def seq(tk, fr, wF, wP, q, iface):
    if iface == 'c':
        ids = [BOS]
        for k, o in fr:
            ids += [(tok1(tk, wF), tok1(tk, wP), NONE)[k], (UP, DOWN, SAME)[o]]
        return ids + tk.encode(q)
    S = BW['s_sentences']; parts = []
    for k, o in fr:
        if k < 2:
            parts += [S['eat'].replace('{w}', ' ' + (wF, wP)[k]), (S['up'], S['down'])[o]]
        else:
            parts.append(S['nothing'])
    return [BOS] + tk.encode(' '.join(parts) + q)


def logits(m, seqs, cols, dev, bs):
    """Логиты слов cols (на вход) на последней позиции; одинаковые входы — один раз."""
    from meaning_bind import read_logits
    return read_logits(m, seqs, cols, dev, bs)[0]


VAR = {'s': ['q1', 'q1_perm', 'q2', 'q3'], 'c': ['q1', 'q1_perm']}


def measures(ws, lg_ref, lg_oth):
    """lg_ref, lg_oth — логиты слов опорного и другого вида."""
    y = np.array([w['ref'] == 0 for w in ws]); cond = np.array([w['cond'] for w in ws]); n = np.array([w['n'] for w in ws])
    sc = lg_ref - lg_oth; d = np.where(y, sc, -sc)                                  # d — к истинной еде
    r = {'auc_neutral': auc(sc[cond == 2], y[cond == 2]), 'd_lying': float(d[cond == 1].mean()), 'd_truthful': float(d[cond == 0].mean()),
         'd_neutral': float(d[cond == 2].mean()), 'by_n': {}}
    for k in NS:
        s_ = n == k
        r['by_n'][str(k)] = {'auc_neutral': auc(sc[s_ & (cond == 2)], y[s_ & (cond == 2)]), 'd_lying': float(d[s_ & (cond == 1)].mean()),
                             'd_truthful': float(d[s_ & (cond == 0)].mean()),
                             'frac_lying_to_experience': float((d[s_ & (cond == 1)] > 0).mean())}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--model', default='models/pythia-6.9b')
    ap.add_argument('--ifaces', nargs='+', default=['s'])
    ap.add_argument('--sets', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--weights', nargs='+', default=['pretrained', 'random'])
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--out', default='results_synth')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.check:
        check(a.sets); return
    from transformers import AutoTokenizer
    dev = a.device; commit, dirty = git_info(); t0 = time.time()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    os.makedirs(a.out, exist_ok=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    for k in ('food', 'harm', 'neutral', 'direct_food', 'direct_harm'):
        for w in W[k]:
            tok1(tk, w)
    tag = os.path.basename(a.model.rstrip('/')).split('-')[-1]
    path = os.path.join(a.out, f'synth_{tag}.json')
    res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'model': a.model, 'runs': [], 'sets': {}, 'n0': {}}
    res['runs'].append({'git_commit': commit, 'git_dirty': dirty, 'sets': a.sets, 'weights': a.weights, 'ifaces': a.ifaces})
    WS = {i: windows(i) for i in a.sets}
    dir_ids = [tok1(tk, w) for w in W['direct_food'] + W['direct_harm']]
    pool_f, pool_h = [tok1(tk, w) for w in W['food']], [tok1(tk, w) for w in W['harm']]
    pre_norm = res.get('pre_norm'); m = None
    for wts in a.weights:
        for i in a.sets:
            todo = [f for f in a.ifaces if f'{wts}/{f}/{i}' not in res['sets']]
            if not todo:
                continue
            if wts == 'pretrained' and m is None:
                m = load(a.model, 'pretrained', dev); pre_norm = res['pre_norm'] = m.get_input_embeddings().weight.norm(dim=1).mean().item()
            if wts == 'random':
                del m; m = None; torch.cuda.empty_cache() if dev == 'cuda' else None
                m = load(a.model, 'random', dev, seed=i, pre_norm=pre_norm)
            ws = WS[i]; ts = time.time()
            cols = np.array([[tok1(tk, w['wF'] if w['ref'] == 0 else w['wP']), tok1(tk, w['wP'] if w['ref'] == 0 else w['wF'])] for w in ws])
            for f in todo:
                out, raw = {}, {}
                for v in VAR[f]:
                    q = BW['questions'][v.split('_')[0]]
                    seqs = [seq(tk, w['frp'] if v.endswith('_perm') else w['fr'], w['wF'], w['wP'], q, f) for w in ws]
                    lg = logits(m, seqs, cols, dev, a.bs)
                    out[v] = measures(ws, lg[:, 0], lg[:, 1]); raw[v] = lg.astype(np.float32)
                res['sets'][f'{wts}/{f}/{i}'] = {'measures': out, 'sec': round(time.time() - ts), 'git_commit': commit, 'git_dirty': dirty}
                np.savez_compressed(os.path.join(a.out, f'raw_{tag}_{wts}_{f}_{i}.npz'), **raw)
                # n = 0: один вход на интерфейс и вопрос — прямые слова и пары пулов «еда / вред» (описательно)
                key0 = f'{wts}/{f}/{i}' if wts == 'random' else f'{wts}/{f}'
                if key0 not in res['n0']:
                    r0 = {}
                    for qn in ('q1', 'q2', 'q3'):
                        s0 = seq(tk, [(2, 2)] * N, 'apple', 'mold', BW['questions'][qn], f)
                        l0 = logits(m, [s0], np.array([dir_ids + pool_f + pool_h]), dev, 1)[0]
                        ld, lf, lh = l0[:11], l0[11:41], l0[41:]
                        r0[qn] = {'M_direct': float((ld[:5, None] > ld[None, 5:]).mean()), 'M_pools': float((lf[:, None] > lh[None, :]).mean())}
                    res['n0'][key0] = r0
            json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
            print(f"{time.strftime('%H:%M:%S')} {tag} {wts} набор {i} ({', '.join(todo)}): {time.time() - ts:.0f} с (всего {time.time() - t0:.0f} с)", flush=True)
        del m; m = None; torch.cuda.empty_cache() if dev == 'cuda' else None


if __name__ == '__main__':
    main()
