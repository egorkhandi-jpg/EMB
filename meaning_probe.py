"""Шлюз к варианту C: знают ли крупные модели смысл слов «еда / вред» и доступно ли это знание в интерфейсе стенда.
Пререгистрация — docs/RESULTS_2026-10-05_meaning.md; слова и шаблоны — meaning_words.json (до расчёта). Без стенда и ES.

Проверка 1 (язык):
  s(w) — среднее по 4 парам шаблонов [log P(пищевое предложение) − log P(вредное предложение)] (сумма log-вероятностей токенов
         предложения после BOS);
  зонд — логистическая регрессия L2 по последнему скрытому состоянию (после финальной нормировки) на позиции слова в «The{w}.»,
         исключение одного слова (λ — по внутренней 5-кратной CV на обучающих словах); AUC «еда против вреда» по словам.
  Положительный контроль — тот же зонд с исключением слова на прямых словах (5 пищевых, 6 вредных).
Проверка 2 (интерфейс): окна событий как в стенде (c + BOS, окно 32, flip-2, слепая политика), символ вида A и B — слова:
  у эпизода пищевое слово и вредное слово из одной из 5 групп (по 6 слов), какое досталось виду A — монетка эпизода.
  Запросы — такты {32, 40, …, 96}, где в окне есть событие вида A. Метка — «слово вида A — пищевое».
  Зонд по скрытому состоянию на последней позиции окна (основной — эту позицию читает голова стенда) и на позиции последнего
  символа вида A (описательно); 5-кратная CV по группам слов (слова теста не встречаются в обучении).
Случайные веса той же архитектуры (эмбеддинги приведены к средней норме предобученных), сеть от посева 1.

python meaning_probe.py --model EleutherAI/pythia-1b
"""
import argparse, json, math, os, time
import numpy as np
import torch

from env_flip import FlipBodies, T
import stand_words as sw
import body_head as bh
from readout import fit_probe
from probe_fast import _newton, LAMBDAS
from stand_gpu import git_info

W = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meaning_words.json'), encoding='utf-8'))
NONE, UP, DOWN, SAME, BOS = 5293, 598, 1066, 1072, 0
N, QT, EPISODES = 32, list(range(32, 97, 8)), 200


def tok1(tk, w):
    ids = tk.encode(' ' + w)
    assert len(ids) == 1, f'слово {w!r} не однотокенное: {ids}'
    return ids[0]


def load(model, weights, dev, pre_norm=None):
    from transformers import AutoModelForCausalLM, AutoConfig
    if weights == 'pretrained':
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).to(dev)
    else:
        torch.manual_seed(1)
        with torch.device(dev):
            m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model), dtype=torch.float32)
        with torch.no_grad():
            E = m.get_input_embeddings().weight; E.mul_(pre_norm / E.norm(dim=1).mean())
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def forward(m, ids_list, dev, bs, want='logp'):
    """ids_list — списки токенов. want: 'logp' — сумма log P токенов после первого; 'hid' — скрытые состояния (все позиции)."""
    out = []
    for i in range(0, len(ids_list), bs):
        chunk = ids_list[i:i + bs]; L = max(len(x) for x in chunk)
        x = torch.full((len(chunk), L), 0, dtype=torch.long); mask = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, s in enumerate(chunk):
            x[j, :len(s)] = torch.tensor(s); mask[j, :len(s)] = 1
        x, mask = x.to(dev), mask.to(dev)
        with torch.no_grad():
            o = m(input_ids=x, attention_mask=mask, output_hidden_states=(want == 'hid'))
        if want == 'logp':
            lp = torch.log_softmax(o.logits[:, :-1].float(), -1).gather(2, x[:, 1:, None]).squeeze(2) * mask[:, 1:]
            out += lp.sum(1).cpu().tolist()
        else:
            h = o.hidden_states[-1]                                         # после финальной нормировки
            if not torch.isfinite(h).all():
                raise FloatingPointError('нечисловое скрытое состояние')
            out += [h[j, :len(chunk[j])].cpu().numpy() for j in range(len(chunk))]
    return out


def auc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def fit_lambda(X, y, dev, groups=None, folds=5):
    """λ по внутренней CV (по группам, если заданы), затем веса на всех данных; возвращает функцию решений."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    mu, sd = Xt.mean(0), Xt.std(0, unbiased=False) + 1e-6
    X1 = torch.cat([(Xt - mu) / sd, torch.ones(len(y), 1, device=dev)], 1)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    ug = np.unique(g); r = np.random.RandomState(0); r.shuffle(ug)
    best, bl = -1, LAMBDAS[0]
    for lam in LAMBDAS:
        dec = np.zeros(len(y))
        for f in range(folds):
            te = np.isin(g, ug[f::folds]); tr = torch.as_tensor(~te, device=dev)
            if te.sum() == 0 or (~te).sum() == 0:
                continue
            w = _newton(X1[tr], yt[tr], lam, torch.zeros(X1.shape[1], device=dev))
            dec[te] = (X1[torch.as_tensor(te, device=dev)] @ w).cpu().numpy()
        a = auc(dec, y)
        if a > best:
            best, bl = a, lam
    w = _newton(X1, yt, bl, torch.zeros(X1.shape[1], device=dev))
    return lambda Z: (((torch.as_tensor(Z, dtype=torch.float32, device=dev) - mu) / sd) @ w[:-1] + w[-1]).cpu().numpy(), bl


def loo_words(H, y, dev):
    dec = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        f, _ = fit_lambda(H[tr], y[tr], dev)
        dec[i] = f(H[i:i + 1])[0]
    return dec


def check1(m, tk, dev, bs):
    words = {k: W[k] for k in ('food', 'harm', 'neutral', 'direct_food', 'direct_harm')}
    allw = [w for k in words for w in words[k]]
    sents, idx = [], []
    for w in allw:
        for a_, b_ in W['template_pairs']:
            sents += [[BOS] + tk.encode(a_.replace('{w}', ' ' + w)), [BOS] + tk.encode(b_.replace('{w}', ' ' + w))]
    lp = np.array(forward(m, sents, dev, bs, 'logp')).reshape(len(allw), len(W['template_pairs']), 2)
    s = dict(zip(allw, (lp[:, :, 0] - lp[:, :, 1]).mean(1)))
    neu = [[BOS] + tk.encode(W['neutral_template'].replace('{w}', ' ' + w)) for w in allw]
    hid = forward(m, neu, dev, bs, 'hid')
    wid = {w: tok1(tk, w) for w in allw}
    Hw = {}
    for w, seq, h in zip(allw, neu, hid):
        Hw[w] = h[seq.index(wid[w])]
    fh = words['food'] + words['harm']; y = np.array([1] * len(words['food']) + [0] * len(words['harm']))
    H = np.stack([Hw[w] for w in fh])
    dec = loo_words(H, y, dev)
    dd = words['direct_food'] + words['direct_harm']; yd = np.array([1] * len(words['direct_food']) + [0] * len(words['direct_harm']))
    dec_d = loo_words(np.stack([Hw[w] for w in dd]), yd, dev)
    f_all, _ = fit_lambda(H, y, dev)
    neu_dec = f_all(np.stack([Hw[w] for w in words['neutral']]))
    return {'s': {w: float(v) for w, v in s.items()}, 'auc_s': auc([s[w] for w in fh], y), 'auc_s_direct': auc([s[w] for w in dd], yd),
            'probe_dec': dict(zip(fh, dec.tolist())), 'auc_probe': auc(dec, y), 'auc_probe_direct': auc(dec_d, yd),
            'probe_dec_direct': dict(zip(dd, dec_d.tolist())), 'neutral_probe_dec': dict(zip(words['neutral'], neu_dec.tolist())),
            'mean_s': {k: float(np.mean([s[w] for w in words[k]])) for k in words}}


def check2(m, tk, dev, bs):
    food, harm = W['food'], W['harm']
    gid_f = [food[i::5] for i in range(5)]; gid_h = [harm[i::5] for i in range(5)]
    seeds = [900000 + e for e in range(EPISODES)]
    b = FlipBodies(EPISODES, seeds, True, 'flip-2')
    rng = np.random.RandomState(12345)
    fold = np.arange(EPISODES) % 5
    fw = [gid_f[fold[e]][rng.randint(6)] for e in range(EPISODES)]; hw = [gid_h[fold[e]][rng.randint(6)] for e in range(EPISODES)]
    a_food = rng.rand(EPISODES) < 0.5
    tokA = np.array([tok1(tk, fw[e] if a_food[e] else hw[e]) for e in range(EPISODES)])
    tokB = np.array([tok1(tk, hw[e] if a_food[e] else fw[e]) for e in range(EPISODES)])
    kt, et = [], []
    for t in range(T):
        s = b.sense()
        kind, en = sw.events(s)
        kt.append(np.where(kind == 0, tokA, np.where(kind == 1, tokB, NONE))); et.append(np.array([UP, DOWN, SAME])[en])
        act = sw.reflex(s, np.stack([np.where(s[:, 2] <= s[:, 5], 1.0, -1.0), np.ones(EPISODES)], 1)); act[~b.alive] = 0
        b.step(act)
    kt, et = np.stack(kt, 1), np.stack(et, 1)
    seqs, posA, meta = [], [], []
    for e in range(EPISODES):
        for t in QT:
            win = [x for u in range(t - N + 1, t + 1) for x in (kt[e, u], et[e, u])]
            if tokA[e] not in win:
                continue
            seq = [BOS] + [int(x) for x in win]
            seqs.append(seq); posA.append(len(seq) - 1 - seq[::-1].index(int(tokA[e]))); meta.append((e, t))
    hid = forward(m, seqs, dev, bs, 'hid')
    Hl = np.stack([h[-1] for h in hid]); Ha = np.stack([h[p] for h, p in zip(hid, posA)])
    ep = np.array([x[0] for x in meta]); y = a_food[ep].astype(int); fq = fold[ep]
    res = {'n_queries': int(len(y)), 'n_episodes': EPISODES}
    for nm, H in (('last', Hl), ('kindA', Ha)):
        dec = np.zeros(len(y))
        for f in range(5):
            te = fq == f
            fn, _ = fit_lambda(H[~te], y[~te], dev, groups=ep[~te])
            dec[te] = fn(H[te])
        res[f'auc_{nm}'] = auc(dec, y); res[f'dec_{nm}'] = dec.tolist()
    res['episode'] = ep.tolist(); res['label'] = y.tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='EleutherAI/pythia-1b')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--out', default='results_meaning')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    dev = a.device; commit, dirty = git_info(); t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    for k in ('food', 'harm', 'neutral', 'direct_food', 'direct_harm'):
        for w in W[k]:
            tok1(tk, w)                                                         # однотокенность для токенизатора этой модели
    tag = a.model.split('-')[-1]
    res = {'model': a.model, 'git_commit': commit, 'git_dirty': dirty}
    m = load(a.model, 'pretrained', dev)
    pre_norm = m.get_input_embeddings().weight.norm(dim=1).mean().item()
    res['pretrained'] = {'check1': check1(m, tk, dev, a.bs), 'check2': check2(m, tk, dev, a.bs)}
    print(f"{tag} пред: s(w) AUC {res['pretrained']['check1']['auc_s']:.3f}, зонд {res['pretrained']['check1']['auc_probe']:.3f}, прямые {res['pretrained']['check1']['auc_probe_direct']:.3f} | "
          f"интерфейс: последняя {res['pretrained']['check2']['auc_last']:.3f}, вид A {res['pretrained']['check2']['auc_kindA']:.3f} | {time.time() - t0:.0f} с", flush=True)
    del m; torch.cuda.empty_cache() if dev == 'cuda' else None
    m = load(a.model, 'random', dev, pre_norm)
    res['random'] = {'check1': check1(m, tk, dev, a.bs), 'check2': check2(m, tk, dev, a.bs)}
    print(f"{tag} случ: s(w) AUC {res['random']['check1']['auc_s']:.3f}, зонд {res['random']['check1']['auc_probe']:.3f} | интерфейс: последняя "
          f"{res['random']['check2']['auc_last']:.3f}, вид A {res['random']['check2']['auc_kindA']:.3f} | {time.time() - t0:.0f} с", flush=True)
    res['sec'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(a.out, f'meaning_{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
