"""Репликация ключевых результатов на других семействах (2026-10-06; docs/RESULTS_2026-10-06_replication.md).
Протоколы A3/A4 (R1, R2), A22 (R3), A24 (R4), A27 (R5) — без изменений, кроме переноса на токенизатор модели:
номера токенов берутся из токенизатора (все слова однотокенные с ведущим пробелом — проверяется при запуске), начальный
токен — «<|endoftext|>» модели (у Pythia и OLMo-2 это BOS, у Qwen2.5 BOS по умолчанию нет — берётся токен конца документа),
доступ к модели — m.base_model(...).last_hidden_state (после финальной нормировки) вместо m.gpt_neox.
R3–R5 вызывают исходные функции report_body / meaning_probe / meaning_synth с подменёнными модульными константами;
R1–R2 собраны из исходных частей probe_untrained.collect / windows и probe_fast.probe (условия c, c_arb1, c_arb2 при N = 32,
последний слой — как в probe_interface / probe_words; скрипта статистики A3 / A4 в репозитории нет, тесты — summarize_replication.py).

python replicate.py --model models/Qwen2.5-1.5B --parts r12 r3 r4 r5
python replicate.py --model models/Qwen2.5-1.5B --show      # декодирование входов всех протоколов (проверка до расчёта)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

import probe_untrained as pu
from probe_fast import probe
import stand_words as sw
import report_body as rb
import meaning_probe as mp
import meaning_bind as mbd
import meaning_synth as ms
from stand_gpu import git_info

ARB = {'c': (('A', 'B', 'none'), ('up', 'down', 'same')),
       'c_arb1': (('cat', 'table', 'stone'), ('river', 'chair', 'door')),
       'c_arb2': (('lamp', 'cloud', 'brick'), ('pencil', 'garden', 'window'))}


def tok1(tk, w):
    ids = tk.encode(' ' + w, add_special_tokens=False)
    assert len(ids) == 1, f'{w!r} не однотокенное: {ids}'
    return ids[0]


def setup(tk):
    """Подмена модульных констант протоколов на номера токенов этой модели."""
    begin = tk.convert_tokens_to_ids('<|endoftext|>')
    assert isinstance(begin, int) and begin >= 0 and tk.convert_ids_to_tokens(begin) == '<|endoftext|>'
    T = {w: tok1(tk, w) for w in ('A', 'B', 'none', 'up', 'down', 'same')}
    enc = lambda s: tk.encode(s, add_special_tokens=False)
    # A22 (report_body)
    rb.A, rb.B, rb.NONE, rb.UP, rb.DOWN, rb.SAME = T['A'], T['B'], T['none'], T['up'], T['down'], T['same']
    rb.Q = {'zs1': enc(' Food:'), 'zs2': enc(' Safe to eat:'), 'zs3': enc(' Good kind:'), 'en': enc(' Energy:')}
    sw.BOS = begin
    rb.run_lm = run_lm
    # A24 (meaning_probe)
    mp.NONE, mp.UP, mp.DOWN, mp.SAME, mp.BOS = T['none'], T['up'], T['down'], T['same'], begin
    mp.forward = forward
    # A27 (meaning_synth, чтение логитов — meaning_bind.read_logits)
    ms.BOS, ms.NONE, ms.UP, ms.DOWN, ms.SAME = begin, T['none'], T['up'], T['down'], T['same']
    mbd.read_logits = read_logits
    words = {c: tuple(tuple(tok1(tk, w) for w in grp) for grp in v) for c, v in ARB.items()}
    for k in ('food', 'harm', 'neutral', 'direct_food', 'direct_harm'):
        for w in mp.W[k]:
            tok1(tk, w)
    return begin, words


# ---------------- общий доступ к модели ----------------
def last_hidden(m, x):
    return m.base_model(input_ids=x).last_hidden_state


def run_lm(m, ids, ans, dev, chunk):
    """Как report_body.run_lm: последнее скрытое состояние и разность логитов ответов."""
    Wout = m.get_output_embeddings().weight; H, D = [], []
    for i in range(0, len(ids), chunk):
        x = ids[i:i + chunk].to(dev)
        with torch.no_grad():
            h = last_hidden(m, x)[:, -1]
            if not torch.isfinite(h).all():
                raise FloatingPointError('нечисловое скрытое состояние')
            lg = h @ Wout[list(ans)].T
        H.append(h.cpu().numpy()); D.append((lg[:, 0] - lg[:, 1]).cpu().numpy())
    return np.concatenate(H), np.concatenate(D)


def forward(m, ids_list, dev, bs, want='logp'):
    """Как meaning_probe.forward; скрытые состояния — last_hidden_state (после финальной нормировки)."""
    out = []
    for i in range(0, len(ids_list), bs):
        chunk = ids_list[i:i + bs]; L = max(len(x) for x in chunk)
        x = torch.full((len(chunk), L), 0, dtype=torch.long); mask = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, s in enumerate(chunk):
            x[j, :len(s)] = torch.tensor(s); mask[j, :len(s)] = 1
        x, mask = x.to(dev), mask.to(dev)
        with torch.no_grad():
            if want == 'logp':
                o = m(input_ids=x, attention_mask=mask)
                lp = torch.log_softmax(o.logits[:, :-1].float(), -1).gather(2, x[:, 1:, None]).squeeze(2) * mask[:, 1:]
                out += lp.sum(1).cpu().tolist()
            else:
                h = m.base_model(input_ids=x, attention_mask=mask).last_hidden_state
                if not torch.isfinite(h).all():
                    raise FloatingPointError('нечисловое скрытое состояние')
                out += [h[j, :len(chunk[j])].cpu().numpy() for j in range(len(chunk))]
    return out


def read_logits(m, seqs, cols, dev, bs):
    """Как meaning_bind.read_logits (одинаковые входы — один раз, дополнение справа), через base_model."""
    uniq, inv = {}, []
    for s in seqs:
        inv.append(uniq.setdefault(tuple(s), len(uniq)))
    U = list(uniq); order = sorted(range(len(U)), key=lambda j: len(U[j]))
    H = torch.zeros(len(U), m.config.hidden_size, device=dev)
    for a in range(0, len(order), bs):
        idx = order[a:a + bs]; L = max(len(U[j]) for j in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long); last = torch.tensor([len(U[j]) - 1 for j in idx])
        for r, j in enumerate(idx):
            x[r, :len(U[j])] = torch.tensor(U[j])
        with torch.no_grad():
            h = last_hidden(m, x.to(dev))[torch.arange(len(idx)), last.to(dev)]
        if not torch.isfinite(h).all():
            raise FloatingPointError('нечисловое скрытое состояние')
        H[idx] = h
    Wout = m.get_output_embeddings().weight
    inv = torch.as_tensor(inv, device=dev); C = torch.as_tensor(cols, device=dev)
    return torch.einsum('nd,nkd->nk', H[inv], Wout[C]).cpu().numpy(), len(U)


# ---------------- R1, R2 (A3 / A4) ----------------
def load_base(model, weights, seed, dev, ref_norm=None):
    from transformers import AutoModelForCausalLM, AutoConfig
    if weights == 'pretrained':
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, device_map=dev)
    else:
        torch.manual_seed(seed)
        with torch.device(dev):
            m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model), dtype=torch.float32)
        with torch.no_grad():
            E = m.get_input_embeddings().weight; E.mul_(ref_norm / E.norm(dim=1).mean())
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def r12_seqs(W, L, words, begin):
    """Как probe_words.LM.seq для условий c / c_arb: начальный токен + (вид, исход) на такт, последние l кадров окна."""
    (tA, tB, tN), (tU, tD, tS) = words
    ate = np.where(W[:, :, 10] > 0, tA, np.where(W[:, :, 11] > 0, tB, tN))
    en = np.where(W[:, :, 7] > 0.5, tU, np.where(W[:, :, 7] < -0.5, tD, tS))
    pair = np.stack([ate, en], 2).reshape(len(W), -1)                      # (n, 2N)
    return pair, L


def r12_features(m, W, L, words, begin, dev, chunk):
    n, N, _ = W.shape; pair, _ = r12_seqs(W, L, words, begin); X = np.zeros((n, m.config.hidden_size), np.float32)
    for l in np.unique(L):
        sel = np.where(L == l)[0]
        for i in range(0, len(sel), chunk):
            s = sel[i:i + chunk]
            ids = np.concatenate([np.full((len(s), 1), begin), pair[s, 2 * (N - l):]], 1)
            with torch.no_grad():
                h = last_hidden(m, torch.as_tensor(ids, device=dev))[:, -1]
            if not torch.isfinite(h).all():
                raise FloatingPointError('нечисловое скрытое состояние')
            X[s] = h.cpu().numpy()
    return X


def r12(model, words, begin, dev, out, chunk):
    path = os.path.join(out, 'r12.json'); commit, dirty = git_info()
    res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'model': model, 'results': []}
    done = {(r['set'], r['weights'], r['cond']) for r in res['results']}
    pre = load_base(model, 'pretrained', 0, dev); ref = pre.get_input_embeddings().weight.norm(dim=1).mean().item()
    for s in range(1, 11):
        ts = time.time()
        D_ = pu.collect('flip-2', True, 200, seed0=31000 + 1000 * s)
        idx = np.where(D_['valid']); groups = idx[0]; y = D_['sign'][idx].astype(float)
        W, L = pu.windows(D_['S'], idx, 32)
        rnd = None
        for wts in ('pretrained', 'random'):
            if all((s, wts, c) in done for c in ARB):
                continue
            if wts == 'random':
                rnd = load_base(model, 'random', s, dev, ref)
            m = pre if wts == 'pretrained' else rnd
            for c in ARB:
                if (s, wts, c) in done:
                    continue
                X = r12_features(m, W, L, words[c], begin, dev, chunk)
                _, acc = probe(X, y, groups, dev)
                res['results'].append({'set': s, 'weights': wts, 'cond': c, 'bacc': float(acc), 'n': int(len(y)), 'git_commit': commit, 'git_dirty': dirty})
                done.add((s, wts, c))
        del rnd; torch.cuda.empty_cache()
        json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f"{time.strftime('%H:%M:%S')} r12 набор {s}: {time.time() - ts:.0f} с", flush=True)


# ---------------- показ входов ----------------
def show(tk, begin, words):
    dec = lambda ids: tk.decode(list(map(int, ids)))
    D_ = pu.collect('flip-2', True, 20, seed0=32000); idx = np.where(D_['valid']); W, L = pu.windows(D_['S'], idx, 32)
    j = int(np.argmax(L)); pair, _ = r12_seqs(W[j:j + 1], L[j:j + 1], words['c'], begin)
    for c in ARB:
        pair, _ = r12_seqs(W[j:j + 1], L[j:j + 1], words[c], begin)
        print(f'R1/R2 {c}:', repr(dec([begin] + list(pair[0]))[:400]))
    tr = rb.trajectories([800000 + 1000 + e for e in range(20)]); qs = rb.queries(tr)
    for mode in ('zs1', 'zs2', 'zs3'):
        print(f'R3 {mode}:', repr(dec(rb.build(tr, qs[:1], mode)[0])[-160:]), '| ответы', repr(dec([rb.A])), repr(dec([rb.B])))
    print('R4 шаблон:', repr(dec([mp.BOS] + tk.encode(mp.W['neutral_template'].replace('{w}', ' apple'), add_special_tokens=False))))
    ws = ms.windows(1); w = [x for x in ws if x['cond'] == 1 and x['n'] == 4][0]
    print('R5 s:', repr(dec(ms.seq(tk, w['fr'], w['wF'], w['wP'], ms.BW['questions']['q1'], 's')).replace('Nothing happened. ', '')))
    print('R5 s перест.:', repr(dec(ms.seq(tk, w['frp'], w['wF'], w['wP'], ms.BW['questions']['q1'], 's')).replace('Nothing happened. ', '')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--parts', nargs='+', default=['r12', 'r3', 'r4', 'r5'])
    ap.add_argument('--out', default='results_replication')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model)
    begin, words = setup(tk)
    print(f"{a.model}: начальный токен {begin} {tk.convert_ids_to_tokens(begin)!r}; c = {words['c']}", flush=True)
    if a.show:
        show(tk, begin, words); return
    short = os.path.basename(a.model.rstrip('/')); out = os.path.join(a.out, short); os.makedirs(out, exist_ok=True)
    commit, dirty = git_info()
    if dirty:
        print(f'ВНИМАНИЕ: незакоммиченный код (HEAD {commit[:8]})', flush=True)
    for part in a.parts:
        t0 = time.time()
        if part == 'r12':
            r12(a.model, words, begin, a.device, out, a.chunk)
        elif part == 'r3':
            for wts in ('pretrained', 'random'):
                sys.argv = ['report_body', '--model', a.model, '--weights', wts, '--modes', 'zs1', 'zs2', 'zs3',
                            '--probe-modes', 'zs1', 'zs2', 'zs3', '--out', out, '--chunk', str(a.chunk), '--device', a.device]
                rb.main()
        elif part == 'r4':
            sys.argv = ['meaning_probe', '--model', a.model, '--out', out, '--bs', '16', '--device', a.device]
            mp.main()
        elif part == 'r5':
            sys.argv = ['meaning_synth', '--model', a.model, '--ifaces', 's', '--out', out, '--bs', '32', '--device', a.device]
            ms.main()
        print(f"{time.strftime('%H:%M:%S')} {short} {part}: {time.time() - t0:.0f} с", flush=True)


if __name__ == '__main__':
    main()
