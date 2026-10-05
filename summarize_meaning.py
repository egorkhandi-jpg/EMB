"""Сводка шлюза к варианту C по пререгистрации (docs/RESULTS_2026-10-05_meaning.md).
K1: AUC зонда предобученной > AUC случайной в проверке 1 (исключение слова); K2: то же в проверке 2 (последняя позиция окна).
p — односторонний бутстрэп (10 000): по словам со стратификацией по классу (K1) и по эпизодам (K2), на решениях вне выборки.
Холм по шести (3 модели × K1, K2). «Знание есть»: AUC предобученной ≥ 0,8 и K выполнено (p Холма < 0,05).
Положительный контроль: зонд на прямых словах у предобученной AUC ≥ 0,8 — иначе метод для этой модели негоден.

python summarize_meaning.py results_meaning > results_meaning/summary.txt
"""
import glob, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results_meaning'
B = 10000


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); p, n = s[y], s[~y]
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def boot_words(dp, dr, y, rng):
    y = np.asarray(y, bool); ip, ineg = np.nonzero(y)[0], np.nonzero(~y)[0]; diffs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))])
        diffs.append(auc(dp[idx], y[idx]) - auc(dr[idx], y[idx]))
    return float((np.array(diffs) <= 0).mean())


def boot_episodes(dp, dr, y, ep, rng):
    ue = np.unique(ep); groups = {e: np.nonzero(ep == e)[0] for e in ue}; diffs = []
    for _ in range(B):
        idx = np.concatenate([groups[e] for e in rng.choice(ue, len(ue))])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(auc(dp[idx], y[idx]) - auc(dr[idx], y[idx]))
    return float((np.array(diffs) <= 0).mean())


def holm(ps):
    order = np.argsort(ps); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


R = {}
for f in sorted(glob.glob(os.path.join(D, 'meaning_*.json'))):
    r = json.load(open(f, encoding='utf-8')); R[r['model'].split('-')[-1]] = r
order = [m for m in ('1b', '2.8b', '6.9b') if m in R]
rng = np.random.RandomState(0)
tests = []
for m in order:
    r = R[m]; P, Q = r['pretrained'], r['random']
    c1p, c1r, c2p, c2r = P['check1'], Q['check1'], P['check2'], Q['check2']
    print(f"=== {m} | {r['git_commit'][:8]}{' ГРЯЗН' if r['git_dirty'] else ''} | {r.get('sec', '?')} с")
    valid = c1p['auc_probe_direct'] >= 0.8
    print(f"  положительный контроль (прямые слова, зонд): пред {c1p['auc_probe_direct']:.3f}, случ {c1r['auc_probe_direct']:.3f}; s(w) прямых: пред {c1p['auc_s_direct']:.3f}, случ {c1r['auc_s_direct']:.3f} → {'метод годен' if valid else 'МЕТОД НЕГОДЕН'}")
    print(f"  проверка 1: зонд пред {c1p['auc_probe']:.3f}, случ {c1r['auc_probe']:.3f} | s(w) пред {c1p['auc_s']:.3f}, случ {c1r['auc_s']:.3f}")
    print(f"     среднее s(w) по пулам (пред): " + ', '.join(f"{k} {v:+.2f}" for k, v in c1p['mean_s'].items()))
    neu = np.array(list(c1p['neutral_probe_dec'].values())); fd = np.array([c1p['probe_dec'][w] for w in R[m]['pretrained']['check1']['probe_dec']])
    print(f"     зонд на нейтральных (обучен на всех 60): медиана {np.median(neu):+.2f}; доля «еда» {np.mean(neu > 0):.2f}")
    print(f"  проверка 2 ({c2p['n_queries']} запросов): последняя позиция — пред {c2p['auc_last']:.3f}, случ {c2r['auc_last']:.3f} | позиция вида A — пред {c2p['auc_kindA']:.3f}, случ {c2r['auc_kindA']:.3f}")
    words = list(c1p['probe_dec'])
    y1 = np.array([1 if w in set(json.load(open('meaning_words.json', encoding='utf-8'))['food']) else 0 for w in words])
    dp1 = np.array([c1p['probe_dec'][w] for w in words]); dr1 = np.array([c1r['probe_dec'][w] for w in words])
    p1 = boot_words(dp1, dr1, y1, rng)
    dp2, dr2 = np.array(c2p['dec_last']), np.array(c2r['dec_last']); y2 = np.array(c2p['label']); ep = np.array(c2p['episode'])
    p2 = boot_episodes(dp2, dr2, y2, ep, rng)
    tests += [(m, 'K1', c1p['auc_probe'], c1p['auc_probe'] - c1r['auc_probe'], p1, valid), (m, 'K2', c2p['auc_last'], c2p['auc_last'] - c2r['auc_last'], p2, valid)]
if tests:
    adj = holm([t[4] for t in tests])
    print('\nОСНОВНАЯ СЕМЬЯ (односторонний бутстрэп, Холм по шести; «знание есть» — AUC пред ≥ 0,8 и p Холма < 0,05):')
    for (m, k, a_, d, p, v), pa in zip(tests, adj):
        ok = a_ >= 0.8 and pa < 0.05 and v
        print(f"  {m:5s} {k}: AUC пред {a_:.3f}, Δ к случайной {d:+.3f}, p {p:.4f}, p(Холм) {pa:.4f} → {'ЗНАНИЕ ЕСТЬ' if ok else 'нет'}{'' if v else ' (метод негоден)'}")
