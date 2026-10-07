"""Проверка F3 (2026-10-07): почему AUC случайных весов на позиции символа вида бывает заметно ниже 0,5 (Pythia-2.8B 0,31,
Qwen2.5 0,34). Без пересчёта моделей: решения зонда уже сохранены (meaning_*.json, проверка 2).
1) Восстанавливаются слова эпизодов (тот же генератор, что в meaning_probe.check2) и проверяется совпадение меток.
2) Моделирование той же схемы (5 частей по группам слов, λ по внутренней проверке с группами-эпизодами, meaning_probe.fit_lambda)
   на признаках «случайный вектор слова вида A + слабая примесь слова вида B + шум»: есть ли систематическое смещение ниже 0,5
   и каков разброс AUC между независимыми случайными «моделями».
3) Интервалы: бутстреп по эпизодам (как было в F3) против кластерного бутстрепа по словам вида A (единица обобщения — слово).

python f3_check.py > results_meaning/f3_check.txt
"""
import json, os
import numpy as np
import meaning_probe as mp

EP = 200
W = mp.W


def episode_words():
    food, harm = W['food'], W['harm']
    gid_f = [food[i::5] for i in range(5)]; gid_h = [harm[i::5] for i in range(5)]
    rng = np.random.RandomState(12345); fold = np.arange(EP) % 5
    fw = [gid_f[fold[e]][rng.randint(6)] for e in range(EP)]; hw = [gid_h[fold[e]][rng.randint(6)] for e in range(EP)]
    a_food = rng.rand(EP) < 0.5
    wA = [fw[e] if a_food[e] else hw[e] for e in range(EP)]; wB = [hw[e] if a_food[e] else fw[e] for e in range(EP)]
    return np.array(wA), np.array(wB), a_food, fold


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); p, n = s[y], s[~y]
    return float(((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean()))


def boot(dec, y, unit, B=2000, seed=0):
    dec, y, unit = np.asarray(dec), np.asarray(y, bool), np.asarray(unit)
    uu = np.unique(unit); grp = {u: np.nonzero(unit == u)[0] for u in uu}; rng = np.random.RandomState(seed); out = []
    for _ in range(B):
        idx = np.concatenate([grp[u] for u in rng.choice(uu, len(uu))])
        if len(np.unique(y[idx])) == 2:
            out.append(auc(dec[idx], y[idx]))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def simulate(ep_q, wA, wB, a_food, fold, n_sim=60, D=256, mix=0.3, noise=0.5):
    """Случайные «модели»: вектор на слово (гауссов), признак на позиции вида A = v[wA] + mix·v[wB] + шум."""
    import torch
    words = sorted(set(W['food']) | set(W['harm'])); idx = {w: i for i, w in enumerate(words)}
    y = a_food[ep_q].astype(int); fq = fold[ep_q]; res = []
    for k in range(n_sim):
        r = np.random.RandomState(1000 + k); V = r.randn(len(words), D)
        X = np.stack([V[idx[wA[e]]] + mix * V[idx[wB[e]]] for e in ep_q]) + noise * r.randn(len(ep_q), D)
        dec = np.zeros(len(y))
        for f in range(5):
            te = fq == f
            fn, _ = mp.fit_lambda(X[~te], y[~te], 'cpu', groups=ep_q[~te])
            dec[te] = fn(X[te])
        res.append(auc(dec, y))
    return np.array(res)


def main():
    wA, wB, a_food, fold = episode_words()
    srcs = [('results_meaning/meaning_1b.json', 'Pythia-1B'), ('results_meaning/meaning_2.8b.json', 'Pythia-2.8B'), ('results_meaning/meaning_6.9b.json', 'Pythia-6.9B')]
    for d, nm in (('Qwen2.5-1.5B', 'Qwen2.5-1.5B'), ('OLMo-2-0425-1B', 'OLMo-2-1B')):
        f = [x for x in os.listdir(f'results_replication/{d}') if x.startswith('meaning_')][0]; srcs.append((f'results_replication/{d}/{f}', nm))
    print('1) Восстановление слов эпизодов:')
    c2 = json.load(open(srcs[0][0], encoding='utf-8'))['pretrained']['check2']; ep_q = np.array(c2['episode'])
    print(f"   метки совпадают с сохранёнными: {bool((a_food[ep_q].astype(int) == np.array(c2['label'])).all())}; запросов {len(ep_q)}, эпизодов {len(np.unique(ep_q))}, "
          f"разных слов вида A {len(np.unique(wA[ep_q]))} (из 60), запросов на слово: медиана {int(np.median(np.unique(wA[ep_q], return_counts=True)[1]))}")
    print('\n2) Моделирование схемы на случайных векторах слов (60 независимых «моделей», D 256):')
    sim = simulate(ep_q, wA, wB, a_food, fold)
    print(f"   AUC: среднее {sim.mean():.3f}, ст. откл. {sim.std():.3f}, 2,5–97,5 % [{np.percentile(sim, 2.5):.3f}; {np.percentile(sim, 97.5):.3f}], доля < 0,40: {np.mean(sim < 0.40):.2f}")
    print('\n3) Интервалы F3: бутстреп по эпизодам против кластерного по словам вида A (95 %):')
    print('   модель           веса        позиция   AUC    по эпизодам        по словам')
    for p, nm in srcs:
        r = json.load(open(p, encoding='utf-8'))
        for wts in ('pretrained', 'random'):
            c = r[wts]['check2']; e = np.array(c['episode']); y = np.array(c['label'])
            assert (a_food[e].astype(int) == y).all()
            for pos in ('kindA', 'last'):
                dec = np.array(c[f'dec_{pos}']); a = auc(dec, y)
                le, he = boot(dec, y, e); lw, hw = boot(dec, y, wA[e])
                print(f"   {nm:15s}  {wts:10s}  {pos:6s}   {a:.3f}  [{le:.3f}; {he:.3f}]   [{lw:.3f}; {hw:.3f}]")


if __name__ == '__main__':
    main()
