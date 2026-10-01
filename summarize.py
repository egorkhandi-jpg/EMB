"""Сводка по папке результатов: python summarize.py [results]"""
import json, glob, sys, numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else 'results'
res = {}
for f in sorted(glob.glob(f'{d}/result_*.json')):
    r = json.load(open(f)); res.setdefault(r['tag'], []).append(r)
print('ориентиры: стоять 0.52 | случайно 0.59 | жадно к пище 1.46')
print(f"{'условие':40s} {'n':>3s} {'тест ср.':>9s} {'± ст.откл':>10s} {'пок. до 1.2 (медиана)':>22s} {'не дошли':>9s}")
for tag, rs in sorted(res.items()):
    t = np.array([r['test'] for r in rs])
    g = [next((i for i, x in enumerate(r['hist']) if x >= 1.2), None) for r in rs]
    gg = [x for x in g if x is not None]
    print(f"{tag:40s} {len(rs):3d} {t.mean():9.3f} {t.std():10.3f} {(np.median(gg) if gg else float('nan')):22.1f} {sum(x is None for x in g):9d}")
