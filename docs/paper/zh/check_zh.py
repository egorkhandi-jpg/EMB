"""Сверка чисел английского и китайского PDF (2026-10-07, программа Егора, п. 3): все числа английского PDF должны быть в
китайском — по разделам и в том же порядке. Текст обоих PDF извлекается одним способом (pdf_numbers_check.extract_pages);
номера страниц (последнее число страницы) отбрасываются; подписи к рисункам вырезаются из потока текста и сверяются
отдельно (плавающие объекты в двух версиях могут оказаться в разных разделах); титульный блок (до аннотации) не сверяется.
Числа — строки цифр с внутренними «.» и «,» (1,024; 0.983); знак не учитывается.

python docs/paper/zh/check_zh.py          # печатает отчёт и пишет docs/paper/zh/CHECK_zh.md
"""
import difflib, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pdf_numbers_check import extract_pages                                    # noqa: E402

SECTIONS = [  # (имя, заголовок в английском PDF, заголовок в китайском PDF) — ищутся по порядку
    ('Abstract', r'Abstract', r'摘要'),
    ('1 Introduction', r'1\s+Introduction', r'1\s*引言'),
    ('2 Setup', r'2\s+Setup', r'2\s*设置'),
    ('3 Results', r'3\s+Results', r'3\s*结果'),
    ('4 Pitfalls and controls', r'4\s+Pitfalls and controls', r'4\s*陷阱与对照'),
    ('5 Discussion', r'5\s+Discussion', r'5\s*讨论'),
    ('Disclosure', r'Disclosure', r'声明'),
    ('Code and data availability', r'Code and data availability', r'代码与数据获取'),
    ('References', r'References', r'参考文献'),
    ('A Replication and tokenizer transfer', r'A\s+Replication and tokenizer transfer', r'A\s*复现与分词器迁移'),
    ('B Registry', r'B\s+Registry of experiments', r'B\s*本文所用实验的登记表'),
    ('C Compute and costs', r'C\s+Compute and costs', r'C\s*计算与成本'),
]
CAP_EN = (r'Figure (\d):', [r'\(protocol of A22\)\.', r'\(protocols of A3 and A4\)\.', r'threshold 0\.8\.', r'\(a different instance\)\.'])
CAP_ZH = (r'图\s*(\d)\s*[:：]', [r'A22\s*的协议\s*）\s*。', r'A3\s*和\s*A4\s*的协议\s*）\s*。', r'虚线：预注册阈值\s*0\.8\s*。', r'另一个实例\s*）\s*。'])
NUM = re.compile(r'\d+(?:[.,]\d+)*')


def load(path, cap):
    pages = [re.sub(r'\s*\d+\s*$', '', p) for p in extract_pages(path)]       # номер страницы
    t = ' '.join(pages)
    caps = {}
    start_rx, ends = cap
    for m in list(re.finditer(start_rx, t)):
        k = int(m.group(1))
        if k in caps:
            continue
        found = [x for x in (re.compile(r).search(t, m.end()) for r in ends) if x]   # номер рисунка в PDF ≠ F-номер
        assert found, (path, k)
        e = min(found, key=lambda x: x.start())
        caps[k] = t[m.end():e.end()]
    for k in sorted(caps, reverse=True):                                       # вырезать подписи из потока
        t = t.replace(caps[k], ' ', 1)
    t = re.sub(start_rx, ' ', t)
    return t, caps


def split(t, col):
    out, pos = {}, 0
    marks = []
    for name, en, zh in SECTIONS:
        m = re.compile(en if col == 1 else zh).search(t, pos)
        assert m, (name, col)
        marks.append((name, m.start(), m.end())); pos = m.end()
    for i, (name, s, e) in enumerate(marks):
        out[name] = t[e:marks[i + 1][1]] if i + 1 < len(marks) else t[e:]
    return out


def nums(s):
    return NUM.findall(s)


def compare(a, b):
    """Список расхождений: недостающие в zh, лишние в zh, и места, где порядок другой."""
    ca, cb = Counter(a), Counter(b)
    miss, extra = dict(ca - cb), dict(cb - ca)
    order = []
    if not miss and not extra and a != b:
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if op != 'equal':
                order.append(f'en[{i1}:{i2}] {a[i1:i2]} ↔ zh[{j1}:{j2}] {b[j1:j2]}')
    return miss, extra, order


def main():
    en_t, en_c = load(os.path.join(os.path.dirname(HERE), 'latex', 'paper.pdf'), CAP_EN)
    zh_t, zh_c = load(os.path.join(HERE, 'main_zh.pdf'), CAP_ZH)
    E, Z = split(en_t, 1), split(zh_t, 2)
    rows, lines = [], []
    for name, _, _ in SECTIONS:
        rows.append((name, nums(E[name]), nums(Z[name])))
    for k in sorted(en_c):
        rows.append((f'Figure {k} (caption)', nums(en_c[k]), nums(zh_c.get(k, ''))))
    total_bad = 0
    lines.append('# Сверка чисел: английский PDF → китайский PDF\n')
    lines.append('Источник: `docs/paper/latex/paper.pdf` и `docs/paper/zh/main_zh.pdf`; скрипт `docs/paper/zh/check_zh.py`.\n')
    lines.append('| Раздел | Чисел (en) | Чисел (zh) | Совпадает с порядком | Расхождения |')
    lines.append('| --- | --- | --- | --- | --- |')
    for name, a, b in rows:
        miss, extra, order = compare(a, b)
        ok = (a == b)
        total_bad += (not ok)
        d = []
        if miss:
            d.append('нет в zh: ' + ', '.join(f'{k}×{v}' if v > 1 else k for k, v in miss.items()))
        if extra:
            d.append('лишние в zh: ' + ', '.join(f'{k}×{v}' if v > 1 else k for k, v in extra.items()))
        if order:
            d.append('порядок: ' + '; '.join(order))
        lines.append(f'| {name} | {len(a)} | {len(b)} | {"да" if ok else "нет"} | {" / ".join(d) or "—"} |')
    lines.append(f'\nРазделов с расхождениями: {total_bad} из {len(rows)}.')
    rep = '\n'.join(lines) + '\n'
    open(os.path.join(HERE, 'CHECK_zh.md'), 'w', encoding='utf-8', newline='\n').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
