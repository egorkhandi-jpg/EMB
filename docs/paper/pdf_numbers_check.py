"""Сверка чисел в собранном PDF с draft_full.md (2026-10-07), без внешних библиотек: распаковка потоков (zlib), таблицы ToUnicode
шрифтов, текст из операторов Tj/TJ. Сравнивается мультимножество чисел тела статьи (до приложений) с draft_full.md без
комментариев (номера разделов и годы ссылок в LaTeX выглядят иначе — печатаются отдельно).

python docs/paper/pdf_numbers_check.py
"""
import os, re, zlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def extract_pages(path):
    """Текст страниц PDF (список строк) — для этой сверки и для сверки китайской версии (zh/check_zh.py)."""
    pdf = open(path, 'rb').read()

    objs = {int(m.group(1)): m.group(2) for m in re.finditer(rb'(\d+) 0 obj(.*?)endobj', pdf, flags=re.S)}


    def _objstm():
        """Объекты из сжатых потоков объектов (/Type /ObjStm): заголовок «номер смещение …», затем тела с /First."""
        for body in list(objs.values()):
            if b'/ObjStm' not in body:
                continue
            data = stream(body); first = int(re.search(rb'/First\s+(\d+)', body).group(1))
            nums = list(map(int, data[:first].split()))
            pairs = list(zip(nums[0::2], nums[1::2]))
            for k, (on, off) in enumerate(pairs):
                end = pairs[k + 1][1] if k + 1 < len(pairs) else len(data) - first
                objs.setdefault(on, data[first + off:first + end])


    def stream(body):
        m = re.search(rb'stream\r?\n(.*)endstream', body, flags=re.S)
        if not m:
            return None
        data = m.group(1)
        try:
            return zlib.decompress(data) if b'FlateDecode' in body else data
        except zlib.error:
            return zlib.decompressobj().decompress(data)


    def cmap(data):
        mp = {}
        for blk in re.findall(rb'beginbfchar(.*?)endbfchar', data, flags=re.S):
            for a, b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                mp[int(a, 16)] = bytes.fromhex(b.decode()).decode('utf-16-be', 'replace')
        for blk in re.findall(rb'beginbfrange(.*?)endbfrange', data, flags=re.S):
            for a, b, c in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                base = bytes.fromhex(c.decode()).decode('utf-16-be', 'replace')
                for i, code in enumerate(range(int(a, 16), int(b, 16) + 1)):
                    try:
                        mp[code] = base[:-1] + chr(ord(base[-1]) + i) if base else ''
                    except ValueError:                                              # вне Юникода (служебные глифы) — пропуск
                        pass
        return mp


    _objstm()
    fonts = {}                                                                     # имя ресурса шрифта → (cmap, ширина кода)
    for n, body in objs.items():
        for name, ref in re.findall(rb'/(F\d+)\s+(\d+) 0 R', body):
            fb = objs.get(int(ref), b'')
            tu = re.search(rb'/ToUnicode\s+(\d+) 0 R', fb)
            if tu:
                two = b'Identity-H' in fb or b'Type0' in fb
                fonts[name.decode()] = (cmap(stream(objs[int(tu.group(1))]) or b''), 2 if two else 1)

    pages_text = []
    for n, body in sorted(objs.items()):
        if b'/Type /Page' in body or b'/Type/Page' in body:
            cont = re.search(rb'/Contents\s*(\[[^\]]*\]|\d+ 0 R)', body)
            if not cont:
                continue
            data = b''.join(stream(objs[int(r)]) or b'' for r in re.findall(rb'(\d+) 0 R', cont.group(1)))
            out, cur = [], None
            for tok in re.finditer(rb'/(F\d+)\s+[\d.]+\s+Tf|\[(.*?)\]\s*TJ|<([0-9A-Fa-f]+)>\s*Tj|\(((?:\\.|[^\\)])*)\)\s*Tj|(T\*|Td|TD|ET)', data, flags=re.S):
                if tok.group(1):
                    cur = fonts.get(tok.group(1).decode()); continue
                if tok.group(5):
                    out.append(' '); continue
                parts = []
                if tok.group(2) is not None:
                    for h, num in re.findall(rb'<([0-9A-Fa-f]*)>|(-?[\d.]+)', tok.group(2)):
                        if h:
                            parts.append(h)
                        elif num and float(num) < -250:
                            parts.append(b' ')
                elif tok.group(3):
                    parts.append(tok.group(3))
                if cur is None:
                    continue
                mp, w = cur
                for h in parts:
                    if h == b' ':
                        out.append(' '); continue
                    hx = h.decode()
                    for i in range(0, len(hx), 2 * w):
                        out.append(mp.get(int(hx[i:i + 2 * w], 16), ''))
            pages_text.append(''.join(out))
    return pages_text


if __name__ == '__main__':
    pages_text = extract_pages(os.path.join(HERE, 'latex', 'paper.pdf'))
    text = '\n'.join(pages_text)
    open(os.path.join(HERE, 'latex', 'paper_extracted.txt'), 'w', encoding='utf-8').write(text)
    body = re.split(r'Appendix(?! B refer)', text, maxsplit=1)[0]                  # «Appendix B refer…» — ссылка в Code and data availability
    body = re.split(r'\bA\s+Replication and tokenizer transfer', body)[0]
    full = re.sub(r'<!--.*?-->', '', open(os.path.join(HERE, 'draft_full.md'), encoding='utf-8').read(), flags=re.S)
    full = full.split('## Abstract', 1)[1].split('## References', 1)[0] + full.split('## Figure captions', 1)[1]
    full = full.replace('Title options', '')
    norm = lambda s: s.replace('−', '-').replace('–', '-')
    num = lambda s: Counter(re.findall(r'\d+(?:\.\d+)?', norm(s)))
    a, b = num(full), num(body.split('References')[0] + (body.split('References')[1] if 'References' in body else ''))
    print('страниц с текстом:', len(pages_text), '| знаков:', len(text))
    print('только в draft_full:', dict(a - b))
    print('только в PDF:', dict(b - a))
