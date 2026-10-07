"""Сборка LaTeX-препринта из Markdown-черновика (2026-10-07): docs/paper/latex/paper.tex, refs.bib, figures/*.pdf.
Источник текста — те же части, что у draft_full.md (build_draft.py), плюс draft_appendix.md. Комментарии-источники
<!-- … --> из LaTeX убираются (версия с ними — docs/paper/draft_full.md). Стандартный одноколоночный article, natbib (автор–год).
В конце — сверка: числа в теле paper.tex против draft_full.md (без комментариев) и приложений.

python docs/paper/build_latex.py
"""
import os, re, shutil
import build_draft as bd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..'); OUT = os.path.join(HERE, 'latex')
read = lambda p: open(p, encoding='utf-8').read().replace('\r\n', '\n')

BIB = {   # ключ → (поля); авторы и годы — по страницам arXiv (RELATED_WORK.md; год — первой версии)
    'Ahn 2022': ('ahn2022', 'Ahn, Michael and others', '{Do As I Can, Not As I Say}: Grounding Language in Robotic Affordances', '2204.01691'),
    'Azaria 2023': ('azaria2023', 'Azaria, Amos and Mitchell, Tom', "The Internal State of an {LLM} Knows When It's Lying", '2304.13734'),
    'Belinkov 2021': ('belinkov2021', 'Belinkov, Yonatan', 'Probing Classifiers: Promises, Shortcomings, and Advances', '2102.12452'),
    'Burns 2022': ('burns2022', 'Burns, Collin and others', 'Discovering Latent Knowledge in Language Models Without Supervision', '2212.03827'),
    'Dave 2024': ('dave2024', 'Dave, Akshay J. and others', 'Integrating {LLMs} for Explainable Fault Diagnosis in Complex Systems', '2402.06695'),
    'Elazar 2020': ('elazar2020', 'Elazar, Yanai and others', 'Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals', '2006.00995'),
    'Feng 2023': ('feng2023', 'Feng, Jiahai and Steinhardt, Jacob', 'How do Language Models Bind Entities in Context?', '2310.17191'),
    'Gekhman 2025': ('gekhman2025', 'Gekhman, Zorik and others', 'Inside-Out: Hidden Factual Knowledge in {LLMs}', '2503.15299'),
    'Hewitt 2019': ('hewitt2019', 'Hewitt, John and Liang, Percy', 'Designing and Interpreting Probes with Control Tasks', '1909.03368'),
    'Hu 2023': ('hu2023', 'Hu, Jennifer and Levy, Roger', 'Prompting is not a substitute for probability measurements in large language models', '2305.13264'),
    'Huang 2022': ('huang2022', 'Huang, Wenlong and others', 'Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents', '2201.07207'),
    'Kim 2023': ('kim2023', 'Kim, Najoung and Schuster, Sebastian', 'Entity Tracking in Language Models', '2305.02363'),
    'Li 2022': ('li2022', 'Li, Shuang and others', 'Pre-Trained Language Models for Interactive Decision-Making', '2202.01771'),
    'Li 2026': ('li2026', 'Li, Baoxue and Zhao, Chunhui', '{S2S-FDD}: Bridging Industrial Time Series and Natural Language for Explainable Zero-shot Fault Diagnosis', '2603.08048'),
    'Liu 2025': ('liu2025', 'Liu, Yichen and others', 'Evaluation of Large Language Models for Numeric Anomaly Detection in Power Systems', '2511.21371'),
    'Orgad 2024': ('orgad2024', 'Orgad, Hadas and others', '{LLMs} Know More Than They Show: On the Intrinsic Representation of {LLM} Hallucinations', '2410.02707'),
    'Reid 2022': ('reid2022', 'Reid, Machel and others', 'Can {Wikipedia} Help Offline Reinforcement Learning?', '2201.12122'),
    'Sun 2024': ('sun2024', 'Sun, Mingjie and Chen, Xinlei and Kolter, J. Zico and Liu, Zhuang', 'Massive Activations in Large Language Models', '2402.17762'),
    'Zhong 2024': ('zhong2024', 'Zhong, Ziqian and Andreas, Jacob', 'Algorithmic Capabilities of Random Transformers', '2410.04368'),
}

UNI = [('−', '$-$'), ('≈', '$\\approx$'), ('≤', '$\\le$'), ('≥', '$\\ge$'), ('×', '$\\times$'), ('±', '$\\pm$'), ('→', '$\\to$'),
       ('↔', '$\\leftrightarrow$'), ('…', '\\ldots{}'), ('—', '---'), ('–', '--'), ('~', '$\\sim$')]


def esc(t):
    t = t.replace('\\', '\\textbackslash{}')
    for a, b in (('&', '\\&'), ('%', '\\%'), ('#', '\\#'), ('_', '\\_'), ('$', '\\$'), ('^', '\\^{}'), ('{', '\\{'), ('}', '\\}')):
        t = t.replace(a, b)
    return t


FIGNAME = {'F1': 'probe', 'F2': 'words', 'F3': 'meaning', 'F4': 'synth'}


def inline(t):
    t = re.sub(r'<!--.*?-->', '', t, flags=re.S)
    t = re.sub(r'[ \t]+([.,;:)])', r'\1', t)                               # следы удалённых комментариев: пробел перед знаком
    t = re.sub(r'\([ \t]+', '(', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    figrefs = []
    t = re.sub(r'Fig\. (F[1-4])', lambda m: figrefs.append(FIGNAME[m.group(1)]) or f'\x02{len(figrefs) - 1}\x02', t)
    codes = []
    t = re.sub(r'`([^`]+)`', lambda m: codes.append(m.group(1)) or f'\x00{len(codes) - 1}\x00', t)
    cites = []
    def cite(m):
        keys = [k.strip() for k in m.group(1).split(';')]
        if all(k in BIB for k in keys):
            cites.append(','.join(BIB[k][0] for k in keys)); return f'\x01{len(cites) - 1}\x01'
        return m.group(0)
    t = re.sub(r'\[([A-Z][A-Za-z\-]+ \d{4}(?:; [A-Z][A-Za-z\-]+ \d{4})*)\]', cite, t)
    t = esc(t)
    sup = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789'))
    t = re.sub(r'(\d(?:\.\d+)?)·10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]+)', lambda m: '$' + m.group(1) + '\\cdot 10^{-' + ''.join(sup[c] for c in m.group(2)) + '}$', t)
    t = t.replace('∈', '$\\in$')
    t = t.replace('·', '$\\cdot$')
    for a, b in UNI:
        t = t.replace(a, b)
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?![\w*])', r'\\emph{\1}', t)
    t = re.sub(r'\x00(\d+)\x00', lambda m: '\\texttt{' + esc(codes[int(m.group(1))]) + '}', t)
    t = re.sub(r'\x01(\d+)\x01', lambda m: '\\citep{' + cites[int(m.group(1))] + '}', t)
    t = re.sub(r'\x02(\d+)\x02', lambda m: 'Figure~\\ref{fig:' + figrefs[int(m.group(1))] + '}', t)
    return t


def md2tex(md, figs=None):
    """Небольшой преобразователь для наших частей: заголовки ##/###, абзацы, списки (вложенные), таблицы."""
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    out, lines, i = [], md.split('\n'), 0
    stack = []                                                                  # открытые списки: (отступ, тип)

    def close_lists(level=-1):
        while stack and stack[-1][0] > level:
            out.append('\\end{%s}' % stack.pop()[1])
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            close_lists(); out.append(''); i += 1; continue
        m = re.match(r'^(#{2,3}) (.*)$', ln)
        if m:
            close_lists()
            title = re.sub(r'^[0-9A-C]+(\.\d+)? ', '', m.group(2)).replace('Appendix ', '')
            title = re.sub(r'^[A-C] ', '', title)
            out.append(('\\section{%s}' if len(m.group(1)) == 2 else '\\subsection{%s}') % inline(title))
            key = m.group(2).split(' ')[0]
            if figs and key in figs:
                out.append(figs[key])
            i += 1; continue
        if ln.startswith('|'):
            close_lists(); rows = []
            while i < len(lines) and lines[i].startswith('|'):
                if not re.match(r'^\|\s*-', lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            n = len(rows[0]); spec = 'p{%.2f\\linewidth}' % (0.92 / n)
            out.append('\\begin{center}\\small\\begin{tabular}{' + ' '.join([spec] * n) + '}\\hline')
            out.append(' & '.join('\\textbf{%s}' % inline(c) for c in rows[0]) + ' \\\\ \\hline')
            for r in rows[1:]:
                out.append(' & '.join(inline(c) for c in r) + ' \\\\')
            out.append('\\hline\\end{tabular}\\end{center}'); continue
        m = re.match(r'^(\s*)([-*]|\d+\.) (.*)$', ln)
        if m:
            ind, typ = len(m.group(1)), 'itemize' if m.group(2) in '-*' else 'enumerate'
            if not stack or ind > stack[-1][0]:
                out.append('\\begin{%s}' % typ); stack.append((ind, typ))
            else:
                close_lists(ind)
            item = m.group(3); i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r'^\s*([-*]|\d+\.) ', lines[i]) and not lines[i].startswith(('|', '#')) \
                    and (len(lines[i]) - len(lines[i].lstrip())) > ind:
                item += ' ' + lines[i].strip(); i += 1
            out.append('\\item ' + inline(item)); continue
        close_lists(); para = ln.strip(); i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|')) and not re.match(r'^\s*([-*]|\d+\.) ', lines[i]):
            para += ' ' + lines[i].strip(); i += 1
        out.append(inline(para))
    close_lists()
    return '\n'.join(out)


def caption_blocks():
    caps = {}
    for blk in bd.CAPTIONS.strip().split('\n\n'):
        m = re.match(r'\*\*Figure (F\d) — ([^*]+)\*\*\s*(.*?)\s*File: `([^`]+)`\.', blk, flags=re.S)
        fid, title, text, path = m.groups(); caps[fid] = (title.strip(), text.strip(), path)
    return caps


def main():
    os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
    caps = caption_blocks(); figs = {}
    place = {'3.1': 'F2', '3.3': 'F1', '3.4': 'F3', '3.5': 'F4'}
    for sec, fid in place.items():
        title, text, path = caps[fid]
        shutil.copy(os.path.join(ROOT, path), os.path.join(OUT, 'figures', os.path.basename(path)))
        figs[sec] = ('\\begin{figure}[t]\\centering\\includegraphics[width=\\linewidth]{figures/%s}\n\\caption{\\textbf{%s.} %s}\\label{fig:%s}\\end{figure}'
                     % (os.path.basename(path), inline(title.rstrip('.')), inline(text), FIGNAME[fid]))
    outline = read(os.path.join(ROOT, 'docs', 'PAPER_OUTLINE.md'))
    abstract = outline.split('## 1. Аннотация\n', 1)[1].split('\n### Сверка', 1)[0].strip().split('\n\n', 1)[1]
    disclosure = outline.split('## 4. Раскрытие\n', 1)[1].split('\n## ', 1)[0].strip()
    parts = [read(os.path.join(HERE, f)) for f in ('draft_intro.md', 'draft_setup_results.md', 'draft_pitfalls.md', 'draft_discussion.md')]
    parts[1] = parts[1].split('## 2 Setup', 1)[1]; parts[1] = '## 2 Setup' + parts[1]
    body = '\n\n'.join(parts)
    appendix = read(os.path.join(HERE, 'draft_appendix.md'))
    used = []
    for grp in re.findall(r'\[([A-Z][^\]]*?\d{4}[^\]]*)\]', body):
        for k in grp.split(';'):
            k = k.strip()
            if k in BIB and k not in used:
                used.append(k)
    tex = r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{iftex}
\ifXeTeX\usepackage{fontspec}\else\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{lmodern}\fi
\usepackage{amsmath,amssymb,graphicx,booktabs,array}
\usepackage[authoryear,round]{natbib}
\usepackage[hidelinks]{hyperref}
\bibliographystyle{plainnat}
\setcitestyle{authoryear,round,semicolon}
% Название — выбор автора; второй вариант: Represented but Not Reported: Probing Frozen Language Models as Embodied Readers
\title{Knowing More Than It Can Say: Frozen Language Models in Minimal Bodies}
\author{Egor Grigorev\\ ORCID: \href{https://orcid.org/0009-0000-2725-5466}{0009-0000-2725-5466}}
\date{Preprint, October 2026}
\begin{document}
\maketitle
\begin{abstract}
''' + inline(' '.join(abstract.split())) + r'''
\end{abstract}

''' + md2tex(body, figs) + r'''

\section*{Disclosure}
''' + inline(' '.join(disclosure.split())) + r'''

\bibliography{refs}

\appendix
''' + md2tex(appendix) + r'''

\end{document}
'''
    tex = tex.replace('\\section{Introduction}', '\\section{Introduction}\\label{sec:intro}')
    open(os.path.join(OUT, 'paper.tex'), 'w', encoding='utf-8', newline='\n').write(tex)
    bib = '\n'.join('@misc{%s,\n  author = {%s},\n  title = {%s},\n  year = {%s},\n  eprint = {%s},\n  archivePrefix = {arXiv},\n  howpublished = {arXiv:%s}\n}\n'
                    % (BIB[k][0], BIB[k][1], BIB[k][2], k.split()[1], BIB[k][3], BIB[k][3]) for k in sorted(used))
    open(os.path.join(OUT, 'refs.bib'), 'w', encoding='utf-8', newline='\n').write(bib)
    # сверка чисел: тело paper.tex (без приложений) против draft_full.md без комментариев
    num = lambda s: sorted(re.findall(r'\d+(?:\.\d+)?', s))
    full = re.sub(r'<!--.*?-->', '', read(os.path.join(HERE, 'draft_full.md')), flags=re.S)
    full = full.split('## Abstract', 1)[1].split('## References', 1)[0] + full.split('## Figure captions', 1)[1]
    full = full.replace('Title options', '')
    tbody = tex.split('\\begin{abstract}', 1)[1].split('\\appendix', 1)[0]
    tbody = re.sub(r'\\(includegraphics|label|citep|bibliography)\{[^}]*\}', '', tbody)
    a, b = num(full), num(tbody)
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    print('paper.tex:', len(tex), 'знаков; ссылок в refs.bib:', len(used))
    print('числа только в draft_full:', dict(ca - cb)); print('числа только в paper.tex:', dict(cb - ca))


if __name__ == '__main__':
    import sys; sys.path.insert(0, HERE); main()
