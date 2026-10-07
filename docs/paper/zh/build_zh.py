"""Китайская версия препринта (2026-10-07, программа Егора «ссылка на репозиторий и китайская версия», п. 2).
Источник — перевод английского текста после п. 1: docs/paper/zh/*_zh.md (структура заголовков и таблиц — как у английских
частей). Преобразование Markdown → LaTeX — те же функции, что у английской версии (docs/paper/build_latex.py: md2tex, inline,
BIB); библиография — та же refs.bib (на английском). Рисунки — figures/zh/ (make_figures.py --lang zh).
XeLaTeX + ctex; шрифты Noto Serif CJK SC / Noto Sans CJK SC из docs/paper/zh/fonts/ (вне git, OFL, github.com/notofonts/noto-cjk).

python docs/paper/zh/build_zh.py          # пишет docs/paper/zh/main_zh.tex
tectonic -X compile docs/paper/zh/main_zh.tex
"""
import os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
sys.path.insert(0, PAPER)
import build_latex as bl                                                        # noqa: E402

read = lambda f: open(os.path.join(HERE, f), encoding='utf-8').read().replace('\r\n', '\n')
FONTS = ['NotoSerifCJKsc-Regular.otf', 'NotoSerifCJKsc-Bold.otf', 'NotoSansCJKsc-Regular.otf', 'NotoSansCJKsc-Bold.otf']
NOTE = '本文为英文预印本的中文译本，由 AI 辅助翻译。如有出入，以英文原文为准。'
TITLE_ZH = '所知多于所言：极简身体中的冻结语言模型'
TITLE_EN = 'Knowing More Than It Can Say: Frozen Language Models in Minimal Bodies'


def captions():
    caps = {}
    for blk in read('captions_zh.md').strip().split('\n\n'):
        m = re.match(r'\*\*Figure (F\d) — ([^*]+)\*\*\s*(.*?)\s*File: `([^`]+)`\.', blk, flags=re.S)
        fid, title, text, path = m.groups(); caps[fid] = (title.strip(), text.strip(), path)
    return caps


def main():
    for f in FONTS:
        assert os.path.exists(os.path.join(HERE, 'fonts', f)), f'нет шрифта fonts/{f}'
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    caps = captions(); figs = {}
    place = {'3.1': 'F2', '3.3': 'F1', '3.4': 'F3', '3.5': 'F4'}               # как в английской версии
    for sec, fid in place.items():
        title, text, path = caps[fid]
        shutil.copy(os.path.join(ROOT, path), os.path.join(HERE, 'figures', os.path.basename(path)))
        figs[sec] = ('\\begin{figure}[t]\\centering\\includegraphics[width=\\linewidth]{figures/%s}\n\\caption{\\textbf{%s。}%s}\\label{fig:%s}\\end{figure}'
                     % (os.path.basename(path), bl.inline(title.rstrip('。')), bl.inline(text), bl.FIGNAME[fid]))
    body = '\n\n'.join(read(f) for f in ('intro_zh.md', 'setup_results_zh.md', 'pitfalls_zh.md', 'discussion_zh.md'))
    tex = r'''\documentclass[UTF8,fontset=none,a4paper]{ctexart}
\usepackage[a4paper,margin=2.5cm]{geometry}
\setCJKmainfont{NotoSerifCJKsc-Regular.otf}[Path=fonts/,BoldFont=NotoSerifCJKsc-Bold.otf]
\setCJKsansfont{NotoSansCJKsc-Regular.otf}[Path=fonts/,BoldFont=NotoSansCJKsc-Bold.otf]
\setCJKmonofont{NotoSansCJKsc-Regular.otf}[Path=fonts/]
\usepackage{amsmath,amssymb,graphicx,booktabs,array}
\usepackage[authoryear,round]{natbib}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\def\UrlBreaks{\do\/\do-\do.}
\bibliographystyle{plainnat}
\setcitestyle{authoryear,round,semicolon}
\ctexset{section/format=\Large\bfseries\raggedright}
\pagestyle{plain}
\title{''' + TITLE_ZH + r'''\\[0.5em]\large ''' + TITLE_EN + r'''\\[0.8em]\normalsize ''' + NOTE + r'''}
\author{Egor Grigorev\\ ORCID: \href{https://orcid.org/0009-0000-2725-5466}{0009-0000-2725-5466}}
\date{预印本，2026 年 10 月}
\begin{document}
\maketitle
\begin{abstract}
''' + bl.inline(' '.join(read('abstract_zh.md').split())) + r'''
\end{abstract}

''' + bl.md2tex(body, figs) + r'''

\section*{声明}
''' + bl.inline(' '.join(read('disclosure_zh.md').split())) + r'''

\section*{代码与数据获取}
''' + bl.inline(' '.join(read('availability_zh.md').split())) + r'''

\bibliography{refs}

\appendix
''' + bl.md2tex(read('appendix_zh.md')) + r'''

\end{document}
'''
    tex = tex.replace('Figure~\\ref{', '图~\\ref{')
    open(os.path.join(HERE, 'main_zh.tex'), 'w', encoding='utf-8', newline='\n').write(tex)
    shutil.copy(os.path.join(PAPER, 'latex', 'refs.bib'), os.path.join(HERE, 'refs.bib'))
    print('main_zh.tex:', len(tex), 'знаков')


if __name__ == '__main__':
    main()
