# Сверка чисел: английский PDF → китайский PDF

Источник: `docs/paper/latex/paper.pdf` и `docs/paper/zh/main_zh.pdf`; скрипт `docs/paper/zh/check_zh.py`.

| Раздел | Чисел (en) | Чисел (zh) | Совпадает с порядком | Расхождения |
| --- | --- | --- | --- | --- |
| Abstract | 17 | 17 | да | — |
| 1 Introduction | 18 | 18 | да | — |
| 2 Setup | 49 | 49 | да | — |
| 3 Results | 201 | 201 | да | — |
| 4 Pitfalls and controls | 34 | 34 | да | — |
| 5 Discussion | 17 | 17 | да | — |
| Disclosure | 0 | 0 | да | — |
| Code and data availability | 1 | 1 | да | — |
| References | 39 | 39 | да | — |
| A Replication and tokenizer transfer | 88 | 88 | да | — |
| B Registry | 111 | 111 | да | — |
| C Compute and costs | 41 | 41 | да | — |
| Figure 1 (caption) | 6 | 6 | да | — |
| Figure 2 (caption) | 7 | 7 | да | — |
| Figure 3 (caption) | 16 | 16 | да | — |
| Figure 4 (caption) | 13 | 13 | да | — |

Разделов с расхождениями: 0 из 16.

## Приложение B: en и zh построчно, rev-parse в свежем клоне EMB (правки после чистки, 2026-10-07)

| Id | раздел en/zh | метки | коммиты | даты | rev-parse в EMB |
| --- | --- | --- | --- | --- | --- |
| A2 | 3.1 / 3.1 | = v2-scale | = 6af7ab7 | = 2026-10-01 | v2-scale→6af7ab7; 6af7ab7:6af7ab7 ✓ |
| A3 | 3.1 / 3.1 | = v2-interface | = 5476edf | = 2026-10-02 | v2-interface→5476edf; 5476edf:5476edf ✓ |
| A4 | 3.2 / 3.2 | = v3-words | = ebab915 | = 2026-10-02 | v3-words→ebab915; ebab915:ebab915 ✓ |
| A13 | 4 / 4 | = v3-replay | = bcffc98 | = 2026-10-03 | v3-replay→bcffc98; bcffc98:bcffc98 ✓ |
| A22 | 3.3 / 3.3 | = v4-report | = 1f2be62 | = 2026-10-04 | v4-report→1f2be62; 1f2be62:1f2be62 ✓ |
| A23 | 3.3 / 3.3 | = v4-icl | = 9cc7aea | = 2026-10-04 | v4-icl→9cc7aea; 9cc7aea:9cc7aea ✓ |
| A24 | 3.4 / 3.4 | = v5-meaning | = bfb0c62 | = 2026-10-05 | v5-meaning→bfb0c62; bfb0c62:bfb0c62 ✓ |
| A25 | 4 / 4 | = — | = f114250 | = 2026-10-05 | f114250:f114250 ✓ |
| A26 | 3.5, 4 / 3.5, 4 | = v5-bind | = ac18bce | = 2026-10-06 | v5-bind→ac18bce; ac18bce:ac18bce ✓ |
| A27 | 3.5, 4 / 3.5, 4 | = v5-synth | = 1719022, 6462459, d0e7b01, efedb20 | = 2026-10-06 | v5-synth→d0e7b01; 1719022:1719022; 6462459:6462459; d0e7b01:d0e7b01; efedb20:efedb20 ✓ |
| A28 | all / 全部 | = v7-replication | = ec88b6f | = 2026-10-06 | v7-replication→ec88b6f; ec88b6f:ec88b6f ✓ |

строк en 11, zh 11; с расхождениями: 0

Ссылки в обоих PDF после пересборки: ORCID, репозиторий, ATTEMPTS.md по метке v1.1-preprint — все 200. Переносов с дефисом и тире в китайском PDF нет: названия с дефисом, даты и диапазоны собраны в \mbox; автоматические переносы латинских слов отключены (\hyphenpenalty=10000 xhyphenpenalty=10000, mergencystretch=3em).
