# План статьи — фокус 1: «модель знает больше, чем может сказать»

> **Статус 2026-10-07.** Разделы 1, 2 и 4 — текст Егора, вставлен дословно (язык работы — английский). Сверка чисел — под аннотацией; расхождения отмечены, текст не правился.

## 1. Аннотация
Title options:
- Knowing More Than It Can Say: Frozen Language Models in Minimal Bodies
- Represented but Not Reported: Probing Frozen Language Models as Embodied Readers

Frozen pretrained language models are increasingly used as components of embodied agents, yet it
is unclear what pretraining contributes in such settings and how much of what the model internally
represents is available through its output. We study this in a minimal two-dimensional foraging
environment where an agent's body events are presented to a frozen language model as a short
sequence of discrete tokens, and compare each pretrained model against the same architecture with
random weights and matched embedding norms. Across three model families (Pythia, Qwen2.5, OLMo-2;
1–1.5B parameters, with Pythia up to 6.9B for some analyses), we find that: (i) the advantage of
pretraining comes from sequence-processing machinery rather than word meaning — arbitrary event
words work as well as or better than meaningful ones; (ii) a latent episode variable that is never
stated explicitly in the context is linearly decodable from hidden states with AUC 0.93–0.99, but
the model's own answer to a direct question recovers it only at AUC 0.61–0.64, and in Pythia,
in-context examples from other episodes do not close this gap; (iii) the meaning of a word (food
vs. harmful) is fully present at that word's position but does not reach the final position where
an episode-level readout would use it (0.97–1.00 vs. 0.51–0.62); and (iv) when word meaning
conflicts with in-context experience, a statistically reliable trace of the experience reaches the
answer but is substantially weaker than the meaning effect (4–27% of it) and leaves the answers
essentially unchanged. All analyses were preregistered with timestamped commits, use exact
permutation tests with Holm correction, and include controls for confounds that produced false
positives during the project, such as survivorship-induced frequency cues. Taken together, the
results suggest that frozen language models in this regime represent substantially more about
their embodied situation than they can express, and that the bottleneck lies in routing and
expression rather than in representation.

### Сверка чисел аннотации и плана с разделом 5 (2026-10-07; текст выше не правился)
| Место в тексте | В тексте | По отчётам | Итог |
| --- | --- | --- | --- |
| Аннотация: модели | Pythia, Qwen2.5, OLMo-2; 1–1.5B, Pythia до 6.9B | A24, A26, A27 — Pythia 1B–6.9B; A28 — Qwen2.5-1.5B, OLMo-2-1B, Pythia-1B | совпадает |
| (i) произвольные слова | as well as or better | Δ +0,016 / +0,078 / +0,024; A4 (160M) +0,016 | совпадает |
| (ii) зонд | AUC 0.93–0.99 | 0,931–0,993 (A28, три вопроса, три модели); A22 0,98–0,99 | совпадает |
| (ii) ответ | AUC 0.61–0.64 | 0,607–0,637 (A28); A22 0,61 | совпадает при округлении |
| (ii) примеры из других эпизодов | in Pythia … do not close this gap | A23: AUC ≈ 0,5 у Pythia 160M и 1B | совпадает |
| (iii) позиция слова / последняя позиция | 0.97–1.00 vs. 0.51–0.62 | 0,972–0,999 / 0,514–0,620 (A24, A28) | совпадает |
| (iii) «does not reach» | не доходит | у Pythia-6.9B на последней позиции 0,620; «значимо выше случайной (Холм 0,0009)» в A24 опиралось на бутстреп по эпизодам, который занижает интервалы; по словам — 0,620 [0,51; 0,73] против 0,447 [0,33; 0,56] у случайной | **оговорка:** «доходит самое большее слабо»; значимость разницы со случайной у 6.9B не установлена |
| (iv) доля сдвига от эффекта смысла | 4–27% | сдвиг / \|d без связки\| на лживых окнах, q1, интерфейс s: Pythia-6.9B 0,020 / 0,527 = 3,8 % (A27); Pythia-1B 0,022 / 0,284 = 7,7 %; OLMo-2 0,044 / 0,701 = 6,3 %; Qwen2.5 0,098 / 0,355 = 27,6 % (A28) | **расхождение в округлении:** верхняя граница 27,6 % → «28 %»; нижняя 3,8 % → «4 %» верно. В разделе 5 было «4–11 %» (A27 описательно, с правдивыми окнами и интерфейсом c) — это другая мера; для лживых окон и q1 диапазон 4–28 % |
| (iv) ответы не меняются | essentially unchanged | доля лживых окон с ответом к опыту меняется на 0,004–0,020 | совпадает |
| «All analyses were preregistered» | все | основные семьи A3, A4, A22, A23, A24, A26, A27, A28 — да; **анализ копирования A26 (`bind_posthoc.py`) — после расчёта**, помечен так в отчёте | **оговорка:** «all primary analyses»; анализ A26 — после данных |
| «exact permutation tests with Holm» | все | A3, A4, A22, A23, A26, A27, A28 — точные перестановки, Холм; **A24 — односторонний бутстреп** (10 000), Холм по шести | **расхождение:** для A24 — бутстреп |
| План, п. 2: «fixed standardization» | фиксированная | фиксированная стандартизация — у головы агента стенда; в зондах A3/A4/A28 (R1, R2) признаки стандартизуются внутри частей CV (`probe_fast._std`), в A22 / A24 — так же по обучающей части | **расхождение:** для зондов — стандартизация по обучающей части |
| План, п. 3.3: случайные веса, зонд | 0.89–0.94 | 0,892–0,937 (A28, R3; средние по наборам) | совпадает при округлении |
| План, п. 3.3: «pretraining mainly adds to the answer channel» | — | ответ: случайные 0,50–0,53 → предобученные 0,61–0,64 (+0,09…+0,13); зонд: 0,89–0,94 → 0,93–0,99 (+0,04…+0,06) | согласуется (описательно, без теста) |
| План, п. 3.1: massive activations | — | касается артефакта съёма v0 (п. 1, стандартизация), а не A3 | **оговорка:** в 3.1 это контекст, а не результат A3 |

## 2. План
1. Introduction. Frozen LMs as agent components; what pretraining contributes and how much of the
   internal state is accessible through output. Contributions: three-layer finding (representation,
   routing, expression); cross-family replication; preregistered methodology with documented pitfalls.
2. Setup. Environment (flip-2); interfaces (continuous projector vs. event words with a start token;
   synthetic windows); models and controls (random weights with matched embedding norms; fixed
   standardization; fp32); measures (probes with episode-grouped CV, logit-based answers);
   exact permutation tests, Holm; preregistration and attempt registry.
3. Results.
   3.1 Interface determines whether pretraining helps (A3; massive activations; R1 — replicates on
       OLMo-2, symbol-set-dependent on Qwen2.5).
   3.2 Machinery, not meaning (A4, R2 — all three families).
   3.3 Represented but not reported (A22, R3; A23 — no in-context learning from other episodes,
       Pythia only). Note: random-weight probes are also high (0.89–0.94); pretraining mainly adds
       to the answer channel.
   3.4 Meaning stays local (A24, R4).
   3.5 Meaning vs. experience: a weak trace that does not change answers (A27, R5).
4. Pitfalls and controls. Survivorship-induced frequency cue (A26); symmetric conditions against
   random-weight baselines; information loss in symbolic interfaces; massive activations and readout
   standardization.
5. Discussion. Three layers: representation, routing, expression. Relation to probing-vs-accessibility
   and symbol-binding work (RELATED_WORK.md). Limitations: model sizes, toy environment, frozen models
   only, no trained report channel.
6. Appendices. Replication and tokenizer transfer; Pythia-only results; preregistration timeline;
   attempt registry; compute and costs.

## 3. Графики (готовы; `make_figures.py` → `figures/F*.pdf`, `figures/F*.png`)
| № | Что показывает | Данные | n и интервалы |
| --- | --- | --- | --- |
| F1 | Зонд против ответа: AUC линейного зонда на позиции вопроса и AUC ответа (разность логитов «A» − «B») против знака эпизода, три вопроса без примеров; предобученная и случайные веса | A28 (протокол A22): `results_replication/<модель>/report_*` — Pythia-1B, Qwen2.5-1.5B, OLMo-2-1B, один экземпляр 5090 | 10 наборов × 100 эпизодов; 95 % t-интервал по наборам |
| F2 | Осмысленные слова событий против произвольных и случайных весов: точность зонда знака | A28 (протоколы A3, A4): `results_replication/<модель>/r12.json` | 10 наборов × 200 эпизодов; 95 % t-интервал по наборам |
| F3 | Зонд смысла «еда / вред» по позициям: позиция символа вида и последняя позиция окна; предобученная и случайные веса | A24: `results_meaning/` (Pythia 1B / 2.8B / 6.9B); A28: `results_replication/` (Qwen2.5-1.5B, OLMo-2-1B) — **разные экземпляры 5090, сравнение описательное** | 200 эпизодов, один прогон, ≈ 56 слов; 95 % ДИ — **кластерный бутстреп по словам вида A** (2000; исправлено 2026-10-07 — бутстреп по эпизодам занижал интервалы, см. `RESULTS_2026-10-05_meaning.md`, «Уточнение 2026-10-07») |
| F4 | Синтетические окна без частотного канала: а) эффект смысла (\|d\| без связки) против сдвига к опыту (d − d без связки), у предобученной и у случайной; б) доля лживых окон с ответом к опыту — с исходами и с переставленными | A28 (протокол A27): `results_replication/` (Pythia-1B, Qwen2.5-1.5B, OLMo-2-1B); A27: `results_synth/synth_6.9b.json` (Pythia-6.9B, **другой экземпляр**, помечен на графике) | 10 наборов × 648 окон; 95 % t-интервал по наборам |

**Отступление от программы.** В списке источников графиков (results_replication, results_report, results_meaning, results_bind) нет `results_synth/`, а Pythia-6.9B в синтетических окнах есть только там (A27). F4 берёт её оттуда. `results_report/` (A22 на ноутбуке) и `results_bind/` (A26) в графиках не использованы: F1 строится по A28 (одна машина для трёх семейств), а A26 — анализ копирования, его в списке графиков нет.

**Оформление:**
- в F1–F3 ось не от нуля, поэтому точки с интервалами, а не столбцы; в F4 — столбцы от нуля;
- на каждом графике указаны модель, n и тип интервала;
- F3: у случайных весов значения ниже 0,5 (Pythia-2.8B 0,315, Qwen2.5 0,343) — не артефакт разбиения, а разброс одиночной случайной сети при ≈ 56 словах: моделирование той же схемы даёт среднее 0,495 и разброс 0,36–0,63; у Pythia-6.9B то же значение выше 0,5 (0,645). Подпись под графиком это поясняет;
- английские подписи — `python make_figures.py --lang en` → `figures/en/`; русские — `--lang ru` → `figures/`;
- цвета по ролям: синий — зонд / эффект смысла, оранжевый — ответ / опыт; полые маркеры и штриховка — случайные веса или контроль без связки.

## 4. Раскрытие
Code implementation and experiment execution were performed with an AI coding assistant (Claude Code)
under the author's direction; hypotheses, preregistrations, decisions to stop or continue each line,
and interpretation are the author's. All code, preregistration commits, and results are released.

## 5. Сверочная таблица чисел (по отчётам и реестру; для проверки аннотации)
| Утверждение | Число | Источник |
| --- | --- | --- |
| Знак эпизода в представлении: зонд на позиции вопроса | 0,993 / 0,932 / 0,975 (Pythia-1B / Qwen2.5-1.5B / OLMo-2-1B, zs1); у Pythia 160M и 1B в A22 — 0,98–0,99 | `RESULTS_2026-10-06_replication.md`; `RESULTS_2026-10-04_report.md` |
| Ответ без примеров (AUC разности логитов) | 0,607–0,612 / 0,623–0,637 / 0,615–0,627 по трём вопросам; A22: 0,61 (≈ 22 % от зонда в шкале AUC − 0,5) | там же |
| Разрыв выражения: (ответ − 0,5) ≤ 0,5·(зонд − 0,5) | выполнен у всех трёх моделей по всем трём вопросам | A28, R3 |
| Предобученная лучше случайной на событиях словами | Pythia-160M 0,950 против 0,813 (A3, Холм 0,004); Pythia-1B 0,976 против 0,906; OLMo-2 0,963 против 0,938 (Холм 0,006); **Qwen2.5 0,906 против 0,921 — не воспроизвелось** | `RESULTS_2026-10-01_flip.md` (v2-interface); A28, R1 |
| Произвольные слова не хуже осмысленных | Pythia-1B 0,992 (Δ +0,016), Qwen2.5 0,983 (Δ +0,078), OLMo-2 0,987 (Δ +0,024); все > случайной (Холм 0,006); A4 (160M): −0,016 в пользу произвольных | A28, R2; A4 |
| Смысл слов «еда / вред» по словам (зонд с исключением слова) | 1,000 / 1,000 / 0,999 (Pythia 1B / 2.8B / 6.9B); Qwen2.5 1,000, OLMo-2 1,000 | A24; A28, R4 |
| Смысл на позиции символа вида / на последней позиции окна | Pythia 0,983–0,999 / 0,526–0,620; Qwen2.5 0,972 / 0,595; OLMo-2 0,996 / 0,514 | A24; A28, R4 |
| Положительный контроль (прямые слова) | Pythia 1,000 / 0,967 / 0,967; Qwen2.5 0,900; OLMo-2 0,967 | A24; A28 |
| Смысл без опыта (M) | 0,83–0,95 во всех 6 ячейках (1B / 2.8B / 6.9B × c / s) | A26 |
| Сдвиг к опыту при реальных траекториях — копирование | +1,1…+1,9 логита на упоминание; частотный ориентир 0,62 (отбор выживших); при равных упоминаниях B ≈ 0 в 5 из 6 | A26 (анализ после расчёта — выведено) |
| Связка «слово ↔ исход» без частотного канала | 6.9B s: сдвиг +0,020 логита (Холм 0,002), 4 % от эффекта смысла (−0,51); описательно 4–11 % (1B, 6.9B; s, c) | A27 |
| То же на трёх семействах | сдвиг Pythia-1B +0,022, Qwen2.5 +0,098, OLMo-2 +0,044; у случайных ≈ 0 (Холм 0,006) | A28, R5 |
| Ответ при конфликте не меняется | доля лживых окон с ответом к опыту: 6.9B 0,36 и 0,36; Pythia-1B 0,450 / 0,444; Qwen2.5 0,464 / 0,453; OLMo-2 0,392 / 0,372 (с исходами / без связки) | A27; A28 |
| Доля сдвига к опыту от эффекта смысла (лживые окна, q1, s) | 3,8 % (Pythia-6.9B, A27); 7,7 % (Pythia-1B), 6,3 % (OLMo-2), 27,6 % (Qwen2.5) (A28) | A27; A28 (`results_replication/*/synth_*`, `results_synth/synth_6.9b.json`) |
| Рост с размером | связка: нет (1B → 6.9B); смысл на последней позиции: 0,53 → 0,59 → 0,62 (ниже 0,8; значимость разницы со случайной у 6.9B после уточнения интервалов не установлена) | A27; A24 (уточнение 2026-10-07) |
