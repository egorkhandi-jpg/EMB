# Связанные работы — фокус 1: «модель знает больше, чем может сказать»

Включены только работы с проверенным идентификатором.
- **Раздел A** — работы, уже проверенные в проекте (`docs/THESES.md` п. 1.7, `docs/FINAL_REPORT.md` п. 12; страницы открыты 2026-10-04 и 2026-10-05).
- **Раздел B** — новые кандидаты. Страница arXiv каждой открыта 2026-10-07 (HTTP 200); название и авторы — из метаданных страницы, дата — из истории версий, место публикации — только из поля Comments, если авторы его указали (Journal-ref нет ни у одной). Краткое содержание — пересказ аннотации; полные тексты не читались.

Непроверяемое не включено.

**Год в ссылках** (правило с 2026-10-07): год первой версии arXiv (по истории версий, совпадает с префиксом идентификатора); место публикации — отдельно и только по полю Comments. Поэтому, например, Elazar et al. — **2020** (arXiv 2006.00995, v1 1 июня 2020; TACL — по Comments), Burns et al. — 2022 (ICLR 2023 — по Comments).

## A. Уже проверенные в проекте
| Работа | Идентификатор | Связь с фокусом 1 |
| --- | --- | --- |
| Zhong, Andreas. Algorithmic Capabilities of Random Transformers (NeurIPS 2024) | arXiv 2410.04368 | случайный трансформер без обучения уже хранит знак (A1) — контроль «случайные веса» в F1–F3 |
| Pacela, Joshi, Camacho, Lacoste-Julien, Klindt. Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation | arXiv 2603.28744 | ограничения линейных зондов (A22, A24): «зонд читает» ≠ «модель использует» |
| Lindsey. Emergent Introspective Awareness in Large Language Models | arXiv 2601.01828 | модель иногда замечает и называет внедрённое понятие, ненадёжно — к разрыву «представление → отчёт» (A22, A23) |
| Lin, Hofmann, Wan, …, Pierrehumbert. Can Large Language Models Generalize Procedures Across Representations? (ICML 2026) | arXiv 2602.03542 | знание в одном представлении не переносится само в другое — к A24 (смысл есть, не доходит до читаемой позиции) |
| Işık, Li. Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning (ICML 2026) | arXiv 2601.23169 | привязка к символам, а не к роли — к A4 / R2 (произвольные слова не хуже) и A20–A21 |
| Ankner, Renda, Carbin. Renamer: A Transformer Architecture Invariant to Variable Renaming (NeurIPS 2023 Workshop MLSys) | OpenReview AilSfYM8wN | обычные трансформеры зависят от имён там, где не должны — к A4 / R2 |
| Takagi. On the Effect of Pre-training for Transformer in Different Modality on Offline Reinforcement Learning (NeurIPS 2022) | arXiv 2211.09817 | предобучение в чужой модальности — к A1–A2 |
| Sun, Chen, Kolter, Liu. Massive Activations in Large Language Models (COLM 2024) | arXiv 2402.17762 | сверхбольшие активации — артефакт съёма, отсюда стандартизация признаков |
| Lu, Grover, Abbeel, Mordatch. Pretrained Transformers as Universal Computation Engines | arXiv 2103.05247 | замороженный предобученный трансформер в задачах вне языка |
| Mirchandani et al. Large Language Models as General Pattern Machines (CoRL 2023) | arXiv 2307.04721 | LLM продолжают последовательности токенов без смысла — к интерфейсу событий словами |
| Driess et al. PaLM-E: An Embodied Multimodal Language Model | arXiv 2303.03378 | сенсорные модальности во входе LM (в отличие от нашей замороженной LM) |

## B. Новые кандидаты (страницы открыты 2026-10-07)
### Зонды против доступности знания
| Работа | Идентификатор | По аннотации | Связь |
| --- | --- | --- | --- |
| Hewitt, Liang. Designing and Interpreting Probes with Control Tasks (EMNLP 2019, по Comments) | arXiv 1909.03368 | Контрольные задачи со случайными метками и «избирательность» зонда; популярные зонды не избирательны | методический: наш контроль — та же архитектура со случайными весами, а не случайные метки |
| Belinkov. Probing Classifiers: Promises, Shortcomings, and Advances (Computational Linguistics, squib, по Comments) | arXiv 2102.12452 | Критический обзор метода зондов и его ограничений | рамка для «зонд читает ≠ модель использует» |
| Elazar et al. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals (TACL, по Comments) | arXiv 2006.00995 | Точность зонда не говорит, использует ли модель свойство; проверка удалением свойства | к разрыву зонд / ответ (F1, F3) |
| Burns et al. Discovering Latent Knowledge in Language Models Without Supervision (ICLR 2023, по Comments) | arXiv 2212.03827 | Направление в активациях без учителя (логическая согласованность) превосходит ответы без примеров | ближайший по духу: «знает» отделено от «говорит» |
| Azaria, Mitchell. The Internal State of an LLM Knows When It's Lying | arXiv 2304.13734 | Классификатор по скрытым состояниям определяет истинность лучше вероятности предложения | к F1: зонд по представлению сильнее ответа |
| Orgad et al. LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations | arXiv 2410.02707 | Модель может кодировать верный ответ и при этом порождать неверный | прямой аналог формулировки фокуса 1 (на фактах, а не на опыте тела) |
| Kadavath et al. Language Models (Mostly) Know What They Know | arXiv 2207.05221 | Крупные модели откалиброваны и могут оценивать собственную правоту | контраст: у нас без обучения отчёт о состоянии тела слаб |
| Hu, Levy. Prompting is not a substitute for probability measurements in large language models (EMNLP 2023, по Comments) | arXiv 2305.13264 | Метаязыковые суждения через запрос хуже прямых вероятностей; отрицательный результат запроса не доказывает отсутствия знания | методический: почему мера ответа — логиты, а не текст; и почему низкий ответ ≠ «не знает» |
| Gekhman et al. Inside-Out: Hidden Factual Knowledge in LLMs (COLM 2025, по Comments) | arXiv 2503.15299 | Внутреннее и внешнее знание формально различены; внутренне кодируется заметно больше фактов, чем выражается | количественная параллель к F1 (на фактах) |

### Связывание символов в контексте
| Работа | Идентификатор | По аннотации | Связь |
| --- | --- | --- | --- |
| Feng, Steinhardt. How do Language Models Bind Entities in Context? | arXiv 2310.17191 | «Идентификаторы связывания» — векторы, прикрепляющие атрибуты к сущностям, у достаточно крупных Pythia и LLaMA | к A27 / R5 (слабая связка «слово вида ↔ исход») и A24 (смысл не доходит до последней позиции) |
| Olsson et al. In-context Learning and Induction Heads | arXiv 2209.11895 | Индукционные головы как возможный основной механизм обучения в контексте | наш A7: удаление верхних индукционных голов чтение знака не ухудшает; A26 — копирование чаще упомянутого |
| Kim, Schuster. Entity Tracking in Language Models (ACL 2023, по Comments) | arXiv 2305.02363 | Слежение за состоянием сущностей после серии операций; у моделей, предобученных только на тексте, способность не проявляется | к A27 (вывод из последовательности событий без обучения слаб) |
| Prakash et al. Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking (ICLR 2024, по Comments) | arXiv 2402.14811 | Дообучение усиливает уже существующую цепь слежения за сущностями, а не заменяет её | к открытому вопросу «звено 4 после обучения» |

### Замороженные или предобученные LM в агентах
| Работа | Идентификатор | По аннотации | Связь |
| --- | --- | --- | --- |
| Ahn et al. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances | arXiv 2204.01691 | Знание LLM о задаче сочетается с функциями ценности навыков робота (SayCan) | LM как планировщик сверху; у нас — замороженная LM внутри контура тела |
| Huang et al. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents | arXiv 2201.07207 | Крупные LM без обучения разбивают задачу на шаги; шаги переводятся в допустимые действия | то же; знание задачи из языка, а не из опыта тела |
| Li et al. Pre-Trained Language Models for Interactive Decision-Making | arXiv 2202.01771 | Политика, инициализированная LM, выигрывает; важны последовательный вход и инициализация LM, а не то, язык ли вход | близко к A4 / R2: польза LM не зависит от смысла входных символов |
| Reid et al. Can Wikipedia Help Offline Reinforcement Learning? | arXiv 2201.12122 | Предобучение на языке ускоряет офлайн-RL при дообучении | предобученная последовательная модель как основа агента (с дообучением, у нас — без) |

### LLM в промышленном мониторинге и диагностике (страницы открыты 2026-10-07; для введения)
| Работа | Идентификатор | По аннотации | Связь |
| --- | --- | --- | --- |
| Dave et al. Integrating LLMs for Explainable Fault Diagnosis in Complex Systems (2024; Comments: 4 pages) | arXiv 2402.06695 | Физическая диагностическая система соединена с LLM, чтобы не только находить неисправности, но и объяснять операторам их причины и последствия; мотивация — АЭС; показано на установке с расплавленной солью (связь неисправностей с данными датчиков, ответы на вопросы оператора, разбор прошлых аномалий) | ближе всего к роли «объясняющий слой над датчиками для операторов» во введении |
| Li, Zhao. S2S-FDD: Bridging Industrial Time Series and Natural Language for Explainable Zero-shot Fault Diagnosis (2026) | arXiv 2603.08048 | Многомерные временные ряды датчиков переводятся в текстовые описания; LLM ведёт многошаговую диагностику с опорой на документы обслуживания и обратную связь человека; проверено на многофазном потоке | LLM как диагностический и объясняющий слой над промышленными рядами (не энергетика) |
| Liu et al. Evaluation of Large Language Models for Numeric Anomaly Detection in Power Systems (2025) | arXiv 2511.21371 | Оценка GPT-OSS-20B на поиске аномалий в многомерной телеметрии энергосистемы (IEEE 14-bus) в разных режимах (без примеров, с примерами, дообучение, гибрид с классическими детекторами); оцениваются и качество обнаружения, и обоснования модели | энергосистемы; LLM обнаруживает аномалии и объясняет их |

Проверены, но не включены: Jing, Rahman, arXiv 2407.08836 (диагностика энергосетей через подсказки GPT-4; короткий препринт без места публикации); Qaid et al., arXiv 2412.01218 (FD-LLM — LLM как классификатор по вибрации, без объяснений); Peng et al., arXiv 2408.11281 (BearLLM, AAAI 2025 по Comments — подшипники, вибрация); Wu, arXiv 2510.03815 (байесовская сеть + LLM; данные в аннотации не названы). Все страницы открыты 2026-10-07.

**Примечания к проверке:**
- У трёх работ полное название шире рабочего: Elazar et al. (подзаголовок; первоначальное название было другим), Orgad et al., Huang et al.
- «SayCan» — название метода, в заголовке работы его нет.
- Места публикации у работ раздела B — только со слов авторов (поле Comments).
- Авторы в разделе B: полностью — где их не больше двух; иначе «первый автор et al.» (проверены по метаданным страницы).
