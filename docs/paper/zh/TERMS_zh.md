# Китайская версия: термины вне словаря и решения перевода (для проверки носителем)

Перевод — `docs/paper/zh/*_zh.md` → `main_zh.tex` → `main_zh.pdf` (2026-10-07). Словарь Егора применён везде одинаково.

## Как применён словарь
- **Английский в скобках** — при первом употреблении термина из словаря в каждом разделе верхнего уровня: аннотация, 1–5, приложения A–C, подписи. Пример: 探针（probe）, 回合（episode）.
- **Пояснения из самого словаря** даны при первом употреблении в статье или в разделе с определением:
  - «回合符号（episode sign，哪类物品为食物）» — в п. 2.1, где знак эпизода определяется;
  - 线性探针 — там, где в оригинале «linear probe»;
  - 自助法（bootstrap）.
- **routing.** В словаре «传递（路由）». В тексте использовано 传递, при первом употреблении — 传递（routing). Вариант 路由 в текст не вставлялся — **проверить**.
- **Не переведены:** названия моделей, KONTUR, flip-2, токены и вопросы (" A", " up", " Food:"), примеры слов (apple, mold, cat…), обозначения (A2–A28, R1–R5, K1, K2, H1, H2), формулы (d(outcomes) − d(permuted), (AUC_answer − 0.5) ≤ 0.5 · (AUC_probe − 0.5)), метки и хэши.

## Термины, которых не было в словаре
| English | Выбранный перевод | Где / замечание |
| --- | --- | --- |
| machinery (sequence-processing machinery) | 机制（序列处理机制） | **совпадает с переводом «mechanism» (机制)** — в оригинале это разные слова; возможно, лучше 机理 / 运算机制 для одного из них |
| zero-shot / few-shot | 零样本 / 少样本 | 3.3, F1 |
| in-context examples / in-context learning | 上下文示例 / 上下文学习 | 3.3, прил. B |
| confidence interval / t-interval | 置信区间 / t 区间 | 2.5, подписи |
| sets (10 sets) | 组 | везде |
| cell (of the design) | 单元 | 2.2, 3.5, 4 |
| fold / held-out words | 折 / 留出词 | 2.4, 3.4 |
| arbitrary words / meaningful words | 任意词 / 有意义的词 | 3.1, 3.2, F2 |
| readout / readout head | 读出 / 读出头 | 4 |
| continuous projector | 连续投影器 | 3.1, 4 |
| embodied agent / foraging environment | 具身智能体 / 觅食环境 | аннотация, 1, 2.1 |
| token / tokenizer / single-token word | 标记 / 分词器 / 单标记词 | 2.2, 2.3, прил. A |
| end-of-document token | 文档结束标记 | 2.3, прил. A |
| hidden state | 隐藏状态 | везде |
| frequency cue / frequency reference | 频率线索 / 频率参照 | 2.2, 3.5, 4 |
| last-mentioned / first-mentioned reference | 最后提及参照 / 首次提及参照 | 2.2 |
| conflict windows | 冲突窗口 | 3.5, 4, F4 |
| truthful / lying / neutral kind words | 真实的 / 说谎的 / 中性的种类词 | 3.5 |
| meaning effect / shift toward experience | 词义效应 / 向经验的偏移 | 3.5, F4 |
| out-of-loop evaluation | 闭环外评估 | 4, прил. B |
| massive activations | 巨量激活 | 5 |
| null result | 零结果 | 4, 5, прил. B |
| model family / preregistered family (of tests) | 模型家族 / 检验族 | различены намеренно |
| sign-flip permutation test; one-sided / two-sided | 符号翻转置换检验；单侧 / 双侧 | 2.5 |
| post hoc | 事后分析（post hoc） | 2.5, 3.5, 4 |
| threshold | 阈值 | 3.4, прил. A |
| non-inferiority margin | 非劣效界值 | прил. A |
| descriptively | 描述性地 / 描述性结果 | везде |
| fitness; seed | 适应度；种子 | 4 |
| instruction-tuned models | 指令微调模型 | 5 |
| entity tracking | 实体追踪 | 5 |
| registry of all attempts | 所有尝试的登记表 | 1, 2.5, «代码与数据获取», прил. B |
| stopping rule / amendment before computation | 停止规则 / 计算前的补充 | прил. B |
| event alphabet; self/world separation | 事件字母表；自我/世界分离 | прил. B |
| queue time / billed time / instance | 队列时间 / 计费时间 / 实例 | прил. A, C |
| gas-engine power generation | 燃气发动机发电设施 | 1 |
| plant | 设备 (1) / 电厂 (5) | по контексту |
| technician-facing explanations | 为技术人员生成解释 | 1 |
| AI coding assistant | AI 编程助手 | 5, 声明 |
| hatched (bars) / filled / hollow markers | 斜线填充 / 实心 / 空心 | подписи |
| Disclosure (section) | 声明 | — |
| Figure (ссылки в тексте) | 图 | LaTeX: 图~\ref |

## Оформление
- Название: 所知多于所言：极简身体中的冻结语言模型. Под ним английское название и примечание о переводе. Автор и ORCID — как в английской версии.
- Список литературы — на английском, тот же `refs.bib`. Ссылки в тексте — (Ahn et al., 2022).
- **Единицы в приложениях A и C оставлены как в оригинале:** h, min, GB, $.
- **Отклонение от прямого перевода — порядок слов.** В нескольких местах (≈ 12) фраза перестроена так, чтобы числа шли в том же порядке, что в английском (требование сверки), например «0.983–0.999（Pythia）». Смысл не менялся.
