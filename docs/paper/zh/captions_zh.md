**Figure F1 — 隐含回合符号的探针与回答对比。** 针对回合符号（episode sign）的 AUC：问题位置上的线性探针（probe）（蓝色），以及模型的回答，即 logit 差值（logit difference）" A" − " B"（橙色），对应三个零样本问题。实心标记：预训练（pretrained）；空心：相同架构、随机权重（random weights）且嵌入范数匹配。flip-2，32 步事件词窗口；每个模型 10 组 × 100 个回合；各组的均值和 95% t 区间。数据：A28（A22 的协议）。File: `figures/zh/F1_probe_vs_answer.pdf`.

**Figure F2 — 有意义的事件词与任意事件词对比。** 窗口最终位置上回合符号线性探针的平衡准确率（balanced accuracy），使用有意义的事件词（A, B, none / up, down, same；蓝色）或任意单标记词（两组词的平均；橙色）；实心：预训练，空心：随机权重。每个模型 10 组 × 200 个回合；各组的均值和 95% t 区间。数据：A28（A3 和 A4 的协议）。File: `figures/zh/F2_words_vs_random.pdf`.

**Figure F3 — 按位置的词义。** 针对"种类词表示食物"的线性探针的 AUC，在训练中从未见过的词上测试，分别在种类词自身的位置（蓝色）和窗口最终位置（橙色）读取；实心：预训练，空心：随机权重。Pythia 1B / 2.8B / 6.9B（来自 A24）与 Qwen2.5-1.5B / OLMo-2-1B（来自 A28）在不同的 RTX 5090 实例上计算；这两组之间的比较是描述性的。200 个回合，≈56 个不同的测试词；按词的 95% 聚类自助法（cluster bootstrap）区间。随机权重：每个模型一个网络；在独立随机网络之间，AUC 分布在 ≈0.36–0.63（模拟）。虚线：预注册阈值 0.8。File: `figures/zh/F3_meaning_by_position.pdf`.

**Figure F4 — 合成窗口中的词义对抗经验。** (a) 在冲突窗口上（种类词的含义与结果相矛盾）：词义效应，即结果被置换的同一批窗口上的 |d|（蓝色）；向经验的偏移，d(outcomes) − d(permuted)（橙色）；随机权重的同一偏移（灰色）；d 是真实食物词与毒物词之间的 logit 差值。(b) 回答偏向经验的冲突窗口比例，有结果（橙色）与结果被置换（斜线填充）。句子接口，每个窗口 2–6 个事件，10 组 × 648 个窗口；各组的均值和 95% t 区间。Pythia-1B、Qwen2.5-1.5B、OLMo-2-1B：A28（同一实例）；Pythia-6.9B：A27（另一个实例）。File: `figures/zh/F4_synthetic_binding.pdf`.
