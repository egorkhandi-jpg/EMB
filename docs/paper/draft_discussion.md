## 5 Discussion

**A three-layer interpretation.** We read the results as pointing to three layers with very different
capacities, while noting that this is an interpretation of the measurements rather than a measured
mechanism. *Representation* is rich: the episode sign is linearly decodable almost perfectly, and so is
the meaning of a word. *Routing* is limited: word meaning is present at the word's own position but
barely reaches the position that summarizes the episode. *Expression* is narrow: the answer channel
recovers the latent sign only weakly, and when word meaning conflicts with in-context experience, the
answer follows the word. On this reading, the bottleneck of a frozen model reading an embodied situation
lies not in what it registers but in what reaches its output.

High probe accuracy alone does not show that a model uses the information [Hewitt 2019; Belinkov 2021;
Elazar 2021], and our random-weight controls make this point concrete: random networks also support
probes at 0.89–0.94 while answering at chance. What distinguishes pretrained models in our data is
mainly the answer channel. We therefore treat the answer as the measure of use, and the probe as an
upper bound on what could be used — in line with the observation that prompted judgments can
understate what a model encodes [Hu 2023], and with reports that models encode more about facts than
they express [Orgad 2024; Gekhman 2025].

**Why expression might lag.** One account, consistent with the data but not tested here, is that the
answer channel is shaped by the statistics of pretraining text: it maps contexts to words through
associations learned before the episode began. Word meaning is such an association, and it dominated
the answers in every model we tested (Section 3.5); in Pythia, a post hoc analysis suggests that mention
frequency acted the same way (Section 3.5). The episode sign is not such an association: it is a binding
of a kind to its outcomes that exists only in the current context. Work on entity tracking and binding
[Kim 2023; Feng 2023] suggests that such in-context bindings are fragile in models pretrained only on
text; our results add that the bound variable can be well represented and still fail to reach the output.

**Implications for monitoring critical systems.** The setting is minimal, and we do not claim direct
transfer to deployed systems. Three practical implications nonetheless follow for any role in which a
language model's output is taken as evidence about a physical process. First, the correspondence between
output and internal state should be measured, not assumed: a fluent explanation may express less than
the model has registered, or something other than it. Second, where ground truth is available, reading
internal states with calibrated probes may be more faithful than reading the model's answer. Third, names
carry priors: in our data the meaning of a word outweighed in-context evidence about the thing it named.
In a monitoring setting, labels of components, alarms or states could likewise pull an answer toward
what the label suggests rather than what the evidence shows — a failure mode worth testing explicitly
before a model's explanations are put in front of operators.

**Methodological lessons.** Several of our early positive results were false, and each was caught by a
specific control: a readout artifact from massive activations [Sun 2024], shortcuts through the agent's
own closed-loop behavior, a frequency cue created by survivorship, information lost by a symbolic
interface, and a condition against random weights that silently required a sign. We list these not as
contributions in themselves but as risks that apply to any evaluation of language models embedded in
agents, where the data a model sees are shaped by the agent that produces them.

**Limitations.** Models are frozen and small: replication covers 1–1.5B parameters, and Pythia extends to
6.9B for some analyses. We did not study instruction-tuned models, whose answer channel is trained to
report and may behave differently — the most important open question for the practical implications
above. The environment is a toy designed to separate mechanisms, not to resemble a real process. Probes
are linear; answers are read as a logit difference between two tokens rather than as free text. The
cross-episode in-context test was run on Pythia only. Code was implemented and run with an AI coding
assistant under the author's direction; all preregistrations, code and results are released.

**Future work.** Three directions follow directly: instruction-tuned and larger models; the minimal
trainable bridge from internal state to answer — how much plasticity a frozen model needs to report what
it represents; and the same protocol on real sensor streams from an operating plant.
