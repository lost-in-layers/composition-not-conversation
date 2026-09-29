# Layered-VQA Experimental Roadmap

This document defines the experiments, hypotheses, controls, and recommended
execution order for the Layered-VQA study. The project has three connected
research aims:

1. evaluate VQA answers together with faithful supporting visual layers;
2. extend multi-turn information-sharding analysis from text to vision and
   joint vision-language input;
3. test whether progressive evidence integration delays a **Visual Evidence
   Cliff** compared with single-pass full-RGB reasoning.

The experiments should be run in the order below. Later stages depend on the
controls and annotations established in earlier stages.

---

## 1. Experiment A — Layer-grounded VQA

### Research question

Can a model answer correctly **and** identify the complete supporting visual
evidence, rather than only producing a short answer from the fused RGB image?

### Conditions

1. **RGB Answer Only** — full RGB image; predict the answer.
2. **Full img + All Layers + Answer + Layer Selection** — full image, background, and all retained layers; predict the answer and supporting layer IDs.
3. **Oracle Gold Layers** — only a gold minimal sufficient layer set; predict the answer. This is an upper-bound diagnostic, not a fair retrieval result.

### Required outputs

- final answer;
- selected supporting layer IDs;
- optional confidence;
- concise evidence-grounded explanation for analysis

### Metrics

- answer accuracy or normalized exact match;
- layer precision, recall, and F1;
- exact evidence-set match;
- minimal-set success: whether a predicted set matches any accepted minimal sufficient set;
- answer-and-grounding joint accuracy;
- evidence completeness;
- excess/distractor layer selection rate;
- calibration or selective accuracy when confidence is requested.

### Main comparison

Compare ordinary RGB answer accuracy with joint answer-and-layer accuracy. The claim is not merely that layers improve answer accuracy; it is that layer-space evaluation exposes whether a correct answer is supported by complete and faithful evidence.

---

## 2. Experiment B — Text and Visual sharding multi-turn replication and 

Do vision-language models exhibit the text-side multi-turn degradation reported by *LLMs Get Lost in Multi-Turn Conversation* when visual input is held fixed? and What happens when the question stays complete but visual evidence arrives incrementally?
### Conditions

1. **FULL** — full image and full question. --> From Exp-A
#### Text
2. **CONCAT_Q** — full image and all question shards concatenated in their canonical order in one turn.
3. **SHARDED_Q** — full RGB remains available; one new question shard arrives per turn.
4. **RECAP_Q** — run `SHARDED_Q`, then provide the full question again.
5. **SNOWBALL_Q** — each turn repeats all previous question information and adds one new shard.
#### Visual
6. **CONCAT_V** — full question and all all visual layers together.
7. **SHARDED_V** — full question remains; one new visual layer arrives per turn and prior layers remain in conversation context.
8. **RECAP_V** — after `SHARDED_V`, provide the full RGB image again.
9. **SNOWBALL_V** — each turn explicitly repeats accumulated visual inputs and adds one new layer.
#### Main
10. **SHARDED_VQ** - 
   - Aligned_VQ: sharded question one per turn along with their supporting layers
   - Misaligned_VQ: sharded question one per turn along with their non-supporting layers
#### Additional 
11. **EVIDENCE_FIRST_V** — necessary evidence layers arrive before distractors.
12. **EVIDENCE_LAST_V** — distractors arrive before necessary evidence.
13. **SHUFFLE_Q-CONCAT** — full image and the non-intent shards concatenated in randomized order.


## Exact scheduling for `SHARDED_VQ`

Use `question_shard_support` as the shard-to-layer alignment between question shards and visual layers.

For question shard `q_t`, let its mapped layer set be `S_t`. Under cumulative delivery:

```text
new_layers(t) = S_t − previously_revealed_layers
available_layers(t) = union(S_1 ... S_t)
```

Therefore:

- a shard can introduce several layers;
- the same layer can support several shards;
- a shard may introduce no new image when all its supporting layers were already shown;
- unequal numbers of text shards and visual layers require no artificial one-to-one pairing.

#### Optional Information-matched all-layer track

Ensure that all candidate layers have arrived by the final turn:
- introduce mapped evidence layers at their earliest supporting shard;
- schedule distractors into otherwise empty turns or according to a fixed, seeded policy;
- keep the final set identical to `CONCAT_V`.

---

## 3. Experiment C — Visual Evidence Demand and the Visual Evidence Cliff

### Working definition

**Visual evidence demand** is the complexity of the evidence set that must be selected and integrated to answer a question faithfully. It includes:

- number of necessary evidence layers;
- number and similarity of distractor layers;
- reference/dependency-chain length;
- number of relations or hops;
- occlusion severity;
- evidence size or visibility;
- background dependence;
- OCR difficulty when applicable;
- controlled decomposition noise.

A **Visual Evidence Cliff** is a sharp nonlinear decline in answer accuracy, grounding completeness, faithfulness, or reliability once visual evidence demand exceeds a model's effective integration capacity.

### Demand levels

Construct a controlled slice such as:

| Level | Typical construction |
|---:|---|
| 1 | One necessary layer, no meaningful distractor |
| 2 | One necessary layer plus one or two distractors |
| 3 | Two necessary layers or one explicit relation |
| 4 | Two to three evidence units, occlusion, or a two-hop reference |
| 5 | Three or more evidence units plus similar distractors, small evidence, or controlled layer noise |

Use the underlying evidence program/annotation rather than raw layer count.
Compare demand within compatible question families so category difficulty is
not mistaken for a cliff.


### Maybe this happens but not sure. 

1. Single-pass full-RGB accuracy and grounding will decline earlier as evidence demand rises.
2. Properly aligned progressive evidence will shift the failure point to a higher demand level.
3. Misalignment, evidence-last delivery, and distractors will move the cliff earlier.
4. Recap will recover memory/integration failures but not genuinely missing evidence.
5. Layered inputs reduce evidence-selection demand, but they do not necessarily reduce reasoning demand when many layers must still be composed.


## 4. Experiment D — Token and reasoning-efficiency analysis

The goal is to determine which processing actually contributes to successful evidence integration, not merely whether a model emits a long explanation.

Measure:

- input tokens per turn and total input tokens;
- output/reasoning tokens per turn and total;
- number of model calls;
- latency and peak context size;
- tokens consumed before each required evidence layer becomes available;
- answer accuracy and grounding gain per additional token/call;
- redundant repeated information in `SNOWBALL`;
- recovery efficiency in `RECAP`.

Compare single-pass versus multi-call conditions at both:

- matched final information;
- approximately matched token/call budgets where feasible.

---

## 5. Statistical and qualitative reporting

### Statistical reporting

- paired comparisons because every condition uses the same questions;
- bootstrap confidence intervals over questions/scenes;
- scene-clustered tests when several questions share one image;
- correction for the main family of planned comparisons;
- effect sizes in addition to significance;
- macro averages across categories as well as micro averages;
- repeated-run reliability and variance.

### Error taxonomy

Manually inspect errors from:

- wrong target selection;
- incomplete evidence set;
- distractor capture;
- relation reversal;
- reference-resolution failure;
- occlusion failure;
- count-scope failure;
- text-shard forgetting;
- visual-layer forgetting;
- cross-modal misbinding;
- correct answer with unfaithful evidence;
- recovery after recap;
- answer degradation after unnecessary reasoning.

Include qualitative examples showing both successes and failures. Avoid showing only favorable examples.

---

## 6. Claims

1. **Grounding claim:** Answer accuracy alone hides incomplete or unfaithful visual grounding; layer-level evidence evaluation measures this directly.
2. **Conversation claim:** Models respond differently when equivalent textual, visual, or joint evidence is distributed over turns, extending text-only multi-turn analysis into multimodal settings.
3. **Capacity claim:** Increasing evidence demand produces a measurable Visual Evidence Cliff, and correctly scheduled progressive evidence can delay or soften that cliff under controlled final-information conditions.
