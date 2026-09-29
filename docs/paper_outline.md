# Paper Skeleton

## Title

LEAF-VQA: Layer-Evidence-Aware Faithful Visual Question Answering

Alternative:

Layer-Grounded VQA: Faithful Visual Question Answering through Compositional Visual Evidence

## Abstract Draft

Grounded visual question answering systems often produce correct answers and plausible regions, but their grounding may be incomplete or unfaithful. We argue that this limitation is partly representational: answer-critical evidence is frequently compositional, overlapping, text-bearing, or occluded in the final RGB image. We introduce Layer-Grounded VQA, a task requiring models to answer questions while identifying supporting evidence in both image space and layer space. We build a benchmark by decomposing images into fixed RGBA layer stacks and annotating answer, image evidence, layer evidence, evidence completeness, and counterfactual necessity. We propose LEAF-VQA, an evidence-first model that retrieves question-conditioned layers, reasons over a layer graph, answers from selected evidence, and verifies faithfulness through layer interventions. Experiments compare against strong RGB-grounded VLMs and layer-selection baselines, showing improved evidence completeness and faithfulness, especially for OCR, occlusion, clutter, and multi-layer reasoning.

## Contributions

1. We define Layer-Grounded VQA, a grounded VQA task over compositional layer evidence and projected image evidence.
2. We introduce a benchmark with answer labels, image-space evidence, layer-space evidence, evidence type, completeness labels, and counterfactual intervention annotations.
3. We propose LEAF-VQA, an evidence-first model that retrieves and reasons over layer evidence before answering.
4. We evaluate faithfulness with sufficiency, comprehensiveness, and counterfactual layer-intervention metrics.

## Figure 1

Show one hard image:

1. original image and question
2. RGB-grounding baseline highlights one plausible but incomplete box
3. Qwen-Image-Layered decomposition
4. gold/LEAF selected evidence layers
5. selected-only image still answers correctly
6. removing selected evidence breaks the answer

## Sections

### 1. Introduction

- Grounded VQA needs faithful evidence, not only plausible boxes.
- RGB evidence can be entangled.
- Layered decomposition gives editable, compositional visual units.
- State task, benchmark, method, and intervention evaluation.

### 2. Related Work

- VQA and compositional VQA: VQA v2, GQA, OK-VQA, A-OKVQA, TextVQA.
- Faithful grounding: FPVG and intervention-style evaluation.
- Grounded MLLMs: Qwen2.5-VL, Shikra, Ferret, Kosmos-2, Florence-2.
- Segmentation and layer decomposition: SAM 2, Qwen-Image-Layered.

### 3. Layer-Grounded VQA

- Formal input/output.
- Evidence sets.
- Single-layer vs multi-layer evidence.
- Faithfulness criteria.

### 4. Benchmark

- Source data.
- Layer decomposition protocol.
- Annotation UI and guidelines.
- Quality filtering.
- Dataset statistics.
- Agreement analysis.

### 5. Method

- Layer decomposition.
- Layer graph construction.
- Question-conditioned evidence retrieval.
- Evidence-first answering.
- Intervention losses.

### 6. Experiments

- Baselines.
- Metrics.
- Main results.
- Category breakdown.
- Ablations.
- Counterfactual intervention analysis.
- Qualitative examples.

### 7. Limitations

- Layer decomposition can fail.
- Layer evidence depends on fixed generated decompositions.
- Annotation cost is nontrivial.
- Knowledge-heavy questions may test retrieval/world knowledge more than visual evidence.

### 8. Conclusion

Layered visual representations can turn grounded VQA from post-hoc localization on fused images into evidence-first reasoning over compositional visual units.

