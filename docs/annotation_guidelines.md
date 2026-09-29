# Layer-Grounded VQA Annotation Guidelines

## Goal

For each image-question-answer example, mark the visual evidence needed to answer the question. Evidence must be annotated in both:

- **Layer space:** which decomposed RGBA layers are necessary.
- **Image space:** where the evidence appears in the final RGB image.

The key rule is necessity: select evidence that the answer depends on, not every object that looks related.

## Inputs Shown To Annotator

- original RGB image
- decomposed layer montage
- each RGBA layer shown separately on checkerboard and on black/white backgrounds
- question
- answer
- optional source annotations such as object boxes, OCR boxes, or scene graph objects

## Layer Selection Rules

Select a layer if removing it would make the answer impossible, ambiguous, or likely wrong.

Do not select a layer only because:

- it is visually salient
- it is near the evidence
- it is part of the general scene but not needed
- it makes the image look natural but does not support the answer

If multiple layers jointly support the answer, select all necessary layers.

Examples:

- "What number is on the jersey?" Select the jersey/text-bearing layer. Select the person layer only if needed to identify which jersey.
- "What is the man holding?" Select the man/hand layer and held-object layer if both are needed to establish the relation.
- "What is written on the sign behind the cyclist?" Select the sign/text layer and the cyclist/background layer if needed to identify the correct sign.
- "What sport is being played?" Select visual evidence for equipment, players, field/court, or text if they jointly determine the answer.

## Image Evidence Rules

Draw or accept the tightest region that contains the answer-supporting evidence.

Preferred order:

1. mask, if the object/text region is separable
2. box, if mask is too slow or ambiguous
3. point, only for tiny evidence or when the source annotation provides points

For multi-layer evidence, provide one evidence region per necessary piece when possible.

## Evidence Type Labels

Use one or more:

- `single_layer`
- `multi_layer`
- `ocr`
- `occlusion`
- `overlap`
- `clutter`
- `tiny_evidence`
- `spatial_relation`
- `attribute`
- `counting`
- `external_knowledge`
- `decomposition_failure`

## Completeness Label

Choose:

- `single`: one evidence piece is sufficient
- `multi_complete`: all selected pieces are jointly necessary
- `partial`: some evidence is visible but decomposition or annotation cannot capture it cleanly
- `failure`: layers do not expose usable evidence

## Counterfactual Labels

For selected layers:

- `selected_only_should_answer`: yes/no
- `remove_selected_should_break`: yes/no
- `editable_counterfactual_possible`: yes/no
- `edit_type`: color/text/count/object/removal/none

Only mark counterfactuals as valid if the edited or removed composition remains visually interpretable enough for the question.

## Quality Flags

Use these flags to filter examples:

- `good`: evidence clearly appears in selected layers
- `mixed`: evidence is split or partially entangled but usable
- `poor`: selected layer also contains major irrelevant evidence
- `unusable`: decomposition fails for this example

The main benchmark should use `good` and selected `mixed` examples. Keep `poor` and `unusable` for diagnostic analysis only.

