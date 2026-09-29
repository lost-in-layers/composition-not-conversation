import { useState } from "react";
import { retainedLayerIds } from "../draftUtils";
import type { Relation, ReviewDraft, Scene } from "../types";

function quote(value: string | null | undefined): string {
  return JSON.stringify((value ?? "").trim());
}

function relationKey(relation: Pick<Relation, "subject_layer_id" | "predicate" | "object_layer_id">): string {
  return `${relation.subject_layer_id}\u0000${relation.predicate}\u0000${relation.object_layer_id}`;
}

function reviewedRelations(scene: Scene, draft: ReviewDraft, retained: Set<number>): Relation[] {
  const removed = new Set(
    draft.relations_removed.map(([subject, predicate, object]) => `${subject}\u0000${predicate}\u0000${object}`),
  );
  const original = (scene.relations ?? []).filter(
    (relation) =>
      retained.has(relation.subject_layer_id) &&
      retained.has(relation.object_layer_id) &&
      !removed.has(relationKey(relation)),
  );
  const added = draft.relations_added.filter(
    (relation) => retained.has(relation.subject_layer_id) && retained.has(relation.object_layer_id),
  );
  return [...original, ...added];
}

export function buildSceneTemplate(scene: Scene, draft: ReviewDraft): string {
  const retained = new Set(retainedLayerIds(scene, draft));
  const layerLines = scene.layers
    .filter((layer) => retained.has(layer.layer_id))
    .map((layer) => {
      const review = draft.layer_reviews[String(layer.layer_id)];
      return `L${layer.layer_id} - ${quote(review?.label)}: ${quote(review?.caption)}`;
    });
  const relationLines = reviewedRelations(scene, draft, retained).map((relation) => {
    const score = typeof relation.score === "number" ? ` (${relation.score.toFixed(2)})` : "";
    return `L${relation.subject_layer_id} ${relation.predicate} L${relation.object_layer_id}${score}`;
  });

  return [
    "Now this:",
    "",
    `img: ${quote(draft.reviewed_image_caption)} and ${quote(draft.reviewed_image_caption_llava)}`,
    "",
    `bg: ${quote(draft.reviewed_background_caption)} and ${quote(draft.reviewed_background_caption_llava)}`,
    "",
    ...layerLines,
    "",
    ...relationLines,
    "",
    "now for this.",
  ].join("\n");
}

export function ScenePromptTemplate(props: { scene: Scene; draft: ReviewDraft }) {
  const [copied, setCopied] = useState(false);
  const template = buildSceneTemplate(props.scene, props.draft);

  const copy = async () => {
    await navigator.clipboard.writeText(template);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div>
      <div className="section-title">Scene Template</div>
      <div className="card scene-template-card">
        <div>
          <div className="field-label">Copyable reviewed scene context</div>
          <p className="small-note">
            Built automatically from completed Steps 1–3. Continue to Categories when you are done copying it.
          </p>
        </div>
        <button className={copied ? "success small" : "primary small"} onClick={() => void copy()}>
          {copied ? "Copied ✓" : "Copy template"}
        </button>
      </div>
      <textarea
        className="scene-template-output"
        value={template}
        readOnly
        spellCheck={false}
        aria-label="Copyable scene template"
      />
    </div>
  );
}
