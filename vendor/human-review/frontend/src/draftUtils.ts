// Draft hydration, per-step completion logic and local validation mirrors of
// the backend rules (the backend remains authoritative at finalize time).
import {
  CATEGORIES,
  CATEGORY_LABELS,
  emptyDraft,
  emptyLayerReview,
  type Category,
  type CategorySuggestion,
  type LayerReview,
  type ReviewDraft,
  type Scene,
  type ScenePayload,
} from "./types";

export function hydrateDraft(payload: ScenePayload): ReviewDraft {
  const base = emptyDraft();
  const stored = payload.draft ?? payload.decision?.review ?? null;
  const draft: ReviewDraft = stored
    ? {
        ...base,
        ...stored,
        reviewed_image_caption: !isBlank(stored.reviewed_image_caption)
          ? stored.reviewed_image_caption
          : payload.scene.captions?.blip2 ?? null,
        reviewed_image_caption_llava:
          "reviewed_image_caption_llava" in stored && !isBlank(stored.reviewed_image_caption_llava)
            ? stored.reviewed_image_caption_llava
            : payload.scene.captions?.llava ?? null,
        reviewed_background_caption: !isBlank(stored.reviewed_background_caption)
          ? stored.reviewed_background_caption
          : payload.scene.background?.captions?.blip2 ?? null,
        reviewed_background_caption_llava:
          "reviewed_background_caption_llava" in stored && !isBlank(stored.reviewed_background_caption_llava)
            ? stored.reviewed_background_caption_llava
            : payload.scene.background?.captions?.llava ?? null,
        human_questions: (() => {
          if ("human_questions" in stored && stored.human_questions) return stored.human_questions;
          const legacy = stored as ReviewDraft & { human_question?: string | null; general_notes?: string | null };
          const question = legacy.human_question ?? legacy.general_notes ?? null;
          return !isBlank(question) && stored.primary_category ? { [stored.primary_category]: question } : {};
        })(),
        human_answers: "human_answers" in stored && stored.human_answers ? stored.human_answers : {},
      }
    : {
        ...base,
        reviewed_image_caption: payload.scene.captions?.blip2 ?? null,
        reviewed_image_caption_llava: payload.scene.captions?.llava ?? null,
        reviewed_background_caption: payload.scene.background?.captions?.blip2 ?? null,
        reviewed_background_caption_llava: payload.scene.background?.captions?.llava ?? null,
      };
  const layerReviews: Record<string, LayerReview> = {};
  for (const layer of payload.scene.layers) {
    const key = String(layer.layer_id);
    const storedLayer = draft.layer_reviews?.[key] as (Partial<LayerReview> & { decision?: string }) | undefined;
    const rawDecision = (storedLayer as { decision?: string } | undefined)?.decision;
    layerReviews[key] = {
      ...emptyLayerReview(),
      ...(storedLayer ?? {}),
      decision: rawDecision === "keep_minor" ? "keep" : (rawDecision as LayerReview["decision"] | undefined) ?? null,
      caption: storedLayer && !isBlank(storedLayer.caption) ? storedLayer.caption ?? null : layer.captions?.blip2 ?? null,
    };
  }
  return { ...draft, layer_reviews: layerReviews };
}

export function isBlank(value: string | null | undefined): boolean {
  return value == null || value.trim() === "";
}

export function questionAnswerProblems(draft: ReviewDraft): string[] {
  const problems: string[] = [];
  const filled = draft.suitable_categories.filter((category) => !isBlank(draft.human_questions?.[category]));
  if (filled.length === 0) {
    problems.push("Write at least one question for a selected category.");
    return problems;
  }
  for (const category of filled) {
    if (isBlank(draft.human_answers?.[category])) {
      problems.push(`Answer the ${CATEGORY_LABELS[category]} question.`);
    }
  }
  return problems;
}

export function retainedLayerIds(scene: Scene, draft: ReviewDraft): number[] {
  return scene.layers
    .filter((layer) => {
      const lr = draft.layer_reviews[String(layer.layer_id)];
      return lr && lr.decision === "keep";
    })
    .map((layer) => layer.layer_id);
}

export function excludedLayerIds(scene: Scene, draft: ReviewDraft): number[] {
  return scene.layers
    .filter((layer) => draft.layer_reviews[String(layer.layer_id)]?.decision === "exclude")
    .map((layer) => layer.layer_id);
}

export function validBbox(
  bbox: [number, number, number, number] | null,
  width: number,
  height: number,
): boolean {
  if (bbox == null) return true;
  const [x1, y1, x2, y2] = bbox;
  return x1 >= 0 && y1 >= 0 && x2 <= width && y2 <= height && x1 < x2 && y1 < y2;
}

export function clampBbox(
  bbox: [number, number, number, number],
  width: number,
  height: number,
): [number, number, number, number] {
  let [x1, y1, x2, y2] = bbox.map((v) => Math.round(v)) as [number, number, number, number];
  x1 = Math.max(0, Math.min(x1, width - 1));
  y1 = Math.max(0, Math.min(y1, height - 1));
  x2 = Math.max(x1 + 1, Math.min(x2, width));
  y2 = Math.max(y1 + 1, Math.min(y2, height));
  return [x1, y1, x2, y2];
}

export interface StepStatus {
  complete: boolean;
  problems: string[];
}

export function step1Status(scene: Scene, draft: ReviewDraft): StepStatus {
  const problems: string[] = [];
  if (draft.overall_quality == null) problems.push("Rate the overall decomposition quality (1–5).");
  for (const layer of scene.layers) {
    const lr = draft.layer_reviews[String(layer.layer_id)];
    if (!lr?.decision) {
      problems.push(`Layer ${layer.layer_id}: choose Keep or Exclude.`);
    } else if (lr.decision === "exclude") {
      if (isBlank(lr.exclusion_reason)) problems.push(`Layer ${layer.layer_id}: pick an exclusion reason.`);
      else if (lr.exclusion_reason === "other" && isBlank(lr.exclusion_note))
        problems.push(`Layer ${layer.layer_id}: exclusion reason "other" needs a note.`);
    }
  }
  return { complete: problems.length === 0, problems };
}

export function step2Status(scene: Scene, draft: ReviewDraft): StepStatus {
  const problems: string[] = [];
  if (isBlank(draft.reviewed_image_caption)) problems.push("Write the reviewed overall image caption.");
  if (isBlank(draft.reviewed_image_caption_llava)) problems.push("Review the LLaVA overall image caption.");
  if (isBlank(draft.reviewed_background_caption)) problems.push("Write the reviewed background caption.");
  if (isBlank(draft.reviewed_background_caption_llava)) problems.push("Review the LLaVA background caption.");
  for (const id of retainedLayerIds(scene, draft)) {
    const lr = draft.layer_reviews[String(id)];
    if (isBlank(lr?.label)) problems.push(`Layer ${id}: canonical short label is required.`);
    if (isBlank(lr?.caption)) problems.push(`Layer ${id}: reviewed caption is required.`);
  }
  return { complete: problems.length === 0, problems };
}

export function step3Status(scene: Scene, draft: ReviewDraft): StepStatus {
  const problems: string[] = [];
  for (const id of retainedLayerIds(scene, draft)) {
    const lr = draft.layer_reviews[String(id)];
    if (!lr?.geometry_status) {
      problems.push(`Layer ${id}: set the geometry status.`);
    } else if (lr.geometry_status === "major_problem") {
      problems.push(`Layer ${id}: geometry marked as a major problem — fix it, exclude the layer, or reject the scene.`);
    }
    if (!validBbox(lr?.bbox_override ?? null, scene.width, scene.height)) {
      problems.push(`Layer ${id}: corrected bbox must satisfy x1 < x2, y1 < y2 inside the image.`);
    }
  }
  return { complete: problems.length === 0, problems };
}

export function step4Status(
  scene: Scene,
  draft: ReviewDraft,
  suggestions: CategorySuggestion[],
): StepStatus {
  const problems: string[] = [];
  if (draft.suitable_categories.length === 0 && !draft.category_decision_made) {
    problems.push("Make an explicit category-suitability decision.");
  }
  if (!draft.primary_category) {
    problems.push("Choose exactly one primary category.");
  } else {
    if (!draft.suitable_categories.includes(draft.primary_category)) {
      problems.push("The primary category must also be marked suitable.");
    }
  }
  return { complete: problems.length === 0, problems };
}

export function acceptanceProblems(
  scene: Scene,
  draft: ReviewDraft,
  suggestions: CategorySuggestion[],
  blockingUnresolved: string[],
): string[] {
  const problems: string[] = [];
  const retained = retainedLayerIds(scene, draft);
  if (retained.length === 0) problems.push("At least one retained foreground layer is required.");
  problems.push(...step1Status(scene, draft).problems);
  problems.push(...step2Status(scene, draft).problems);
  problems.push(...step3Status(scene, draft).problems);
  problems.push(...step4Status(scene, draft, suggestions).problems);
  problems.push(...questionAnswerProblems(draft));
  problems.push(...blockingUnresolved);
  return problems;
}

/** Blocking source findings that the current draft has not neutralised. */
export function unresolvedBlocking(payload: ScenePayload, draft: ReviewDraft): string[] {
  const out: string[] = [];
  for (const item of payload.missing_fields.blocking) {
    if (item.layer_id != null) {
      const lr = draft.layer_reviews[String(item.layer_id)];
      if (lr?.decision === "exclude") continue;
      if (item.field === "captions" && !isBlank(lr?.caption)) continue;
      out.push(`Layer ${item.layer_id} (${item.field}): ${item.detail}`);
    } else {
      if (
        item.field === "captions" &&
        (!isBlank(draft.reviewed_image_caption) || !isBlank(draft.reviewed_image_caption_llava))
      ) continue;
      if (
        item.field === "background.captions" &&
        (!isBlank(draft.reviewed_background_caption) || !isBlank(draft.reviewed_background_caption_llava))
      ) continue;
      out.push(`${item.field}: ${item.detail}`);
    }
  }
  return out;
}

export function allStepStatuses(
  scene: Scene,
  draft: ReviewDraft,
  suggestions: CategorySuggestion[],
): StepStatus[] {
  return [
    step1Status(scene, draft),
    step2Status(scene, draft),
    step3Status(scene, draft),
    step4Status(scene, draft, suggestions),
  ];
}

export function toggleSuitable(draft: ReviewDraft, category: Category): ReviewDraft {
  const has = draft.suitable_categories.includes(category);
  const suitable = has
    ? draft.suitable_categories.filter((c) => c !== category)
    : [...draft.suitable_categories, category].sort(
        (a, b) => CATEGORIES.indexOf(a) - CATEGORIES.indexOf(b),
      );
  return {
    ...draft,
    suitable_categories: suitable,
    category_decision_made: true,
    primary_category: has && draft.primary_category === category ? null : draft.primary_category,
  };
}

export function setPrimary(draft: ReviewDraft, category: Category): ReviewDraft {
  const suitable = draft.suitable_categories.includes(category)
    ? draft.suitable_categories
    : [...draft.suitable_categories, category].sort(
        (a, b) => CATEGORIES.indexOf(a) - CATEGORIES.indexOf(b),
      );
  return {
    ...draft,
    suitable_categories: suitable,
    primary_category: category,
    category_decision_made: true,
  };
}
