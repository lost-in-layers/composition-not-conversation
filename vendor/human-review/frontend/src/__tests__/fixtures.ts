import { hydrateDraft } from "../draftUtils";
import type { ReviewDraft, Scene, ScenePayload } from "../types";

export function fixtureScene(overrides: Partial<Scene> = {}): Scene {
  const renderDir = "data/pre-data/renders/lvis/lvis/000000000025";
  return {
    scene_id: "lvis_000000000025",
    subset: "lvis",
    image_id: "000000000025",
    width: 640,
    height: 426,
    full_path: `${renderDir}/full.png`,
    render_dir: renderDir,
    captions: { blip2: "two giraffes standing by a tree", llava: "A long llava caption." },
    background: {
      layer_id: 0,
      z_index: 0,
      path: `${renderDir}/layer_0.png`,
      captions: { blip2: "a fallen tree", llava: "A forest scene." },
    },
    layers: [
      {
        layer_id: 1,
        role: "instance",
        z_index: 1,
        path: `${renderDir}/layer_1.png`,
        captions: { blip2: "single giraffe standing" },
        label: null,
        label_source: null,
        caption_quality: "unverified",
        geometry: { bbox_xyxy: [52, 356, 184, 414], support_centroid_xy: [119.3, 385.2], support_area_pixels: 2264, connected_components_4: 1 },
        mask: { native_alpha_sha256: null, matches_native_alpha: null },
      },
      {
        layer_id: 2,
        role: "instance",
        z_index: 2,
        path: `${renderDir}/layer_2.png`,
        captions: { blip2: "a shot of two flowers" },
        label: null,
        label_source: null,
        caption_quality: "unverified",
        geometry: { bbox_xyxy: [316, 88, 398, 133], support_centroid_xy: [356.2, 111.8], support_area_pixels: 1927, connected_components_4: 2 },
        mask: { native_alpha_sha256: null, matches_native_alpha: null },
      },
      {
        layer_id: 3,
        role: "instance",
        z_index: 3,
        path: `${renderDir}/layer_3.png`,
        captions: { blip2: "a giraffe standing alone" },
        label: null,
        label_source: null,
        caption_quality: "unverified",
        geometry: { bbox_xyxy: [386, 64, 557, 357], support_centroid_xy: [467.7, 243.9], support_area_pixels: 18625, connected_components_4: 1 },
        mask: { native_alpha_sha256: null, matches_native_alpha: null },
      },
    ],
    relations: [
      { subject_layer_id: 1, predicate: "left_of", object_layer_id: 2, score: 0.79, source: "nonoverlapping_bboxes" },
      { subject_layer_id: 3, predicate: "possible_occludes", object_layer_id: 2, score: 0.05, source: "mask_overlap_and_stack_order" },
    ],
    pairwise_geometry: [
      { layer_ids: [1, 2], intersection_pixels: 0 },
      { layer_ids: [2, 3], intersection_pixels: 105 },
    ],
    annotation: { actual_sha256: null },
    inventory: { error: "" },
    validation: { status: "valid" },
    ...overrides,
  };
}

export function fixturePayload(overrides: Partial<ScenePayload> = {}): ScenePayload {
  const scene = overrides.scene ?? fixtureScene();
  return {
    scene,
    draft: null,
    decision: null,
    status: "unseen",
    missing_fields: {
      human_required: [
        { field: "label", detail: "Canonical short label must be assigned by the reviewer.", layer_id: 1 },
        { field: "reviewed_image_caption", detail: "Reviewed overall image caption is required." },
      ],
      technical_ok: [
        { field: "mask.native_alpha_sha256", detail: "No native alpha checksum was available.", layer_id: 1 },
        { field: "annotation.actual_sha256", detail: "No native checksum available." },
      ],
      blocking: [],
      counts: { human_required: 2, technical_ok: 2, blocking: 0 },
      per_layer: {
        "1": { human_required: 1, technical_ok: 1, blocking: 0 },
        "2": { human_required: 1, technical_ok: 0, blocking: 0 },
        "3": { human_required: 1, technical_ok: 0, blocking: 0 },
      },
    },
    category_suggestions: [
      { category: "direct_ask", label: "Direct Ask", suggested: true, eligible: true, explanation: "3 retained layers" },
      { category: "attribute_binding", label: "Attribute Binding", suggested: true, eligible: true, explanation: "relations exist" },
      { category: "spatial", label: "Spatial", suggested: true, eligible: true, explanation: "directional relations exist" },
      { category: "occlusion", label: "Occlusion", suggested: true, eligible: true, explanation: "overlap exists" },
      { category: "counting", label: "Counting", suggested: true, eligible: true, explanation: "2+ layers" },
      { category: "multi_hop_reference", label: "Multi-hop Reference", suggested: false, eligible: true, explanation: "needs 3 layers and 2 relations" },
    ],
    queue_position: 0,
    queue_total: 100,
    file_checks: { full: true, background: true, layer_1: true, layer_2: true, layer_3: true, progressive_strip: true },
    prev_scene_id: null,
    next_scene_id: "laion_000000003",
    ...overrides,
  };
}

/** A draft with every requirement satisfied (accept-ready). */
export function completeDraft(payload: ScenePayload): ReviewDraft {
  const draft = hydrateDraft(payload);
  draft.overall_quality = 4;
  for (const layer of payload.scene.layers) {
    draft.layer_reviews[String(layer.layer_id)] = {
      ...draft.layer_reviews[String(layer.layer_id)],
      decision: "keep",
      label: `object ${layer.layer_id}`,
      caption: `reviewed caption ${layer.layer_id}`,
      geometry_status: "correct",
    };
  }
  draft.reviewed_image_caption = "a reviewed image caption";
  draft.reviewed_image_caption_llava = "a reviewed LLaVA image caption";
  draft.reviewed_background_caption = "a reviewed background caption";
  draft.reviewed_background_caption_llava = "a reviewed LLaVA background caption";
  draft.suitable_categories = ["direct_ask"];
  draft.primary_category = "direct_ask";
  draft.category_decision_made = true;
  draft.human_questions = { direct_ask: "What is next to the tree?" };
  draft.human_answers = { direct_ask: "A giraffe." };
  return draft;
}
