// Shared types mirroring the backend JSON contracts.

export type Category =
  | "direct_ask"
  | "attribute_binding"
  | "spatial"
  | "occlusion"
  | "counting"
  | "multi_hop_reference";

export const CATEGORIES: Category[] = [
  "direct_ask",
  "attribute_binding",
  "spatial",
  "occlusion",
  "counting",
  "multi_hop_reference",
];

export const CATEGORY_LABELS: Record<Category, string> = {
  direct_ask: "Direct Ask",
  attribute_binding: "Attribute Binding",
  spatial: "Spatial",
  occlusion: "Occlusion",
  counting: "Counting",
  multi_hop_reference: "Multi-hop Reference",
};

export const CATEGORY_EXAMPLES: Record<Category, string> = {
  direct_ask: "What color is the bag?",
  attribute_binding: "What color is the object left of the bicycle?",
  spatial: "Which object is above the table?",
  occlusion: "What object is behind the person?",
  counting: "How many objects are right of the chair?",
  multi_hop_reference: "What is behind the object nearest to the person?",
};

export const EXCLUSION_REASONS = [
  ["incorrect_segmentation", "Incorrect segmentation"],
  ["fused_objects", "Fused objects"],
  ["fragmented_object", "Fragmented object"],
  ["duplicate", "Duplicate"],
  ["irrelevant_or_hallucinated", "Irrelevant / hallucinated"],
  ["unusable_transparency", "Unusable transparency"],
  ["severe_artifact", "Severe artifact"],
  ["meaningless_layer", "Meaningless layer"],
  ["other", "Other (note required)"],
] as const;

export interface Geometry {
  bbox_xyxy?: [number, number, number, number];
  support_centroid_xy?: [number, number];
  alpha_weighted_centroid_xy?: [number, number];
  support_area_pixels?: number;
  perimeter_pixels_4?: number;
  connected_components_4?: number;
  touches_image_border?: boolean;
  [key: string]: unknown;
}

export interface SceneLayer {
  layer_id: number;
  native_instance_key?: string | null;
  role: string;
  z_index: number;
  path: string;
  captions?: Record<string, string>;
  label: string | null;
  label_source: string | null;
  caption_quality?: string;
  mask?: Record<string, unknown>;
  geometry?: Geometry;
}

export interface Relation {
  subject_layer_id: number;
  predicate: string;
  object_layer_id: number;
  score?: number | null;
  source?: string;
}

export interface PairwiseGeometry {
  layer_ids: [number, number];
  intersection_pixels?: number;
  mask_iou?: number;
  [key: string]: unknown;
}

export interface Scene {
  scene_id: string;
  subset: "lvis" | "laion";
  image_id: string;
  width: number;
  height: number;
  full_path: string;
  render_dir: string;
  image_ref?: string;
  categories?: string[];
  captions?: Record<string, string>;
  background?: {
    layer_id: number;
    z_index: number;
    path: string;
    captions?: Record<string, string>;
  };
  layers: SceneLayer[];
  pairwise_geometry?: PairwiseGeometry[];
  relations?: Relation[];
  annotation?: Record<string, unknown>;
  inventory?: Record<string, unknown>;
  validation?: Record<string, unknown>;
}

export interface MissingItem {
  field: string;
  detail: string;
  layer_id?: number;
}

export interface MissingReport {
  human_required: MissingItem[];
  technical_ok: MissingItem[];
  blocking: MissingItem[];
  counts: { human_required: number; technical_ok: number; blocking: number };
  per_layer: Record<string, { human_required: number; technical_ok: number; blocking: number }>;
}

export interface CategorySuggestion {
  category: Category;
  label: string;
  suggested: boolean;
  eligible: boolean;
  explanation: string;
}

export interface LayerReview {
  decision: "keep" | "exclude" | null;
  exclusion_reason: string | null;
  exclusion_note: string | null;
  label: string | null;
  caption: string | null;
  geometry_status: "correct" | "minor_correction" | "major_problem" | null;
  bbox_override: [number, number, number, number] | null;
  centroid_override: [number, number] | null;
  geometry_note: string | null;
}

export interface ReviewDraft {
  reviewer_id?: string | null;
  overall_quality: number | null;
  layer_reviews: Record<string, LayerReview>;
  reviewed_image_caption: string | null;
  reviewed_image_caption_llava: string | null;
  reviewed_background_caption: string | null;
  reviewed_background_caption_llava: string | null;
  relations_removed: Array<[number, string, number]>;
  relations_added: Array<{ subject_layer_id: number; predicate: string; object_layer_id: number }>;
  suitable_categories: Category[];
  primary_category: Category | null;
  category_decision_made: boolean;
  human_questions: Partial<Record<Category, string | null>>;
  human_answers: Partial<Record<Category, string | null>>;
  step_reached: number;
  duration_seconds: number;
  started_at?: string;
}

export interface ScenePayload {
  scene: Scene;
  draft: ReviewDraft | null;
  decision: { decision: "accepted" | "rejected"; review?: ReviewDraft } | null;
  status: "unseen" | "draft" | "deferred" | "accepted" | "rejected" | "finalized";
  missing_fields: MissingReport;
  category_suggestions: CategorySuggestion[];
  queue_position: number | null;
  queue_total: number;
  file_checks: Record<string, boolean>;
  prev_scene_id: string | null;
  next_scene_id: string | null;
}

export interface Progress {
  target_total: number;
  target_per_category: number;
  reviewed: number;
  accepted: number;
  rejected: number;
  deferred: number;
  drafts: number;
  finalized: number;
  remaining_to_target: number;
  queue_total: number;
  queue_cursor: number;
  category_counts: Record<Category, number>;
  category_quota_full: Category[];
  accepted_by_subset: { lvis: number; laion: number };
  reviewer_id: string | null;
  reviewer_slot?: string | null;
}

export type ReviewerSlot = "r1" | "r2";

export interface ReviewerSummary {
  reviewer: ReviewerSlot;
  reviewer_id: string;
  label: string;
  assigned: number;
  reviewed: number;
  accepted: number;
  rejected: number;
}

export interface Bootstrap {
  index_ready: boolean;
  ready: boolean;
  reviewer: ReviewerSlot | null;
  reviewer_id: string | null;
  guide: { do_not_autoshow: boolean };
  warnings: string[];
  seed: number;
  reviewers?: ReviewerSummary[];
  progress?: Progress;
  current_scene_id?: string | null;
  sources?: Record<string, { exists: boolean; path: string; count?: number }>;
}

export interface DashboardStats {
  quality_distribution: Record<string, number>;
  exclusion_reasons: Record<string, number>;
  rejection_reasons: Record<string, number>;
  caption_edit_rate: number | null;
  missing_label_completion_rate: number | null;
  geometry_correction_rate: number | null;
  mean_review_duration_seconds: number | null;
}

export interface FinalizeResult {
  ok: boolean;
  problems?: string[];
  decision?: string;
  next_scene_id?: string | null;
  progress?: Progress;
}

export function emptyLayerReview(): LayerReview {
  return {
    decision: null,
    exclusion_reason: null,
    exclusion_note: null,
    label: null,
    caption: null,
    geometry_status: null,
    bbox_override: null,
    centroid_override: null,
    geometry_note: null,
  };
}

export function emptyDraft(): ReviewDraft {
  return {
    overall_quality: null,
    layer_reviews: {},
    reviewed_image_caption: null,
    reviewed_image_caption_llava: null,
    reviewed_background_caption: null,
    reviewed_background_caption_llava: null,
    relations_removed: [],
    relations_added: [],
    suitable_categories: [],
    primary_category: null,
    category_decision_made: false,
    human_questions: {},
    human_answers: {},
    step_reached: 1,
    duration_seconds: 0,
    started_at: new Date().toISOString(),
  };
}
