import { setPrimary, toggleSuitable } from "../../draftUtils";
import {
  CATEGORIES,
  CATEGORY_EXAMPLES,
  CATEGORY_LABELS,
  type Progress,
  type ReviewDraft,
  type ScenePayload,
} from "../../types";
import type { GuideSection } from "../GuideModal";

const CATEGORY_GUIDANCE: Record<string, string> = {
  direct_ask: "≥1 coherent recognizable object.",
  attribute_binding: "Target + anchor + attribute + a unique identifying relation.",
  spatial: "≥2 retained layers with an unambiguous spatial relation.",
  occlusion: "Meaningful overlap plus reliable front/behind ordering (z-index alone is not enough).",
  counting: "Separately represented objects forming a complete countable set.",
  multi_hop_reference: "≥3 retained layers and a unique chain of ≥2 relations.",
};

export function Step4Categories(props: {
  payload: ScenePayload;
  draft: ReviewDraft;
  updateDraft: (updater: (d: ReviewDraft) => ReviewDraft) => void;
  readOnly: boolean;
  progress: Progress | null;
  openGuide: (section: GuideSection) => void;
}) {
  const { payload, draft, updateDraft, readOnly, progress } = props;

  return (
    <div>
      <div className="section-title">
        Step 4 · Category Suitability
        <button className="help-icon" title="Category definitions" onClick={() => props.openGuide("categories")}>
          ?
        </button>
        <span className="small-note">
          Mark every suitable category, then choose exactly one primary. Only the primary counts toward its 100-scene quota.
          Suggestions are automatic hints — your decision is authoritative.
        </span>
      </div>

      <div className="card-grid">
        {CATEGORIES.map((cat) => {
          const suggestion = payload.category_suggestions.find((s) => s.category === cat);
          const suitable = draft.suitable_categories.includes(cat);
          const isPrimary = draft.primary_category === cat;
          const count = progress?.category_counts?.[cat] ?? 0;
          const target = progress?.target_per_category ?? 100;
          const full = count >= target;
          return (
            <div
              key={cat}
              className={`card category-card ${suitable ? "selected" : ""} ${isPrimary ? "primary" : ""}`}
              data-testid={`category-card-${cat}`}
            >
              <div className="cat-head">
                <label style={{ display: "flex", alignItems: "center", gap: 7, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={suitable}
                    disabled={readOnly}
                    aria-label={`${CATEGORY_LABELS[cat]} suitable`}
                    onChange={() => updateDraft((d) => toggleSuitable(d, cat))}
                  />
                  <span className="name">{CATEGORY_LABELS[cat]}</span>
                </label>
                {isPrimary && <span className="badge ok">PRIMARY</span>}
                <span style={{ marginLeft: "auto" }}>
                  <button
                    className={`small ${isPrimary ? "success" : ""}`}
                    disabled={readOnly}
                    onClick={() => updateDraft((d) => setPrimary(d, cat))}
                    aria-label={`Set ${CATEGORY_LABELS[cat]} as primary`}
                  >
                    {isPrimary ? "Primary ✓" : "Set primary"}
                  </button>
                </span>
              </div>
              <div className="example">“{CATEGORY_EXAMPLES[cat]}”</div>
              <div className="hint">Review guidance: {CATEGORY_GUIDANCE[cat]}</div>
              {suggestion && (
                <div className={`suggestion ${suggestion.suggested ? "yes" : ""}`}>
                  {suggestion.suggested ? "Suggested" : "Not suggested"} — {suggestion.explanation}
                </div>
              )}
              <div className="quota">
                <div className="quota-track">
                  <div className={`quota-fill ${full ? "full" : ""}`} style={{ width: `${Math.min(100, (count / target) * 100)}%` }} />
                </div>
                <div className="quota-text">
                  <span>
                    {count}/{target} accepted
                  </span>
                  {full && <span style={{ color: "var(--warn)" }}>quota full — prefer another primary</span>}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {draft.primary_category && progress && (progress.category_counts?.[draft.primary_category] ?? 0) >= (progress.target_per_category ?? 100) && (
        <div className="warn-banner" data-testid="quota-warning">
          The quota for “{CATEGORY_LABELS[draft.primary_category]}” is already full ({progress.category_counts[draft.primary_category]}/
          {progress.target_per_category}). You can still accept — the decision is yours — but an underfilled category may be a better primary.
        </div>
      )}

    </div>
  );
}
