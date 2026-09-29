import type { GuideSection } from "../GuideModal";
import { EXCLUSION_REASONS, type ReviewDraft, type ScenePayload } from "../../types";

export function Step1LayerQuality(props: {
  payload: ScenePayload;
  draft: ReviewDraft;
  updateDraft: (updater: (d: ReviewDraft) => ReviewDraft) => void;
  readOnly: boolean;
  selectedLayerId: number | null;
  onSelectLayer: (layerId: number) => void;
  openGuide: (section: GuideSection) => void;
  onRejectNow: () => void;
}) {
  const { payload, draft, updateDraft, readOnly } = props;
  const scene = payload.scene;

  const setLayer = (layerId: number, patch: Partial<ReviewDraft["layer_reviews"][string]>) =>
    updateDraft((d) => ({
      ...d,
      layer_reviews: {
        ...d.layer_reviews,
        [String(layerId)]: { ...d.layer_reviews[String(layerId)], ...patch },
      },
    }));

  return (
    <div>
      <div className="section-title">
        Step 1 · Layer Quality
        <button className="help-icon" title="Layer-quality definitions" onClick={() => props.openGuide("layer-quality")}>
          ?
        </button>
      </div>

      <div className="card">
        <div className="field-label">Overall decomposition quality (1 = unusable, 5 = excellent)</div>
        <div className="quality-stars" role="radiogroup" aria-label="Overall quality">
          {[1, 2, 3, 4, 5].map((value) => (
            <button
              key={value}
              className={draft.overall_quality != null && draft.overall_quality >= value ? "lit" : ""}
              disabled={readOnly}
              aria-label={`Quality ${value}`}
              onClick={() => updateDraft((d) => ({ ...d, overall_quality: value }))}
            >
              ★
            </button>
          ))}
          {draft.overall_quality != null && <span className="muted" style={{ alignSelf: "center", fontSize: 12 }}>{draft.overall_quality}/5</span>}
        </div>
        <p className="small-note" style={{ marginBottom: 0 }}>
          The background cannot be excluded — if it is unusable, or no foreground layer is worth keeping, reject the scene.
        </p>
        <div className="inline-actions">
          <button className="danger small" disabled={readOnly} onClick={props.onRejectNow}>
            Reject scene
          </button>
        </div>
      </div>

      {scene.layers.map((layer) => {
        const lr = draft.layer_reviews[String(layer.layer_id)];
        const decision = lr?.decision ?? null;
        const fileMissing = payload.file_checks[`layer_${layer.layer_id}`] === false;
        return (
          <div
            key={layer.layer_id}
            className="card"
            onClick={() => props.onSelectLayer(layer.layer_id)}
            style={props.selectedLayerId === layer.layer_id ? { borderColor: "var(--accent)" } : undefined}
            data-testid={`step1-layer-${layer.layer_id}`}
          >
            <div className="section-title" style={{ margin: 0, fontSize: 13 }}>
              Layer {layer.layer_id} <span className="z-chip">z{layer.z_index}</span>
              <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>
                {layer.captions?.blip2 ?? "(no source caption)"}
              </span>
              {fileMissing && <span className="badge blocking">image file missing</span>}
            </div>
            <div className="radio-row" style={{ marginTop: 8 }}>
              <button
                className={decision === "keep" ? "on-good" : ""}
                disabled={readOnly}
                onClick={() => setLayer(layer.layer_id, { decision: "keep", exclusion_reason: null, exclusion_note: null })}
              >
                Keep
              </button>
              <button
                className={decision === "exclude" ? "on-bad" : ""}
                disabled={readOnly}
                onClick={() => setLayer(layer.layer_id, { decision: "exclude" })}
              >
                Exclude
              </button>
            </div>
            {decision === "exclude" && (
              <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                <select
                  value={lr?.exclusion_reason ?? ""}
                  disabled={readOnly}
                  aria-label={`Exclusion reason for layer ${layer.layer_id}`}
                  onChange={(e) => setLayer(layer.layer_id, { exclusion_reason: e.target.value || null })}
                >
                  <option value="">— exclusion reason —</option>
                  {EXCLUSION_REASONS.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                {lr?.exclusion_reason === "other" && (
                  <input
                    type="text"
                    placeholder="note (required for 'other')"
                    value={lr?.exclusion_note ?? ""}
                    disabled={readOnly}
                    onChange={(e) => setLayer(layer.layer_id, { exclusion_note: e.target.value })}
                    style={{ minWidth: 240 }}
                  />
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
