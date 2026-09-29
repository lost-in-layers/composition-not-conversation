import { isBlank, retainedLayerIds } from "../../draftUtils";
import type { LayerReview, ReviewDraft, ScenePayload } from "../../types";
import type { GuideSection } from "../GuideModal";

function EditableCaption(props: {
  label: string;
  value: string | null;
  required?: boolean;
  readOnly: boolean;
  ariaLabel: string;
  onChange: (value: string | null) => void;
}) {
  return (
    <div style={{ marginTop: 8 }}>
      <div className="field-label">{props.label} {props.required && "(required)"}</div>
      <textarea
        className={props.required && isBlank(props.value) ? "invalid" : ""}
        value={props.value ?? ""}
        disabled={props.readOnly}
        aria-label={props.ariaLabel}
        onChange={(event) => props.onChange(event.target.value || null)}
      />
    </div>
  );
}

export function Step2Captions(props: {
  payload: ScenePayload;
  draft: ReviewDraft;
  updateDraft: (updater: (d: ReviewDraft) => ReviewDraft) => void;
  readOnly: boolean;
  onSelectLayer: (layerId: number) => void;
  openGuide: (section: GuideSection) => void;
}) {
  const { payload, draft, updateDraft, readOnly } = props;
  const scene = payload.scene;
  const retained = new Set(retainedLayerIds(scene, draft));

  const setLayer = (layerId: number, patch: Partial<LayerReview>) =>
    updateDraft((d) => ({
      ...d,
      layer_reviews: { ...d.layer_reviews, [String(layerId)]: { ...d.layer_reviews[String(layerId)], ...patch } },
    }));

  return (
    <div>
      <div className="section-title">
        Step 2 · Captions &amp; Labels
        <button className="help-icon" title="Caption and label definitions" onClick={() => props.openGuide("captions")}>
          ?
        </button>
      </div>

      <div className="card">
        <div className="section-title" style={{ margin: 0, fontSize: 13 }}>Overall image captions</div>
        <EditableCaption
          label="BLIP-2 caption"
          value={draft.reviewed_image_caption}
          required
          readOnly={readOnly}
          ariaLabel="Editable BLIP-2 overall image caption"
          onChange={(value) => updateDraft((d) => ({ ...d, reviewed_image_caption: value }))}
        />
        <EditableCaption
          label="LLaVA caption"
          value={draft.reviewed_image_caption_llava}
          required
          readOnly={readOnly}
          ariaLabel="Editable LLaVA overall image caption"
          onChange={(value) => updateDraft((d) => ({ ...d, reviewed_image_caption_llava: value }))}
        />
      </div>

      <div className="card">
        <div className="section-title" style={{ margin: 0, fontSize: 13 }}>Background captions (background only; ignore foreground objects)</div>
        <EditableCaption
          label="BLIP-2 caption"
          value={draft.reviewed_background_caption}
          required
          readOnly={readOnly}
          ariaLabel="Editable BLIP-2 background caption"
          onChange={(value) => updateDraft((d) => ({ ...d, reviewed_background_caption: value }))}
        />
        <EditableCaption
          label="LLaVA caption"
          value={draft.reviewed_background_caption_llava}
          required
          readOnly={readOnly}
          ariaLabel="Editable LLaVA background caption"
          onChange={(value) => updateDraft((d) => ({ ...d, reviewed_background_caption_llava: value }))}
        />
      </div>

      {scene.layers.map((layer) => {
        const lr = draft.layer_reviews[String(layer.layer_id)];
        const isRetained = retained.has(layer.layer_id);
        const excluded = lr?.decision === "exclude";
        return (
          <div
            key={layer.layer_id}
            className="card"
            style={excluded ? { opacity: 0.55 } : undefined}
            onClick={() => props.onSelectLayer(layer.layer_id)}
            data-testid={`step2-layer-${layer.layer_id}`}
          >
            <div className="section-title" style={{ margin: 0, fontSize: 13 }}>
              Layer {layer.layer_id} <span className="z-chip">z{layer.z_index}</span>
              {excluded && <span className="badge exclude">excluded — no completion needed</span>}
              {!excluded && !lr?.decision && <span className="badge pending">decide quality in Step 1</span>}
              {layer.label === null && !excluded && (
                <span className="badge missing" title="Source label is null and must be completed by you">
                  label: null in source
                </span>
              )}
            </div>
            {!excluded && (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 10, marginTop: 8 }}>
                  <div>
                    <div className="field-label">Canonical short label {isRetained && "(required)"}</div>
                    <input
                      type="text"
                      className={isRetained && isBlank(lr?.label) ? "invalid" : ""}
                      placeholder='e.g. "giraffe", "red bag"'
                      value={lr?.label ?? ""}
                      disabled={readOnly}
                      aria-label={`Canonical label for layer ${layer.layer_id}`}
                      onChange={(e) => setLayer(layer.layer_id, { label: e.target.value || null })}
                      style={{ width: "100%" }}
                    />
                    <div className="small-note" style={{ marginTop: 3 }}>
                      label_source becomes <code>human_review</code> on acceptance.
                    </div>
                  </div>
                  <div>
                    <EditableCaption
                      label="BLIP-2 layer caption"
                      value={lr?.caption ?? null}
                      required={isRetained}
                      readOnly={readOnly}
                      ariaLabel={`Editable BLIP-2 caption for layer ${layer.layer_id}`}
                      onChange={(value) => setLayer(layer.layer_id, { caption: value })}
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
