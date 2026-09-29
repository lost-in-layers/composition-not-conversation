import { useState } from "react";
import { clampBbox, retainedLayerIds, validBbox } from "../../draftUtils";
import type { LayerReview, ReviewDraft, ScenePayload } from "../../types";
import type { GuideSection } from "../GuideModal";

const PREDICATES = ["left_of", "right_of", "above", "below", "near", "contains", "overlaps", "possible_occludes"];

export function Step3Geometry(props: {
  payload: ScenePayload;
  draft: ReviewDraft;
  updateDraft: (updater: (d: ReviewDraft) => ReviewDraft) => void;
  readOnly: boolean;
  selectedLayerId: number | null;
  onSelectLayer: (layerId: number) => void;
  openGuide: (section: GuideSection) => void;
}) {
  const { payload, draft, updateDraft, readOnly } = props;
  const scene = payload.scene;
  const retained = retainedLayerIds(scene, draft);
  const retainedSet = new Set(retained);
  const [newRel, setNewRel] = useState<{ subject: string; predicate: string; object: string }>({
    subject: "",
    predicate: "left_of",
    object: "",
  });

  const setLayer = (layerId: number, patch: Partial<LayerReview>) =>
    updateDraft((d) => ({
      ...d,
      layer_reviews: { ...d.layer_reviews, [String(layerId)]: { ...d.layer_reviews[String(layerId)], ...patch } },
    }));

  const removedKeys = new Set(draft.relations_removed.map((r) => r.join("|")));
  const activeRelations = (scene.relations ?? []).filter(
    (rel) => retainedSet.has(rel.subject_layer_id) && retainedSet.has(rel.object_layer_id),
  );

  return (
    <div>
      <div className="section-title">
        Step 3 · Geometry &amp; Relations
        <button className="help-icon" title="Geometry definitions" onClick={() => props.openGuide("geometry")}>
          ?
        </button>
        <span className="small-note">
          Corrections are stored as overrides — the original geometry and alpha masks are never modified. Select a layer and
          drag its box/corners/centroid in the viewer, or type exact values here.
        </span>
      </div>

      {scene.layers.map((layer) => {
        const lr = draft.layer_reviews[String(layer.layer_id)];
        if (lr?.decision === "exclude") {
          return (
            <div key={layer.layer_id} className="card" style={{ opacity: 0.5 }}>
              Layer {layer.layer_id} <span className="badge exclude">excluded</span>{" "}
              <span className="small-note">geometry review not required</span>
            </div>
          );
        }
        const geo = layer.geometry ?? {};
        const bbox = geo.bbox_xyxy;
        const override = lr?.bbox_override ?? null;
        const bboxInvalid = !validBbox(override, scene.width, scene.height);
        const status = lr?.geometry_status ?? null;
        return (
          <div
            key={layer.layer_id}
            className="card"
            onClick={() => props.onSelectLayer(layer.layer_id)}
            style={props.selectedLayerId === layer.layer_id ? { borderColor: "var(--accent)" } : undefined}
            data-testid={`step3-layer-${layer.layer_id}`}
          >
            <div className="section-title" style={{ margin: 0, fontSize: 13 }}>
              Layer {layer.layer_id} <span className="z-chip">z{layer.z_index}</span>
              <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>{lr?.label || layer.captions?.blip2}</span>
            </div>

            <table className="kv" style={{ marginTop: 6 }}>
              <tbody>
                <tr>
                  <td>original bbox [x1,y1,x2,y2]</td>
                  <td className="mono">{bbox ? `[${bbox.join(", ")}]` : "—"}</td>
                </tr>
                <tr>
                  <td>original centroid</td>
                  <td className="mono">
                    {geo.support_centroid_xy ? `[${geo.support_centroid_xy.map((v) => (+v).toFixed(1)).join(", ")}]` : "—"}
                  </td>
                </tr>
                <tr>
                  <td>mask area / components (read-only)</td>
                  <td className="mono">
                    {geo.support_area_pixels ?? "—"} px · {geo.connected_components_4 ?? "—"} component(s)
                  </td>
                </tr>
              </tbody>
            </table>

            <div className="radio-row" style={{ marginTop: 8 }}>
              <button className={status === "correct" ? "on-good" : ""} disabled={readOnly} onClick={() => setLayer(layer.layer_id, { geometry_status: "correct" })}>
                Correct
              </button>
              <button className={status === "minor_correction" ? "on-warn" : ""} disabled={readOnly} onClick={() => setLayer(layer.layer_id, { geometry_status: "minor_correction" })}>
                Minor correction
              </button>
              <button className={status === "major_problem" ? "on-bad" : ""} disabled={readOnly} onClick={() => setLayer(layer.layer_id, { geometry_status: "major_problem" })}>
                Major problem
              </button>
              <span className="small-note">A major problem blocks acceptance — correct it, exclude the layer, or reject.</span>
            </div>

            <div style={{ display: "flex", gap: 14, marginTop: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
              <div>
                <div className="field-label">Corrected bbox (override)</div>
                <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                  {["x1", "y1", "x2", "y2"].map((name, i) => (
                    <input
                      key={name}
                      type="number"
                      aria-label={`${name} for layer ${layer.layer_id}`}
                      className={bboxInvalid ? "invalid" : ""}
                      style={{ width: 72 }}
                      placeholder={name}
                      disabled={readOnly}
                      value={override ? override[i] : ""}
                      onChange={(e) => {
                        const base: [number, number, number, number] =
                          override ?? (bbox as [number, number, number, number]) ?? [0, 0, scene.width, scene.height];
                        const next = [...base] as [number, number, number, number];
                        next[i] = Number(e.target.value);
                        setLayer(layer.layer_id, { bbox_override: next });
                      }}
                    />
                  ))}
                  <button
                    className="small"
                    disabled={readOnly || !override}
                    onClick={() =>
                      override && setLayer(layer.layer_id, { bbox_override: clampBbox(override, scene.width, scene.height) })
                    }
                    title="Clamp to image bounds and fix ordering"
                  >
                    Clamp
                  </button>
                  <button className="small" disabled={readOnly || !override} onClick={() => setLayer(layer.layer_id, { bbox_override: null })}>
                    Reset
                  </button>
                </div>
                {bboxInvalid && (
                  <div className="small-note" style={{ color: "var(--bad)" }} data-testid={`bbox-invalid-${layer.layer_id}`}>
                    Must satisfy 0 ≤ x1 &lt; x2 ≤ {scene.width}, 0 ≤ y1 &lt; y2 ≤ {scene.height}.
                  </div>
                )}
              </div>
              <div>
                <div className="field-label">Corrected centroid (override)</div>
                <div style={{ display: "flex", gap: 4 }}>
                  {["x", "y"].map((name, i) => (
                    <input
                      key={name}
                      type="number"
                      aria-label={`centroid ${name} for layer ${layer.layer_id}`}
                      style={{ width: 84 }}
                      placeholder={name}
                      disabled={readOnly}
                      value={lr?.centroid_override ? lr.centroid_override[i] : ""}
                      onChange={(e) => {
                        const base = lr?.centroid_override ?? (geo.support_centroid_xy as [number, number]) ?? [0, 0];
                        const next = [...base] as [number, number];
                        next[i] = Number(e.target.value);
                        setLayer(layer.layer_id, { centroid_override: next });
                      }}
                    />
                  ))}
                  <button className="small" disabled={readOnly || !lr?.centroid_override} onClick={() => setLayer(layer.layer_id, { centroid_override: null })}>
                    Reset
                  </button>
                </div>
              </div>
              <div style={{ flex: 1, minWidth: 200 }}>
                <div className="field-label">Geometry note</div>
                <input
                  type="text"
                  placeholder="optional…"
                  value={lr?.geometry_note ?? ""}
                  disabled={readOnly}
                  onChange={(e) => setLayer(layer.layer_id, { geometry_note: e.target.value || null })}
                  style={{ width: "100%" }}
                />
              </div>
            </div>

            <details className="drawer">
              <summary>Full geometry &amp; mask record (read-only)</summary>
              <div className="drawer-body">
                <pre>{JSON.stringify({ geometry: layer.geometry, mask: layer.mask }, null, 1)}</pre>
              </div>
            </details>
          </div>
        );
      })}

      <div className="card">
        <div className="section-title" style={{ margin: 0, fontSize: 13 }}>
          Relations between retained layers
        </div>
        {activeRelations.length === 0 && <p className="small-note">No source relations among retained layers.</p>}
        {activeRelations.map((rel) => {
          const key: [number, string, number] = [rel.subject_layer_id, rel.predicate, rel.object_layer_id];
          const removed = removedKeys.has(key.join("|"));
          return (
            <div key={key.join("|")} className="bar-row" style={{ opacity: removed ? 0.45 : 1 }}>
              <span className="mono" style={{ width: 300 }}>
                L{rel.subject_layer_id} <b style={{ color: "var(--accent)" }}>{rel.predicate}</b> L{rel.object_layer_id}
                {rel.score != null && <span className="small-note"> ({Number(rel.score).toFixed(2)})</span>}
              </span>
              <span className="small-note" style={{ width: 210 }}>{rel.source}</span>
              <button
                className="small"
                disabled={readOnly}
                onClick={() =>
                  updateDraft((d) => ({
                    ...d,
                    relations_removed: removed
                      ? d.relations_removed.filter((r) => r.join("|") !== key.join("|"))
                      : [...d.relations_removed, key],
                  }))
                }
              >
                {removed ? "Restore" : "Mark invalid"}
              </button>
            </div>
          );
        })}
        {draft.relations_added.map((rel, index) => (
          <div key={`added-${index}`} className="bar-row">
            <span className="mono" style={{ width: 300 }}>
              L{rel.subject_layer_id} <b style={{ color: "var(--good)" }}>{rel.predicate}</b> L{rel.object_layer_id}
            </span>
            <span className="small-note" style={{ width: 210 }}>human_review (added)</span>
            <button
              className="small"
              disabled={readOnly}
              onClick={() => updateDraft((d) => ({ ...d, relations_added: d.relations_added.filter((_, i) => i !== index) }))}
            >
              Remove
            </button>
          </div>
        ))}
        {retained.length >= 2 && !readOnly && (
          <div className="inline-actions" style={{ marginTop: 8 }}>
            <select value={newRel.subject} aria-label="New relation subject" onChange={(e) => setNewRel({ ...newRel, subject: e.target.value })}>
              <option value="">subject…</option>
              {retained.map((id) => (
                <option key={id} value={id}>{`L${id}`}</option>
              ))}
            </select>
            <select value={newRel.predicate} aria-label="New relation predicate" onChange={(e) => setNewRel({ ...newRel, predicate: e.target.value })}>
              {PREDICATES.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            <select value={newRel.object} aria-label="New relation object" onChange={(e) => setNewRel({ ...newRel, object: e.target.value })}>
              <option value="">object…</option>
              {retained.map((id) => (
                <option key={id} value={id}>{`L${id}`}</option>
              ))}
            </select>
            <button
              className="small"
              disabled={!newRel.subject || !newRel.object || newRel.subject === newRel.object}
              onClick={() => {
                updateDraft((d) => ({
                  ...d,
                  relations_added: [
                    ...d.relations_added,
                    { subject_layer_id: Number(newRel.subject), predicate: newRel.predicate, object_layer_id: Number(newRel.object) },
                  ],
                }));
                setNewRel({ subject: "", predicate: "left_of", object: "" });
              }}
            >
              + Add relation
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
