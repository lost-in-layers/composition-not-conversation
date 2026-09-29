import { useEffect, useRef, useState } from "react";
import { Modal } from "./Modals";

export type GuideSection = "workflow" | "layer-quality" | "captions" | "geometry" | "categories" | "checklist";

const SECTIONS: Array<[GuideSection, string]> = [
  ["workflow", "Workflow"],
  ["layer-quality", "Layer quality"],
  ["captions", "Captions & labels"],
  ["geometry", "Geometry"],
  ["categories", "Categories"],
  ["checklist", "Acceptance checklist"],
];

export function GuideModal(props: {
  section: GuideSection;
  doNotAutoshow: boolean;
  onPreference: (value: boolean) => void;
  onClose: () => void;
}) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const [checked, setChecked] = useState(props.doNotAutoshow);

  useEffect(() => {
    const el = bodyRef.current?.querySelector(`[data-guide-section="${props.section}"]`);
    if (el) el.scrollIntoView({ block: "start" });
  }, [props.section]);

  const jump = (section: GuideSection) => {
    const el = bodyRef.current?.querySelector(`[data-guide-section="${section}"]`);
    if (el) el.scrollIntoView({ block: "start", behavior: "smooth" });
  };

  return (
    <Modal
      title="Review Guide"
      onClose={props.onClose}
      footer={
        <>
          <label style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 13 }}>
            <input
              type="checkbox"
              checked={checked}
              onChange={(e) => {
                setChecked(e.target.checked);
                props.onPreference(e.target.checked);
              }}
            />
            Do not automatically show again
          </label>
          <span style={{ flex: 1 }} />
          <button className="primary" onClick={props.onClose}>
            Close
          </button>
        </>
      }
    >
      <div className="guide-nav">
        {SECTIONS.map(([id, label]) => (
          <button key={id} onClick={() => jump(id)}>
            {label}
          </button>
        ))}
      </div>
      <div className="modal-body" ref={bodyRef}>
        <div className="guide-section" data-guide-section="workflow">
          <h3>Complete workflow</h3>
          <ol>
            <li><b>Review layer quality first.</b> Look at the full image, the background, and every foreground layer.</li>
            <li><b>Reject immediately</b> if the decomposition is unusable — do not waste time on later steps.</li>
            <li><b>Exclude individual bad layers</b> when the rest of the scene remains useful.</li>
            <li><b>Fill missing canonical labels and correct captions.</b> Every retained layer starts with <code>label: null</code> — you supply the label.</li>
            <li><b>Verify geometry and relations</b> (boxes, centroids, stack order, spatial relations).</li>
            <li><b>Select suitable categories and exactly one primary category.</b></li>
            <li><b>Review category-specific question-and-answer areas and the final summary.</b> Step 5 creates one area per category selected in Step 4; complete at least one pair, and answer every question you fill.</li>
          </ol>
          <p className="example">
            Missing-field indicators: <span className="badge missing">n missing</span> = values you must complete;{" "}
            <span className="badge blocking">blocking</span> = a missing source value (e.g. a missing image file) that can only be
            solved by excluding the layer, rejecting, or deferring; <span className="badge tech">tech-null</span> = optional
            technical/provenance fields (e.g. a native checksum that never existed) that legitimately stay null — never invent
            values for those.
          </p>
        </div>

        <div className="guide-section" data-guide-section="layer-quality">
          <h3>Layer-quality definitions</h3>
          <ul>
            <li><b>Good (Keep):</b> a coherent, correctly isolated object.</li>
            <li><b>Exclude layer:</b> incorrect segmentation, fused objects, fragmented object, duplicate, irrelevant or hallucinated content, unusable transparency, severe artifact, or a meaningless layer.</li>
            <li><b>Reject scene:</b> unusable background/decomposition, no useful foreground layers, or anything that makes the scene unreliable for VQA.</li>
          </ul>
          <p>The background layer cannot be excluded. If the background is unusable, reject the scene. Excluding every foreground layer also means the scene must be rejected.</p>
          <p>Exclusions never modify PNGs — original layer IDs and z-indexes are preserved; excluded layers simply leave the curated <code>layers</code> list (and their relations are dropped).</p>
        </div>

        <div className="guide-section" data-guide-section="captions">
          <h3>Caption and label definitions</h3>
          <ul>
            <li><b>Overall BLIP-2 and LLaVA captions</b> are editable in place; correct each source caption directly.</li>
            <li><b>Background BLIP-2 and LLaVA captions</b> describe only the background (ignore foreground objects) and are editable in place.</li>
            <li><b>Canonical layer label</b> is a short object name such as <code>giraffe</code>, <code>red bag</code>, or <code>bicycle</code>.</li>
            <li><b>Layer caption</b> describes only that layer's content.</li>
            <li>Prefer concise, visually grounded language.</li>
            <li>Do <b>not</b> preserve hallucinated MuLAn text — the source captions frequently describe things that are not there ("in the dark", "two flowers in the black sky").</li>
            <li>Do <b>not</b> add details that are not visible.</li>
            <li>Original <code>label: null</code> / <code>label_source: null</code> must be completed for every retained layer; on acceptance <code>label_source</code> becomes <code>human_review</code>. Original nulls stay untouched in provenance.</li>
          </ul>
          <p>If you cannot confidently assign a canonical label to a layer, exclude the layer. If that makes the scene unusable, reject or defer it.</p>
        </div>

        <div className="guide-section" data-guide-section="geometry">
          <h3>Geometry definitions</h3>
          <ul>
            <li>Bounding box uses <code>[x1, y1, x2, y2]</code> pixel coordinates (x1 &lt; x2, y1 &lt; y2, clamped to the image).</li>
            <li>The centroid marks the central visible location of the object.</li>
            <li><b>Z-index is ordinal stack order, not metric depth.</b> A larger z-index means the layer is composited later — on top.</li>
            <li>Overlap between masks plus z-order may <i>suggest</i> occlusion, but confirm it visually.</li>
            <li>Your corrections are stored as <b>overrides</b>; the original geometry and the alpha masks are never modified.</li>
            <li>Mask-derived area/perimeter values are read-only. If a mask is fundamentally unusable, exclude the layer instead of trying to fix numbers.</li>
          </ul>
        </div>

        <div className="guide-section" data-guide-section="categories">
          <h3>Category definitions</h3>
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th>Meaning &amp; example</th>
                <th>Review guidance</th>
                <th>Suitable when…</th>
                <th>Unsuitable when…</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><b>Direct Ask</b></td>
                <td>Ask directly about an object. <i>“What color is the bag?”</i></td>
                <td>≥1 coherent recognizable object.</td>
                <td>A clearly identifiable object with askable attributes.</td>
                <td>All objects are ambiguous blobs.</td>
              </tr>
              <tr>
                <td><b>Attribute Binding</b></td>
                <td>Attribute of an object identified by a relation. <i>“What color is the object left of the bicycle?”</i></td>
                <td>Target, anchor, attribute, and a unique identifying relation.</td>
                <td>“The cup left of the laptop” picks out exactly one cup.</td>
                <td>Multiple objects satisfy the same reference.</td>
              </tr>
              <tr>
                <td><b>Spatial</b></td>
                <td>Relative position. <i>“Which object is above the table?”</i></td>
                <td>≥2 retained layers and an unambiguous spatial relation.</td>
                <td>Clear left/right/above/below arrangement.</td>
                <td>Objects nearly coincide; the relation is debatable.</td>
              </tr>
              <tr>
                <td><b>Occlusion</b></td>
                <td>Front/behind ordering. <i>“What object is behind the person?”</i></td>
                <td>Meaningful overlap and reliable front/behind ordering; <b>z-index alone is insufficient</b>.</td>
                <td>Visible partial covering with obvious depth order.</td>
                <td>Tiny 1-px overlaps, or ordering only implied by z-index.</td>
              </tr>
              <tr>
                <td><b>Counting</b></td>
                <td>Count objects. <i>“How many objects are right of the chair?”</i></td>
                <td>Separately represented objects forming a complete countable set.</td>
                <td>Every instance of the counted class has its own layer.</td>
                <td>Some instances are fused into one layer or missing.</td>
              </tr>
              <tr>
                <td><b>Multi-hop Reference</b></td>
                <td>Chained references. <i>“What is behind the object nearest to the person?”</i></td>
                <td>≥3 retained layers and a unique chain of ≥2 relations.</td>
                <td>Each hop resolves to exactly one object.</td>
                <td>A hop is ambiguous or the chain collapses to one object.</td>
              </tr>
            </tbody>
          </table>
          <p>All six categories are always selectable. Automatic suggestions are guidance only; your judgment decides suitability. A scene may support several categories, but exactly one is primary and only the primary counts toward its 100-scene quota.</p>
        </div>

        <div className="guide-section" data-guide-section="checklist">
          <h3>Final acceptance checklist</h3>
          <ul>
            <li>At least one retained foreground layer.</li>
            <li>Every layer has a quality decision (keep / exclude).</li>
            <li>Every retained layer has a human canonical label <b>and</b> a reviewed caption.</li>
            <li>BLIP-2 and LLaVA overall/background captions are reviewed.</li>
            <li>Geometry reviewed for every retained layer (no unresolved major problems).</li>
            <li>Exactly one primary category is selected.</li>
            <li>At least one scene-grounded question–answer pair is completed under a category selected in Step 4.</li>
            <li>Every filled question has a non-empty answer.</li>
            <li>No unresolved blocking missing fields (e.g. missing image files).</li>
          </ul>
          <p>Reject a scene directly from Step 1. It is exported and the next scene opens; rejected scenes never count toward the 600-scene target.</p>
        </div>
      </div>
    </Modal>
  );
}
