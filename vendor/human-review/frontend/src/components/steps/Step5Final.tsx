import { excludedLayerIds, isBlank, questionAnswerProblems, retainedLayerIds, unresolvedBlocking, type StepStatus } from "../../draftUtils";
import { CATEGORY_LABELS, type ReviewDraft, type ScenePayload } from "../../types";
import type { GuideSection } from "../GuideModal";

export function Step5Final(props: {
  payload: ScenePayload;
  draft: ReviewDraft;
  updateDraft: (updater: (d: ReviewDraft) => ReviewDraft) => void;
  readOnly: boolean;
  problems: string[];
  stepStatuses: StepStatus[];
  openGuide: (section: GuideSection) => void;
  onAccept: () => void;
}) {
  const { payload, draft, updateDraft, readOnly } = props;
  const scene = payload.scene;
  const retained = retainedLayerIds(scene, draft);
  const excluded = excludedLayerIds(scene, draft);
  const blocking = unresolvedBlocking(payload, draft);
  const stepsIncomplete = props.stepStatuses.some((s) => !s.complete);
  const qaProblems = questionAnswerProblems(draft);
  const acceptable = !stepsIncomplete && retained.length > 0 && blocking.length === 0 && qaProblems.length === 0;

  return (
    <div>
      <div className="section-title">
        Step 5 · Final Decision
        <button className="help-icon" title="Final acceptance checklist" onClick={() => props.openGuide("checklist")}>
          ?
        </button>
      </div>

      {props.problems.length > 0 && (
        <div className="problem-list" data-testid="finalize-problems">
          <b>Cannot finalize yet:</b>
          <ul>
            {props.problems.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="card">
        <div className="field-label">Summary</div>
        <table className="kv">
          <tbody>
            <tr>
              <td>retained layers</td>
              <td>
                {retained.length > 0 ? retained.map((id) => {
                  const lr = draft.layer_reviews[String(id)];
                  return (
                    <span key={id} className="badge keep" style={{ marginRight: 6 }}>
                      L{id} {lr?.label ? `“${lr.label}”` : "(no label!)"}
                    </span>
                  );
                }) : <span className="badge blocking">none — scene cannot be accepted</span>}
              </td>
            </tr>
            <tr>
              <td>excluded layers</td>
              <td>
                {excluded.length > 0
                  ? excluded.map((id) => (
                      <span key={id} className="badge exclude" style={{ marginRight: 6 }}>
                        L{id} ({draft.layer_reviews[String(id)]?.exclusion_reason ?? "?"})
                      </span>
                    ))
                  : "none"}
              </td>
            </tr>
            <tr>
              <td>overall / background captions</td>
              <td>
                {draft.reviewed_image_caption && draft.reviewed_image_caption_llava ? "✓ BLIP-2 and LLaVA reviewed" : "✗ image caption missing"}
                {" · "}
                {draft.reviewed_background_caption && draft.reviewed_background_caption_llava ? "✓ background captions reviewed" : "✗ background caption missing"}
              </td>
            </tr>
            <tr>
              <td>geometry</td>
              <td>
                {retained.map((id) => {
                  const s = draft.layer_reviews[String(id)]?.geometry_status;
                  const cls = s === "correct" ? "ok" : s === "minor_correction" ? "missing" : "blocking";
                  return (
                    <span key={id} className={`badge ${cls}`} style={{ marginRight: 6 }}>
                      L{id}: {s ?? "unreviewed"}
                      {draft.layer_reviews[String(id)]?.bbox_override ? " (bbox edited)" : ""}
                    </span>
                  );
                })}
                {draft.relations_removed.length > 0 && ` · ${draft.relations_removed.length} relation(s) marked invalid`}
                {draft.relations_added.length > 0 && ` · ${draft.relations_added.length} relation(s) added`}
              </td>
            </tr>
            <tr>
              <td>categories</td>
              <td>
                {draft.suitable_categories.length > 0
                  ? draft.suitable_categories.map((c) => (
                      <span key={c} className={`badge ${draft.primary_category === c ? "ok" : "tech"}`} style={{ marginRight: 6 }}>
                        {CATEGORY_LABELS[c]}
                        {draft.primary_category === c ? " (primary)" : ""}
                      </span>
                    ))
                  : "none marked suitable"}
              </td>
            </tr>
            <tr>
              <td>missing fields</td>
              <td>
                {blocking.length === 0 && !stepsIncomplete ? (
                  <span className="badge ok">all required fields complete</span>
                ) : (
                  <>
                    {props.stepStatuses.map((s, i) =>
                      s.complete ? null : (
                        <span key={i} className="badge missing" style={{ marginRight: 6 }}>
                          step {i + 1}: {s.problems.length} open
                        </span>
                      ),
                    )}
                    {blocking.map((b, i) => (
                      <span key={`b${i}`} className="badge blocking" style={{ marginRight: 6 }}>
                        {b}
                      </span>
                    ))}
                  </>
                )}
              </td>
            </tr>
            <tr>
              <td>quota impact</td>
              <td>
                {draft.primary_category
                  ? `accepting adds 1 to “${CATEGORY_LABELS[draft.primary_category]}”`
                  : "no primary category chosen"}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="section-title" style={{ margin: 0, fontSize: 13 }}>Human questions by selected category</div>
        <p className="small-note">
          Step 4 selected {draft.suitable_categories.length} categor{draft.suitable_categories.length === 1 ? "y" : "ies"}.
          A question and answer area is shown for each one. Complete at least one pair; every filled question must have an answer.
        </p>
        {draft.suitable_categories.length === 0 && (
          <div className="warn-banner">Choose at least one suitable category in Step 4 first.</div>
        )}
        {draft.suitable_categories.map((category) => {
          const value = draft.human_questions?.[category] ?? "";
          const answer = draft.human_answers?.[category] ?? "";
          return (
            <div key={category} style={{ marginTop: 10 }}>
              <div className="field-label">
                {CATEGORY_LABELS[category]} question
                {draft.primary_category === category && <span className="badge ok" style={{ marginLeft: 8 }}>PRIMARY</span>}
              </div>
              <textarea
                className={qaProblems.length > 0 && isBlank(value) ? "invalid" : ""}
                placeholder={`Write a ${CATEGORY_LABELS[category]} question grounded in this scene…`}
                value={value}
                disabled={readOnly}
                aria-label={`${CATEGORY_LABELS[category]} human question`}
                onChange={(event) =>
                  updateDraft((d) => ({
                    ...d,
                    human_questions: {
                      ...d.human_questions,
                      [category]: event.target.value || null,
                    },
                  }))
                }
              />
              <div className="field-label" style={{ marginTop: 8 }}>
                Answer {!isBlank(value) && "(required)"}
              </div>
              <textarea
                className={!isBlank(value) && isBlank(answer) ? "invalid" : ""}
                placeholder={isBlank(value) ? "Enter a question above first…" : "Write the answer to this question…"}
                value={answer}
                disabled={readOnly || isBlank(value)}
                aria-label={`${CATEGORY_LABELS[category]} human answer`}
                onChange={(event) =>
                  updateDraft((d) => ({
                    ...d,
                    human_answers: {
                      ...d.human_answers,
                      [category]: event.target.value || null,
                    },
                  }))
                }
              />
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
        <button
          className="success"
          disabled={readOnly || !acceptable}
          onClick={props.onAccept}
          title={acceptable ? "Accept this scene" : "Complete all steps first"}
          data-testid="accept-button"
        >
          ✓ Accept &amp; Next
        </button>
      </div>
      {!acceptable && !readOnly && (
        <p className="small-note" style={{ marginTop: 8 }}>
          Accept is disabled until every step is complete, at least one layer is retained, and no blocking missing field is
          unresolved.
        </p>
      )}
    </div>
  );
}
