import { useState } from "react";
import type { Progress, ScenePayload } from "../types";

export function Header(props: {
  progress: Progress | null;
  payload: ScenePayload | null;
  saveStatus: "idle" | "unsaved" | "saving" | "saved" | "error";
  reviewerId: string | null;
  onSwitchReviewer: () => void;
  view: "review" | "dashboard";
  onToggleView: () => void;
  onOpenGuide: () => void;
  onJump: (sceneId: string) => void;
  onResume: () => void;
  onUndo: () => void;
  onBackup: () => void;
  onExport: () => void;
  onKeyboardHelp: () => void;
}) {
  const { progress, payload, saveStatus, reviewerId } = props;
  const [jump, setJump] = useState("");
  const reviewerLabel = reviewerId === "reviewer_2" ? "Reviewer 2" : reviewerId === "reviewer_1" ? "Reviewer 1" : reviewerId;

  const statusText: Record<typeof saveStatus, string> = {
    idle: "",
    unsaved: "Unsaved changes…",
    saving: "Saving draft…",
    saved: "Draft saved ✓",
    error: "Save failed!",
  };

  return (
    <div className="header">
      <span className="title">
        MuLAn Review
        {payload && (
          <small className="mono">
            {payload.scene.scene_id}
            {payload.queue_position != null && ` · #${payload.queue_position + 1}/${payload.queue_total}`}
            {` · ${payload.scene.subset.toUpperCase()}`}
          </small>
        )}
      </span>

      {reviewerLabel && (
        <span className="progress-chip" title="Active reviewer (fixed, non-overlapping image set)">
          👤 {reviewerLabel}
          <button
            className="small ghost"
            style={{ marginLeft: 6 }}
            onClick={props.onSwitchReviewer}
            title="Switch to the other reviewer"
          >
            Switch
          </button>
        </span>
      )}

      {progress && (
        <>
          <span className="progress-chip" title="Scenes reviewed out of this reviewer's assigned set">
            <b>{progress.reviewed}</b>/{progress.target_total} reviewed
            <span className="mini-track">
              <span
                className="mini-fill"
                style={{ width: `${progress.target_total ? Math.min(100, (progress.reviewed / progress.target_total) * 100) : 0}%` }}
              />
            </span>
          </span>
          <span className="progress-chip" title="Accepted / rejected / deferred / drafts">
            ✓ {progress.accepted} · ✗ {progress.rejected} · ⏸ {progress.deferred} · ✎ {progress.drafts}
          </span>
          <span className="progress-chip" title="Accepted by subset">
            LVIS {progress.accepted_by_subset.lvis} · LAION {progress.accepted_by_subset.laion}
          </span>
        </>
      )}

      <span className="spacer" />
      <span className={`save-status ${saveStatus}`} data-testid="save-status">
        {statusText[saveStatus]}
      </span>

      <form
        className="jump-box"
        onSubmit={(e) => {
          e.preventDefault();
          if (jump.trim()) {
            props.onJump(jump.trim());
            setJump("");
          }
        }}
      >
        <input
          type="text"
          placeholder="Jump to scene ID…"
          value={jump}
          onChange={(e) => setJump(e.target.value)}
          aria-label="Jump to scene ID"
        />
      </form>

      <button className="small" onClick={props.onResume} title="Return to the queue's current scene">
        Resume
      </button>
      <button className="small" onClick={props.onUndo} title="Undo the most recent final decision">
        Undo last
      </button>
      <button className="small" onClick={props.onExport} title="Rewrite both export files now">
        Export now
      </button>
      <button className="small" onClick={props.onBackup} title="Create a timestamped backup of review_state.json">
        Backup
      </button>
      <button className="small" onClick={props.onToggleView}>
        {props.view === "review" ? "Dashboard" : "Back to review"}
      </button>
      <button className="small" onClick={props.onKeyboardHelp} title="Keyboard shortcuts (?)">
        ⌨
      </button>
      <button className="small primary" onClick={props.onOpenGuide} title="Open the Review Guide (g)">
        Review Guide
      </button>
    </div>
  );
}
