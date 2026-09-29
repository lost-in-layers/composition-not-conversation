import { useEffect, useState, type ReactNode } from "react";
import type { ReviewerSlot, ReviewerSummary } from "../types";

export function Modal(props: {
  title: string;
  onClose?: () => void;
  children: ReactNode;
  footer?: ReactNode;
  small?: boolean;
  headExtra?: ReactNode;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && props.onClose) props.onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props.onClose]);

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && props.onClose) props.onClose();
      }}
    >
      <div className={`modal ${props.small ? "small" : ""}`} role="dialog" aria-label={props.title}>
        <div className="modal-head">
          <h2>{props.title}</h2>
          {props.headExtra}
          {props.onClose && (
            <button className="close-x ghost" onClick={props.onClose} aria-label="Close">
              ✕
            </button>
          )}
        </div>
        {props.children}
        {props.footer && <div className="modal-foot">{props.footer}</div>}
      </div>
    </div>
  );
}

export function ConfirmModal(props: {
  title: string;
  body: string;
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <Modal
      title={props.title}
      onClose={props.onCancel}
      small
      footer={
        <>
          <span className="spacer" style={{ flex: 1 }} />
          <button onClick={props.onCancel}>Cancel</button>
          <button className={props.danger ? "danger" : "success"} onClick={props.onConfirm}>
            {props.confirmLabel}
          </button>
        </>
      }
    >
      <div className="modal-body">
        <p>{props.body}</p>
      </div>
    </Modal>
  );
}

export function ReviewerPicker(props: {
  reviewers: ReviewerSummary[];
  onSelect: (slot: ReviewerSlot) => Promise<void>;
  onClose?: () => void;
}) {
  const [busy, setBusy] = useState<ReviewerSlot | null>(null);

  const pick = async (slot: ReviewerSlot) => {
    if (busy) return;
    setBusy(slot);
    try {
      await props.onSelect(slot);
    } finally {
      setBusy(null);
    }
  };

  const cards =
    props.reviewers.length > 0
      ? props.reviewers
      : ([
          { reviewer: "r1", reviewer_id: "reviewer_1", label: "Reviewer 1", assigned: 0, reviewed: 0, accepted: 0, rejected: 0 },
          { reviewer: "r2", reviewer_id: "reviewer_2", label: "Reviewer 2", assigned: 0, reviewed: 0, accepted: 0, rejected: 0 },
        ] as ReviewerSummary[]);

  return (
    <Modal title="Who is reviewing?" onClose={props.onClose} small>
      <div className="modal-body">
        <p className="muted">
          Each reviewer has a fixed, non-overlapping set of images. Your decisions,
          drafts and exports are kept separate per reviewer.
        </p>
        <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
          {cards.map((rv) => {
            const remaining = Math.max(0, rv.assigned - rv.reviewed);
            const pct = rv.assigned ? Math.round((rv.reviewed / rv.assigned) * 100) : 0;
            return (
              <button
                key={rv.reviewer}
                className="card"
                onClick={() => void pick(rv.reviewer)}
                disabled={!!busy}
                style={{ flex: 1, textAlign: "left", cursor: "pointer", padding: 14 }}
              >
                <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 6 }}>{rv.label}</div>
                <div className="muted" style={{ fontSize: 13 }}>
                  {rv.reviewed}/{rv.assigned} reviewed · {remaining} left
                </div>
                <div className="mini-track" style={{ marginTop: 8 }}>
                  <span className="mini-fill" style={{ width: `${pct}%` }} />
                </div>
                <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                  ✓ {rv.accepted} accepted · ✗ {rv.rejected} rejected
                </div>
                <div style={{ marginTop: 10, fontWeight: 600, color: "var(--accent, #4c8bf5)" }}>
                  {busy === rv.reviewer ? "Opening…" : "Review as this reviewer →"}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </Modal>
  );
}

const SHORTCUTS: Array<[string, string]> = [
  ["?", "Open this shortcut list"],
  ["g", "Open the Review Guide"],
  ["1 – 5", "Jump to review step"],
  ["[", "Previous unfinished scene"],
  ["]", "Next unfinished scene"],
  ["d", "Defer current scene"],
  ["Esc", "Close any open dialog"],
];

export function KeyboardHelpModal(props: { onClose: () => void }) {
  return (
    <Modal title="Keyboard shortcuts" onClose={props.onClose} small>
      <div className="modal-body">
        <table className="kv">
          <tbody>
            {SHORTCUTS.map(([key, desc]) => (
              <tr key={key}>
                <td>
                  <span className="kbd">{key}</span>
                </td>
                <td>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small-note" style={{ marginTop: 10 }}>
          Shortcuts are ignored while typing in a text field.
        </p>
      </div>
    </Modal>
  );
}
