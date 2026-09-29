import { useEffect, useState } from "react";
import { api } from "../api";
import { CATEGORIES, CATEGORY_LABELS, type DashboardStats, type Progress } from "../types";

function Bar(props: { name: string; value: number; max: number; suffix?: string }) {
  const pct = props.max > 0 ? Math.min(100, (props.value / props.max) * 100) : 0;
  return (
    <div className="bar-row">
      <span className="name" title={props.name}>{props.name}</span>
      <span className="track">
        <span className="fill" style={{ width: `${pct}%`, display: "block" }} />
      </span>
      <span className="val">
        {props.value}
        {props.suffix ?? ""}
      </span>
    </div>
  );
}

function pct(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function Dashboard() {
  const [data, setData] = useState<{ progress: Progress; dashboard: DashboardStats } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.dashboard().then(setData).catch((err) => setError(String(err)));
  }, []);

  if (error) return <div className="dashboard"><p className="muted">{error}</p></div>;
  if (!data) return <div className="dashboard"><div className="spin" /></div>;
  const { progress, dashboard } = data;

  return (
    <div className="dashboard" data-testid="dashboard">
      <h2>Review dashboard</h2>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="label">Accepted (target {progress.target_total})</div>
          <div className="big" style={{ color: "var(--good)" }}>{progress.accepted}</div>
          <div className="quota-track" style={{ marginTop: 6 }}>
            <div className="quota-fill" style={{ width: `${Math.min(100, (progress.accepted / progress.target_total) * 100)}%` }} />
          </div>
        </div>
        <div className="stat-card">
          <div className="label">Rejected</div>
          <div className="big" style={{ color: "var(--bad)" }}>{progress.rejected}</div>
        </div>
        <div className="stat-card">
          <div className="label">Deferred / Drafts</div>
          <div className="big">{progress.deferred} / {progress.drafts}</div>
        </div>
        <div className="stat-card">
          <div className="label">Remaining to target</div>
          <div className="big">{progress.remaining_to_target}</div>
        </div>
        <div className="stat-card">
          <div className="label">Source balance (accepted)</div>
          <div className="big">
            {progress.accepted_by_subset.lvis} <span className="muted" style={{ fontSize: 14 }}>LVIS</span> ·{" "}
            {progress.accepted_by_subset.laion} <span className="muted" style={{ fontSize: 14 }}>LAION</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="label">Queue position</div>
          <div className="big">{progress.queue_cursor}<span className="muted" style={{ fontSize: 14 }}>/{progress.queue_total}</span></div>
          <div className="small-note">reviewer: {progress.reviewer_id ?? "—"}</div>
        </div>
      </div>

      <hr className="soft" />
      <h3>Category progress (primary category of accepted scenes; {progress.target_per_category} each)</h3>
      {CATEGORIES.map((cat) => (
        <Bar key={cat} name={CATEGORY_LABELS[cat]} value={progress.category_counts[cat] ?? 0} max={progress.target_per_category} />
      ))}

      <hr className="soft" />
      <div className="stat-grid">
        <div className="stat-card">
          <div className="label">Caption edit rate</div>
          <div className="big">{pct(dashboard.caption_edit_rate)}</div>
          <div className="small-note">reviewed caption differs from the source caption</div>
        </div>
        <div className="stat-card">
          <div className="label">Missing-label completion</div>
          <div className="big">{pct(dashboard.missing_label_completion_rate)}</div>
          <div className="small-note">retained layers with a human canonical label (must be 100%)</div>
        </div>
        <div className="stat-card">
          <div className="label">Geometry correction rate</div>
          <div className="big">{pct(dashboard.geometry_correction_rate)}</div>
          <div className="small-note">retained layers with corrections/overrides</div>
        </div>
        <div className="stat-card">
          <div className="label">Mean review duration</div>
          <div className="big">
            {dashboard.mean_review_duration_seconds == null ? "—" : `${Math.round(dashboard.mean_review_duration_seconds)}s`}
          </div>
        </div>
      </div>

      <hr className="soft" />
      <h3>Quality distribution (accepted scenes)</h3>
      {Object.entries(dashboard.quality_distribution).map(([stars, count]) => (
        <Bar key={stars} name={`${stars} star${stars === "1" ? "" : "s"}`} value={count} max={Math.max(1, progress.accepted)} />
      ))}

      <hr className="soft" />
      <h3>Layer exclusion reasons</h3>
      {Object.keys(dashboard.exclusion_reasons).length === 0 && <p className="small-note">No exclusions yet.</p>}
      {Object.entries(dashboard.exclusion_reasons)
        .sort((a, b) => b[1] - a[1])
        .map(([reason, count]) => (
          <Bar key={reason} name={reason} value={count} max={Math.max(...Object.values(dashboard.exclusion_reasons), 1)} />
        ))}
    </div>
  );
}
