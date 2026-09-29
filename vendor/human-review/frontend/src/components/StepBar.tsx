import type { StepStatus } from "../draftUtils";

const STEP_NAMES = ["Layer Quality", "Captions & Labels", "Geometry", "Categories", "Final Decision"];

export function StepBar(props: {
  step: number;
  onStep: (step: number) => void;
  statuses: StepStatus[];
}) {
  return (
    <div className="step-bar" role="tablist" aria-label="Review steps">
      {STEP_NAMES.map((name, index) => {
        const number = index + 1;
        const status = props.statuses[index];
        const complete = number <= 4 && status?.complete;
        return (
          <span key={name} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            {index > 0 && <span className="arrow">›</span>}
            <button
              className={`step ghost ${props.step === number ? "active" : ""} ${complete ? "complete" : ""}`}
              role="tab"
              aria-selected={props.step === number}
              onClick={() => props.onStep(number)}
            >
              <span className="num">{complete ? "✓" : number}</span>
              {name}
            </button>
          </span>
        );
      })}
    </div>
  );
}
