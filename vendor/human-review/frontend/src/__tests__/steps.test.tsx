import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { StepBar } from "../components/StepBar";
import { Step1LayerQuality } from "../components/steps/Step1LayerQuality";
import { Step2Captions } from "../components/steps/Step2Captions";
import { Step3Geometry } from "../components/steps/Step3Geometry";
import { Step4Categories } from "../components/steps/Step4Categories";
import { Step5Final } from "../components/steps/Step5Final";
import { allStepStatuses, hydrateDraft } from "../draftUtils";
import type { Progress, ReviewDraft, ScenePayload } from "../types";
import { completeDraft, fixturePayload } from "./fixtures";

/** Harness that owns draft state like App does. */
function Harness(props: {
  payload: ScenePayload;
  initial?: ReviewDraft;
  children: (draft: ReviewDraft, updateDraft: (u: (d: ReviewDraft) => ReviewDraft) => void) => React.ReactNode;
}) {
  const [draft, setDraft] = useState<ReviewDraft>(props.initial ?? hydrateDraft(props.payload));
  const updateDraft = (updater: (d: ReviewDraft) => ReviewDraft) => setDraft((d) => updater(d));
  return <>{props.children(draft, updateDraft)}</>;
}

const noop = () => {};

function fakeProgress(overrides: Partial<Progress> = {}): Progress {
  return {
    target_total: 600,
    target_per_category: 100,
    reviewed: 4,
    accepted: 3,
    rejected: 1,
    deferred: 0,
    drafts: 1,
    finalized: 4,
    remaining_to_target: 596,
    queue_total: 600,
    queue_cursor: 5,
    category_counts: {
      direct_ask: 3,
      attribute_binding: 0,
      spatial: 0,
      occlusion: 0,
      counting: 0,
      multi_hop_reference: 0,
    },
    category_quota_full: [],
    accepted_by_subset: { lvis: 2, laion: 1 },
    reviewer_id: "tester",
    ...overrides,
  };
}

describe("StepBar", () => {
  it("navigates between steps and shows completion", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    const statuses = allStepStatuses(payload.scene, draft, payload.category_suggestions);
    const onStep = vi.fn();
    render(<StepBar step={2} onStep={onStep} statuses={statuses} />);
    await user.click(screen.getByRole("tab", { name: /Geometry/ }));
    expect(onStep).toHaveBeenCalledWith(3);
    // Completed steps show a check.
    expect(within(screen.getByRole("tab", { name: /Layer Quality/ })).getByText("✓")).toBeInTheDocument();
  });
});

describe("Step 1 — layer quality", () => {
  it("records decisions and requires an exclusion reason", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    render(
      <Harness payload={payload}>
        {(draft, updateDraft) => (
          <Step1LayerQuality
            payload={payload}
            draft={draft}
            updateDraft={updateDraft}
            readOnly={false}
            selectedLayerId={1}
            onSelectLayer={noop}
            openGuide={noop}
            onRejectNow={noop}
          />
        )}
      </Harness>,
    );
    const layer2 = screen.getByTestId("step1-layer-2");
    await user.click(within(layer2).getByRole("button", { name: "Exclude" }));
    const reason = within(layer2).getByLabelText("Exclusion reason for layer 2");
    expect(reason).toBeInTheDocument();
    await user.selectOptions(reason, "duplicate");
    expect((reason as HTMLSelectElement).value).toBe("duplicate");

    const layer1 = screen.getByTestId("step1-layer-1");
    await user.click(within(layer1).getByRole("button", { name: "Keep" }));
    expect(within(layer1).getByRole("button", { name: "Keep" })).toHaveClass("on-good");
  });

  it("offers immediate rejection", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    const onReject = vi.fn();
    render(
      <Harness payload={payload}>
        {(draft, updateDraft) => (
          <Step1LayerQuality
            payload={payload}
            draft={draft}
            updateDraft={updateDraft}
            readOnly={false}
            selectedLayerId={null}
            onSelectLayer={noop}
            openGuide={noop}
            onRejectNow={onReject}
          />
        )}
      </Harness>,
    );
    await user.click(screen.getByRole("button", { name: "Reject scene" }));
    expect(onReject).toHaveBeenCalled();
  });
});

describe("Step 2 — captions and labels", () => {
  it("edits source captions directly and skips excluded layers", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    const initial = hydrateDraft(payload);
    initial.layer_reviews["3"] = { ...initial.layer_reviews["3"], decision: "exclude", exclusion_reason: "duplicate" };
    initial.layer_reviews["1"].decision = "keep";
    render(
      <Harness payload={payload} initial={initial}>
        {(draft, updateDraft) => (
          <Step2Captions
            payload={payload}
            draft={draft}
            updateDraft={updateDraft}
            readOnly={false}
            onSelectLayer={noop}
            openGuide={noop}
          />
        )}
      </Harness>,
    );

    const label = screen.getByLabelText("Canonical label for layer 1");
    await user.type(label, "giraffe");
    expect(label).toHaveValue("giraffe");

    const blip2 = screen.getByLabelText("Editable BLIP-2 overall image caption");
    const llava = screen.getByLabelText("Editable LLaVA overall image caption");
    expect(blip2).toHaveValue("two giraffes standing by a tree");
    expect(llava).toHaveValue("A long llava caption.");
    await user.clear(blip2);
    await user.type(blip2, "a corrected image caption");
    expect(blip2).toHaveValue("a corrected image caption");
    expect(screen.queryByRole("button", { name: "Copy source" })).not.toBeInTheDocument();

    // Excluded layer shows no inputs.
    const layer3 = screen.getByTestId("step2-layer-3");
    expect(within(layer3).getByText(/excluded — no completion needed/)).toBeInTheDocument();
    expect(within(layer3).queryByLabelText("Canonical label for layer 3")).not.toBeInTheDocument();

    // Null-source badge shown for pending layers.
    expect(within(screen.getByTestId("step2-layer-1")).getByText(/label: null in source/)).toBeInTheDocument();
  });
});

describe("Step 3 — geometry", () => {
  it("validates bbox overrides and flags invalid values", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    const initial = completeDraft(payload);
    render(
      <Harness payload={payload} initial={initial}>
        {(draft, updateDraft) => (
          <Step3Geometry
            payload={payload}
            draft={draft}
            updateDraft={updateDraft}
            readOnly={false}
            selectedLayerId={1}
            onSelectLayer={noop}
            openGuide={noop}
          />
        )}
      </Harness>,
    );
    const x1 = screen.getByLabelText("x1 for layer 1");
    await user.clear(x1);
    await user.type(x1, "9999");
    expect(screen.getByTestId("bbox-invalid-1")).toBeInTheDocument();
  });

  it("marks relations invalid and restores them", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    render(
      <Harness payload={payload} initial={completeDraft(payload)}>
        {(draft, updateDraft) => (
          <Step3Geometry
            payload={payload}
            draft={draft}
            updateDraft={updateDraft}
            readOnly={false}
            selectedLayerId={1}
            onSelectLayer={noop}
            openGuide={noop}
          />
        )}
      </Harness>,
    );
    const buttons = screen.getAllByRole("button", { name: "Mark invalid" });
    await user.click(buttons[0]);
    expect(screen.getByRole("button", { name: "Restore" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Restore" }));
    expect(screen.queryByRole("button", { name: "Restore" })).not.toBeInTheDocument();
  });
});

describe("Step 4 — categories", () => {
  it("enforces a single primary and keeps every category selectable", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    render(
      <Harness payload={payload} initial={completeDraft(payload)}>
        {(draft, updateDraft) => (
          <Step4Categories
            payload={payload}
            draft={draft}
            updateDraft={updateDraft}
            readOnly={false}
            progress={fakeProgress()}
            openGuide={noop}
          />
        )}
      </Harness>,
    );

    // direct_ask is primary from the fixture; switch primary to spatial.
    await user.click(screen.getByRole("button", { name: "Set Spatial as primary" }));
    const spatialCard = screen.getByTestId("category-card-spatial");
    expect(within(spatialCard).getByText("PRIMARY")).toBeInTheDocument();
    const directCard = screen.getByTestId("category-card-direct_ask");
    expect(within(directCard).queryByText("PRIMARY")).not.toBeInTheDocument();

    // Automatic hints never disable a category.
    const multiHop = screen.getByTestId("category-card-multi_hop_reference");
    const multiHopPrimary = within(multiHop).getByRole("button", { name: /Set Multi-hop Reference as primary/ });
    expect(multiHopPrimary).toBeEnabled();
    await user.click(multiHopPrimary);
    expect(within(multiHop).getByText("PRIMARY")).toBeInTheDocument();
  });

  it("warns when the primary quota is full", () => {
    const payload = fixturePayload();
    const progress = fakeProgress({
      category_counts: { ...fakeProgress().category_counts, direct_ask: 100 },
      category_quota_full: ["direct_ask"],
    });
    render(
      <Harness payload={payload} initial={completeDraft(payload)}>
        {(draft, updateDraft) => (
          <Step4Categories payload={payload} draft={draft} updateDraft={updateDraft} readOnly={false} progress={progress} openGuide={noop} />
        )}
      </Harness>,
    );
    expect(screen.getByTestId("quota-warning")).toBeInTheDocument();
  });
});

describe("Step 5 — final decision", () => {
  it("disables Accept until requirements are met and shows problems", () => {
    const payload = fixturePayload();
    const incomplete = hydrateDraft(payload);
    render(
      <Harness payload={payload} initial={incomplete}>
        {(draft, updateDraft) => (
          <Step5Final
            payload={payload}
            draft={draft}
            updateDraft={updateDraft}
            readOnly={false}
            problems={["Layer 1: canonical short label is required for retained layers."]}
            stepStatuses={allStepStatuses(payload.scene, draft, payload.category_suggestions)}
            openGuide={noop}
            onAccept={noop}
          />
        )}
      </Harness>,
    );
    expect(screen.getByTestId("accept-button")).toBeDisabled();
    expect(screen.getByTestId("finalize-problems")).toHaveTextContent("canonical short label");
  });

  it("enables Accept for a complete draft and triggers callbacks", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    const onAccept = vi.fn();
    render(
      <Harness payload={payload} initial={draft}>
        {(d, updateDraft) => (
          <Step5Final
            payload={payload}
            draft={d}
            updateDraft={updateDraft}
            readOnly={false}
            problems={[]}
            stepStatuses={allStepStatuses(payload.scene, d, payload.category_suggestions)}
            openGuide={noop}
            onAccept={onAccept}
          />
        )}
      </Harness>,
    );
    const accept = screen.getByTestId("accept-button");
    expect(accept).toBeEnabled();
    await user.click(accept);
    expect(onAccept).toHaveBeenCalled();
    expect(screen.queryByTestId("reject-button")).not.toBeInTheDocument();
  });

  it("shows one question-answer area per selected category and requires one complete pair", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    draft.suitable_categories = ["direct_ask", "spatial"];
    draft.human_questions = {};
    render(
      <Harness payload={payload} initial={draft}>
        {(d, updateDraft) => (
          <Step5Final
            payload={payload}
            draft={d}
            updateDraft={updateDraft}
            readOnly={false}
            problems={[]}
            stepStatuses={allStepStatuses(payload.scene, d, payload.category_suggestions)}
            openGuide={noop}
            onAccept={noop}
          />
        )}
      </Harness>,
    );

    expect(screen.getByLabelText("Direct Ask human question")).toBeInTheDocument();
    expect(screen.getByLabelText("Spatial human question")).toBeInTheDocument();
    expect(screen.queryByLabelText("Counting human question")).not.toBeInTheDocument();
    expect(screen.getByTestId("accept-button")).toBeDisabled();
    await user.type(screen.getByLabelText("Spatial human question"), "Which object is above the table?");
    expect(screen.getByTestId("accept-button")).toBeDisabled();
    await user.type(screen.getByLabelText("Spatial human answer"), "The lamp.");
    expect(screen.getByTestId("accept-button")).toBeEnabled();
  });
});
