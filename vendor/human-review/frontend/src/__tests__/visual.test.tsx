import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Header } from "../components/Header";
import { VisualPane } from "../components/VisualPane";
import { hydrateDraft } from "../draftUtils";
import { completeDraft, fixturePayload } from "./fixtures";

const noop = () => {};

describe("VisualPane layer rail", () => {
  it("renders thumbnails with z-index, badges and review status", () => {
    const payload = fixturePayload();
    const draft = hydrateDraft(payload);
    draft.layer_reviews["1"] = { ...draft.layer_reviews["1"], decision: "keep", label: "giraffe", caption: "a giraffe", geometry_status: "correct" };
    draft.layer_reviews["2"] = { ...draft.layer_reviews["2"], decision: "exclude", exclusion_reason: "duplicate" };
    render(
      <VisualPane
        payload={payload}
        draft={draft}
        selectedLayerId={1}
        onSelectLayer={noop}
        viewMode="composite"
        onViewMode={noop}
        editableGeometry={false}
        onBboxChange={noop}
        onCentroidChange={noop}
      />,
    );
    const card1 = screen.getByTestId("layer-card-1");
    expect(within(card1).getByText("z1")).toBeInTheDocument();
    expect(within(card1).getByText("complete")).toBeInTheDocument();
    expect(within(card1).getByText("giraffe")).toBeInTheDocument();
    expect(within(card1).getByText("tech-null")).toBeInTheDocument();

    const card2 = screen.getByTestId("layer-card-2");
    expect(within(card2).getByText("excluded")).toBeInTheDocument();
    expect(card2.className).toContain("excluded");

    const card3 = screen.getByTestId("layer-card-3");
    expect(within(card3).getByText("unreviewed")).toBeInTheDocument();
    expect(within(card3).getByText(/missing/)).toBeInTheDocument();
  });

  it("shows a placeholder when the image file is missing", () => {
    const payload = fixturePayload();
    payload.file_checks = { ...payload.file_checks, full: false };
    render(
      <VisualPane
        payload={payload}
        draft={hydrateDraft(payload)}
        selectedLayerId={null}
        onSelectLayer={noop}
        viewMode="composite"
        onViewMode={noop}
        editableGeometry={false}
        onBboxChange={noop}
        onCentroidChange={noop}
      />,
    );
    expect(screen.getByTestId("missing-image")).toHaveTextContent("full.png is missing");
  });

  it("renders the geometry overlay with per-layer boxes", () => {
    const payload = fixturePayload();
    render(
      <VisualPane
        payload={payload}
        draft={completeDraft(payload)}
        selectedLayerId={1}
        onSelectLayer={noop}
        viewMode="composite"
        onViewMode={noop}
        editableGeometry={false}
        onBboxChange={noop}
        onCentroidChange={noop}
      />,
    );
    const overlay = screen.getByTestId("geometry-overlay");
    expect(overlay.querySelectorAll("rect.bbox")).toHaveLength(3);
    expect(overlay).toHaveTextContent("L1 z1");
  });
});

describe("Header", () => {
  it("shows progress and save status, and submits jumps", async () => {
    const user = userEvent.setup();
    const payload = fixturePayload();
    const onJump = vi.fn();
    render(
      <Header
        progress={{
          target_total: 600,
          target_per_category: 100,
          reviewed: 49,
          accepted: 42,
          rejected: 7,
          deferred: 3,
          drafts: 2,
          finalized: 49,
          remaining_to_target: 551,
          queue_total: 600,
          queue_cursor: 60,
          category_counts: { direct_ask: 10, attribute_binding: 5, spatial: 9, occlusion: 6, counting: 7, multi_hop_reference: 5 },
          category_quota_full: [],
          accepted_by_subset: { lvis: 20, laion: 22 },
          reviewer_id: "reviewer_1",
          reviewer_slot: "r1",
        }}
        payload={payload}
        saveStatus="saved"
        reviewerId="reviewer_1"
        onSwitchReviewer={noop}
        view="review"
        onToggleView={noop}
        onOpenGuide={noop}
        onJump={onJump}
        onResume={noop}
        onUndo={noop}
        onBackup={noop}
        onExport={noop}
        onKeyboardHelp={noop}
      />,
    );
    expect(screen.getByText("49")).toBeInTheDocument();
    expect(screen.getByTestId("save-status")).toHaveTextContent("Draft saved");
    const input = screen.getByLabelText("Jump to scene ID");
    await user.type(input, "lvis_000000000099{Enter}");
    expect(onJump).toHaveBeenCalledWith("lvis_000000000099");
  });
});
