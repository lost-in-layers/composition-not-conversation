import { describe, expect, it } from "vitest";
import {
  acceptanceProblems,
  clampBbox,
  hydrateDraft,
  retainedLayerIds,
  setPrimary,
  step1Status,
  step2Status,
  step3Status,
  step4Status,
  toggleSuitable,
  unresolvedBlocking,
  validBbox,
} from "../draftUtils";
import { completeDraft, fixturePayload } from "./fixtures";

describe("hydrateDraft", () => {
  it("creates layer review slots for every layer", () => {
    const payload = fixturePayload();
    const draft = hydrateDraft(payload);
    expect(Object.keys(draft.layer_reviews)).toEqual(["1", "2", "3"]);
    expect(draft.layer_reviews["1"].decision).toBeNull();
  });

  it("keeps stored draft values", () => {
    const payload = fixturePayload();
    payload.draft = { ...hydrateDraft(payload), overall_quality: 5, reviewed_image_caption: "kept" };
    const draft = hydrateDraft(payload);
    expect(draft.overall_quality).toBe(5);
    expect(draft.reviewed_image_caption).toBe("kept");
  });

  it("restores source captions when an older draft saved null caption fields", () => {
    const payload = fixturePayload();
    const oldDraft = hydrateDraft(payload);
    oldDraft.reviewed_image_caption = null;
    oldDraft.reviewed_image_caption_llava = null;
    oldDraft.reviewed_background_caption = null;
    oldDraft.reviewed_background_caption_llava = null;
    oldDraft.layer_reviews["1"].caption = null;
    payload.draft = oldDraft;

    const draft = hydrateDraft(payload);
    expect(draft.reviewed_image_caption).toBe("two giraffes standing by a tree");
    expect(draft.reviewed_image_caption_llava).toBe("A long llava caption.");
    expect(draft.reviewed_background_caption).toBe("a fallen tree");
    expect(draft.reviewed_background_caption_llava).toBe("A forest scene.");
    expect(draft.layer_reviews["1"].caption).toBe("single giraffe standing");
  });
});

describe("bbox validation", () => {
  it("accepts a valid bbox and rejects bad ones", () => {
    expect(validBbox([0, 0, 10, 10], 100, 100)).toBe(true);
    expect(validBbox([10, 0, 10, 10], 100, 100)).toBe(false); // x1 == x2
    expect(validBbox([-1, 0, 10, 10], 100, 100)).toBe(false);
    expect(validBbox([0, 0, 101, 10], 100, 100)).toBe(false);
    expect(validBbox(null, 100, 100)).toBe(true); // no override is fine
  });

  it("clamps into bounds preserving ordering", () => {
    expect(clampBbox([-5, -5, 500, 500], 100, 80)).toEqual([0, 0, 100, 80]);
    const [x1, , x2] = clampBbox([99, 0, 98, 50], 100, 80);
    expect(x1).toBeLessThan(x2);
  });
});

describe("step statuses", () => {
  it("step1 requires quality and per-layer decisions", () => {
    const payload = fixturePayload();
    const draft = hydrateDraft(payload);
    const status = step1Status(payload.scene, draft);
    expect(status.complete).toBe(false);
    expect(status.problems[0]).toContain("Rate the overall");
    expect(status.problems).toHaveLength(4); // quality + 3 undecided layers
  });

  it("step1 flags exclusion without reason", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    draft.layer_reviews["2"] = { ...draft.layer_reviews["2"], decision: "exclude", exclusion_reason: null };
    const status = step1Status(payload.scene, draft);
    expect(status.problems.some((p) => p.includes("exclusion reason"))).toBe(true);
  });

  it("step2 requires labels and captions only for retained layers", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    draft.layer_reviews["2"] = {
      ...draft.layer_reviews["2"],
      decision: "exclude",
      exclusion_reason: "duplicate",
      label: null,
      caption: null,
    };
    expect(step2Status(payload.scene, draft).complete).toBe(true);

    draft.layer_reviews["1"].label = "";
    const status = step2Status(payload.scene, draft);
    expect(status.complete).toBe(false);
    expect(status.problems[0]).toContain("Layer 1");
  });

  it("step3 blocks on major problems and invalid overrides", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    draft.layer_reviews["1"].geometry_status = "major_problem";
    expect(step3Status(payload.scene, draft).complete).toBe(false);

    draft.layer_reviews["1"].geometry_status = "correct";
    draft.layer_reviews["1"].bbox_override = [50, 50, 10, 60];
    expect(step3Status(payload.scene, draft).complete).toBe(false);
  });

  it("step4 enforces one primary inside suitable without blocking categories", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    expect(step4Status(payload.scene, draft, payload.category_suggestions).complete).toBe(true);

    draft.primary_category = "multi_hop_reference";
    draft.suitable_categories = ["multi_hop_reference"];
    const status = step4Status(payload.scene, draft, payload.category_suggestions);
    expect(status.complete).toBe(true);

    draft.primary_category = null;
    draft.suitable_categories = [];
    draft.category_decision_made = false;
    const empty = step4Status(payload.scene, draft, payload.category_suggestions);
    expect(empty.problems.some((p) => p.includes("primary"))).toBe(true);
  });
});

describe("category selection helpers", () => {
  it("toggling off the primary's suitability clears primary", () => {
    const payload = fixturePayload();
    let draft = completeDraft(payload);
    draft = toggleSuitable(draft, "direct_ask");
    expect(draft.suitable_categories).not.toContain("direct_ask");
    expect(draft.primary_category).toBeNull();
  });

  it("setting a primary auto-marks it suitable", () => {
    const payload = fixturePayload();
    let draft = hydrateDraft(payload);
    draft = setPrimary(draft, "spatial");
    expect(draft.primary_category).toBe("spatial");
    expect(draft.suitable_categories).toContain("spatial");
  });
});

describe("acceptance", () => {
  it("complete draft has no problems", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    expect(acceptanceProblems(payload.scene, draft, payload.category_suggestions, [])).toEqual([]);
  });

  it("requires at least one question among the selected categories", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    draft.suitable_categories = ["direct_ask", "spatial"];
    draft.human_questions = { counting: "How many objects are visible?" };
    expect(acceptanceProblems(payload.scene, draft, payload.category_suggestions, []))
      .toContain("Write at least one question for a selected category.");

    draft.human_questions.spatial = "Which object is above the table?";
    draft.human_answers.spatial = "The lamp.";
    expect(acceptanceProblems(payload.scene, draft, payload.category_suggestions, [])).toEqual([]);
  });

  it("excluding all layers is unacceptable", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    for (const key of Object.keys(draft.layer_reviews)) {
      draft.layer_reviews[key] = { ...draft.layer_reviews[key], decision: "exclude", exclusion_reason: "duplicate" };
    }
    const problems = acceptanceProblems(payload.scene, draft, payload.category_suggestions, []);
    expect(problems.some((p) => p.includes("At least one retained"))).toBe(true);
    expect(retainedLayerIds(payload.scene, draft)).toEqual([]);
  });

  it("blocking findings resolve via exclusion or reviewer captions", () => {
    const payload = fixturePayload();
    payload.missing_fields.blocking = [
      { field: "captions", detail: "Layer has no source caption.", layer_id: 2 },
    ];
    payload.scene.layers[1].captions = {};
    const draft = hydrateDraft(payload);
    expect(unresolvedBlocking(payload, draft)).toHaveLength(1);
    draft.layer_reviews["2"].caption = "written by human";
    expect(unresolvedBlocking(payload, draft)).toHaveLength(0);
    draft.layer_reviews["2"].caption = null;
    draft.layer_reviews["2"].decision = "exclude";
    expect(unresolvedBlocking(payload, draft)).toHaveLength(0);
  });
});
