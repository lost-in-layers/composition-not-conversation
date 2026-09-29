import { describe, expect, it } from "vitest";
import { buildSceneTemplate } from "../components/ScenePromptTemplate";
import { completeDraft, fixturePayload } from "./fixtures";

describe("scene prompt template", () => {
  it("uses reviewed captions, retained labels, and two-decimal relation scores", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    const text = buildSceneTemplate(payload.scene, draft);

    expect(text).toContain('img: "a reviewed image caption" and "a reviewed LLaVA image caption"');
    expect(text).toContain('bg: "a reviewed background caption" and "a reviewed LLaVA background caption"');
    expect(text).toContain('L1 - "object 1": "reviewed caption 1"');
    expect(text).toContain("L1 left_of L2 (0.79)");
    expect(text).toContain("L3 possible_occludes L2 (0.05)");
  });

  it("omits excluded layers and removed relations while including added relations", () => {
    const payload = fixturePayload();
    const draft = completeDraft(payload);
    draft.layer_reviews["3"].decision = "exclude";
    draft.relations_removed = [[1, "left_of", 2]];
    draft.relations_added = [{ subject_layer_id: 2, predicate: "above", object_layer_id: 1 }];
    const text = buildSceneTemplate(payload.scene, draft);

    expect(text).not.toContain('L3 - "object 3"');
    expect(text).not.toContain("L1 left_of L2");
    expect(text).not.toContain("possible_occludes");
    expect(text).toContain("L2 above L1");
  });
});
