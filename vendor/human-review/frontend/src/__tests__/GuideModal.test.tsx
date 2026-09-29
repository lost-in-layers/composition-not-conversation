import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GuideModal } from "../components/GuideModal";

function renderGuide(section: Parameters<typeof GuideModal>[0]["section"] = "workflow") {
  const onClose = vi.fn();
  const onPreference = vi.fn();
  render(<GuideModal section={section} doNotAutoshow={false} onPreference={onPreference} onClose={onClose} />);
  return { onClose, onPreference };
}

describe("GuideModal", () => {
  it("renders every required section", () => {
    renderGuide();
    expect(screen.getByRole("dialog", { name: "Review Guide" })).toBeInTheDocument();
    expect(screen.getByText("Complete workflow")).toBeInTheDocument();
    expect(screen.getByText("Layer-quality definitions")).toBeInTheDocument();
    expect(screen.getByText("Caption and label definitions")).toBeInTheDocument();
    expect(screen.getByText("Geometry definitions")).toBeInTheDocument();
    expect(screen.getByText("Category definitions")).toBeInTheDocument();
    expect(screen.getByText("Final acceptance checklist")).toBeInTheDocument();
    // Key content spot-checks.
    expect(screen.getByText(/Z-index is ordinal stack order, not metric depth/)).toBeInTheDocument();
    expect(screen.getByText(/What color is the object left of the bicycle\?/)).toBeInTheDocument();
  });

  it("closes with the Close button, the X button, and Escape", async () => {
    const user = userEvent.setup();
    let handles = renderGuide();
    await user.click(screen.getByText("Close", { selector: "button" }));
    expect(handles.onClose).toHaveBeenCalledTimes(1);
    cleanup();

    handles = renderGuide();
    await user.click(screen.getByText("✕", { selector: "button" }));
    expect(handles.onClose).toHaveBeenCalledTimes(1);
    cleanup();

    handles = renderGuide();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(handles.onClose).toHaveBeenCalledTimes(1);
  });

  it("deep-links to the requested section", () => {
    const spy = vi.spyOn(Element.prototype, "scrollIntoView");
    renderGuide("geometry");
    expect(spy).toHaveBeenCalled();
    const target = spy.mock.instances[0] as unknown as Element;
    expect(target.getAttribute("data-guide-section")).toBe("geometry");
    spy.mockRestore();
  });

  it("saves the do-not-autoshow preference", async () => {
    const user = userEvent.setup();
    const { onPreference } = renderGuide();
    await user.click(screen.getByLabelText("Do not automatically show again"));
    expect(onPreference).toHaveBeenCalledWith(true);
  });
});
