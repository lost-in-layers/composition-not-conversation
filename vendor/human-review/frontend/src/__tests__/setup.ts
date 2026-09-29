import "@testing-library/jest-dom";

// jsdom lacks a few browser APIs the app touches.
Element.prototype.scrollIntoView = Element.prototype.scrollIntoView ?? (() => {});
window.HTMLElement.prototype.setPointerCapture = window.HTMLElement.prototype.setPointerCapture ?? (() => {});

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver ?? (MockResizeObserver as unknown as typeof ResizeObserver);
