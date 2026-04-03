/**
 * Unit tests for pure-math helpers exported from useTelemetry.ts
 *
 * These run without any React environment — no DOM, no hooks, no mocks.
 */
import { describe, it, expect } from "vitest";
import {
  computeMouseStats,
  selectCurrentParagraph,
  computeWindowIdle,
  checkTelemetrySanity,
  findCurrentParagraphFromDOM,
  type Point,
  type IntersectionEntry,
  type UiContext,
  type InteractionZone,
} from "../hooks/useTelemetry";

// ─── Signed scroll sum helpers (inline, mirrors hook accumulator logic) ────────

/**
 * Simulate the hook's accumulator for a sequence of deltaY values.
 * Returns { posSum, negSum, absSum, netSum }.
 */
function computeSignedScrollSums(deltas: number[]) {
  let posSum = 0;
  let negSum = 0;
  let absSum = 0;
  let netSum = 0;
  for (const dy of deltas) {
    netSum += dy;
    absSum += Math.abs(dy);
    if (dy > 0) posSum += dy;
    else if (dy < 0) negSum += Math.abs(dy);
  }
  return { posSum, negSum, absSum, netSum };
}

/**
 * Compute viewport-normalised scroll velocity (mirrors baseline.py logic).
 * vel_norm = scroll_delta_abs_sum / (viewport_height_px * window_seconds)
 */
function normalizedScrollVelocity(
  scrollDeltaAbsSum: number,
  viewportHeightPx: number,
  windowSeconds: number = 2.0,
): number {
  if (viewportHeightPx <= 0) return 0;
  return scrollDeltaAbsSum / (viewportHeightPx * windowSeconds);
}

describe("computeSignedScrollSums", () => {
  it("all forward scrolling: posSum = absSum, negSum = 0", () => {
    const { posSum, negSum, absSum } = computeSignedScrollSums([100, 200, 50]);
    expect(posSum).toBeCloseTo(350);
    expect(negSum).toBe(0);
    expect(absSum).toBeCloseTo(350);
  });

  it("all backward scrolling: negSum = absSum, posSum = 0", () => {
    const { posSum, negSum, absSum } = computeSignedScrollSums([-80, -120]);
    expect(posSum).toBe(0);
    expect(negSum).toBeCloseTo(200);
    expect(absSum).toBeCloseTo(200);
  });

  it("mixed scrolling: posSum + negSum = absSum", () => {
    const { posSum, negSum, absSum } = computeSignedScrollSums([300, -100, 200, -50]);
    expect(posSum).toBeCloseTo(500);      // 300 + 200
    expect(negSum).toBeCloseTo(150);      // 100 + 50
    expect(absSum).toBeCloseTo(posSum + negSum);
  });

  it("zero deltas are ignored in both sums", () => {
    const { posSum, negSum } = computeSignedScrollSums([0, 0, 100]);
    expect(posSum).toBeCloseTo(100);
    expect(negSum).toBe(0);
  });

  it("regress_rate = negSum / (negSum + posSum)", () => {
    const { posSum, negSum } = computeSignedScrollSums([300, -100]);
    const regressRate = negSum / (negSum + posSum);
    // 100 / (100 + 300) = 0.25
    expect(regressRate).toBeCloseTo(0.25);
  });

  it("net sum = posSum - negSum", () => {
    const { posSum, negSum, netSum } = computeSignedScrollSums([200, -80]);
    expect(netSum).toBeCloseTo(posSum - negSum);
  });
});

// ─── computeMouseStats ────────────────────────────────────────────────────────

describe("computeMouseStats", () => {
  it("returns zeros for empty array", () => {
    expect(computeMouseStats([])).toEqual({ pathPx: 0, netPx: 0 });
  });

  it("returns zeros for single point", () => {
    expect(computeMouseStats([{ x: 10, y: 20 }])).toEqual({ pathPx: 0, netPx: 0 });
  });

  it("computes correct path for two points (horizontal move)", () => {
    const points: Point[] = [{ x: 0, y: 0 }, { x: 30, y: 0 }];
    const { pathPx, netPx } = computeMouseStats(points);
    expect(pathPx).toBeCloseTo(30);
    expect(netPx).toBeCloseTo(30);
  });

  it("computes correct path for two points (diagonal move)", () => {
    // 3-4-5 right triangle
    const points: Point[] = [{ x: 0, y: 0 }, { x: 3, y: 4 }];
    const { pathPx, netPx } = computeMouseStats(points);
    expect(pathPx).toBeCloseTo(5);
    expect(netPx).toBeCloseTo(5);
  });

  it("pathPx > netPx for a zig-zag trajectory", () => {
    // Mouse went right, then down, then left — zig-zag increases path
    const points: Point[] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },   // right 100
      { x: 100, y: 100 }, // down 100
      { x: 0, y: 100 },   // left 100
    ];
    const { pathPx, netPx } = computeMouseStats(points);
    // pathPx = 100 + 100 + 100 = 300
    expect(pathPx).toBeCloseTo(300);
    // netPx = distance from (0,0) to (0,100) = 100
    expect(netPx).toBeCloseTo(100);
    expect(pathPx).toBeGreaterThan(netPx);
  });

  it("pathPx === netPx for a straight-line trajectory", () => {
    const points: Point[] = [
      { x: 0, y: 0 },
      { x: 50, y: 0 },
      { x: 100, y: 0 },
    ];
    const { pathPx, netPx } = computeMouseStats(points);
    expect(pathPx).toBeCloseTo(100);
    expect(netPx).toBeCloseTo(100);
  });

  it("handles negative coordinates", () => {
    const points: Point[] = [{ x: -10, y: -10 }, { x: 10, y: 10 }];
    const { pathPx, netPx } = computeMouseStats(points);
    // distance = sqrt(20^2 + 20^2) = sqrt(800) ≈ 28.28
    expect(pathPx).toBeCloseTo(Math.sqrt(800));
    expect(netPx).toBeCloseTo(Math.sqrt(800));
  });
});

// ─── selectCurrentParagraph ──────────────────────────────────────────────────

describe("selectCurrentParagraph", () => {
  it("returns null for empty array", () => {
    const { paragraphId } = selectCurrentParagraph([]);
    expect(paragraphId).toBeNull();
  });

  it("falls back to highest-ratio when all below 0.6", () => {
    // A3 fix: no longer returns null — picks the highest-ratio element
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-1", chunkIndex: 0, ratio: 0.3, topOffset: 100 },
      { paragraphId: "chunk-2", chunkIndex: 1, ratio: 0.59, topOffset: 50 },
    ];
    const { paragraphId } = selectCurrentParagraph(entries);
    expect(paragraphId).toBe("chunk-2"); // highest ratio
  });

  it("returns the single element when only one meets threshold", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-3", chunkIndex: 2, ratio: 0.1, topOffset: 200 },
      { paragraphId: "chunk-4", chunkIndex: 3, ratio: 0.75, topOffset: 10 },
    ];
    const result = selectCurrentParagraph(entries);
    expect(result.paragraphId).toBe("chunk-4");
    expect(result.chunkIndex).toBe(3);
  });

  it("returns the element with highest ratio when multiple meet threshold", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-5", chunkIndex: 4, ratio: 0.65, topOffset: 300 },
      { paragraphId: "chunk-6", chunkIndex: 5, ratio: 0.9, topOffset: 100 },
      { paragraphId: "chunk-7", chunkIndex: 6, ratio: 0.72, topOffset: 200 },
    ];
    const result = selectCurrentParagraph(entries);
    expect(result.paragraphId).toBe("chunk-6");
    expect(result.chunkIndex).toBe(5);
  });

  it("treats exactly 0.6 as meeting the threshold", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-8", chunkIndex: 7, ratio: 0.6, topOffset: 0 },
    ];
    const result = selectCurrentParagraph(entries);
    expect(result.paragraphId).toBe("chunk-8");
  });

  it("preserves null chunkIndex when it is null", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-9", chunkIndex: null, ratio: 0.8, topOffset: 0 },
    ];
    const result = selectCurrentParagraph(entries);
    expect(result.paragraphId).toBe("chunk-9");
    expect(result.chunkIndex).toBeNull();
  });

  it("falls back to closest-to-top when all ratios are 0", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-10", chunkIndex: 0, ratio: 0, topOffset: 300 },
      { paragraphId: "chunk-11", chunkIndex: 1, ratio: 0, topOffset: 10 },
    ];
    const { paragraphId } = selectCurrentParagraph(entries);
    expect(paragraphId).toBe("chunk-11"); // closest to top
  });
});

// ─── A1: computeWindowIdle ────────────────────────────────────────────────────

describe("computeWindowIdle (A1 fix)", () => {
  const WINDOW_S = 2.0;

  it("returns WINDOW_S when last interaction was before the window started", () => {
    const now = 10_000;
    const windowStart = 8_000;
    const lastInteraction = 7_500; // before window start
    expect(computeWindowIdle(windowStart, lastInteraction, now, WINDOW_S)).toBeCloseTo(WINDOW_S);
  });

  it("returns 0 when interaction happened at the very end of window", () => {
    const now = 10_000;
    const windowStart = 8_000;
    const lastInteraction = 9_990; // just before now
    const idle = computeWindowIdle(windowStart, lastInteraction, now, WINDOW_S);
    expect(idle).toBeCloseTo(0.01, 1);
  });

  it("never exceeds WINDOW_S", () => {
    const now = 100_000;
    const windowStart = 0;
    const lastInteraction = 0; // ancient
    const idle = computeWindowIdle(windowStart, lastInteraction, now, WINDOW_S);
    expect(idle).toBeLessThanOrEqual(WINDOW_S);
  });

  it("is always >= 0", () => {
    const now = 10_000;
    const windowStart = 9_000;
    const lastInteraction = 10_500; // future (shouldn't happen, but guard)
    const idle = computeWindowIdle(windowStart, lastInteraction, now, WINDOW_S);
    expect(idle).toBeGreaterThanOrEqual(0);
  });

  it("partial idle when user interacted mid-window", () => {
    // Window is 2s. Interaction happened 1s ago = 1s idle in window
    const now = 10_000;
    const windowStart = 8_000;
    const lastInteraction = 9_000; // 1s ago = within window
    const idle = computeWindowIdle(windowStart, lastInteraction, now, WINDOW_S);
    expect(idle).toBeCloseTo(1.0, 1);
  });
});

// ─── A4: checkTelemetrySanity ────────────────────────────────────────────────

describe("checkTelemetrySanity (A4 fix)", () => {
  it("flags idleExceedsWindow when idle > 2.0", () => {
    const w = checkTelemetrySanity(2.1, 0, 0.0, 0.0, null);
    expect(w.idleExceedsWindow).toBe(true);
  });

  it("does not flag idleExceedsWindow when idle <= 2.0", () => {
    const w = checkTelemetrySanity(2.0, 100, 0.0, 0.0, "chunk-1");
    expect(w.idleExceedsWindow).toBe(false);
  });

  it("flags scrollZeroWithProgress when scroll=0 but progress changed > 0.05", () => {
    const w = checkTelemetrySanity(0.5, 0, 0.0, 0.1, "chunk-1", true);
    expect(w.scrollZeroWithProgress).toBe(true);
  });

  it("does not flag scrollZeroWithProgress when scroll events captured", () => {
    const w = checkTelemetrySanity(0.5, 100, 0.0, 0.1, "chunk-1", true);
    expect(w.scrollZeroWithProgress).toBe(false);
  });

  it("does not flag scrollZeroWithProgress when listener not yet ready (loading phase)", () => {
    // Even though scroll=0 and progress changed, no false positive during load
    const w = checkTelemetrySanity(0.5, 0, 0.0, 0.37, "chunk-1", false);
    expect(w.scrollZeroWithProgress).toBe(false);
  });

  it("defaults scrollListenerReady to true for backwards compatibility", () => {
    // Omitting the argument should behave the same as passing true
    const w = checkTelemetrySanity(0.5, 0, 0.0, 0.1, "chunk-1");
    expect(w.scrollZeroWithProgress).toBe(true);
  });

  it("flags paragraphMissing when paragraphId is null", () => {
    const w = checkTelemetrySanity(0.5, 100, 0.0, 0.1, null);
    expect(w.paragraphMissing).toBe(true);
  });

  it("does not flag paragraphMissing when paragraphId present", () => {
    const w = checkTelemetrySanity(0.5, 100, 0.0, 0.1, "chunk-5");
    expect(w.paragraphMissing).toBe(false);
  });
});

// ─── findCurrentParagraphFromDOM ─────────────────────────────────────────────
// These tests require a DOM environment (jsdom, provided by vitest).

function makeParagraphEl(
  paragraphId: string,
  chunkIndex: number,
  offsetTop: number,
  height: number,
): HTMLElement {
  const el = document.createElement("div");
  el.dataset.paragraphId = paragraphId;
  el.dataset.chunkIndex = String(chunkIndex);
  // jsdom doesn't compute layout, so we mock offsetTop via a getter
  Object.defineProperty(el, "offsetTop", { get: () => offsetTop });
  Object.defineProperty(el, "offsetHeight", { get: () => height });
  return el;
}

function makeContainer(
  scrollTop: number,
  clientHeight: number,
  children: HTMLElement[],
): HTMLElement {
  const container = document.createElement("div");
  Object.defineProperty(container, "scrollTop", { get: () => scrollTop });
  Object.defineProperty(container, "clientHeight", { get: () => clientHeight });
  children.forEach((c) => container.appendChild(c));
  return container;
}

describe("findCurrentParagraphFromDOM", () => {
  it("returns null when container has no paragraph elements", () => {
    const container = makeContainer(0, 600, []);
    const { paragraphId } = findCurrentParagraphFromDOM(container);
    expect(paragraphId).toBeNull();
  });

  it("returns the only element when there is one", () => {
    const el = makeParagraphEl("chunk-1", 0, 100, 80);
    const container = makeContainer(0, 600, [el]);
    const { paragraphId, chunkIndex } = findCurrentParagraphFromDOM(container);
    expect(paragraphId).toBe("chunk-1");
    expect(chunkIndex).toBe(0);
  });

  it("returns element closest to viewport centre", () => {
    // Viewport: scrollTop=0, clientHeight=600, centre=300
    // chunk-1 centre: offsetTop 50 + height 100 / 2 = 100  → dist 200
    // chunk-2 centre: offsetTop 250 + height 100 / 2 = 300 → dist 0
    // chunk-3 centre: offsetTop 600 + height 100 / 2 = 650 → dist 350
    const el1 = makeParagraphEl("chunk-1", 0, 50, 100);
    const el2 = makeParagraphEl("chunk-2", 1, 250, 100);
    const el3 = makeParagraphEl("chunk-3", 2, 600, 100);
    const container = makeContainer(0, 600, [el1, el2, el3]);
    const { paragraphId } = findCurrentParagraphFromDOM(container);
    expect(paragraphId).toBe("chunk-2");
  });

  it("correctly handles scrolled-down state", () => {
    // Viewport: scrollTop=500, clientHeight=600, centre=800
    // chunk-4 centre: 750 + 50 = 800 → dist 0 (perfect match)
    // chunk-5 centre: 200 + 50 = 250 → dist 550 (far above)
    const el4 = makeParagraphEl("chunk-4", 3, 750, 100);
    const el5 = makeParagraphEl("chunk-5", 4, 200, 100);
    const container = makeContainer(500, 600, [el4, el5]);
    const { paragraphId } = findCurrentParagraphFromDOM(container);
    expect(paragraphId).toBe("chunk-4");
  });

  it("returns correct chunkIndex", () => {
    const el = makeParagraphEl("chunk-7", 7, 100, 50);
    const container = makeContainer(0, 600, [el]);
    const { chunkIndex } = findCurrentParagraphFromDOM(container);
    expect(chunkIndex).toBe(7);
  });
});

// ─── Viewport-normalised scroll velocity ─────────────────────────────────────

describe("normalizedScrollVelocity", () => {
  it("large viewport → smaller normalised velocity for same delta", () => {
    const small = normalizedScrollVelocity(100, 500);
    const large = normalizedScrollVelocity(100, 1000);
    expect(small).toBeGreaterThan(large);
  });

  it("matches formula: delta / (vh * 2)", () => {
    expect(normalizedScrollVelocity(200, 800)).toBeCloseTo(200 / (800 * 2));
  });

  it("returns 0 when viewport height is 0 (guard against division by zero)", () => {
    expect(normalizedScrollVelocity(100, 0)).toBe(0);
  });

  it("proportional to scroll delta", () => {
    const vh = 800;
    expect(normalizedScrollVelocity(200, vh)).toBeCloseTo(2 * normalizedScrollVelocity(100, vh));
  });
});

// ─── Phase 8: ui_context computation logic ────────────────────────────────────

/**
 * Mirror of the ui_context logic in useTelemetry flush().
 * This is a pure helper extracted for testing.
 */
function computeUiContext(opts: {
  sessionPaused: boolean;
  panelInteractedInWindow: boolean;
  panelOpen: boolean;
}): UiContext {
  if (opts.sessionPaused) return "USER_PAUSED";
  if (opts.panelInteractedInWindow) return "PANEL_INTERACTING";
  if (opts.panelOpen) return "PANEL_OPEN";
  return "READ_MAIN";
}

describe("ui_context computation", () => {
  it("defaults to READ_MAIN when session active and no panel", () => {
    const ctx = computeUiContext({ sessionPaused: false, panelInteractedInWindow: false, panelOpen: false });
    expect(ctx).toBe("READ_MAIN");
  });

  it("returns USER_PAUSED when session paused (regardless of panel state)", () => {
    const ctx = computeUiContext({ sessionPaused: true, panelInteractedInWindow: true, panelOpen: true });
    expect(ctx).toBe("USER_PAUSED");
  });

  it("returns PANEL_INTERACTING when user interacted in panel this window", () => {
    const ctx = computeUiContext({ sessionPaused: false, panelInteractedInWindow: true, panelOpen: true });
    expect(ctx).toBe("PANEL_INTERACTING");
  });

  it("returns PANEL_OPEN when panel is visible but no interaction this window", () => {
    const ctx = computeUiContext({ sessionPaused: false, panelInteractedInWindow: false, panelOpen: true });
    expect(ctx).toBe("PANEL_OPEN");
  });

  it("PANEL_INTERACTING takes priority over PANEL_OPEN", () => {
    const ctx = computeUiContext({ sessionPaused: false, panelInteractedInWindow: true, panelOpen: true });
    expect(ctx).toBe("PANEL_INTERACTING");
  });

  it("USER_PAUSED takes priority over everything", () => {
    // Paused + panel open + interaction = still USER_PAUSED
    const ctx = computeUiContext({ sessionPaused: true, panelInteractedInWindow: true, panelOpen: false });
    expect(ctx).toBe("USER_PAUSED");
  });
});

// ─── Phase 8: interaction_zone defaults ──────────────────────────────────────

describe("interaction_zone type guard", () => {
  it("reader is a valid InteractionZone", () => {
    const zone: InteractionZone = "reader";
    expect(zone).toBe("reader");
  });

  it("panel is a valid InteractionZone", () => {
    const zone: InteractionZone = "panel";
    expect(zone).toBe("panel");
  });

  it("other is a valid InteractionZone", () => {
    const zone: InteractionZone = "other";
    expect(zone).toBe("other");
  });
});

// ─── Phase 8: ui_context type values ─────────────────────────────────────────

describe("UiContext type values", () => {
  const valid: UiContext[] = ["READ_MAIN", "PANEL_OPEN", "PANEL_INTERACTING", "USER_PAUSED"];

  it("has exactly 4 valid values", () => {
    expect(valid).toHaveLength(4);
  });

  it("baseline sessions use READ_MAIN when no panel", () => {
    // Simulate baseline session: no panel, not paused
    const ctx = computeUiContext({ sessionPaused: false, panelInteractedInWindow: false, panelOpen: false });
    expect(ctx).toBe("READ_MAIN");
    expect(valid).toContain(ctx);
  });
});
