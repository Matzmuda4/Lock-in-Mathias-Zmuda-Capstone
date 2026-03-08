/**
 * Unit tests for pure-math helpers exported from useTelemetry.ts
 *
 * These run without any React environment — no DOM, no hooks, no mocks.
 */
import { describe, it, expect } from "vitest";
import { computeMouseStats, selectCurrentParagraph, type Point, type IntersectionEntry } from "../hooks/useTelemetry";

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

  it("returns null when all ratios are below 0.6 threshold", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-1", chunkIndex: 0, ratio: 0.3 },
      { paragraphId: "chunk-2", chunkIndex: 1, ratio: 0.59 },
    ];
    const { paragraphId } = selectCurrentParagraph(entries);
    expect(paragraphId).toBeNull();
  });

  it("returns the single element when only one meets threshold", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-3", chunkIndex: 2, ratio: 0.1 },
      { paragraphId: "chunk-4", chunkIndex: 3, ratio: 0.75 },
    ];
    const result = selectCurrentParagraph(entries);
    expect(result.paragraphId).toBe("chunk-4");
    expect(result.chunkIndex).toBe(3);
  });

  it("returns the element with highest ratio when multiple meet threshold", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-5", chunkIndex: 4, ratio: 0.65 },
      { paragraphId: "chunk-6", chunkIndex: 5, ratio: 0.9 },
      { paragraphId: "chunk-7", chunkIndex: 6, ratio: 0.72 },
    ];
    const result = selectCurrentParagraph(entries);
    expect(result.paragraphId).toBe("chunk-6");
    expect(result.chunkIndex).toBe(5);
  });

  it("treats exactly 0.6 as meeting the threshold", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-8", chunkIndex: 7, ratio: 0.6 },
    ];
    const result = selectCurrentParagraph(entries);
    expect(result.paragraphId).toBe("chunk-8");
  });

  it("preserves null chunkIndex when it is null", () => {
    const entries: IntersectionEntry[] = [
      { paragraphId: "chunk-9", chunkIndex: null, ratio: 0.8 },
    ];
    const result = selectCurrentParagraph(entries);
    expect(result.paragraphId).toBe("chunk-9");
    expect(result.chunkIndex).toBeNull();
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
