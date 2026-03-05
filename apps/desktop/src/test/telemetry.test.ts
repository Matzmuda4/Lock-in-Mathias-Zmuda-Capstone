/**
 * Unit tests for pure-math helpers exported from useTelemetry.ts
 *
 * These run without any React environment — no DOM, no hooks, no mocks.
 */
import { describe, it, expect } from "vitest";
import { computeMouseStats, selectCurrentParagraph, type Point, type IntersectionEntry } from "../hooks/useTelemetry";

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
