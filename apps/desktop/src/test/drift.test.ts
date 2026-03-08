/**
 * Unit tests for drift service helpers.
 * No DOM, no React, no mocks needed.
 */
import { describe, it, expect } from "vitest";
import { driftColor } from "../services/driftService";

// ─── driftColor ───────────────────────────────────────────────────────────────

describe("driftColor", () => {
  it("returns green for drift_ema < 0.30", () => {
    expect(driftColor(0.0)).toBe("green");
    expect(driftColor(0.15)).toBe("green");
    expect(driftColor(0.29)).toBe("green");
  });

  it("returns yellow for drift_ema in [0.30, 0.60)", () => {
    expect(driftColor(0.30)).toBe("yellow");
    expect(driftColor(0.45)).toBe("yellow");
    expect(driftColor(0.599)).toBe("yellow");
  });

  it("returns red for drift_ema >= 0.60", () => {
    expect(driftColor(0.60)).toBe("red");
    expect(driftColor(0.75)).toBe("red");
    expect(driftColor(1.0)).toBe("red");
  });

  it("treats exactly 0.30 as yellow (boundary)", () => {
    expect(driftColor(0.30)).toBe("yellow");
  });

  it("treats exactly 0.60 as red (boundary)", () => {
    expect(driftColor(0.60)).toBe("red");
  });
});

// ─── drift percentage display helper ─────────────────────────────────────────

function driftPct(ema: number): number {
  return Math.round(ema * 100);
}

describe("driftPct", () => {
  it("rounds correctly", () => {
    expect(driftPct(0.0)).toBe(0);
    expect(driftPct(0.354)).toBe(35);
    expect(driftPct(0.999)).toBe(100);
  });

  it("is never negative for valid ema values", () => {
    expect(driftPct(0.0)).toBeGreaterThanOrEqual(0);
  });
});
