/**
 * Tests for the adaptive assistant panel logic.
 *
 * These are pure logic tests — no React rendering.
 * The panel is only relevant for adaptive-mode sessions; all others
 * must behave exactly as before (no panel, READ_MAIN ui_context).
 */
import { describe, it, expect } from "vitest";

// ─── Session mode → panel visibility ─────────────────────────────────────────

function shouldRenderPanel(sessionMode: string): boolean {
  return sessionMode === "adaptive";
}

describe("panel visibility by session mode", () => {
  it("adaptive mode renders the panel", () => {
    expect(shouldRenderPanel("adaptive")).toBe(true);
  });

  it("baseline mode does NOT render the panel", () => {
    expect(shouldRenderPanel("baseline")).toBe(false);
  });

  it("calibration mode does NOT render the panel", () => {
    expect(shouldRenderPanel("calibration")).toBe(false);
  });

  it("standard mode does NOT render the panel", () => {
    expect(shouldRenderPanel("standard")).toBe(false);
  });
});

// ─── ui_context when baseline session (no panel) ──────────────────────────────

type UiContext = "READ_MAIN" | "PANEL_OPEN" | "PANEL_INTERACTING" | "USER_PAUSED";
type InteractionZone = "reader" | "panel" | "other";

function defaultTelemetryContext(sessionMode: string, paused: boolean): {
  ui_context: UiContext;
  interaction_zone: InteractionZone;
} {
  if (paused) return { ui_context: "USER_PAUSED", interaction_zone: "reader" };
  if (sessionMode !== "adaptive") {
    return { ui_context: "READ_MAIN", interaction_zone: "reader" };
  }
  // adaptive, not paused, panel not interacted yet
  return { ui_context: "PANEL_OPEN", interaction_zone: "reader" };
}

describe("default telemetry context per session mode", () => {
  it("baseline session emits READ_MAIN and reader zone by default", () => {
    const ctx = defaultTelemetryContext("baseline", false);
    expect(ctx.ui_context).toBe("READ_MAIN");
    expect(ctx.interaction_zone).toBe("reader");
  });

  it("calibration session emits READ_MAIN and reader zone", () => {
    const ctx = defaultTelemetryContext("calibration", false);
    expect(ctx.ui_context).toBe("READ_MAIN");
    expect(ctx.interaction_zone).toBe("reader");
  });

  it("adaptive session defaults to PANEL_OPEN when panel renders", () => {
    const ctx = defaultTelemetryContext("adaptive", false);
    expect(ctx.ui_context).toBe("PANEL_OPEN");
    expect(ctx.interaction_zone).toBe("reader");
  });

  it("paused session emits USER_PAUSED regardless of mode", () => {
    expect(defaultTelemetryContext("adaptive", true).ui_context).toBe("USER_PAUSED");
    expect(defaultTelemetryContext("baseline", true).ui_context).toBe("USER_PAUSED");
  });
});
