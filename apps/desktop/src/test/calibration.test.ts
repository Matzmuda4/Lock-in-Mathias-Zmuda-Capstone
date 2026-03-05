/**
 * Frontend unit tests for Phase B calibration helpers.
 *
 * Tests two pure pieces of logic:
 *  1. Redirect decision: should redirect when has_baseline=false AND calib_available=true
 *  2. Finish Calibration button enablement based on time + progress thresholds
 */
import { describe, it, expect } from "vitest";

// ─── Redirect logic ───────────────────────────────────────────────────────────

/**
 * Pure function extracted from CalibrationPage / HomePage redirect logic.
 * Given a CalibrationStatus response, decide whether to redirect to /calibration.
 */
function shouldRedirectToCalibration(status: {
  has_baseline: boolean;
  calib_available: boolean;
}): boolean {
  return !status.has_baseline && status.calib_available;
}

describe("shouldRedirectToCalibration", () => {
  it("redirects when no baseline and calib is available", () => {
    expect(
      shouldRedirectToCalibration({ has_baseline: false, calib_available: true }),
    ).toBe(true);
  });

  it("does NOT redirect when baseline already exists", () => {
    expect(
      shouldRedirectToCalibration({ has_baseline: true, calib_available: true }),
    ).toBe(false);
  });

  it("does NOT redirect when calibration PDF is unavailable (server missing)", () => {
    expect(
      shouldRedirectToCalibration({ has_baseline: false, calib_available: false }),
    ).toBe(false);
  });

  it("does NOT redirect when both are false (edge case)", () => {
    expect(
      shouldRedirectToCalibration({ has_baseline: false, calib_available: false }),
    ).toBe(false);
  });
});

// ─── Finish Calibration button enablement ────────────────────────────────────

// Only a small grace period prevents accidental instant clicks.
const CALIB_MIN_SECONDS = 10;

/**
 * Pure function mirroring the CalibrationControls enablement logic.
 * The user can finish as soon as they are done reading — no time minimum
 * beyond the short grace period.
 */
function canFinishCalibration(elapsedSeconds: number): boolean {
  return elapsedSeconds >= CALIB_MIN_SECONDS;
}

describe("canFinishCalibration", () => {
  it("disabled at 0 seconds (grace period)", () => {
    expect(canFinishCalibration(0)).toBe(false);
  });

  it("disabled at 9 seconds (one second short of grace period)", () => {
    expect(canFinishCalibration(9)).toBe(false);
  });

  it("enabled at exactly 10 seconds", () => {
    expect(canFinishCalibration(10)).toBe(true);
  });

  it("enabled at 60 seconds", () => {
    expect(canFinishCalibration(60)).toBe(true);
  });

  it("enabled at 200 seconds", () => {
    expect(canFinishCalibration(200)).toBe(true);
  });
});
