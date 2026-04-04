/**
 * useTelemetry — Phase 7 Telemetry Logging v4
 *
 * Key fixes from v3:
 *   A3+) Paragraph ID now uses a primary DOM-scan strategy (`findCurrentParagraphFromDOM`)
 *        that reads `offsetTop` on every flush. This is reliable regardless of whether
 *        the IntersectionObserver was set up before the content loaded (timing bug fix).
 *        The IntersectionObserver remains as a secondary path and updates between flushes.
 *   A6)  IntersectionObserver now retries if containerRef.current was null on first run
 *        (polls every 200 ms until the element mounts).
 *   A7)  prevProgressRatio is initialised to the container's current scroll position when
 *        the session activates, preventing a false-positive "scroll=0 but progress changed"
 *        warning on the very first flush of a resumed session.
 *   FUTURE: Mouse tracking will need to exclude pointer events inside intervention panels.
 *        Add `data-telemetry-exclude="true"` to any overlay/panel to opt out once
 *        intervention UI is implemented.
 *
 * Key fixes from v2:
 *   A1) idle_seconds is now per-window (0..2s), not cumulative.
 *   A2) Scroll deltas computed from scroll container's scrollTop.
 *   A3) IntersectionObserver uses multi-tier fallback strategy.
 *   A4) Dev-mode sanity warnings surface silent telemetry failures.
 *
 * Signals collected (per 2-second window):
 *   scroll_delta_sum              net signed scroll distance (px)
 *   scroll_delta_abs_sum          total absolute scroll distance (px)
 *   scroll_delta_pos_sum          sum of positive (down) deltas only
 *   scroll_delta_neg_sum          sum of |negative (up) deltas| only
 *   scroll_event_count            raw scroll event count
 *   scroll_direction_changes      direction reversals (backtracking proxy)
 *   scroll_pause_seconds          seconds since last scroll (capped 60 s)
 *   idle_seconds                  seconds idle IN THIS 2s WINDOW (0..2)
 *   idle_since_interaction_seconds total seconds since last interaction (diagnostic)
 *   mouse_path_px                 total cursor path length (px)
 *   mouse_net_px                  straight-line net displacement (px)
 *   window_focus_state            "focused" | "blurred"
 *   current_paragraph_id          data-paragraph-id of most-visible paragraph
 *   current_chunk_index           chunk index of most-visible chunk
 *   viewport_progress_ratio       scrollTop / (scrollHeight - clientHeight)
 *   viewport_height_px            window.innerHeight
 *   viewport_width_px             window.innerWidth
 *   reader_container_height_px    scrollable container clientHeight
 *   ui_context                    READ_MAIN | PANEL_OPEN | PANEL_INTERACTING | USER_PAUSED
 *   interaction_zone              reader | panel | other
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { activityService, type TelemetryBatch } from "../services/activityService";

// ─── Pure helpers (exported for unit tests) ──────────────────────────────────

export interface Point {
  x: number;
  y: number;
}

/** Compute total path length and straight-line net distance from a sequence of points. */
export function computeMouseStats(points: Point[]): { pathPx: number; netPx: number } {
  if (points.length < 2) return { pathPx: 0, netPx: 0 };

  let pathPx = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    pathPx += Math.sqrt(dx * dx + dy * dy);
  }

  const first = points[0];
  const last = points[points.length - 1];
  const dx = last.x - first.x;
  const dy = last.y - first.y;
  const netPx = Math.sqrt(dx * dx + dy * dy);

  return { pathPx, netPx };
}

export interface IntersectionEntry {
  paragraphId: string;
  chunkIndex: number | null;
  ratio: number;
  /** Distance from the top of the scroll container (used as fallback sort key). */
  topOffset: number;
}

/**
 * Given a list of paragraph intersection entries, return the best current paragraph.
 *
 * Strategy (in order):
 *   1. Highest intersectionRatio >= 0.6
 *   2. If none >= 0.6, the paragraph with the highest ratio overall
 *   3. If all ratios are 0, the paragraph whose top is closest to 0 (viewport top)
 */
export function selectCurrentParagraph(entries: IntersectionEntry[]): {
  paragraphId: string | null;
  chunkIndex: number | null;
} {
  if (entries.length === 0) return { paragraphId: null, chunkIndex: null };

  // Strategy 1: threshold pass
  const candidates = entries.filter((e) => e.ratio >= 0.6);
  if (candidates.length > 0) {
    const best = candidates.reduce((a, b) => (b.ratio > a.ratio ? b : a));
    return { paragraphId: best.paragraphId, chunkIndex: best.chunkIndex };
  }

  // Strategy 2: highest ratio (even if < 0.6)
  const allVisible = entries.filter((e) => e.ratio > 0);
  if (allVisible.length > 0) {
    const best = allVisible.reduce((a, b) => (b.ratio > a.ratio ? b : a));
    return { paragraphId: best.paragraphId, chunkIndex: best.chunkIndex };
  }

  // Strategy 3: closest to top by topOffset (ascending)
  const byTop = [...entries].sort((a, b) => Math.abs(a.topOffset) - Math.abs(b.topOffset));
  return { paragraphId: byTop[0].paragraphId, chunkIndex: byTop[0].chunkIndex };
}

/**
 * Find the current paragraph directly from the DOM using element offsets.
 *
 * This is the PRIMARY paragraph-detection mechanism called on every flush.
 * It is reliable regardless of IntersectionObserver timing (e.g. when the
 * container mounts after the initial effect run).
 *
 * Selects the `[data-paragraph-id]` element whose vertical centre is closest
 * to the centre of the container's visible viewport area.
 */
export function findCurrentParagraphFromDOM(container: HTMLElement): {
  paragraphId: string | null;
  chunkIndex: number | null;
} {
  const elements = Array.from(
    container.querySelectorAll("[data-paragraph-id]"),
  ) as HTMLElement[];

  if (elements.length === 0) return { paragraphId: null, chunkIndex: null };

  const viewportCenter = container.scrollTop + container.clientHeight / 2;

  let best: HTMLElement | null = null;
  let bestDist = Infinity;

  for (const el of elements) {
    const elCenter = el.offsetTop + el.offsetHeight / 2;
    const dist = Math.abs(elCenter - viewportCenter);
    if (dist < bestDist) {
      bestDist = dist;
      best = el;
    }
  }

  if (!best) return { paragraphId: null, chunkIndex: null };

  return {
    paragraphId: best.dataset.paragraphId ?? null,
    chunkIndex:
      best.dataset.chunkIndex != null
        ? parseInt(best.dataset.chunkIndex, 10)
        : null,
  };
}

/**
 * Compute per-window idle_seconds (0..WINDOW_S).
 * This is the time in the current window during which there was NO interaction.
 *
 * Given:
 *   - windowStartMs: timestamp when the 2s window began (= previous flush time)
 *   - lastInteractionMs: timestamp of the last interaction event
 *   - nowMs: current time
 *
 * If the last interaction is BEFORE the window started, the user was idle for
 * the entire window (return WINDOW_S = 2.0).
 * If the last interaction is within the window, idle time = now - lastInteraction
 * (they were active at the start of the window, then went idle).
 */
export function computeWindowIdle(
  windowStartMs: number,
  lastInteractionMs: number,
  nowMs: number,
  windowS: number = 2.0,
): number {
  if (lastInteractionMs <= windowStartMs) {
    // No interaction at all in this window
    return windowS;
  }
  // Interaction happened during this window; idle since then
  const idleSinceMs = nowMs - lastInteractionMs;
  return Math.min(windowS, Math.max(0, idleSinceMs / 1000));
}

// ─── Sanity warning conditions (exported for tests) ───────────────────────────

export interface TelemetrySanityWarnings {
  idleExceedsWindow: boolean;       // idle_seconds > 2.0 (should never happen)
  scrollZeroWithProgress: boolean;  // scroll=0 but progress changed significantly
  paragraphMissing: boolean;        // no para_id for extended period
}

export function checkTelemetrySanity(
  idleSeconds: number,
  scrollAbsSum: number,
  prevProgress: number,
  currentProgress: number,
  paragraphId: string | null,
  /** Pass true only after the scroll container has been mounted and the
   *  scroll listener has been attached.  Suppresses false-positive warnings
   *  during the initial loading phase. */
  scrollListenerReady: boolean = true,
): TelemetrySanityWarnings {
  return {
    idleExceedsWindow: idleSeconds > 2.0,
    scrollZeroWithProgress:
      scrollListenerReady &&
      scrollAbsSum === 0 &&
      Math.abs(currentProgress - prevProgress) > 0.05,
    paragraphMissing: paragraphId === null,
  };
}

// ─── Hook ────────────────────────────────────────────────────────────────────

const FLUSH_INTERVAL_MS = 2000;
const WINDOW_S = 2.0;
const MAX_PAUSE_CAP_S = 60;
const DEV = (import.meta as unknown as { env?: { DEV?: boolean } }).env?.DEV ?? false;

export type UiContext = "READ_MAIN" | "PANEL_OPEN" | "PANEL_INTERACTING" | "USER_PAUSED";
export type InteractionZone = "reader" | "panel" | "other";

export interface UseTelemetryOptions {
  sessionId: number;
  token: string | null;
  /** Pass true while the session is active (not paused / ended). */
  active: boolean;
  /** Ref to the scrollable reader container element. */
  containerRef: React.RefObject<HTMLElement | null>;
  /** Whether the session is currently paused (sets ui_context=USER_PAUSED). */
  sessionPaused?: boolean;
  /**
   * Ref to the adaptive side panel container.
   * When provided, events inside this element set interaction_zone="panel".
   * Only pass for adaptive-mode sessions.
   */
  panelContainerRef?: React.RefObject<HTMLElement | null>;
  /** Whether the side panel is currently open/visible (adaptive mode only). */
  panelOpen?: boolean;
}

export interface UseTelemetryReturn {
  /** The last batch that was sent (for the debug panel). */
  lastBatch: TelemetryBatch | null;
  /** Whether telemetry is currently being collected. */
  collecting: boolean;
  /** Dev-mode sanity warnings for the last batch. */
  warnings: TelemetrySanityWarnings | null;
}

export function useTelemetry({
  sessionId,
  token,
  active,
  containerRef,
  sessionPaused = false,
  panelContainerRef,
  panelOpen = false,
}: UseTelemetryOptions): UseTelemetryReturn {
  const activeRef = useRef(active);
  activeRef.current = active;

  const tokenRef = useRef(token);
  tokenRef.current = token;

  // ── Accumulators (reset each flush) ──────────────────────────────────────
  const scrollDeltaSum = useRef(0);
  const scrollDeltaAbsSum = useRef(0);
  const scrollDeltaPosSum = useRef(0);
  const scrollDeltaNegSum = useRef(0);
  const scrollEventCount = useRef(0);
  const scrollDirectionChanges = useRef(0);
  // Initialise to 0 — will be set to Date.now() when the scroll container
  // actually mounts (via attachScroll), preventing inflated scroll_pause_seconds
  // during the loading phase before the container is available.
  const lastScrollTime = useRef<number>(0);
  const lastScrollDir = useRef<"down" | "up" | null>(null);
  // Track last known scrollTop from the container (not window)
  const lastScrollTop = useRef<number>(0);

  // Interaction tracking — for per-window idle computation
  const lastInteraction = useRef<number>(Date.now());
  // windowStart is reset on each flush — used to compute per-window idle
  const windowStart = useRef<number>(Date.now());

  const mousePoints = useRef<Point[]>([]);
  // Throttle: max one mouse-point sample per 50 ms (≤ 20/s vs. potential 100-200/s raw)
  const lastMouseSample = useRef<number>(0);

  const windowFocused = useRef(true);

  // Progress ratio tracking for scroll-capture sanity check
  const prevProgressRatio = useRef<number>(0);
  // Set to true once the scroll container is mounted and listener is attached
  const scrollListenerReady = useRef<boolean>(false);

  // ── IntersectionObserver state ────────────────────────────────────────────
  const intersectionMap = useRef<Map<string, IntersectionEntry>>(new Map());
  const currentParagraphId = useRef<string | null>(null);
  const currentChunkIndex = useRef<number | null>(null);
  // Track how many consecutive batches had no paragraph — for warning
  const missingParaBatches = useRef<number>(0);

  // ── Interaction zone tracking (Phase 8 — panel telemetry) ────────────────
  // Zone of the most recent interaction in this 2s window.
  const lastInteractionZone = useRef<InteractionZone>("reader");
  // True if any interaction happened in the panel during this window.
  const panelInteractedInWindow = useRef<boolean>(false);

  // ── Exposed to UI ──────────────────────────────────────────────────────────
  const [lastBatch, setLastBatch] = useState<TelemetryBatch | null>(null);
  const [warnings, setWarnings] = useState<TelemetrySanityWarnings | null>(null);

  // ── Reset accumulators ────────────────────────────────────────────────────
  const resetAccumulators = useCallback(() => {
    scrollDeltaSum.current = 0;
    scrollDeltaAbsSum.current = 0;
    scrollDeltaPosSum.current = 0;
    scrollDeltaNegSum.current = 0;
    scrollEventCount.current = 0;
    scrollDirectionChanges.current = 0;
    mousePoints.current = [];

    // Sync the last known scrollTop so next delta is relative to now
    const container = containerRef.current;
    if (container) {
      lastScrollTop.current = container.scrollTop;
    }

    // Reset window start for next idle computation
    windowStart.current = Date.now();

    // Reset zone tracking for the next window
    lastInteractionZone.current = "reader";
    panelInteractedInWindow.current = false;
  }, [containerRef]);

  // ── Flush: build and send one batch ───────────────────────────────────────
  const flush = useCallback(() => {
    if (!activeRef.current || !tokenRef.current) return;

    const now = Date.now();
    const container = containerRef.current;

    // Scroll pause: seconds since last scroll event, capped.
    // lastScrollTime = 0 means no scroll has occurred yet this session
    // (container wasn't mounted at effect run time); treat as max pause.
    const scrollPauseSec =
      lastScrollTime.current === 0
        ? MAX_PAUSE_CAP_S
        : Math.min((now - lastScrollTime.current) / 1000, MAX_PAUSE_CAP_S);

    // ── A1: Per-window idle (0..WINDOW_S) ─────────────────────────────────
    // idle_seconds = time in this 2s window with no interaction
    const idleSecWindow = computeWindowIdle(
      windowStart.current,
      lastInteraction.current,
      now,
      WINDOW_S,
    );
    // Diagnostic cumulative value (not used by model, kept for logging)
    const idleSinceInteraction = (now - lastInteraction.current) / 1000;

    // Mouse stats
    const { pathPx, netPx } = computeMouseStats(mousePoints.current);

    // Viewport progress ratio — computed from container scrollTop
    let viewportProgress = 0;
    if (container) {
      const range = container.scrollHeight - container.clientHeight;
      viewportProgress = range > 0 ? Math.min(1, container.scrollTop / range) : 0;
    }

    // ── A3+: DOM-scan paragraph detection (primary, runs every flush) ─────
    // The IntersectionObserver may not have been set up if the container
    // mounted after the initial effect run. Reading offsetTop directly is
    // always reliable and refreshes the "current paragraph" every 2 s.
    if (container) {
      const fromDom = findCurrentParagraphFromDOM(container);
      if (fromDom.paragraphId !== null) {
        currentParagraphId.current = fromDom.paragraphId;
        currentChunkIndex.current = fromDom.chunkIndex;
      }
    }

    // ── A4: Sanity checks ─────────────────────────────────────────────────
    const sanity = checkTelemetrySanity(
      idleSecWindow,
      scrollDeltaAbsSum.current,
      prevProgressRatio.current,
      viewportProgress,
      currentParagraphId.current,
      scrollListenerReady.current,
    );

    if (currentParagraphId.current === null) {
      missingParaBatches.current += 1;
    } else {
      missingParaBatches.current = 0;
    }

    if (DEV) {
      if (sanity.idleExceedsWindow) {
        console.warn("[Telemetry] idle_seconds > 2.0 — bug in window idle computation", idleSecWindow);
      }
      if (sanity.scrollZeroWithProgress) {
        console.warn(
          "[Telemetry] scroll_delta=0 but progress changed",
          prevProgressRatio.current.toFixed(3),
          "→",
          viewportProgress.toFixed(3),
          "— scroll events may not be captured",
        );
      }
      if (missingParaBatches.current >= 5) {
        console.warn(
          `[Telemetry] current_paragraph_id missing for ${missingParaBatches.current} consecutive batches`,
          "— check IntersectionObserver root + data-paragraph-id attributes",
        );
      }
    }

    prevProgressRatio.current = viewportProgress;

    // ── Phase 8: ui_context + interaction_zone ────────────────────────────
    const interactionZone: InteractionZone = lastInteractionZone.current;
    let uiCtx: UiContext;
    if (sessionPaused) {
      uiCtx = "USER_PAUSED";
    } else if (panelInteractedInWindow.current) {
      uiCtx = "PANEL_INTERACTING";
    } else if (panelOpen) {
      uiCtx = "PANEL_OPEN";
    } else {
      uiCtx = "READ_MAIN";
    }

    const batch: TelemetryBatch = {
      session_id: sessionId,
      scroll_delta_sum: scrollDeltaSum.current,
      scroll_delta_abs_sum: scrollDeltaAbsSum.current,
      scroll_delta_pos_sum: scrollDeltaPosSum.current,
      scroll_delta_neg_sum: scrollDeltaNegSum.current,
      scroll_event_count: scrollEventCount.current,
      scroll_direction_changes: scrollDirectionChanges.current,
      scroll_pause_seconds: scrollPauseSec,
      // A1 fix: this is now per-window (0..2), not cumulative
      idle_seconds: idleSecWindow,
      // Diagnostic cumulative value for debugging
      idle_since_interaction_seconds: Math.round(idleSinceInteraction * 10) / 10,
      mouse_path_px: pathPx,
      mouse_net_px: netPx,
      window_focus_state: windowFocused.current ? "focused" : "blurred",
      current_paragraph_id: currentParagraphId.current,
      current_chunk_index: currentChunkIndex.current,
      viewport_progress_ratio: viewportProgress,
      viewport_height_px: window.innerHeight,
      viewport_width_px: window.innerWidth,
      reader_container_height_px: container ? container.clientHeight : window.innerHeight,
      ui_context: uiCtx,
      interaction_zone: interactionZone,
      client_timestamp: new Date().toISOString(),
    };

    if (DEV) {
      console.debug("[Telemetry] batch →", {
        ...batch,
        idle_seconds: idleSecWindow.toFixed(3) + "s (window)",
        idle_since_interaction_seconds: idleSinceInteraction.toFixed(1) + "s (cumulative)",
      });
    }

    activityService.postBatch(tokenRef.current, batch);
    setLastBatch(batch);
    setWarnings(sanity);
    resetAccumulators();
  }, [sessionId, containerRef, resetAccumulators, sessionPaused, panelOpen]);

  // ── Event listeners ────────────────────────────────────────────────────────
  // A2 / A6: The scroll container (`<main ref={contentRef}>`) may not exist yet
  // when this effect first runs — the page renders a loading spinner before data
  // loads, so `containerRef.current` is null at mount time.  Window-level
  // listeners (mouse, keyboard, focus) are attached immediately.  The scroll
  // listener polls every 200 ms until the container element is available.
  useEffect(() => {
    // ── A2: Scroll handler — reads scrollTop delta from the container ─────
    const onScroll = () => {
      if (!activeRef.current) return;
      const el = containerRef.current;
      if (!el) return;

      const currentScrollTop = el.scrollTop;
      const dy = currentScrollTop - lastScrollTop.current;

      // Skip zero-delta events (elastic scroll rebound, programmatic resets)
      if (dy === 0) return;

      scrollDeltaSum.current += dy;
      scrollDeltaAbsSum.current += Math.abs(dy);
      if (dy > 0) scrollDeltaPosSum.current += dy;
      else scrollDeltaNegSum.current += Math.abs(dy);
      scrollEventCount.current += 1;

      const dir: "down" | "up" = dy > 0 ? "down" : "up";
      if (lastScrollDir.current !== null && dir !== lastScrollDir.current) {
        scrollDirectionChanges.current += 1;
      }
      lastScrollDir.current = dir;
      lastScrollTop.current = currentScrollTop;
      lastScrollTime.current = Date.now();
      lastInteraction.current = Date.now();
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!activeRef.current) return;
      const now = Date.now();
      lastInteraction.current = now;
      // Sample at most once per 50 ms — reduces allocations from ~150/s to ~20/s
      if (now - lastMouseSample.current >= 50) {
        mousePoints.current.push({ x: e.clientX, y: e.clientY });
        lastMouseSample.current = now;
      }
    };

    const onKeyDown = () => {
      if (!activeRef.current) return;
      lastInteraction.current = Date.now();
    };

    const onBlur = () => { windowFocused.current = false; };
    const onFocus = () => {
      windowFocused.current = true;
      lastInteraction.current = Date.now();
    };

    // Window-level listeners — attach immediately (always available)
    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("keydown", onKeyDown, { passive: true });
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);

    // Scroll listener — poll until container element is mounted
    let scrollContainer: HTMLElement | null = null;
    let pollId: ReturnType<typeof setInterval> | null = null;

    const attachScroll = () => {
      const container = containerRef.current;
      if (!container) return false;
      // Sync starting scrollTop so first delta is correct
      lastScrollTop.current = container.scrollTop;
      // Sync lastScrollTime so scroll_pause_seconds starts from now,
      // not from component mount (which inflates the "no scroll" duration).
      lastScrollTime.current = Date.now();
      container.addEventListener("scroll", onScroll, { passive: true });
      scrollContainer = container;
      scrollListenerReady.current = true;
      return true;
    };

    if (!attachScroll()) {
      pollId = setInterval(() => {
        if (attachScroll() && pollId !== null) {
          clearInterval(pollId);
          pollId = null;
        }
      }, 200);
    }

    return () => {
      if (pollId !== null) clearInterval(pollId);
      if (scrollContainer) scrollContainer.removeEventListener("scroll", onScroll);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
    };
  }, [containerRef]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Phase 8: Panel interaction zone detection ─────────────────────────────
  // Attach event listeners to the panel container (adaptive sessions only).
  // Any click, mousemove, or keydown inside the panel sets zone="panel".
  // This effect re-runs when panelContainerRef becomes available.
  useEffect(() => {
    const panel = panelContainerRef?.current;
    if (!panel) return;

    const onPanelEvent = () => {
      if (!activeRef.current) return;
      lastInteractionZone.current = "panel";
      panelInteractedInWindow.current = true;
      lastInteraction.current = Date.now();
    };

    panel.addEventListener("click", onPanelEvent);
    panel.addEventListener("mousemove", onPanelEvent, { passive: true });
    panel.addEventListener("keydown", onPanelEvent);

    return () => {
      panel.removeEventListener("click", onPanelEvent);
      panel.removeEventListener("mousemove", onPanelEvent);
      panel.removeEventListener("keydown", onPanelEvent);
    };
  // Intentionally depends on the ref identity — re-runs when panel mounts
  }, [panelContainerRef?.current]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── A3/A6: IntersectionObserver — improved fallback + retry on late mount ──
  // The container element may be null on the first effect run if the page is
  // in a loading state. We poll every 200 ms until it is available, then set
  // up the observer exactly once.
  useEffect(() => {
    let observer: IntersectionObserver | null = null;
    let mo: MutationObserver | null = null;
    let pollId: ReturnType<typeof setInterval> | null = null;

    const setup = (container: HTMLElement) => {
      observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            const el = entry.target as HTMLElement;
            const paragraphId = el.dataset.paragraphId;
            if (!paragraphId) return;

            const chunkIndex =
              el.dataset.chunkIndex != null
                ? parseInt(el.dataset.chunkIndex, 10)
                : null;

            const containerRect = container.getBoundingClientRect();
            const elRect = el.getBoundingClientRect();
            const topOffset = elRect.top - containerRect.top;

            intersectionMap.current.set(paragraphId, {
              paragraphId,
              chunkIndex,
              ratio: entry.intersectionRatio,
              topOffset,
            });
          });

          // Update best-candidate from intersection map (secondary path)
          const { paragraphId, chunkIndex } = selectCurrentParagraph(
            Array.from(intersectionMap.current.values()),
          );
          if (paragraphId !== null) {
            currentParagraphId.current = paragraphId;
            currentChunkIndex.current = chunkIndex;
          }
        },
        {
          root: container,
          threshold: [0, 0.1, 0.25, 0.5, 0.6, 0.75, 1.0],
        },
      );

      const observe = () => {
        container
          .querySelectorAll("[data-paragraph-id]")
          .forEach((el) => observer!.observe(el));
      };
      observe();

      // MutationObserver picks up lazy-loaded chunks
      mo = new MutationObserver(observe);
      mo.observe(container, { childList: true, subtree: true });
    };

    const trySetup = () => {
      const container = containerRef.current;
      if (!container) return false;
      setup(container);
      return true;
    };

    if (!trySetup()) {
      // A6: Container not yet mounted — poll until available
      pollId = setInterval(() => {
        if (trySetup() && pollId !== null) {
          clearInterval(pollId);
          pollId = null;
        }
      }, 200);
    }

    return () => {
      if (pollId !== null) clearInterval(pollId);
      observer?.disconnect();
      mo?.disconnect();
      intersectionMap.current.clear();
    };
  }, [containerRef]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── 2-second flush interval ───────────────────────────────────────────────
  useEffect(() => {
    if (!active) return;

    // Initialise window start when telemetry activates
    windowStart.current = Date.now();

    // A7: Sync starting scroll position so the first flush does not fire a
    // false-positive "scroll=0 but progress changed" warning for sessions
    // that resume mid-document.
    const container = containerRef.current;
    if (container) {
      const range = container.scrollHeight - container.clientHeight;
      prevProgressRatio.current = range > 0 ? container.scrollTop / range : 0;
      lastScrollTop.current = container.scrollTop;
    }

    const id = setInterval(flush, FLUSH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [active, flush, containerRef]);

  return { lastBatch, collecting: active, warnings };
}
