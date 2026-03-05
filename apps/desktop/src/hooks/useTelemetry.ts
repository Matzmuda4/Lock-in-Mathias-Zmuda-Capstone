/**
 * useTelemetry — Phase 5 Telemetry Logging v1
 *
 * Collects user interaction signals every 2 seconds and sends an aggregated
 * batch to POST /activity/batch.  Pure-math helpers are exported so they can
 * be unit-tested independently.
 *
 * Signals collected:
 *   scroll_delta_sum          net signed scroll distance (px)
 *   scroll_delta_abs_sum      total absolute scroll distance (px)
 *   scroll_event_count        raw scroll event count
 *   scroll_direction_changes  direction reversals (backtracking proxy)
 *   scroll_pause_seconds      seconds since last scroll (capped 60 s)
 *   idle_seconds              seconds since any interaction (capped 60 s)
 *   mouse_path_px             total cursor path length (px)
 *   mouse_net_px              straight-line net displacement (px)
 *   window_focus_state        "focused" | "blurred"
 *   current_paragraph_id      data-paragraph-id of most-visible paragraph
 *   current_chunk_index       chunk index of most-visible chunk
 *   viewport_progress_ratio   scrollTop / (scrollHeight - clientHeight)
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
}

/**
 * Given a list of {paragraphId, ratio} pairs, return the id and chunkIndex
 * of the element with the highest intersection ratio >= 0.6.
 * Returns null when no element meets the threshold.
 */
export function selectCurrentParagraph(entries: IntersectionEntry[]): {
  paragraphId: string | null;
  chunkIndex: number | null;
} {
  const candidates = entries.filter((e) => e.ratio >= 0.6);
  if (candidates.length === 0) return { paragraphId: null, chunkIndex: null };
  const best = candidates.reduce((a, b) => (b.ratio > a.ratio ? b : a));
  return { paragraphId: best.paragraphId, chunkIndex: best.chunkIndex };
}

// ─── Hook ────────────────────────────────────────────────────────────────────

const FLUSH_INTERVAL_MS = 2000;
const MAX_PAUSE_CAP_S = 60;
const IDLE_THRESHOLD_MS = 5000;
const DEV = (import.meta as unknown as { env?: { DEV?: boolean } }).env?.DEV ?? false;

export interface UseTelemetryOptions {
  sessionId: number;
  token: string | null;
  /** Pass true while the session is active (not paused / ended). */
  active: boolean;
  /** Ref to the scrollable reader container element. */
  containerRef: React.RefObject<HTMLElement | null>;
}

export interface UseTelemetryReturn {
  /** The last batch that was sent (for the debug panel). */
  lastBatch: TelemetryBatch | null;
  /** Whether telemetry is currently being collected. */
  collecting: boolean;
}

export function useTelemetry({
  sessionId,
  token,
  active,
  containerRef,
}: UseTelemetryOptions): UseTelemetryReturn {
  const activeRef = useRef(active);
  activeRef.current = active;

  const tokenRef = useRef(token);
  tokenRef.current = token;

  // ── Accumulators (reset each flush) ──────────────────────────────────────
  const scrollDeltaSum = useRef(0);
  const scrollDeltaAbsSum = useRef(0);
  const scrollEventCount = useRef(0);
  const scrollDirectionChanges = useRef(0);
  const lastScrollTime = useRef<number>(Date.now());
  const lastScrollDir = useRef<"down" | "up" | null>(null);
  const lastScrollY = useRef<number>(0);

  const lastInteraction = useRef<number>(Date.now());
  const mousePoints = useRef<Point[]>([]);
  const mouseStart = useRef<Point>({ x: 0, y: 0 });

  const windowFocused = useRef(true);

  // ── IntersectionObserver state ────────────────────────────────────────────
  const intersectionMap = useRef<Map<string, IntersectionEntry>>(new Map());
  const currentParagraphId = useRef<string | null>(null);
  const currentChunkIndex = useRef<number | null>(null);

  // ── Exposed to UI ──────────────────────────────────────────────────────────
  const [lastBatch, setLastBatch] = useState<TelemetryBatch | null>(null);

  // ── Reset accumulators ────────────────────────────────────────────────────
  const resetAccumulators = useCallback(() => {
    scrollDeltaSum.current = 0;
    scrollDeltaAbsSum.current = 0;
    scrollEventCount.current = 0;
    scrollDirectionChanges.current = 0;
    mousePoints.current = [];
    const container = containerRef.current;
    mouseStart.current = { x: 0, y: 0 };
    // capture current scrollY as new baseline
    if (container) lastScrollY.current = container.scrollTop;
    else lastScrollY.current = window.scrollY;
  }, [containerRef]);

  // ── Flush: build and send one batch ───────────────────────────────────────
  const flush = useCallback(() => {
    if (!activeRef.current || !tokenRef.current) return;

    const now = Date.now();
    const container = containerRef.current;

    // Scroll pause: seconds since last scroll, capped
    const scrollPauseSec = Math.min(
      (now - lastScrollTime.current) / 1000,
      MAX_PAUSE_CAP_S,
    );

    // Idle: seconds since any interaction, capped
    const idleSec = Math.min(
      (now - lastInteraction.current) / 1000,
      MAX_PAUSE_CAP_S,
    );

    // Mouse stats
    const { pathPx, netPx } = computeMouseStats(mousePoints.current);

    // Viewport progress ratio
    let viewportProgress = 0;
    if (container) {
      const range = container.scrollHeight - container.clientHeight;
      viewportProgress = range > 0 ? Math.min(1, container.scrollTop / range) : 0;
    }

    const batch: TelemetryBatch = {
      session_id: sessionId,
      scroll_delta_sum: scrollDeltaSum.current,
      scroll_delta_abs_sum: scrollDeltaAbsSum.current,
      scroll_event_count: scrollEventCount.current,
      scroll_direction_changes: scrollDirectionChanges.current,
      scroll_pause_seconds: scrollPauseSec,
      idle_seconds: idleSec,
      mouse_path_px: pathPx,
      mouse_net_px: netPx,
      window_focus_state: windowFocused.current ? "focused" : "blurred",
      current_paragraph_id: currentParagraphId.current,
      current_chunk_index: currentChunkIndex.current,
      viewport_progress_ratio: viewportProgress,
      client_timestamp: new Date().toISOString(),
    };

    if (DEV) {
      console.debug("[Telemetry] batch →", batch);
    }

    activityService.postBatch(tokenRef.current, batch);
    setLastBatch(batch);
    resetAccumulators();
  }, [sessionId, containerRef, resetAccumulators]);

  // ── Event listeners ────────────────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;

    const onScroll = (e: Event) => {
      if (!activeRef.current) return;
      const target = e.currentTarget as HTMLElement;
      const currentY = target ? target.scrollTop : window.scrollY;
      const dy = currentY - lastScrollY.current;

      scrollDeltaSum.current += dy;
      scrollDeltaAbsSum.current += Math.abs(dy);
      scrollEventCount.current += 1;

      const dir: "down" | "up" = dy >= 0 ? "down" : "up";
      if (lastScrollDir.current !== null && dir !== lastScrollDir.current) {
        scrollDirectionChanges.current += 1;
      }
      lastScrollDir.current = dir;
      lastScrollY.current = currentY;
      lastScrollTime.current = Date.now();
      lastInteraction.current = Date.now();
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!activeRef.current) return;
      lastInteraction.current = Date.now();
      const pt: Point = { x: e.clientX, y: e.clientY };
      if (mousePoints.current.length === 0) {
        mouseStart.current = pt;
      }
      mousePoints.current.push(pt);
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

    if (container) {
      container.addEventListener("scroll", onScroll, { passive: true });
    }
    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("keydown", onKeyDown, { passive: true });
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);

    return () => {
      if (container) container.removeEventListener("scroll", onScroll);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
    };
    // containerRef.current is stable; re-run if container identity changes
  }, [containerRef]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── IntersectionObserver: track visible paragraph ─────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const el = entry.target as HTMLElement;
          const paragraphId = el.dataset.paragraphId;
          if (!paragraphId) return;
          const chunkIndex = el.dataset.chunkIndex != null
            ? parseInt(el.dataset.chunkIndex, 10)
            : null;
          intersectionMap.current.set(paragraphId, {
            paragraphId,
            chunkIndex,
            ratio: entry.intersectionRatio,
          });
        });

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
        threshold: [0, 0.25, 0.5, 0.6, 0.75, 1.0],
      },
    );

    // Observe all paragraphs already in the container
    const observe = () => {
      container
        .querySelectorAll("[data-paragraph-id]")
        .forEach((el) => observer.observe(el));
    };
    observe();

    // MutationObserver to pick up lazy-loaded chunks
    const mo = new MutationObserver(observe);
    mo.observe(container, { childList: true, subtree: true });

    return () => {
      observer.disconnect();
      mo.disconnect();
    };
  }, [containerRef]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── 2-second flush interval ───────────────────────────────────────────────
  useEffect(() => {
    if (!active) return;
    const id = setInterval(flush, FLUSH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [active, flush]);

  return { lastBatch, collecting: active };
}
