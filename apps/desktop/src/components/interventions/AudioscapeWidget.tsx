/**
 * AudioscapeWidget
 *
 * Rendered in the AssistantPanel when an `ambient_sound` intervention fires.
 * Automatically starts playback on mount (auto-play of the initially-suggested
 * track, defaulting to Nature Sounds).
 *
 * - 3-minute progress bar per track; switching tracks resets the bar.
 * - Pause / resume button pauses both audio and progress simultaneously.
 * - forcePause prop: parent sets this true during a break suggestion so the
 *   audio is silenced for the break duration without the widget being dismissed.
 * - Volume slider for immediate volume control.
 * - Three track switcher buttons.
 * - Dismiss (✕): stops audio and acknowledges the intervention.
 */

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ActiveIntervention } from "../../services/interventionService";

// ─── Track catalogue ──────────────────────────────────────────────────────────

const TRACKS = [
  { id: "nature",     label: "Nature Sounds",   src: "/GamifiedIcons/NatureSounds.mp3"   },
  { id: "pink_rain",  label: "Pink Noise Rain",  src: "/GamifiedIcons/PinkNoiseRain.mp3"  },
  { id: "brown_rain", label: "Brown Noise Rain", src: "/GamifiedIcons/BrownNoiseRain.mp3" },
] as const;

type TrackId = (typeof TRACKS)[number]["id"];

const SOUND_TO_TRACK: Record<string, TrackId> = {
  ambient_shift:   "nature",
  nature:          "nature",
  pink_noise:      "pink_rain",
  pink_noise_rain: "pink_rain",
  brown_noise:     "brown_rain",
  brown_noise_rain:"brown_rain",
};

const DURATION_S = 3 * 60; // 180 seconds

// ─── Props ────────────────────────────────────────────────────────────────────

export interface AudioscapeWidgetProps {
  intervention: ActiveIntervention;
  onDismiss:    (id: number) => void;
  /** When true (e.g. during a break) audio is forcibly paused without dismissal. */
  forcePause?:  boolean;
}

// ─── Colours ──────────────────────────────────────────────────────────────────

const ACCENT = "#6366f1";
const LIGHT  = "#eef2ff";
const BORDER = "#c7d2fe";
const TEXT   = "#312e81";

// ─── Keyframes (once at module load) ──────────────────────────────────────────

if (typeof document !== "undefined") {
  const KF_ID = "audioscape-kf";
  if (!document.getElementById(KF_ID)) {
    const t = document.createElement("style");
    t.id = KF_ID;
    t.textContent = `
      @keyframes as-pulse { 0%,100%{opacity:1} 50%{opacity:0.45} }
      .as-track-btn:hover  { background: #e0e7ff !important; }
      .as-dismiss:hover    { color: #374151 !important; }
      .as-play-btn:hover   { filter: brightness(1.1); }
    `;
    document.head.appendChild(t);
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function AudioscapeWidget({
  intervention,
  onDismiss,
  forcePause = false,
}: AudioscapeWidgetProps) {
  const id = intervention.intervention_id!;

  const initialTrackId = useMemo<TrackId>(() => {
    const sound = (intervention.content as Record<string, string> | null)?.sound ?? "";
    return SOUND_TO_TRACK[sound] ?? "nature";
  }, [intervention.content]);

  const [trackId,   setTrackId]   = useState<TrackId>(initialTrackId);
  const [playing,   setPlaying]   = useState(true);
  const [remaining, setRemaining] = useState(DURATION_S);
  const [volume,    setVolume]    = useState(0.5);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const track = useMemo(() => TRACKS.find((t) => t.id === trackId)!, [trackId]);

  const clearTimer = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  }, []);

  const startTimer = useCallback(() => {
    clearTimer();
    timerRef.current = setInterval(() => {
      setRemaining((r) => {
        if (r <= 1) { clearTimer(); return 0; }
        return r - 1;
      });
    }, 1000);
  }, [clearTimer]);

  // ── Create / swap audio when track changes ────────────────────────────────
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
    }

    const audio = new Audio(track.src);
    audio.loop   = false;
    audio.volume = volume;
    audioRef.current = audio;

    setRemaining(DURATION_S);

    if (playing && !forcePause) {
      audio.play().catch(() => setPlaying(false));
      startTimer();
    }

    audio.addEventListener("ended", () => {
      clearTimer();
      setRemaining(0);
      setPlaying(false);
    });

    return () => {
      audio.pause();
      audio.src = "";
      clearTimer();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trackId]);

  // ── Sync play/pause with user toggle ─────────────────────────────────────
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing && !forcePause) {
      audio.play().catch(() => setPlaying(false));
      if (remaining > 0) startTimer();
    } else {
      audio.pause();
      clearTimer();
    }
  }, [playing]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Force pause from parent (e.g. break suggestion confirmed) ────────────
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (forcePause) {
      audio.pause();
      clearTimer();
    } else if (playing && remaining > 0) {
      audio.play().catch(() => setPlaying(false));
      startTimer();
    }
  }, [forcePause]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Auto-dismiss when time runs out ──────────────────────────────────────
  useEffect(() => {
    if (remaining === 0) onDismiss(id);
  }, [remaining]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => {
    audioRef.current?.pause();
    clearTimer();
  }, [clearTimer]);

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleTogglePlay = useCallback(() => setPlaying((p) => !p), []);

  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    setVolume(v);
    if (audioRef.current) audioRef.current.volume = v;
  }, []);

  const handleSwitchTrack = useCallback((newId: TrackId) => {
    if (newId !== trackId) setTrackId(newId);
  }, [trackId]);

  const handleDismiss = useCallback(() => {
    audioRef.current?.pause();
    clearTimer();
    onDismiss(id);
  }, [id, onDismiss, clearTimer]);

  const progress = 1 - remaining / DURATION_S;
  const effectivelyPlaying = playing && !forcePause;

  return (
    <div style={s.card}>

      {/* Header */}
      <div style={s.header}>
        <div style={s.titleRow}>
          <img src="/GamifiedIcons/AudioscapeIcon.png" alt="Audioscape" style={s.icon} />
          <div>
            <span style={s.overline}>AUDIOSCAPE</span>
            <p style={s.title}>{track.label}</p>
          </div>
        </div>
        <button className="as-dismiss" style={s.dismiss}
          onClick={handleDismiss} type="button" aria-label="Stop audioscape">✕</button>
      </div>

      {/* Progress bar — fills over 3 minutes */}
      <div style={s.progressBg}>
        <div style={{ ...s.progressFill, width: `${progress * 100}%` }} />
      </div>

      {/* Play/pause row */}
      <div style={s.controlRow}>
        <button
          className="as-play-btn"
          style={{
            ...s.playBtn,
            background: effectivelyPlaying ? ACCENT : "#e0e7ff",
            color:      effectivelyPlaying ? "#fff"  : ACCENT,
            opacity:    forcePause ? 0.5 : 1,
          }}
          type="button"
          onClick={handleTogglePlay}
          disabled={forcePause}
          aria-label={effectivelyPlaying ? "Pause" : "Play"}
        >
          {effectivelyPlaying
            ? <span style={{ animation: "as-pulse 1.6s ease-in-out infinite" }}>■</span>
            : "▶"}
        </button>
        {forcePause && (
          <span style={{ fontSize: "11px", color: "#9ca3af", fontStyle: "italic" }}>
            Paused during break
          </span>
        )}
      </div>

      {/* Volume slider */}
      <div style={s.volumeRow}>
        <span style={s.volIcon}>🔈</span>
        <input
          type="range" min={0} max={1} step={0.01}
          value={volume} onChange={handleVolumeChange}
          style={s.slider} aria-label="Volume"
          disabled={forcePause}
        />
        <span style={s.volIcon}>🔊</span>
      </div>

      {/* Track switcher */}
      <div style={s.trackRow}>
        {TRACKS.map((t) => (
          <button
            key={t.id}
            className="as-track-btn"
            style={{
              ...s.trackBtn,
              background: trackId === t.id ? ACCENT : "transparent",
              color:      trackId === t.id ? "#fff"  : ACCENT,
              border:     `1.5px solid ${trackId === t.id ? ACCENT : BORDER}`,
            }}
            type="button"
            onClick={() => handleSwitchTrack(t.id as TrackId)}
            disabled={forcePause}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  card: {
    background: LIGHT,
    border: `1.5px solid ${BORDER}`,
    borderRadius: "14px",
    padding: "13px 12px 14px",
    marginBottom: "12px",
    userSelect: "none",
  },
  header: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    marginBottom: "10px",
    gap: "8px",
  },
  titleRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  icon: {
    width: "36px",
    height: "36px",
    objectFit: "contain",
    borderRadius: "8px",
    background: "#e0e7ff",
    padding: "4px",
    flexShrink: 0,
  },
  overline: {
    display: "block",
    fontSize: "9.5px",
    fontWeight: 700,
    color: ACCENT,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
  },
  title: {
    margin: 0,
    fontSize: "13px",
    fontWeight: 700,
    color: TEXT,
    lineHeight: 1.2,
  },
  dismiss: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "15px",
    color: "#9ca3af",
    padding: "2px 4px",
    borderRadius: "5px",
    lineHeight: 1,
    flexShrink: 0,
    transition: "color 0.15s",
  },
  progressBg: {
    height: "5px",
    background: "#ddd6fe",
    borderRadius: "3px",
    overflow: "hidden",
    marginBottom: "12px",
  },
  progressFill: {
    height: "100%",
    background: ACCENT,
    borderRadius: "3px",
    transition: "width 1s linear",
  },
  controlRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    marginBottom: "10px",
  },
  playBtn: {
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    border: "none",
    cursor: "pointer",
    fontSize: "12px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    transition: "background 0.2s, color 0.2s, filter 0.15s, opacity 0.2s",
  },
  volumeRow: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    marginBottom: "10px",
  },
  volIcon: {
    fontSize: "13px",
    flexShrink: 0,
  },
  slider: {
    flex: 1,
    accentColor: ACCENT,
    cursor: "pointer",
    height: "4px",
  },
  trackRow: {
    display: "flex",
    gap: "5px",
    flexWrap: "wrap",
  },
  trackBtn: {
    flex: "1 1 0",
    minWidth: 0,
    padding: "5px 4px",
    borderRadius: "8px",
    fontSize: "10px",
    fontWeight: 600,
    cursor: "pointer",
    transition: "background 0.18s, color 0.18s",
    textAlign: "center",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
};
