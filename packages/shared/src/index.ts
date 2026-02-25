// Shared TypeScript types — populated as we build each feature

export type SessionMode = "baseline" | "adaptive";

export type ActivityEventType =
  | "scroll_forward"
  | "scroll_backward"
  | "idle"
  | "blur"
  | "focus"
  | "heartbeat";
