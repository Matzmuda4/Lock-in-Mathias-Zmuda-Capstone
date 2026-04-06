/** Badge identifiers — kept in sync with the backend training dataset badge IDs. */
export type BadgeId =
  | "first_focus_streak"
  | "deep_reader"
  | "focus_master"
  | "reading_marathon"
  | "comeback_kid"
  | "no_distraction";

export interface BadgeDef {
  id:          BadgeId;
  label:       string;
  description: string; // shown in the popup
  icon:        string; // public path, e.g. "/GamifiedIcons/2.svg"
  xp:          number; // bonus XP awarded when badge is earned
}

export const BADGE_DEFS: BadgeDef[] = [
  {
    id:          "first_focus_streak",
    label:       "First Focus Streak",
    description: "5 consecutive focused windows — you're finding your rhythm!",
    icon:        "/GamifiedIcons/2.svg",
    xp:          10,
  },
  {
    id:          "deep_reader",
    label:       "Deep Reader",
    description: "8 consecutive focused windows — you're deep in the zone.",
    icon:        "/GamifiedIcons/3.svg",
    xp:          15,
  },
  {
    id:          "focus_master",
    label:       "Focus Master",
    description: "4 consecutive hyperfocused windows — your mind is locked in!",
    icon:        "/GamifiedIcons/4.svg",
    xp:          20,
  },
  {
    id:          "reading_marathon",
    label:       "Reading Marathon",
    description: "You've been reading for 15 minutes — incredible endurance!",
    icon:        "/GamifiedIcons/5.svg",
    xp:          20,
  },
  {
    id:          "comeback_kid",
    label:       "Comeback Kid",
    description: "You drifted but refocused and held it for 3+ windows — resilience rewarded!",
    icon:        "/GamifiedIcons/6.svg",
    xp:          10,
  },
  {
    id:          "no_distraction",
    label:       "No Distraction",
    description: "12 straight windows without losing focus — absolutely flawless!",
    icon:        "/GamifiedIcons/7.svg",
    xp:          15,
  },
];
