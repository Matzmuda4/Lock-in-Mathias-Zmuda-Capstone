/**
 * InterventionList.tsx
 * ─────────────────────
 * Renders the list of currently active foreground interventions inside the
 * AssistantPanel.
 *
 * Open-Closed: to add a new intervention type, create a new Card component
 * and add one case to the switch below — nothing else changes.
 *
 * Currently renders:
 *   focus_point          → FocusPointCard
 *   re_engagement        → ReEngagementCard
 *   comprehension_check  → ComprehensionCheckCard
 *
 * Not yet rendered (coming in next phases):
 *   section_summary, text_reformat, break_suggestion,
 *   gamification, ambient_sound, chime
 */

import type { ActiveIntervention } from "../../services/interventionService";
import { FocusPointCard } from "./FocusPointCard";
import { ReEngagementCard } from "./ReEngagementCard";
import { ComprehensionCheckCard } from "./ComprehensionCheckCard";

interface InterventionListProps {
  interventions: ActiveIntervention[];
  onDismiss:     (id: number) => void;
}

/** Types that are fully implemented as UI cards in this phase. */
const RENDERED_TYPES = new Set([
  "focus_point",
  "re_engagement",
  "comprehension_check",
]);

export function InterventionList({ interventions, onDismiss }: InterventionListProps) {
  // Only render types with a UI component; filter out non-text/passive types
  // (chime, ambient_sound, gamification) which are handled elsewhere.
  const visible = interventions.filter(
    (i) => i.type !== null && RENDERED_TYPES.has(i.type),
  );

  if (visible.length === 0) return null;

  return (
    <div>
      {visible.map((intervention) => {
        switch (intervention.type) {
          case "focus_point":
            return (
              <FocusPointCard
                key={intervention.intervention_id}
                intervention={intervention}
                onDismiss={onDismiss}
              />
            );
          case "re_engagement":
            return (
              <ReEngagementCard
                key={intervention.intervention_id}
                intervention={intervention}
                onDismiss={onDismiss}
              />
            );
          case "comprehension_check":
            return (
              <ComprehensionCheckCard
                key={intervention.intervention_id}
                intervention={intervention}
                onDismiss={onDismiss}
              />
            );
          default:
            return null;
        }
      })}
    </div>
  );
}
