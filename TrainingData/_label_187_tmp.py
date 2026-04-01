#!/usr/bin/env python3
import json
from pathlib import Path

BASE = Path(__file__).parent

# Session 187 — user 249 (real session 2026-03-31T23:19)
# Baseline: wpm=521.5, idle_mean=0.464, idle_std=0.4461
# stag_mu=(11+0.5*5)/30=0.45, regress floor mu=0.05, std=0.06
# idle_damp threshold=0.40, stag_damp threshold=0.65
# AI panel was open throughout (panel_share_30s=1.0), panel_interaction_share shows active use.
#
# Session arc:
#   p0       : focused open (low idle, positive progress)
#   p1-p5    : peak hyperfocus (z_skim=2.2-3.0, pace 2.1-2.9x, near-zero idle)
#   p6       : hyperfocus decaying (z_skim=2.2 but z_pause=3.0)
#   p7-p10   : cognitive overload crash (regression, acute thrash at p10)
#   p11-p12  : mixed recovery
#   p13      : drifting onset (focus_loss=1.562)
#   p14-p16  : heavy tab-switching (33-47% away)
#   p17      : focused recovery
#   p18-p20  : overload rereading (z_regress=3.0, panel opens)
#   p21-p22  : focused AI-assisted recovery
#   p23-p24  : mild overload / focused
#   p25-p26  : focused -> slight overload
#   p27-p29  : second acute overload + recovery
#   p30-p33  : focused
#   p34-p36  : overload / acute episode
#   p37-p38  : focused
#   p39-p42  : heavy tab-switching (26-53% away)
#   p43-p45  : focused recovery
#   p46-p49  : intensive rereading (z_regress=3.0)
#   p50-p52  : extreme freeze-thrash cycles
#   p53-p55  : panel-assisted focused recovery
#   p56-p58  : overload rereading
#   p59-p61  : heavy tab-switching + stagnation
#   p62      : partial return / drifting

timestamps = {
    0:"2026-03-31T23:19:14.250747+00:00", 1:"2026-03-31T23:19:24.266728+00:00",
    2:"2026-03-31T23:19:34.237355+00:00", 3:"2026-03-31T23:19:44.238361+00:00",
    4:"2026-03-31T23:19:54.264349+00:00", 5:"2026-03-31T23:20:04.245738+00:00",
    6:"2026-03-31T23:20:14.242743+00:00", 7:"2026-03-31T23:20:24.278593+00:00",
    8:"2026-03-31T23:20:34.252983+00:00", 9:"2026-03-31T23:20:44.241313+00:00",
    10:"2026-03-31T23:20:54.280627+00:00", 11:"2026-03-31T23:21:04.254841+00:00",
    12:"2026-03-31T23:21:14.247085+00:00", 13:"2026-03-31T23:21:24.266606+00:00",
    14:"2026-03-31T23:21:34.841309+00:00", 15:"2026-03-31T23:21:44.255231+00:00",
    16:"2026-03-31T23:21:54.278883+00:00", 17:"2026-03-31T23:22:04.246736+00:00",
    18:"2026-03-31T23:22:14.246703+00:00", 19:"2026-03-31T23:22:24.263726+00:00",
    20:"2026-03-31T23:22:34.246482+00:00", 21:"2026-03-31T23:22:44.246486+00:00",
    22:"2026-03-31T23:22:54.263160+00:00", 23:"2026-03-31T23:23:04.245140+00:00",
    24:"2026-03-31T23:23:14.248591+00:00", 25:"2026-03-31T23:23:24.269436+00:00",
    26:"2026-03-31T23:23:34.242243+00:00", 27:"2026-03-31T23:23:44.245309+00:00",
    28:"2026-03-31T23:23:54.264234+00:00", 29:"2026-03-31T23:24:04.250042+00:00",
    30:"2026-03-31T23:24:14.326963+00:00", 31:"2026-03-31T23:24:24.269645+00:00",
    32:"2026-03-31T23:24:34.245554+00:00", 33:"2026-03-31T23:24:44.249416+00:00",
    34:"2026-03-31T23:24:54.279066+00:00", 35:"2026-03-31T23:25:04.256048+00:00",
    36:"2026-03-31T23:25:14.254633+00:00", 37:"2026-03-31T23:25:24.278579+00:00",
    38:"2026-03-31T23:25:34.258599+00:00", 39:"2026-03-31T23:25:45.245016+00:00",
    40:"2026-03-31T23:25:54.290245+00:00", 41:"2026-03-31T23:26:04.257536+00:00",
    42:"2026-03-31T23:26:14.257878+00:00", 43:"2026-03-31T23:26:24.278356+00:00",
    44:"2026-03-31T23:26:34.263531+00:00", 45:"2026-03-31T23:26:44.256721+00:00",
    46:"2026-03-31T23:26:54.279310+00:00", 47:"2026-03-31T23:27:04.265704+00:00",
    48:"2026-03-31T23:27:14.255808+00:00", 49:"2026-03-31T23:27:24.282403+00:00",
    50:"2026-03-31T23:27:34.257621+00:00", 51:"2026-03-31T23:27:44.259897+00:00",
    52:"2026-03-31T23:27:54.279580+00:00", 53:"2026-03-31T23:28:04.285174+00:00",
    54:"2026-03-31T23:28:14.260695+00:00", 55:"2026-03-31T23:28:24.283762+00:00",
    56:"2026-03-31T23:28:34.257314+00:00", 57:"2026-03-31T23:28:44.260626+00:00",
    58:"2026-03-31T23:28:54.279937+00:00", 59:"2026-03-31T23:29:05.244707+00:00",
    60:"2026-03-31T23:29:14.260659+00:00", 61:"2026-03-31T23:29:24.278608+00:00",
    62:"2026-03-31T23:29:34.246260+00:00",
}

# (seq, focused, drifting, hyperfocused, cognitive_overload, primary, notes)
labels_data = [
    (0, 72,12,5,11, "focused",
     "Incomplete window (n_batches=4, conf=0.27); labels conservative. Despite being 27% full, progress is already positive (0.0071) and idle_ratio_mean=0.029 -- far below user baseline of 0.464 -- meaning the reader is immediately scrolling with near-zero idle. z_stag=0.333 from 3 paragraphs already covered. All other z-scores at zero. The session opened with strong forward momentum and very low idle."),
    (1, 10,3,72,15, "hyperfocused",
     "Partial window (n_batches=9, conf=0.60). z_skim=2.201, pace_ratio=2.101 (2.1x calibration) with idle=0.030 (near-zero -- well below idle_damp threshold 0.40) and stagnation=0.222 (below stag_damp threshold 0.65). The skim signal is genuinely undampened. 5 paragraphs, progress=0.0079. For this 521-WPM reader the within-user deviation is what matters: a clear early hyperfocused flow state is being established."),
    (2, 8,2,82,8, "hyperfocused",
     "z_skim=3.0 (maxed, undampened), pace_ratio=2.594 (2.6x calibration). idle=0.020 (essentially zero -- far below idle_damp threshold), stagnation=0.214 (far below stag_damp threshold). 8 paragraphs, progress=0.0108. Near-complete window. The reader is flying through content at 2.6x their own calibration pace with virtually no idle or stagnation. Sustained peak hyperfocus."),
    (3, 5,2,88,5, "hyperfocused",
     "z_skim=3.0, pace_ratio=2.617, idle=0.022, stagnation=0.188, 9 paragraphs, progress=0.0128. Full window (conf=1.0). Undampened maximum z_skim with near-zero idle, very low stagnation, excellent progress. The reader is covering 9 paragraphs at 2.6x calibration pace with essentially no pausing. Peak sustained hyperfocus -- the dominant attentional state is unambiguous."),
    (4, 5,2,88,5, "hyperfocused",
     "z_skim=3.0, pace_ratio=2.952 (nearly 3x calibration), idle=0.028, stagnation=0.188, 9 paragraphs, progress=0.0135. The pace ratio has increased further from p3 -- the reader is now at their highest scrolling speed in this session. All z-scores remain clean; z_skim is undampened at maximum. Sustained peak hyperfocus with increasing intensity."),
    (5, 8,5,80,7, "hyperfocused",
     "z_skim=3.0, pace_ratio=2.825. idle=0.214 -- still below the idle_damp threshold (0.40), preserving the undampened skim signal. long_pause_share=0.067 appearing for the first time. 7 paragraphs (narrowing from 9). The hyperfocus arc is softening slightly: idle is no longer near-zero and coverage has reduced, but z_skim remains maxed and the primary state is clearly still hyperfocused."),
    (6, 12,28,45,15, "hyperfocused",
     "z_skim=2.227 (partial dampening: idle=0.503 has crossed the idle_damp threshold 0.40, so idle_damp = 1-(0.503-0.40)/0.40 = 0.742). z_pause=3.0 (pause=3.644s -- very long!). pace_ratio=3.046 is still very fast but long pauses now appear between bursts. The hyperfocus is collapsing into a burst-freeze pattern: fast scrolling bursts alternating with prolonged freezes. engagement=0.200 (suppressed). Hyperfocus is the largest component but with substantial drifting."),
    (7, 12,18,0,70, "cognitive_overload",
     "pace_available=False. z_regress=1.928 (regress_rate=0.166 -- 17% backward batches), z_burst=1.279, z_stag=1.444, z_pause=3.0. idle=0.641. progress=-0.0064 (negative). The hyperfocus has ended: the reader has switched to backward scrolling, elevated stagnation, burstiness, and very long pauses with high idle. The reader overshot content during hyperfocus and is now actively rereading."),
    (8, 8,30,0,62, "cognitive_overload",
     "z_idle=0.727 (idle=0.788), z_regress=1.928 (regress=0.166), z_burst=1.479, z_pause=3.0. pause=6.087s. 4 paragraphs, progress=-0.0076. Very high idle fraction (79%) combined with continued regression and maximum pauses. The rereading is becoming less energetic -- increasing idle between scrolls suggests attentional fatigue mixed into the comprehension struggle."),
    (9, 5,33,0,62, "cognitive_overload",
     "z_idle=0.757 (idle=0.802), z_regress=1.755, z_burst=1.625, z_stag=1.583, z_pause=3.0. pause=6.513s. 4 paragraphs, progress=-0.0071. All disruption channels simultaneously elevated: very high idle, regression, burstiness, stagnation, and maximum pauses. The reader is stuck in a narrow zone oscillating backward with near-maximum pauses between bursts and near-total idle."),
    (10, 3,25,0,72, "cognitive_overload",
     "z_idle=0.967, z_burst=3.0 (maxed -- burstiness=3.445), z_stag=3.0 (maxed -- stagnation=0.938), z_pause=3.0. idle=0.895, 2 paragraphs, progress=0.0022. disrupt=0.784. Triple maximum simultaneous signals: near-total idle (89.5%), maximum burstiness, maximum stagnation. The reader is thrashing violently on 2 paragraphs between prolonged freeze periods -- a textbook acute cognitive overload episode on a specific impenetrable content segment."),
    (11, 20,35,0,45, "cognitive_overload",
     "z_idle=0.467, z_regress=0.221, z_burst=1.784, z_stag=1.167, z_pause=3.0. idle=0.673, progress=0.01372 (positive). 6 paragraphs. After the p10 acute episode, coverage has widened to 6 paragraphs and progress is strongly positive -- the acute lock has broken. But idle, burstiness, stagnation, and pauses remain elevated. Partial recovery with continuing disruption residuals."),
    (12, 28,32,0,40, "cognitive_overload",
     "z_idle=0.085, z_regress=0.221, z_burst=1.454, z_pause=3.0. idle=0.502, regress=0.063. 8 paragraphs, progress=0.003101. Mouse movement detected (25px). disrupt=0.241, engagement=0.488. Disruption is decreasing and coverage expanding to 8 paragraphs with positive progress. The acute overload episode is resolving, though elevated burstiness and maxed pauses persist in this transitional window."),
    (13, 15,52,0,33, "drifting",
     "z_focus_loss=1.562 (focus_loss_rate=0.125 -- first tab-switch detected!), z_regress=0.221, z_burst=1.362, z_pause=1.749. idle=0.353, 8 paragraphs, progress=0.00160. Mouse active (163px). The reader has begun switching to other applications (12.5% of window). Combined with still-elevated burstiness and regression, this signals the onset of attention fragmentation following the cognitive overload episode -- frustration driving escape behavior."),
    (14, 0,75,0,25, "drifting",
     "z_focus_loss=3.0 (focus_loss_rate=0.467 -- 47% of window away!), z_burst=2.843, z_stag=1.444, z_idle=0.438, z_pause=3.0. pace_ratio=0.429. 4 paragraphs, progress=-0.01062 (strongly negative). disrupt=0.901, engagement=0.0. Nearly half the window spent tabbed away; when present, the reader oscillates rapidly through a narrow zone. Severe drifting via sustained application-switching."),
    (15, 0,72,0,28, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.467 -- 46.7% away), z_stag=1.889, z_regress=0.458, z_idle=0.385, z_burst=1.690, z_pause=3.0. pace_ratio=0.275 (27.5% of calibration). 3 paragraphs, progress near-zero. Mouse very active (196px). disrupt=0.911, engagement=0.0. Tab-switching continues at near-peak. When present, the reader is reading at barely a quarter of calibration pace. Progress has stalled completely."),
    (16, 0,70,0,30, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.333 -- 33.3% away), z_idle=0.740 (idle=0.794), z_burst=2.507, z_pause=3.0. 3 paragraphs, progress near-zero. disrupt=0.897, engagement=0.0. Focus loss slightly reduced but very high idle (79%) persists when ostensibly present -- the reader may still be partly mentally absent even when the app is in focus. Tab-switching and idle combine to define a persistent drift state."),
    (17, 42,30,0,28, "focused",
     "z_focus_loss=0.781 (focus_loss=0.0625 -- mostly returned), z_stag=0.333, z_regress=0.392, z_burst=0.395, z_idle=0.155, z_pause=3.0. pace_ratio=0.786, pace_available=True. 6 paragraphs, progress=0.00157. disrupt=0.376, engagement=0.561. Focus loss has cleared to 6.25% -- the reader has largely returned. Engagement recovering to 0.561. Mild residual disruption signals across the board. Transitional recovery window."),
    (18, 15,10,0,75, "cognitive_overload",
     "z_regress=3.0 (regress_rate=0.259 -- 26% backward batches!), z_idle=0.036, z_pause=2.499. pace_available=False. 7 paragraphs, progress=-0.00115. No focus loss. The reader has returned from tabs and immediately begun intensive rereading -- 26% backward scrolling across 7 paragraphs with near-zero idle. Post-distraction remediation: the reader is compensating for missed content by systematically scrolling back."),
    (19, 15,5,0,80, "cognitive_overload",
     "z_regress=3.0 (regress_rate=0.304 -- 30% backward batches!), z_pause=0.036. 8 paragraphs, progress=0.000942. z_idle=0, z_burst=0. Near-zero idle, very short pauses -- the reader is racing backward through 8 paragraphs with near-continuous active scrolling. Maximum regression at maximum velocity. Systematic wide-scope backward review with near-zero idle between scrolls."),
    (20, 18,5,0,77, "cognitive_overload",
     "z_regress=3.0 (regress_rate=0.304), z_pause=0.122. 8 paragraphs, progress=0.000663. panel_interaction_share=0.25 (25% of window on AI panel!). Mouse active (207px). The intensive rereading continues at maximum intensity while the reader simultaneously consults the AI panel -- seeking external clarification for the material that drove the prior drift and overload cycle."),
    (21, 25,8,0,67, "cognitive_overload",
     "z_regress=1.962 (regression reducing from maximum), z_pause=0.423. 9 paragraphs, progress=0.004576. panel=0.25. Regression is decreasing to z=1.962 while coverage expands to 9 paragraphs with better positive progress. Panel interaction at 25% continuing. The AI assistance is helping: the reader is moving forward more effectively with ongoing backward verification."),
    (22, 48,10,0,42, "focused",
     "z_regress=0.724 (significantly reduced), z_burst=0.117, z_pause=0.652. 7 paragraphs, progress=0.003482. panel=0.3125 (31.25%!). Mouse very active (301px, efficiency 0.890). disrupt=0.216 (very low), engagement=0.462. Regression nearly resolved. AI panel use at session high (31.25%). The reader is in an AI-assisted recovery phase -- reading forward with light backward verification and heavy panel consultation."),
    (23, 32,8,0,60, "cognitive_overload",
     "z_regress=1.467 (regression ticking back up despite good coverage). z_pause=0.427. 9 paragraphs, progress=0.002896. panel=0.125 (reducing). Regression has increased somewhat as panel use drops -- the reader is attempting more independent comprehension but encountering content that still requires checking back. Wide coverage with backward sampling is present but z_regress=1.467 keeps overload primary."),
    (24, 52,12,0,36, "focused",
     "z_regress=0.463, z_pause=0.470. 7 paragraphs, progress=0.003042. panel=0.0625 (minimal). Mouse (94px). disrupt=0.196 (very low!), engagement=0.483. Regression nearly cleared. Very low panel use. Good forward progress. The reader has largely recovered to independent focused reading with only minor backward checking remaining."),
    (25, 50,28,0,22, "focused",
     "z_idle=0.228, z_burst=0.508, z_pause=1.451. 5 paragraphs, progress=0.00245. disrupt=0.215. No focus loss, no significant regression, no panel. Moderate signals throughout. Reading is settled -- somewhat elevated idle and pauses but no dominant disruption signal. Consistent forward movement."),
    (26, 33,25,0,42, "cognitive_overload",
     "z_idle=0.418, z_burst=1.240, z_stag=1.167, z_pause=3.0. idle=0.651, stagnation=0.625, burstiness=1.620, pause=4.543. 4 paragraphs, progress=0.00196. disrupt=0.396. Stagnation elevated (z=1.167), burstiness rising, idle elevated, very long pauses on only 4 paragraphs. The reader has slowed significantly and is dwelling more -- encountering denser content that is starting to challenge comprehension."),
    (27, 4,23,0,73, "cognitive_overload",
     "z_idle=0.972, z_burst=3.0 (maxed -- burstiness=2.711), z_stag=3.0 (maxed -- stagnation=0.938), z_pause=3.0. idle=0.898, stagnation=0.938, pause=7.532. 2 paragraphs. disrupt=0.785. A second acute overload episode: near-total idle (90%), maximum burstiness and stagnation, very long pauses on 2 paragraphs -- structurally identical to the episode at p10. The reader is thrashing on a specific content segment with prolonged freezes."),
    (28, 3,20,0,77, "cognitive_overload",
     "z_idle=0.923, z_burst=3.0 (burstiness=3.126 -- above the system cap), z_stag=3.0 (stagnation=1.0), z_pause=3.0. idle=0.876, pause=8.620. 1 paragraph, progress=0.000324. disrupt=0.778. Peak intensity of the second acute episode: stagnation at maximum (1.0), burstiness exceeding the cap, 87.6% idle, 8.6-second pauses, completely locked on 1 paragraph. Maximum thrashing on a single intractable content point."),
    (29, 30,28,0,42, "cognitive_overload",
     "z_idle=0.392, z_burst=2.859, z_stag=1.444, z_pause=3.0, z_pace=0.047. pace_ratio=1.022, pace_available=True. idle=0.639. 3 paragraphs, progress=0.003013. engagement=0.695 (recovering). Some recovery: pace is now available and positive progress resumes, but burstiness, stagnation, and pauses remain elevated. The acute episode is resolving into a mixed transitional state."),
    (30, 50,25,0,25, "focused",
     "z_burst=1.793, z_pace=0.735, z_pause=1.947. pace_ratio=0.716, pace_available=True. idle=0.402, burstiness=1.896, pause=2.663. 4 paragraphs, progress=0.003776. Mouse (85px). engagement=0.800 (ceiling). Engagement at maximum -- pace available below skim threshold signals careful reading. The transition into focused recovery: pace available, positive progress, engagement at ceiling despite mild burstiness."),
    (31, 62,15,0,23, "focused",
     "z_burst=1.178, z_pace=1.413. pace_ratio=0.526, pace_available=True. idle=0.222 (very low!), burstiness=1.589, pause=0.795. 5 paragraphs, progress=0.003990. Mouse (212px, efficiency 0.880). engagement=0.800. Very low idle, pace available, 5 paragraphs with good positive progress. Solid focused recovery -- mild residual burstiness is the only disruption signal remaining."),
    (32, 65,14,0,21, "focused",
     "z_burst=1.019, z_pace=0.617. pace_ratio=0.755, pace_available=True. idle=0.189 (very low!), burstiness=1.510, pause=0.728. 6 paragraphs, progress=0.002556. Mouse (212px). engagement=0.800. Continuing the clean focused recovery -- very low idle, pace available at 75.5% of calibration, 6 paragraphs covered, positive progress."),
    (33, 65,15,0,20, "focused",
     "z_regress=0.232, z_burst=0 (burstiness=0.678 -- below 1.0!), z_pause=0. pace_available=False. idle=0.224, regress=0.063, pause=0.595. 6 paragraphs, progress=0.00218. Mouse (132px). disrupt=0.181 (session low). Disruption at its lowest point in the session. Burstiness below 1.0, near-zero pauses, very low idle. A genuinely clean focused reading window."),
    (34, 30,15,0,55, "cognitive_overload",
     "z_regress=1.389, z_stag=1.000 (stagnation=0.600), z_pause=1.361. pace_available=False. idle=0.456, regress=0.133, stagnation=0.600, pause=2.142. 4 paragraphs, progress=0.00126. disrupt=0.362. Regression has returned (z=1.389), stagnation elevated (z=1.0), moderate pauses. After the clean p33 window, denser content has triggered renewed backward checking and stagnation."),
    (35, 5,22,0,73, "cognitive_overload",
     "z_idle=0.521, z_regress=1.389, z_burst=1.154, z_stag=3.0 (maxed -- stagnation=0.933), z_pause=3.0. idle=0.697, regress=0.133, stagnation=0.933, pause=4.755. 2 paragraphs, progress=0.000501. disrupt=0.747. A third acute overload episode: stagnation maxed on 2 paragraphs with elevated idle, regression, burstiness, and maximum pauses. The reader has narrowed to 2 paragraphs with acute comprehension difficulty."),
    (36, 8,15,0,77, "cognitive_overload",
     "z_idle=0.204, z_regress=2.291, z_burst=0.833, z_stag=2.000, z_pause=3.0. idle=0.555, regress=0.188, stagnation=0.750, pause=4.525. 2 paragraphs, progress=-0.00667 (negative). Mouse (147px). disrupt=0.651. High regression (z=2.291) with stagnation, long pauses, and negative progress concentrated on 2 paragraphs. Active rereading focused on a specific content zone with comprehension difficulty."),
    (37, 40,22,0,38, "focused",
     "z_regress=0.208, z_burst=0.933, z_pause=3.0. pace_available=False. idle=0.319, regress=0.063, burstiness=1.467, pause=3.845. 4 paragraphs, progress=-0.0055 (small negative). Mouse very active (284px, efficiency 0.877). disrupt=0.209, engagement=0.503. Disruption very low despite long pauses and small backward progress. Low idle and active mouse movement suggest engaged exploration. The long pauses here are likely deliberate reflection rather than stuck overload."),
    (38, 52,20,0,28, "focused",
     "z_regress=0.208, z_pause=0.099. pace_available=False. idle=0.092 (very low!), regress=0.063, burstiness=0.935, pause=1.019. 8 paragraphs, progress=-0.00413 (small negative). Mouse very active (406px, efficiency 0.746). disrupt=0.180 (near session low). Very low idle, near-normal burstiness, 8 paragraphs covered, short pauses. The low-efficiency mouse movement and wide coverage with minimal idle suggest active document navigation in a clean engaged state."),
    (39, 5,65,0,30, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.267 -- 26.7% away), z_burst=0.706, z_pause=2.159, z_pace=1.741. focus_loss=0.267, pace_ratio=0.453. 7 paragraphs, progress=0.003010. Mouse very active (415px, efficiency 0.721!). disrupt=0.668, engagement=0.0. Tab-switching has returned (26.7% away). Very low mouse efficiency (72%) suggests frustrated or erratic mouse movement. engagement=0.0 (suppressed by focus_loss=3.0). The third drift episode begins."),
    (40, 0,72,0,28, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.533 -- over half the window away!), z_burst=1.490, z_stag=1.889, z_idle=0.363, z_pause=3.0. 4 paragraphs. disrupt=0.889, engagement=0.0. More than half the window tabbed away; when present: high stagnation, elevated burstiness, maxed pauses. Severe drifting -- the reader is predominantly absent, and when present, remains unfocused."),
    (41, 0,70,0,30, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.500 -- exactly half away), z_burst=2.377, z_stag=2.000, z_idle=0.555, z_pace=1.950, z_pause=3.0. 2 paragraphs. disrupt=0.923, engagement=0.0. Half the window tabbed away; when present: extreme burstiness on 2 paragraphs, high stagnation, maximum pauses. disrupt=0.923. Simultaneous absence AND thrashing when present -- combined disruption near maximum."),
    (42, 5,65,0,30, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.313 -- 31.3% away), z_idle=0.230, z_burst=0.919, z_pause=3.0, z_regress=0.208. 5 paragraphs, progress=0.001235. disrupt=0.737, engagement=0.0. Focus loss reducing from peak (31% vs 50% in p41) -- some return is beginning. Coverage has expanded to 5 paragraphs. Still primarily absent but with increasing engagement when present."),
    (43, 52,25,0,23, "focused",
     "z_focus_loss=0.000 (returned!), z_idle=0.000, z_pause=1.207, z_regress=0.208, z_burst=0.526. idle=0.444, regress=0.063. 6 paragraphs, progress=0.001466. disrupt=0.196 (very low), engagement=0.503. Focus loss has fully cleared. Disruption at very low level. The reader has returned from the third tab-switching episode. Clean recovery momentum beginning."),
    (44, 50,25,0,25, "focused",
     "z_idle=0.194, z_burst=0.698, z_pause=1.109, z_regress=0.208. idle=0.551, regress=0.063, burstiness=1.349, pause=1.917. 6 paragraphs, progress=0.001147. disrupt=0.230, engagement=0.471. Mild moderate signals continuing the recovery. Reading proceeds with minor residual disruption from the prior drift episode."),
    (45, 45,25,0,30, "focused",
     "z_idle=0.020, z_burst=0.942, z_regress=0.797, z_pause=0.655. idle=0.473, regress=0.098, burstiness=1.471, pause=1.513. 5 paragraphs, progress near-zero. disrupt=0.254. Regression has ticked up slightly (z=0.797). Near-zero net progress. Mild signals throughout. The reader is checking backward while making minimal net forward movement -- mild overload content is present."),
    (46, 18,10,0,72, "cognitive_overload",
     "z_regress=3.0 (regress_rate=0.287 -- 28.7% backward batches!), z_burst=0.348, z_pause=0.318. idle=0.384, regress=0.287, burstiness=1.174, pause=1.213. 6 paragraphs, progress near-zero. disrupt=0.418. z_regress has exploded to maximum (3.0). The reader is doing intensive energetic backward scrolling (28.7% of batches) across 6 paragraphs with short pauses and moderate idle -- highly active rereading with low idle distinguishing it from passive drifting."),
    (47, 15,5,0,80, "cognitive_overload",
     "z_regress=3.0 (regress_rate=0.481 -- nearly half of all batches backward!), z_burst=1.305, z_pause=0. idle=0.154 (very low!), regress=0.481, burstiness=1.653, pause=0.515. 8 paragraphs, progress=-0.00847 (strongly negative). Near-zero idle, very short pauses -- the reader is racing backward through 8 paragraphs with near-continuous scrolling and almost 50% of batches in reverse direction. Maximum regression with maximum activity."),
    (48, 12,8,0,80, "cognitive_overload",
     "z_regress=3.0 (regress=0.446 -- 44.6% backward), z_burst=1.646, z_stag=0.333, z_pause=1.863. idle=0.350, regress=0.446, burstiness=1.823, pause=2.588. 6 paragraphs, progress=-0.00815. Continued maximum regression across 6 paragraphs (narrowing from 8). Burstiness is increasing and pauses lengthening as the reader concentrates the backward review on a narrowing zone."),
    (49, 5,20,0,75, "cognitive_overload",
     "z_regress=2.410, z_burst=2.231, z_stag=2.417, z_idle=0.373, z_pause=3.0. regress=0.195, stagnation=0.813, burstiness=2.116, pause=5.649. 4 paragraphs, progress=-0.00864. disrupt=0.786. The wide-scope rereading has transitioned into narrowing acute overload: regression + stagnation + burstiness + maximum pauses concentrated on 4 paragraphs. A fourth acute overload episode is forming."),
    (50, 2,18,0,80, "cognitive_overload",
     "z_idle=1.045, z_burst=3.0 (maxed -- burstiness=4.0, exceeding the cap), z_stag=3.0 (maxed -- stagnation=1.0), z_pause=3.0. idle=0.930, stagnation=1.0, pause=8.749. 1 paragraph, progress=0. disrupt=0.796. Session maximum simultaneous disruption: stagnation at absolute maximum (1.0), burstiness at system cap (4.0), z_idle=1.045, all three z-scores maxed. Completely locked on 1 paragraph with maximum possible oscillation and very long freezes."),
    (51, 3,62,0,35, "drifting",
     "z_idle=1.201, z_stag=3.0, z_pause=3.0, z_burst=0. idle=1.000 (100% idle!), stagnation=1.0, burstiness=0.000, pause=10.000 (system maximum). 1 paragraph, progress=0. The reader has collapsed from maximum burstiness into complete immobility: 100% idle, zero scrolling, 10-second pauses. Attentional withdrawal after the frantic oscillation exhaustion -- the same freeze-after-thrash pattern seen at p8 and p31 in session 186."),
    (52, 3,22,0,75, "cognitive_overload",
     "z_idle=0.969, z_burst=3.0 (maxed -- burstiness=4.0), z_stag=3.0 (maxed -- stagnation=1.0), z_pause=3.0. idle=0.896, stagnation=1.0, burstiness=4.0, regress=0.063, pause=8.422. 1 paragraph. disrupt=0.798. The freeze-thrash cycle repeats: maximum burstiness and stagnation resuming after the total stillness of p51. A second intense oscillation episode on the same single paragraph -- the reader cannot break out of this specific content lock."),
    (53, 18,22,0,60, "cognitive_overload",
     "z_idle=0.254, z_regress=0.208, z_burst=1.574, z_stag=3.0 (still maxed -- stagnation=1.0), z_pause=3.0. idle=0.594, burstiness=1.787, pause=5.575. 1 paragraph, progress near-zero. panel=0.125 (12.5% panel use!). disrupt=0.613, engagement=0.461. Still stagnation-maxed on 1 paragraph, but burstiness has reduced and the AI panel is now being opened (12.5% use). engagement=0.461 (much improved from recent packets). The reader is beginning to seek AI assistance to break the comprehension lock."),
    (54, 44,10,0,46, "cognitive_overload",
     "z_stag=2.833, z_burst=0.682, z_pause=3.0, z_pace=1.883. pace_ratio=0.425, pace_available=True. idle=0.396, stagnation=0.875, pause=3.737. 2 paragraphs. panel=0.375 (37.5%!). Mouse (195px). engagement=0.800 (ceiling). Major shift: AI panel now open 37.5% of the window -- session maximum panel use. engagement=0.800 (pace available below skim threshold = careful reading). Despite still-elevated stagnation (z=2.833) and long pauses, the heavy AI assistance is transforming the reading experience. Marginal overload primary due to stagnation level."),
    (55, 62,8,0,30, "focused",
     "z_stag=0.750, z_burst=0 (burstiness=0.925), z_pause=1.276, z_pace=1.896. pace_ratio=0.422, pace_available=True. idle=0.228 (low!), stagnation=0.563, burstiness=0.925, pause=2.066. 4 paragraphs, progress=0.002170. panel=0.375. disrupt=0.224, engagement=0.800. Stagnation nearly cleared. Low idle. 4 paragraphs with positive progress and heavy AI panel use. disrupt=0.224 (low). The AI-assisted recovery is clearly succeeding."),
    (56, 28,10,0,62, "cognitive_overload",
     "z_regress=2.295, z_burst=0.436, z_pause=0.862. pace_available=False. idle=0.243, regress=0.188, burstiness=1.218, pause=1.697. 4 paragraphs, progress near-zero. panel=0.3125 (31.25%). Mouse (65px). disrupt=0.356, engagement=0.337. Regression has jumped back (z=2.295, 18.8% backward) despite continued panel use (31.25%). The reader is rereading while consulting AI -- active comprehension effort continuing, but the backward checking required to sustain understanding is significant."),
    (57, 20,10,0,70, "cognitive_overload",
     "z_regress=2.601, z_stag=0.750, z_pause=0.623. pace_available=False. idle=0.387, regress=0.206, stagnation=0.563, burstiness=1.160, pause=1.485. 4 paragraphs, progress near-zero. No panel. disrupt=0.466. Regression elevated (z=2.601) with panel closed. The reader is rereading without AI support on 4 paragraphs with near-zero progress -- returning to independent comprehension effort on difficult material."),
    (58, 12,15,0,73, "cognitive_overload",
     "z_idle=0.164, z_regress=2.601, z_burst=1.001, z_stag=1.583, z_pause=1.696. idle=0.537, regress=0.206, stagnation=0.688, burstiness=1.501, pause=2.440. 4 paragraphs, progress=-0.00108. disrupt=0.633. Multiple simultaneous overload signals: elevated regression, stagnation, burstiness, and pauses all active on the same narrow 4-paragraph zone. The reader is locked in a sustained rereading loop."),
    (59, 0,60,0,40, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.267 -- 26.7% away), z_burst=1.397, z_stag=3.0 (maxed -- stagnation=1.0), z_idle=0.625, z_pause=3.0, z_regress=0.596. 1 paragraph, progress near-zero. disrupt=0.956 (near maximum). Session's highest disruption score: simultaneously tabbed away (26.7%) AND stagnation maxed on 1 paragraph when present, with maximum pauses. The reader has lost attention entirely -- both by fleeing the app and remaining frozen when present."),
    (60, 0,65,0,35, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.375 -- 37.5% away), z_burst=2.486, z_stag=2.833, z_idle=0.427, z_pause=3.0. 3 paragraphs, progress=0.002301. disrupt=0.947, engagement=0.0. Focus loss still at 37.5%. When present: near-maxed stagnation, extreme burstiness, maxed pauses. disrupt=0.947. When absent: away 37.5% of the window. Both channels of disruption remain at high intensity."),
    (61, 5,65,0,30, "drifting",
     "z_focus_loss=3.0 (focus_loss=0.375 -- 37.5% still away), z_burst=0.415, z_stag=0.750, z_pause=3.0. 7 paragraphs, progress=0.005654 (positive!). disrupt=0.750, engagement=0.0. Focus loss still maxed (37.5%) but burstiness and stagnation have reduced significantly. Wide 7-paragraph coverage with good positive progress when present -- the reader is beginning to re-engage during the present moments even while still partly absent."),
    (62, 15,55,0,30, "drifting",
     "z_focus_loss=2.343 (focus_loss=0.1875 -- 18.75%, reducing!), z_pause=2.018. pace_ratio=1.284, pace_available=True. idle=0.262, stagnation=0.250, burstiness=1.011. 11 paragraphs!, progress=0.01153. disrupt=0.523, engagement=0.175. Final packet: focus loss is reducing (18.75% vs 37.5%) with the widest paragraph coverage of the late session (11 paragraphs) and above-baseline pace. engagement suppressed by z_focus_loss=2.343 but the trajectory is clearly returning. The session closes in a gradual recovery from the final drift episode."),
]

# Validate
errors = []
for row in labels_data:
    seq, f, d, h, c, primary, notes = row
    total = f + d + h + c
    if total != 100:
        errors.append(f"p{seq}: sum={total}")
    label_dict = {"focused": f, "drifting": d, "hyperfocused": h, "cognitive_overload": c}
    expected = max(label_dict, key=label_dict.get)
    if expected != primary:
        errors.append(f"p{seq}: primary={primary} but argmax={expected} labels={label_dict}")

if errors:
    print("VALIDATION ERRORS:", errors)
    exit(1)

state_counts = {}
for row in labels_data:
    s = row[5]
    state_counts[s] = state_counts.get(s, 0) + 1
print(f"Validated {len(labels_data)} labels OK. Distribution: {state_counts}")

labelled_path = BASE / "labelled.jsonl"
lines = []
for row in labels_data:
    seq, f, d, h, c, primary, notes = row
    obj = {
        "session_id": 187,
        "packet_seq": seq,
        "window_end_at": timestamps[seq],
        "labels": {"focused": f, "drifting": d, "hyperfocused": h, "cognitive_overload": c},
        "primary_state": primary,
        "notes": notes,
    }
    lines.append(json.dumps(obj, ensure_ascii=False))

with open(labelled_path, "a") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"Appended {len(lines)} label rows to labelled.jsonl")
