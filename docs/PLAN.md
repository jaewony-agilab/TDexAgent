# Contact-Annotated Agentic Manipulation — Plan

Status: draft, 2026-10-14. Owner: Jaewon.
New project. No code, sensors or results are inherited from earlier work.

## 1. Claim

A frozen dexterous-hand VLA executor, a VLM judge and an LLM planner all consuming the same **contact-annotated RGB** (per-finger taxel force vectors projected onto the scene/wrist images) outperform the same framework on plain RGB, on contact-rich tasks where the executor alone succeeds 40–70% of the time.

Secondary claim (if compute allows): on the same model (RLDX-1), delivering touch as image-space annotations beats delivering it through the model's native tactile stream.

## 2. Platform

| Component | Choice | Why |
|---|---|---|
| Benchmark | GR-1 Tabletop (Isaac; Fourier GR-1 humanoid with dexterous hands) | Only hand benchmark with a first-party RLDX-1 checkpoint; 58.7% reported |
| Executor | RLDX-1-FT-GR1, frozen | Touch-blind as released; physics stream available for Phase 3 |
| Sensor | Isaac Lab contact sensors as taxels on fingertip and palm links, both hands | Fixed link-anchored locations; force vectors, no optics |
| Fallback | RoboCasa Kitchen + RLDX-1-FT-ROBOCASA (MuJoCo, Panda gripper) | If GR-1 bring-up fails Phase 0 gates |
| Judge | Open-weight VLM (Qwen-VL class), frozen; Haiku as the reference | Reproducible, free per call, fine-tunable later |
| Planner | Haiku, called once at start and on replan | Per-run and global cost caps, every call logged by purpose |
| Compute | Slurm, A100s | Policy server per job; fine-tunes in Phase 3 |

Prior art to cite up front: Visible Touch (CoRL 2026), TAP-VLA, AHA, SAFE, RoboDawn.

## 3. Phases

### Phase 0 — Platform bring-up (screening only)

Goal: RLDX-1 runs on GR-1 Tabletop inside a new evaluation harness with taxels fitted.

1. Install RLDX-1 (uv, Python 3.10) and its GR-1 Tabletop eval path; pin commits. Reproduce the published number on 3–4 tasks (1 seed, 20 episodes each). Stop if more than 10pp below.
2. Confirm the released GR-1 demos can be replayed deterministically in Isaac from stored state (needed for Phase 3 post-hoc overlays). Record yes/no; if no, Phase 3 needs fresh demos and its budget doubles.
3. Harness: episode loop around RLDX-1's inference server with hooks at subtask boundaries. Executor adapter behind the boundary interface `(boundary_type, frames, finger_summaries)`. Boundary for a hand = hand-closure command crossing (aggregate finger-joint command with hysteresis, then a settle window until the fingers stop), per hand; defined on the executor's action space, not on contacts. Privileged state (object poses, contacts) is read only by a recorder and an oracle, enforced by import isolation tests.
4. Taxels: contact sensors on each fingertip link and the palm, both hands, read every step. Fixed per-link anchors. Nothing in the benchmark or RLDX-1 is edited.
5. Sensor-cost check (step time with / without sensors) and a "does the policy still work" check: same 3–4 tasks, 5 init states.
6. Task selection: run every GR-1 Tabletop task once (1 seed, 20 episodes). Keep tasks with executor-alone SR in 30–70% and at least one grasp. Target 4 tasks.

Deliverable: Phase 0 report. Decision: proceed only if ≥3 tasks sit in the band; otherwise switch to the RoboCasa fallback (gripper, MuJoCo).

### Phase 1 — Overlay renderer and offline judge comparison (no training)

Goal: a renderer, and evidence that annotations change judge verdicts.

1. Renderer: per fingertip (and palm), project the link anchor from proprioception + camera matrices; draw one marker per finger (fixed per-finger color, same across hands with a left/right outline), arrow = aggregate force vector for that link (length = magnitude, clipped at the 95th percentile measured on probe data), dashed outline when occluded (scene-camera depth buffer), X when contact lost since the previous boundary. Render on scene and wrist views. One arrow per fingertip only — no phalanx markers, no per-contact points — until clutter is measured in step 4.
2. Collect a boundary set: run the executor alone on the 4 tasks, 5 init states, record every boundary with frames, per-finger summaries and the privileged label (carried / placed / empty) — ~100–150 boundaries.
3. Judge output schema: `outcome ∈ {success, fail, unclear}`, `plan_still_valid ∈ {yes, no}`, `reason` (free text). Inputs: current annotated frames, previous-boundary annotated frame + its verdict, boundary type as text, retry count as text.
4. Offline comparison on the same boundaries, same truth: (a) plain RGB, (b) RGB + taxel summary as text, (c) RGB + overlay. Run with Haiku and with the open VLM. Score outcome accuracy; score `plan_still_valid` against whether a scripted retry converts the episode (from Phase 2).

Decision: if (c) ≤ (b) for both judges, drop the overlay from the judge path and carry text only; the policy-side overlay (Phase 3) is unaffected. If the open VLM cannot exploit (c) but Haiku can, note it and decide later whether to fine-tune the judge.

### Phase 2 — Agentic framework, judge-only (no training) — **the pre-registered comparison**

Goal: the main result.

Architecture: planner (once + on replan) → RLDX-1 frozen → boundary detector → tactile reader → renderer → judge → gate rule → scripted recovery or planner.

Gate rule (code, not prompt):

| judge | gate |
|---|---|
| success | continue |
| fail, valid | scripted local recovery (open, retreat, re-approach, close); cap N |
| fail, invalid / unclear twice / cap hit | replan with the reason text |
| budget cap | abort |

Conditions, same 4 tasks, same init states, pads fitted in all:

- (a) executor alone
- (c) framework, judge on plain RGB
- (t) framework, judge on overlay
- (o) oracle judge (privileged), upper bound

Scripted recovery on a hand = open hand, retreat along the approach direction, re-approach, re-close; implemented as a short joint-space sequence per hand, not a planner call.

Scale: screening first (1 seed, 20 episodes/condition) to confirm plumbing. Then the **one full protocol**: (c) vs (t), 3 seeds × 50 episodes, 4 tasks. Nothing else runs at full scale. Report SR, planner calls per episode, cost, verdict accuracy vs truth, recovery conversion rate.

Ablations at screening scale only: no recovery (recover→abort); recovery by policy resume instead of primitives; touch-only gate (tactile reader flags through the same table, no VLM).

### Phase 3 — Overlay into the policy (training; A100s)

Goal: the first multi-finger overlay result and the physics-stream comparison.

1. Replay GR-1 demos for the 4 tasks in Isaac to recover taxel readings (Phase 0 step 2 must have passed); render overlays post-hoc so all conditions train on identical data.
2. LoRA fine-tune RLDX-1-FT-GR1: (i) clean images, (ii) overlay images, (iii) clean images + `--use-physics` with the per-finger force vectors as the tactile stream. Same steps, same recipe, 2 seeds minimum. Budget before starting: estimate GPU-hours per run on A100 from a 500-step probe.
3. Evaluate each as the executor alone (screening scale), then (ii) and (iii) inside the Phase 2 framework with the overlay judge.

Factorial for the paper: overlay in {policy, judge, both, neither}. Decision: if (ii) ≈ (i), the policy-side overlay is reported as a null result and the paper rests on Phase 2.

### Phase 4 — Gripper transfer (optional)

Same framework on RoboCasa Kitchen + RLDX-1-FT-ROBOCASA with the MuJoCo taxel gripper, screening scale only, to show the result is not hand-specific. Only if a reviewer-facing generality point is needed.

## 4. What is explicitly out of scope

- A trained fusion model (DINOv2 + tactile encoder + fusion transformer): only if Phase 1 shows the frozen judges cannot use the annotations.
- A general tactile tokenizer: separate project.
- Vision-based tactile sensors (gel optics): not used; force-vector taxels only.
- Overlay granularity beyond one arrow per fingertip: only if Phase 1 step 4 shows the judge is not clutter-limited.

## 5. Working rules

- Screening = 1 seed, ~20 episodes. Full protocol = 3 seeds × 50 episodes, once, on the pre-registered (c) vs (t) comparison.
- Planner: per-run and global dollar caps; every call logged by purpose.
- Privileged state only in the recorder and the oracle; isolation tests on from day one.
- Every choice in DECISIONS.md with its reason; claims marked run (observed) or prior (from a paper).

## 6. Cluster workflow (no coding agents on the Slurm cluster)

The cluster is reached only by ssh and git. Any coding agent runs on the local machine against the local checkout; nothing it does touches the cluster directly.

- **Repo is the only channel.** All code, job scripts (`*.sbatch`), configs and install scripts live in git. Local: edit → test what can be tested without a GPU → commit → push. Cluster: `git pull` → `sbatch` by hand. No editing on the cluster except hotfixes that are committed from there and pulled back.
- **Install is a script, not a session.** `scripts/install_*.sh` must run unattended from a clean clone (venvs, pins, checkpoint download to the cluster's cache dir). The agent writes them; you run them once over ssh.
- **Results come back small.** Every job writes a compact summary (JSON/CSV tables, verdict logs, a handful of annotated frames) to `results/<phase>/<job>/`; raw rollouts, videos and checkpoints stay in `$SLURM_LOG_DIR` on the cluster. `results/` is committed (or rsynced) so the agent can read and analyze locally. Keep each summary under a few MB.
- **Analysis runs locally.** Tables, plots and reports are generated from `results/` on the local machine, never on the cluster.
- **Job hygiene.** One sbatch per condition; job ID, commit hash and config path written into every summary so results are traceable without cluster access.
- **Secrets.** Planner API keys live in the cluster environment, never in the repo; the planner adapter reads them from env and refuses to start without the cost-cap file present.

## 7. Immediate next actions

1. Phase 0 steps 1–2 on the Slurm cluster: RLDX-1 + GR-1 Tabletop reproduction and the demo-replay check (1–2 days).
2. Add Isaac Lab contact sensors to the benchmark's hand assets as fingertip and palm taxels (1–2 days).
3. Task-selection sweep (1 job).
4. Write the Phase 1 renderer against the Phase 0 frames; measure overlay clutter on the wrist view before adding anything beyond fingertips.
