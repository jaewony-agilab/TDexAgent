# Decisions

Each entry: the choice, the reason, and whether its basis is **run** (observed in this project) or **prior** (from code, docs or a paper). Newest last.

## 2026-10-14 — project scaffold

**D1. The GR-1 Tabletop simulator is MuJoCo, not Isaac.** PLAN.md §2 says "Isaac; Isaac Lab contact sensors". RLDX-1's GR-1 eval path (`run_scripts/eval/gr1_tabletop/`, commit `cf67c31`) runs `robocasa-gr1-tabletop-tasks` on robosuite 1.5.1 / MuJoCo. *prior (RLDX-1 source).* Consequences:
- Taxels are read from MuJoCo contacts (`mj_contactForce`) on the fingertip/palm links (`tdex/sim/taxels.py`), not Isaac Lab contact sensors. Reading `mjData` is passive, so the sensors cannot change dynamics; the Phase 0 step 5 "does the policy still work" check is kept anyway as a check on the harness.
- The fallback (RoboCasa + Panda) is also MuJoCo, so the sensor code carries over.
- Phase 0 step 2 (replay from stored state) means MuJoCo state restore, not Isaac. See D15.

**D2. One policy camera; wrist views are judge-only.** The GR-1 key converter feeds RLDX-1 only `video.ego_view_pad_res256_freq20` (egoview 1280×800, padded to square, resized to 256). The GR1 MJCF has `eye_in_{left,right}_hand` cameras; they are rendered at boundaries only, for the judge, and never reach the policy. The judge gets the scene view and the wrist view of the hand at the boundary. *prior (robocasa source).*

**D3. All 24 tasks are pick-and-place.** 18 rearrangement plus 6 articulated "…Close" tasks (place, then close a door). The "at least one grasp" selection criterion therefore holds for every task, so only the 30–70% band filters. *prior.*

**D4. Reproduction subset fixed before seeing results** (`configs/gr1.yaml: repro_tasks`): 2 articulated + 2 novel-object PnP tasks. The stop rule compares each against its published per-task number (to fill in from the RLDX-1 paper or model card before the run). *prior.*

**D5. The harness steps one action at a time but queries the policy every 16 steps.** It uses RLDX-1's `MultiStepWrapper` with `n_action_steps=1` and replays each 16-step chunk itself. The observation history at each query matches what the stock 16-step wrapper produces, so policy inputs are identical. Boundaries are detected per step, and a recovery or replan flushes the rest of the chunk. Phase 0 step 1 checks this by running RLDX-1's own client and our harness on the same seeds. *prior; to be confirmed by run.*

**D6. Seeding mirrors RLDX-1's `VideoRecordingWrapper`.** Episode k with base seed s resets with `s*100000 + k`, after the global RNGs and the robosuite env RNG are reseeded. Episode ids then select the same init states as the published eval, and every condition in a comparison shares init states. *prior.*

**D7. Episodes stop at first success.** RLDX-1 scores an episode a success if it succeeds at any step, so stopping early does not change success rate (SR). It saves judge and planner calls. *prior.*

**D8. Closure signal = mean of hand-command dims 1–4.** The hand command is 6-D. Robosuite's order is (pinky, ring, middle, index, thumb-bend, thumb-yaw), and robocasa reverses the hand *state* order. Dims 1–4 are flexion under both orders, so the signal holds whichever order the action uses. `open_value`/`closed_value` are placeholders until `scripts/inspect_env.py` reports command percentiles. *prior.*

**D9. One taxel per link: the summed force from all non-robot bodies on that link, world frame, as felt by the link.** The anchor is the link's collision-geom center from forward kinematics. Contact partner identity is never exposed. Self-contacts are excluded by body-name prefix (`robot0_`, `gripper0_`, to confirm in Phase 0). The sign and magnitude match a resting-weight test in real MuJoCo (`tests/test_mujoco.py`). *run (unit test).*

**D10. Projection convention checked against MuJoCo rendering.** Pinhole from `cam_fovy` and `cam_xmat`, with pixel rows top-down as in the env's flipped frames. It matches MuJoCo's offscreen render to within 1.5 px. Still to do on the cluster: the egoview pad/resize mapping and the robosuite depth-buffer conversion. Check them visually with the `mostclosed_*_overlay.png` frames from `inspect_env.py`. *run (unit test) + pending.*

**D11. Privileged labels.** A close is labelled `carried` or `empty`. An open is labelled `placed`, `dropped`, or `carried` (release failed). This extends PLAN.md's three labels because an open boundary needs its own failure modes. Truth: a close succeeds iff `carried`; an open succeeds iff `placed`. Raw facts are stored next to the label: grasp per hand, object position, in-receptacle, task success. Hand closures that are pushes (closing a drawer) can be mislabelled `empty`; review these in Phase 1 before scoring. *prior.*

**D12. The planner never rewrites the executor instruction.** RLDX-1-FT-GR1 was fine-tuned on the benchmark's own instruction strings, so rephrasing them risks moving the policy out of distribution. The planner (a) lists the expected hand events, which the judge sees as "planned current step", and (b) on replan chooses `resume`, `recover_then_resume <hand>` or `abort`. *prior (design).*

**D13. Gate details not fixed by PLAN.md.** A single `unclear` continues, and two in a row trigger a replan. A success resets the unclear streak. Retry caps are per hand (N=2). The recovery's own release does not count as a task boundary. *design.*

**D14. Scripted recovery is a joint-space replay.** It opens the hand (flexion dims only), replays the last 20 commanded arm steps in reverse (retreat), replays them forward (re-approach), and then re-closes to the last commanded hand pose. This needs no IK, and the approach direction is the arm's own motion. *design.*

**D15. Phase 3 is not built yet.** It depends on Phase 0 step 2. The released GR-1 training data (GR00T-X-Embodiment-Sim, LeRobot) probably carries no MuJoCo state. `scripts/check_replay.py` reports this for both LeRobot and robomimic-hdf5 inputs. If the answer is no, Phase 3 needs fresh demos (budget ×2, per PLAN.md). *prior; pending run.*

**D16. Judges.** The reference is Claude Haiku 5.5 (`claude-haiku-5-5`) with effort `low` and JSON-schema structured output; a refusal counts as `unclear`. The open model is Qwen2.5-VL-7B-Instruct via vLLM 0.8.5 (first pin, not yet verified on the cluster), temperature 0, JSON-schema constrained. Prices live in the cost-cap file, not in code. *prior.*

**D17. Overlay encoding.** Colors are Okabe-Ito (colorblind-safe), one per finger, with a white outline for the left hand and black for the right. Arrows have a halo in the hand color, and their length is clipped at `force_clip_n` (10 N placeholder until the 95th percentile is measured). Occlusion is a dashed outline from a depth test, and a red X marks lost contact. *design.*

**D18. Touch-only ablation rule.** A close succeeds iff the thumb and at least 2 fingertips are in contact. An open succeeds iff no fingertip is in contact. `plan_still_valid` is always `yes`. *design.*
