# TDexAgent — contact-annotated agentic manipulation

Frozen RLDX-1 (dexterous VLA) + VLM judge + LLM planner on GR-1 Tabletop, with per-finger
taxel forces drawn onto the judge's images. Plan: [docs/PLAN.md](docs/PLAN.md). Every
deviation and choice: [DECISIONS.md](DECISIONS.md). Read D1 first: the benchmark runs in MuJoCo, not Isaac.

## Layout

```
tdex/                    library (Python ≥3.10)
  types.py               Boundary, FingerSummary, BoundaryPacket, Verdict
  boundary.py            hand-closure boundary detector (hysteresis + settle window)
  reader.py              taxel stream -> per-finger summaries (contact lost, peak)
  camera.py, render.py   projection + overlay renderer
  judge/                 prompt per condition (plain/text/overlay), Claude + vLLM backends, touch-only
  planner.py             Claude planner (plan once, replan on gate)
  gate.py, recovery.py   gate rule table; scripted open/retreat/re-approach/re-close
  cost.py                per-run + global dollar caps, every call logged by purpose
  harness.py             episode loop (sim-agnostic; injected env/sensor/frames/judge)
  executor.py            RLDX-1 ZeroMQ client adapter
  sim/                   MuJoCo side: GR1Env, TaxelSensor, FrameMaker      (cluster only)
  privileged/            recorder + oracle — imported by scripts/ only    (cluster only)
  analysis/              local tables from results/
scripts/                 install_*.sh, run_episodes.py, inspect_env.py, check_replay.py,
                         sensor_cost.py, offline_judge.py, summarize.py
slurm/                   one sbatch per phase step; common.sh starts policy / vLLM servers
configs/gr1.yaml         tasks, taxel links, closure thresholds (CALIBRATE fields), judge/planner
results/<phase>/<job>/   compact summaries (committed); raw data stays in $SLURM_LOG_DIR
```

Isolation is enforced by `tests/test_isolation.py`: only `scripts/` import `tdex.privileged`, the
judge path imports no simulator code, and `tdex.sim` never reads object state.

## Local (no GPU)

```bash
uv venv .venv --python 3.10 && uv pip install --python .venv/bin/python -e ".[dev,llm]" mujoco
.venv/bin/python -m pytest -q                    # unit + fake-env harness + MuJoCo checks
.venv/bin/python scripts/summarize.py results/phase0/sweep --select
```

## Cluster (ssh + git only)

One-time, from a clean clone:

```bash
srun --cpus-per-task=8 --mem=32G -t 3:00:00 bash scripts/install_rldx.sh   # CPU node is fine; RLDX-1 @ pinned commit, model venv, GR-1 sim venv, tdex, checkpoint
bash scripts/install_vlm.sh       # vLLM + Qwen3-VL-8B (Phase 1+); CPU node is fine
cp configs/costcaps.example.yaml configs/costcaps.yaml   # edit caps; planner/API judge refuse to run without it
export ANTHROPIC_API_KEY=...      # in the cluster environment, never in the repo
```

Pass partition/account on the command line (`sbatch -p <part> -A <acct> ...`) or via `SBATCH_PARTITION`/`SBATCH_ACCOUNT`.

Phase 0, in order:

```bash
sbatch slurm/phase0_inspect.sbatch     # obs/action keys, body/camera names, hand-command + force percentiles, overlay check frames
#  -> edit configs/gr1.yaml CALIBRATE fields from results/phase0/inspect/*/inspect.json, look at the PNGs, commit
sbatch slurm/phase0_repro.sbatch       # RLDX-1 own client vs our harness, 4 tasks x 20 eps
GR1_LEROBOT=/path/to/task_dir sbatch --export=ALL slurm/phase0_replay.sbatch
sbatch slurm/phase0_sensorcost.sbatch
sbatch slurm/phase0_sweep.sbatch       # 24 tasks x 20 eps, executor alone
# locally: python scripts/summarize.py results/phase0/sweep --select  -> set selected_tasks
```

Phase 1 / 2:

```bash
sbatch slurm/phase1_collect.sbatch                                    # boundary set -> $SLURM_LOG_DIR/boundary_set
sbatch --export=ALL,JUDGE=vlm,CONDITION=overlay slurm/phase1_judge.sbatch   # x {claude,vlm} x {plain,text,overlay}
sbatch --export=ALL,CONDITION=overlay,JUDGE=claude slurm/phase2.sbatch      # screening, per condition
sbatch --export=ALL,CONDITION=overlay,JUDGE=claude,SEED=43,EPISODES=50,STAGE=full slurm/phase2.sbatch
```

Each job writes `summary.json` with the Slurm job id, commit hash and config path, plus
`episodes.jsonl`, `calls.jsonl` and a few annotated frames.
