# LangProBe-CodeEvolver (AppWorld + ALFWorld-vision)

Benchmark seed repository for [CodeEvolver](https://github.com/julianghadially/CodeEvolver)
research runs on two agentic benchmarks:

- **AppWorld** — interactive API-based task completion in a simulated world of
  everyday apps (Trivedi et al., 2024). Task model: DeepSeek V4 Flash via GMI,
  reasoning effort high, temperature 0.
- **ALFWorld (vision)** — embodied household tasks; the program sees the
  AI2-THOR camera frame each step and picks a text action. Task model:
  Qwen 3.6 35B-A3B via OpenRouter (throughput-sorted provider routing),
  reasoning effort medium, max_tokens 8192, temperature 0.

The CodeEvolver seed programs, metrics, and simulator runtimes live in
`codeevolver_benchmarks/`; fixed model/provider settings are in
`codeevolver_benchmarks/runtime_config.json`. The ALFWorld runtime uses the
qualified lifecycle-v2 server (`scripts/alfworld_server.py` +
`scripts/alfworld_worker_supervisor.py`): one persistent Unity/AI2-THOR
controller per worker with in-place task resets, episode leases, readiness
endpoints, and fail-closed infrastructure errors.

Dataset splits (task-id / game-file lists only — ground truth lives in the
simulators) are in `data/`: `*_trainval.json` is the engine-3 training pool,
`*_test.json` the held-out test set, `*_smoke.json` / `*_smoke_test.json`
one-row wiring checks.

## Setup

See `CODEEVOLVER_SETUP.md`. `scripts/prepare_wsl.sh` builds the two simulator
runtimes (an `appworld==0.1.3` venv + simulator data, and a python-3.9
`alfworld[full]==0.4.2` venv + game data); in the CodeEvolver fleet these are
baked into the benchmark container images instead and located via the
`APPWORLD_*` / `ALFWORLD_*` environment variables.

## Entry points

- AppWorld: `codeevolver_benchmarks.appworld_program.AppWorldProgram` /
  `codeevolver_benchmarks.appworld_metric.appworld_metric`
- ALFWorld-vision: `codeevolver_benchmarks.alfworld_vision_program.AlfWorldVisionProgram` /
  `codeevolver_benchmarks.alfworld_vision_metric.alfworld_vision_metric`

This repository derives from [LangProBe](https://github.com/Shangyint/langProBe)
(Tan et al., 2025); the unrelated LangProBe benchmark suites were removed to
keep the seed clean for optimization runs.
