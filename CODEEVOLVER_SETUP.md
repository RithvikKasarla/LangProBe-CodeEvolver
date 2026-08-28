# CodeEvolver benchmark setup

This working copy adapts LangProBe's AppWorld and ALFWorld action loops to the
CodeEvolver callable and metric contracts.

## Runtime

Run from Ubuntu under WSL2:

```bash
bash scripts/prepare_wsl.sh
```

Run seed-level checks with the workspace on `PYTHONPATH`:

```bash
PYTHONPATH=. env/bin/pytest tests -q
PYTHONPATH=. env/bin/python scripts/smoke_seed.py appworld --max-steps 2
PYTHONPATH=. env/bin/python scripts/smoke_seed.py alfworld_vision --max-steps 1
```

The script creates isolated, ignored runtimes under `.runtime/`, installs
AppWorld 0.1.3 and the pinned ALFWorld commit, downloads benchmark data, and
exports deterministic JSON splits under `data/`.

AppWorld uses GMI Cloud `deepseek-ai/DeepSeek-V4-Flash`. Visual ALFWorld uses
DeepInfra `Qwen/Qwen3.6-27B`. Both request high reasoning. These settings live
in `codeevolver_benchmarks/runtime_config.json` and should be included in
CodeEvolver's deny paths together with metrics and held-out data.

AppWorld evaluation is process-isolated because AppWorld's in-process global
state is not thread-safe. ALFWorld creates a THOR environment per row and sends
the current RGB frame to the multimodal model.

## Operator checklist

Before a run:

1. Start from the committed `codex/benchmark-setup` branch with no prior run
   branches or untracked optimizer output.
2. Keep runtime configuration, metrics, tracing, datasets, and provider/model
   selection in CodeEvolver's deny paths.
3. Start the local API with one strategy (`asa`, `greedy`, or
   `pareto_evolution`). Change strategies only by stopping and restarting the
   API; do not change the experiment configuration.
4. Submit one smoke job, then inspect its health, detailed reasoning log, OTel
   traces, memory, history mirror, and meta-reflection artifacts before a full
   run.

During a run, use `scripts/check_run_health.py <job_id> --watch` from the
CodeEvolver engine repo. Local artifacts are saved under
`~/.codeevolver/{logs,history,memory,traces,architect_traces}`.

After a run, save results before cleanup. Use the engine's
`scripts/reset_repo.py --save --delete --dry-run` first, inspect the proposed
actions, and then rerun without `--dry-run`. Disconnecting Git or deleting the
working copy is a final bench-sanitization step only; CodeEvolver needs the Git
repository while it is optimizing.

## Machine limits

Use four AppWorld evaluator threads. Use one ALFWorld visual worker on this
machine: AI2-THOR under WSL/Xvfb renders through CPU `llvmpipe`, and additional
workers can saturate the CPU and make Windows unresponsive. A GPU-backed Linux
host is strongly preferred for the 30-example visual subsample.
