# Adaptive MTP draft-depth benchmark - frozen scripts

Snapshot taken 2026-07-01. Findings and interpretation live in `../Adaptive_MTP.md`.
These are reference copies; the working originals (at repo root and `tmp/`) drift over time.

## Files

- `gemma-4-31B-it-GGUF.sh` - launcher for `unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL`
  (shared-memory MTP path). Base server config is what matters for reproducibility; the
  sweep overrides `n_max`/`p_min` via CLI. Ships the recommended `n_max=16, p_min=0.8`.
- `Qwen3.6-27B-MTP-GGUF.sh` - launcher for `unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K`
  (single-head, growing-KV path). Ships the recommended `n_max=16, p_min=0.8`.
- `nmax-pmin-sweep.sh` - the harness. 2D sweep of `n_max` x `p_min`, restarts the server per
  config, scrapes tg/acc/mean_len from the server log.

## Reproduce

Run the harness from the repo root (its relative paths assume it):

    cp bench-adaptive-mtp/nmax-pmin-sweep.sh tmp/
    ./tmp/nmax-pmin-sweep.sh <task> [from] [to]        # task = refactor|replace|autotype|mixed
    LAUNCHER=./Qwen3.6-27B-MTP-GGUF.sh ./tmp/nmax-pmin-sweep.sh replace
    GRID_SPEC="0.80:12,16,24 0.85:12,16,24" ./tmp/nmax-pmin-sweep.sh replace

## results/

Named `sweep-<task>[-<model>]-results.tsv`. Untagged files predate the model-tagging change:
- `sweep-results.tsv` - first run, refactor task, n_predict=600 (superseded; short-response).
- `sweep-{refactor,replace,autotype}-results.tsv` - the 3-task run, n_predict=4096.
- `sweep-*-gemma-*` / `sweep-*-Qwen*` - model-tagged ceiling (n=16/24) and cross-model runs.
- `sweep-mixed-gemma-*` - annotated-recitation (interleaved reason+echo) run.

Columns: `p_min  n_max  tg_t/s  n_dec  acc  gen  mean_len`. Single-shot per config; treat
sub-10% gaps as noise.
