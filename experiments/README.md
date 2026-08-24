# TemporalFocus experiments

Reproducible studies of the TemporalFocus attention kernels. Experiment
dependencies (plotting, RNG, statistics) live in `experiments/Project.toml`
only — the root `Project.toml` `[deps]` stays empty.

## Setup

```bash
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

## Running

```bash
julia --project=experiments experiments/jitter_test.jl
```

Each experiment writes `config.toml`, `metrics.csv`, `summary.md`, and
`figure.png` to `experiments/results/<slug>/` via the helpers in
`experiments/src/ExperimentUtils.jl`.

## Experiments

| Slug | Script | Question |
| --- | --- | --- |
| `jitter_test` | `jitter_test.jl` | How much spike-timestamp jitter can each kernel absorb before the selected focus changes? |
