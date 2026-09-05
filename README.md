# TemporalFocus.jl

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://rmems.github.io/TemporalFocus.jl/dev/)

Pure spike-native temporal interaction primitives for the Spikenaut ecosystem.

**Repository:** [github.com/rmems/TemporalFocus.jl](https://github.com/rmems/TemporalFocus.jl)

**Wiki:** GitHub wiki is enabled for this repo but not yet populated (no pages as of the
Limen-Neural → rmems transfer). When content exists, clone
`git@github.com:rmems/TemporalFocus.jl.wiki.git` to
`~/rmems/limen-return/wiki/TemporalFocus.jl.wiki`.

## Scope

This package is intentionally narrow.

It owns:
- spike events, spike trains, and temporal buffers
- coincidence-based and temporally decayed spike interaction
- temporal attention kernels with recency weighting
- attention normalization (L1, max)
- synaptic/readout application over spike-derived weights

It does not own:
- STDP, Hebbian learning, or reward-modulated plasticity
- eligibility traces or neuromodulatory signals
- distillation or routing mechanisms
- encoding or decoding logic
- runtime execution or event-loop scheduling
- transformer dimensions, token embeddings, or gating mechanisms
- projector weights between SNN and LLM spaces
- LLM-side fusion logic
- finance/HFT semantics such as order books, positions, PnL, market data, or trading signals

If a feature requires knowledge of tokens, embeddings, dense attention semantics,
model-space projection weights, synaptic plasticity rules, or market/trading semantics,
it belongs outside this repository.

### Planned scope change

[ADR 0001](docs/adr/0001-consolidate-neuropulse-and-spikestream.md) records an accepted
decision to consolidate `rmems/NeuroPulse.jl` and `rmems/SpikeStream.jl` into this
repository, which broadens the boundary above to add spike-stream feature extraction and
an activity-routing kernel. **That change is not yet in effect** — the scope
statement above describes what this package ships today, and the ADR describes what it
will own once the migration lands. Everything else in the "does not own" list stays
excluded, including runtime, plasticity, dense/LLM, and finance semantics.

## Interface Contract

Inputs to this package should be pure SNN quantities:

- `SpikeTrain`
- `TemporalBuffer`
- synaptic or readout matrices defined over neuron indices

Outputs from this package should remain pure SNN quantities or direct neuron-space readouts:

- spike-derived weight vectors
- neuron-space readout vectors

## Current API

- `spike_attention_discrete`
- `spike_attention_temporal`
- `spike_attention_continuous`
- `temporal_weight`
- `normalize_l1!`
- `normalize_max!`
- `prune!`

## Examples

Runnable scripts live under [`examples/`](examples/). From the repo root:

```bash
julia --project=. examples/discrete_attention.jl
julia --project=. examples/temporal_attention.jl
julia --project=. examples/continuous_buffer.jl
julia --project=. examples/normalize_readout.jl
```

| Script | Demonstrates |
|--------|----------------|
| `examples/discrete_attention.jl` | `SpikeTrain` + `spike_attention_discrete` coincidence attention |
| `examples/temporal_attention.jl` | `spike_attention_temporal` with small vs large τ recency decay |
| `examples/continuous_buffer.jl` | `TemporalBuffer`, `prune!`, and `spike_attention_continuous` |
| `examples/normalize_readout.jl` | In-place `normalize_l1!` and `normalize_max!` on weight vectors |

All examples use `Float32` spike values and require no extra dependencies.

## Migration Note

**STDP and plasticity removed in v0.1.0:**

The `stdp_update!` function has been removed from this package. Synaptic plasticity, including STDP, Hebbian learning, reward-modulated plasticity, and eligibility traces, should be implemented in a dedicated plasticity package such as `plasticity-lab` or a future Julia plasticity adapter.

TemporalFocus.jl focuses exclusively on spike-native temporal attention and does not own learning rules or weight updates.

## Non-Goals

This repository should not accumulate adapter code for:

- tokenization
- embeddings
- transformer attention
- fusion gates
- cross-modal projector training
- hybrid orchestration

## Benchmarks

Local performance microbenchmarks live under `benchmark/` and use
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl). They are
**not** a package dependency and are **not** wired into CI or `Pkg.test()`.

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/run_benchmarks.jl
```

The suite covers:

- `spike_attention_discrete` / `spike_attention_temporal` / `spike_attention_continuous` at small (64) and medium (500) event counts
- `normalize_l1!` / `normalize_max!`
- `prune!`
- optional `temporal_weight` microbenchmark

Event counts are capped to avoid O(n²) blowup in pairwise attention. Output
reports median time and allocations per case.

## Experiments

Reproducible spike-native experiments live under `experiments/`, in an isolated
Julia environment. CairoMakie and other visualization/data dependencies are
**not** package dependencies and are **not** wired into CI or `Pkg.test()`.

```bash
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=experiments experiments/run_all.jl
```

Each experiment emits `config.toml`, `metrics.csv`, `figure.png`, and
`summary.md` into `experiments/results/<slug>/` through the shared harness in
`experiments/src/ExperimentUtils.jl`. Generated results are git-ignored and
rebuilt by the command above; `run_all.jl` runs whichever experiment scripts are
present, in a deterministic order.

See [`experiments/README.md`](experiments/README.md) for the artifact contract,
the harness API, and how to add an experiment.
