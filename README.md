# TemporalFocus.jl

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://rmems.github.io/TemporalFocus.jl/dev/)

Pure spike-native temporal interaction primitives for the Spikenaut ecosystem.

> **Deprecation:** New projects should use **[NeuroPulse.jl](https://github.com/rmems/NeuroPulse.jl)**
> (v0.1.0+). The attention surface that lived here was copied into NeuroPulse at
> commit `2c5c913` (`src/attention/`, sourced from TemporalFocus `eb38c70`). This
> repository is frozen for that API; maintenance and bugfixes land in NeuroPulse.
>
> **Do not load both packages in one session.** `using TemporalFocus` and
> `using NeuroPulse` together makes names such as `SpikeEvent` ambiguous, and
> `TemporalFocus.SpikeEvent` values are not interchangeable with
> `NeuroPulse.SpikeEvent` (distinct types even when fields match).

**Repository:** [github.com/rmems/TemporalFocus.jl](https://github.com/rmems/TemporalFocus.jl) (legacy snapshot)

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

### Successor: NeuroPulse.jl

[ADR 0002](docs/adr/0002-merge-temporalfocus-into-neuropulse.md) planned consolidation
**into** [`rmems/NeuroPulse.jl`](https://github.com/rmems/NeuroPulse.jl) (package UUID
`b7e4c3f2-1d2e-4a5b-8c9d-0e1f2a3b4c5e`). **That import has landed** on NeuroPulse
`main` (`2c5c913`, v0.1.0): coincidence attention, buffers, and normalization live
under `src/attention/` and are exported from the `NeuroPulse` module.

Install and import the maintained surface:

```julia
using Pkg
Pkg.add(url="https://github.com/rmems/NeuroPulse.jl")

using NeuroPulse

src = SpikeTrain([SpikeEvent(1, 0.0f0)])
y = spike_attention_discrete(src, src, Float32[1])
```

This repo’s `src/` has not changed since NeuroPulse copied it at `eb38c70`. **Do not
import NeuroPulse or SpikeStream into this repository** (movement is one-way). Finance/HFT,
runtime, plasticity, and dense/LLM semantics stay excluded in both packages; see the scope
list above.

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

## Experiment Gallery

Spike-native characterization experiments (recency, kernel regimes, distractors,
jitter, streaming focus, τ/window) live under [`experiments/`](experiments/).
The hosted index points at those scripts and the local `experiments/results/`
artifact layout.

**→ [Experiment Gallery](https://rmems.github.io/TemporalFocus.jl/dev/experiments/)**

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
