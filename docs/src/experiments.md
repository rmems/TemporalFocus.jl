# Experiment Gallery

This page is an index of the spike-native characterization experiments that
already live under [`experiments/`](https://github.com/rmems/TemporalFocus.jl/tree/main/experiments).
It does not regenerate results. Each script writes
`config.toml`, `metrics.csv`, `figure.png`, and `summary.md` into
`experiments/results/<slug>/` (git-ignored; rebuild locally).

The artifact contract, harness API, and how to add an experiment are in
[`experiments/README.md`](https://github.com/rmems/TemporalFocus.jl/blob/main/experiments/README.md).

## Reproduce

```bash
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=experiments experiments/run_all.jl              # every landed script
julia --project=experiments experiments/temporal_lens.jl        # one experiment
```

Results land under `experiments/results/<slug>/`. They are not committed;
identical inputs on the same Julia version reproduce byte-identical
`metrics.csv`.

## Landed experiments

| Experiment | Script | Question |
|---|---|---|
| Temporal Lens | [`temporal_lens.jl`](https://github.com/rmems/TemporalFocus.jl/blob/main/experiments/temporal_lens.jl) | How does the recency field change with spike separation `Δt` and time constant `τ`? |
| Three Regimes | [`three_regimes.jl`](https://github.com/rmems/TemporalFocus.jl/blob/main/experiments/three_regimes.jl) | Given the same spikes, what do discrete, temporal, and continuous attention preserve, decay, and reject? |
| Focus Under Fire | [`focus_under_fire.jl`](https://github.com/rmems/TemporalFocus.jl/blob/main/experiments/focus_under_fire.jl) | How robust is temporal attention as stale same-neuron distractors and random noise grow? |
| Jitter Test | [`jitter_test.jl`](https://github.com/rmems/TemporalFocus.jl/blob/main/experiments/jitter_test.jl) | How much timestamp jitter can each kernel absorb before focus flips? |
| Attention Spotlight | [`attention_spotlight.jl`](https://github.com/rmems/TemporalFocus.jl/blob/main/experiments/attention_spotlight.jl) | How does continuous attention move as a `TemporalBuffer` is filled, pruned, and replayed? |
| Memory Gate | [`memory_gate.jl`](https://github.com/rmems/TemporalFocus.jl/blob/main/experiments/memory_gate.jl) | Across the `τ × window` plane, where does continuous attention keep a recent target and suppress a stale distractor? |

`experiments/harness_smoke.jl` is a harness check, not a characterization
experiment. `run_all.jl` runs it first.

## Downstream boundary

This gallery stays spike-native. Domain-specific experiments — trading, market
microstructure, or any other applied setting — belong in a downstream workspace
that depends on TemporalFocus. See the [scope section](index.md).
