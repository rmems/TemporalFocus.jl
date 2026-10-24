# TemporalFocus.jl

Pure spike-native temporal interaction primitives for the Spikenaut ecosystem.

## Scope

TemporalFocus owns:

- Spike events, spike trains, and temporal buffers
- Coincidence-based and temporally decayed spike interaction
- Temporal attention kernels with recency weighting
- Attention normalization (L1, max)
- Synaptic/readout application over spike-derived weights

It does **not** own STDP or other plasticity rules, tokenization/embeddings,
transformer attention, cross-modal projector weights, runtime scheduling, or
LLM-side fusion logic. Those belong in dedicated packages.

See the repository [README](https://github.com/rmems/TemporalFocus.jl)
for the full interface contract and non-goals.

## Overlap with NeuroPulse.jl

`rmems/NeuroPulse.jl` (`f78370eb`, 2026-10-20) already vendors this package's
coincidence-attention surface as `NeuroPulse.Attention` (copied from TemporalFocus
`eb38c70`; see [ADR 0002](https://github.com/rmems/TemporalFocus.jl/blob/main/docs/adr/0002-merge-temporalfocus-into-neuropulse.md))
and re-exports the same public names. The implementations match aside from
formatting. This repository does **not** depend on NeuroPulse.

**Same public names in both trees**

- `SpikeEvent`, `SpikeTrain`, `TemporalBuffer`
- `prune!`
- `temporal_weight`
- `spike_attention_discrete`, `spike_attention_temporal`, `spike_attention_continuous`
- `normalize_l1!`, `normalize_max!`

**NeuroPulse-only (not duplicated here)**

- `ActivityRegion`, `RegionRouter`, `RoutingConfig`
- `update_routing!` / `update_relevance!`
- `routing_diagnostics` / `nero_diagnostics` (legacy NERO alias; there is no `neuro_diagnostics`)
- `adapt_leak!`, `save_state`, `load_state!`

Those routing helpers use a floored softmax-like pass. They are not this
package's spike-coincidence kernel, which stays implemented in TemporalFocus.

**Why there is no NeuroPulse dependency**

NeuroPulse is installable from git (`Pkg.add(url=...)`) and declares
`name = "NeuroPulse"` with `julia = "1.9"`, but it is not in the General
registry and has no tags. There is therefore no versioned `[compat]` entry
that works for this package's 1.9+ / future-registration story. A git pin
would also pull `LinearAlgebra`, `Printf`, and the routing kernel, and it
would invert [ADR 0002](https://github.com/rmems/TemporalFocus.jl/blob/main/docs/adr/0002-merge-temporalfocus-into-neuropulse.md)
(TemporalFocus consolidates *into* NeuroPulse; this repo must not import it).

**Plan**

1. Keep the coincidence kernels (`spike_attention_*`, `temporal_weight`) here.
2. Refresh NeuroPulse's `src/attention/` from this tree when those files drift.
3. Finish ADR 0002 in NeuroPulse, then turn this README into a successor pointer.
4. Do not add `using NeuroPulse` to TemporalFocus.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/rmems/TemporalFocus.jl")
```

Requires Julia 1.9+.

## Quick start

```julia
using TemporalFocus

# Spike events and trains
src = SpikeTrain([SpikeEvent(1, 0.0f0), SpikeEvent(2, 1.0f0)])
ctx = SpikeTrain([SpikeEvent(1, 0.1f0), SpikeEvent(2, 0.9f0)])
readout = Float32[1 0; 0 1]  # identity over 2 neurons

# Coincidence attention (no temporal decay)
y_disc = spike_attention_discrete(src, ctx, readout)

# Temporally decayed attention
y_temp = spike_attention_temporal(src, ctx, readout; τ=5.0f0)

# Continuous attention over sliding buffers
src_buf = TemporalBuffer(10.0f0, [SpikeEvent(1, 0.0f0)])
ctx_buf = TemporalBuffer(10.0f0, [SpikeEvent(1, 0.2f0)])
y_cont = spike_attention_continuous(src_buf, ctx_buf, readout; τ=5.0f0)

# Normalize weight vectors in-place
w = Float32[1, 2, 3]
normalize_l1!(w)
normalize_max!(copy(Float32[1, 2, 3]))

# Recency weight and buffer pruning
wt = temporal_weight(2.0f0, 5.0f0)
prune!(src_buf, 12.0f0)
```

## Documentation

- [Experiment Gallery](experiments.md) — index of landed characterization experiments
- [API reference](api.md) — all public exports
