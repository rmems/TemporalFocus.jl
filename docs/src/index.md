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

- [Experiment Gallery](experiments.md) — reproducible characterization of the
  attention kernels, generated from committed experiment artifacts
- [API reference](api.md) — all public exports
