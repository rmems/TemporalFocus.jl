# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Base.isempty` for `SpikeTrain` and `TemporalBuffer`.
- Julia docstrings for all public exports (module, types, attention kernels, normalization).
- Runnable usage examples under `examples/`.
- Random-based property tests for normalize, temporal_weight, and prune!.

### Changed

- Expanded CI test matrix to macOS and Windows (Julia 1.11) alongside Ubuntu (Julia 1.9–1.12).

## [0.1.0] - 2026-07-05

Initial public release of TemporalFocus.jl: pure spike-native temporal attention
primitives for the Spikenaut ecosystem.

### Added

- Core types: `SpikeEvent`, `SpikeTrain`, `TemporalBuffer`
- Buffer maintenance: `prune!`
- Temporal decay: `temporal_weight`
- Attention kernels:
  - `spike_attention_discrete`
  - `spike_attention_temporal`
  - `spike_attention_continuous`
- Normalization: `normalize_l1!`, `normalize_max!`
- Dual licensing: MIT OR Apache-2.0
- Agent and review guidance: `AGENTS.md`, `REVIEW.md`
- Top-level dual `LICENSE` (SPDX MIT OR Apache-2.0), TagBot workflow, and
  `RELEASING.md` for General registry prep

### Changed

- Project focused as TemporalFocus.jl (spike-native temporal attention kernel)
- CI test matrix on Ubuntu for Julia 1.9–1.12
- Package scope narrowed to spike events, coincidence / temporally decayed
  interaction, attention kernels, normalization, and readout application

### Removed

- `stdp_update!` and all STDP / synaptic-plasticity learning rules from this
  package (see README Migration Note; plasticity belongs in a dedicated package)
- Tokenization, embeddings, transformer attention, and related non-spike
  attention surface area from package scope

[Unreleased]: https://github.com/Limen-Neural/TemporalFocus.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Limen-Neural/TemporalFocus.jl/releases/tag/v0.1.0
