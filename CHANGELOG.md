# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `prune!` documents and handles empty, all-zero, and threshold-equal buffers;
  `NaN` and `-Inf` windows, non-finite event times, and non-finite /
  Float32-overflowing `current_time` raise `ArgumentError`. A `+Inf` window
  stays unbounded. Finite negative windows keep the inclusive age comparison.
- `normalize_l1!` / `normalize_max!` leave empty or non-positive totals
  unchanged (no `NaN` / divide-by-zero) and reject `NaN`/`Inf` weights.
- Attention kernels and `temporal_weight` reject `NaN`/`Inf` spike times,
  values, readout entries, and `dt`, a `τ` that is non-positive or overflows
  `Float32` to `Inf32`, and a spike-time difference that overflows `Float32`,
  with `ArgumentError`. Finite complex readout entries stay valid; only
  non-finite entries are rejected.

### Added

- Docs note listing TemporalFocus names also vendored in `rmems/NeuroPulse.jl`
  (`NeuroPulse.Attention`) and why this package does not depend on NeuroPulse.
- `docs/adr/0002-merge-temporalfocus-into-neuropulse.md`: superseding ADR.
  Surviving package is `rmems/NeuroPulse.jl` (UUID `b7e4c3f2-…`). TemporalFocus
  consolidates into NeuroPulse. Do not implement ADR 0001 / PR #54's
  NP+SpikeStream → TemporalFocus sequence. Planning only — no source code is
  migrated by this change.
- `Base.isempty` for `SpikeTrain` and `TemporalBuffer`.
- `Base.==`, `Base.isequal`, and `Base.hash` for `SpikeEvent`, `SpikeTrain`, and
  `TemporalBuffer` (content equality; float `isequal`/`hash` rules for Set/Dict).
- Julia docstrings for all public exports (module, types, attention kernels, normalization).
- Runnable usage examples under `examples/`.
- Random-based property tests for normalize, temporal_weight, and prune!.
- `docs/adr/0001-consolidate-neuropulse-and-spikestream.md`: historical inventory
  and adapter/precision notes from PR #54. **Superseded by ADR 0002** — do not
  implement its NP+SpikeStream → TemporalFocus sequence.

### Changed

- Boundary docs (`README.md`, `AGENTS.md`, `REVIEW.md`) and ADR 0001 banner:
  ADR 0001 / PR #54 is superseded; pointers now follow ADR 0002.
- Post-transfer hygiene: docs, package metadata, and changelog links now point at
  `rmems/TemporalFocus.jl` (formerly Limen-Neural).
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

[Unreleased]: https://github.com/rmems/TemporalFocus.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rmems/TemporalFocus.jl/releases/tag/v0.1.0
