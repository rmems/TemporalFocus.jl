# AGENTS.md

> Last updated: 2026-09-05

## Project overview

TemporalFocus.jl is a pure spike-native temporal attention kernel for the Spikenaut ecosystem.

See [README Scope](README.md#scope) for the human-facing boundary documentation.

**Scope** — TemporalFocus owns:

- Spike events, spike trains, and temporal buffers
- Coincidence-based and temporally decayed spike interaction
- Temporal attention kernels with recency weighting
- Attention normalization (L1, max)
- Synaptic/readout application over spike-derived weights

**Out of scope** — Features that belong elsewhere:

- STDP (Spike-Timing-Dependent Plasticity), Hebbian learning, reward-modulated plasticity, eligibility traces
- Tokenization, embeddings, transformer attention, gating mechanisms
- Cross-modal projector weights between SNN (Spiking Neural Network) and LLM spaces
- Runtime execution or event-loop scheduling
- LLM-side fusion logic
- Finance/HFT semantics — order books, positions, PnL, market data, trading signals

If a feature requires knowledge of tokens, embeddings, dense attention semantics, synaptic plasticity rules, or market/trading semantics, it belongs outside this repository.

## Dev environment

```bash
# Instantiate project
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Build documentation (Documenter.jl; not a root dependency)
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl

# Run experiments (CairoMakie; not a root dependency, not in CI)
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=experiments experiments/run_all.jl
```

## Code style

- Julia 1.9+ compatible
- Float32 for all spike values and temporal quantities
- Internal helpers prefixed with `_` (e.g., `_check_neuron_id`, `_apply_readout`)
- Public API follows naming conventions:
  - `spike_attention_*` — attention computation variants (use `!` suffix if mutating)
  - `normalize_*!` — in-place normalization functions
  - `temporal_weight` — exponential decay weighting
  - `prune!` — in-place buffer pruning
- SPDX (Software Package Data Exchange) license header at the top of every source file: `# SPDX-License-Identifier: MIT OR Apache-2.0`
- Dual license MIT OR Apache-2.0 — see top-level `LICENSE`, plus `LICENSE-MIT` and `LICENSE-APACHE`

## Testing

- All new public functions should have tests in `test/runtests.jl` (trivial internal helpers excepted)
- Test edge cases: empty trains, single neuron, out-of-range IDs, zero τ
- Run `julia --project=. -e 'using Pkg; Pkg.test()'` before committing
- CI runs on Julia 1.9–1.12 on ubuntu-latest, plus Julia 1.11 on macos-latest and windows-latest

## PR instructions

- Branch naming: `<type>/<description>` (e.g., `feat/continuous-attention`, `fix/buffer-pruning`)
- Commit format: `<type>(<scope>): <description>` (e.g., `feat(attention): add continuous kernel`)
- Resolve all review threads before merge
- All CI checks should pass (Codacy, Kilo, Julia test matrix) unless the change is doc-only

## Boundary enforcement

- If a PR introduces STDP, plasticity, or learning rules, it belongs in a dedicated plasticity package
- If a PR touches tokenization, embeddings, or transformer logic, it belongs in a different repo
- If a PR introduces finance/HFT semantics, redirect it to a downstream application that
  consumes this package (e.g. `rmems/Limen-Capital`, `rmems/DendriteTrader.jl`) — this
  package stays domain-neutral, and the exclusion holds before and after the migration below
- Reviewers should reject scope creep with a redirect to the appropriate package
- **Pending boundary change:** [ADR 0001](docs/adr/0001-consolidate-neuropulse-and-spikestream.md)
  accepts a one-time broadening to also own spike-stream feature extraction and an
  activity-routing kernel (consolidating `rmems/NeuroPulse.jl` and `rmems/SpikeStream.jl`).
  That kernel is deterministic and self-contained but **stateful** — it may mutate only
  the pre-allocated buffers of the router it is handed, with no I/O, clock, ambient or
  global state, hot-path allocation, or weight updates. Enforce that list, not the word
  "pure". The broadening is **not in effect** — enforce the Scope list above until the
  migration lands and updates it. The out-of-scope list is unchanged either way.

## Cursor Cloud specific instructions

- Julia is provided via `juliaup` and the default channel is **1.12**, matching this repo's committed `Manifest.toml`. Run the standard dev commands from "Dev environment" above: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`, then `julia --project=. -e 'using Pkg; Pkg.test()'` (50 tests).

## Release

For General registry registration, TagBot, UUID immutability, and the “do not pre-tag” rule, see [RELEASING.md](RELEASING.md). Release-prep PRs must not create git tags or run Registrator.
