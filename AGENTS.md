# AGENTS.md

> Last updated: 2026-07-22

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

If a feature requires knowledge of tokens, embeddings, dense attention semantics, or synaptic plasticity rules, it belongs outside this repository.

## Dev environment

```bash
# Instantiate project
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Build documentation (Documenter.jl; not a root dependency)
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
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
- Reviewers should reject scope creep with a redirect to the appropriate package

## Cursor Cloud specific instructions

- Julia is provided via `juliaup` and the default channel is **1.12**, matching this repo's committed `Manifest.toml`. Run the standard dev commands from "Dev environment" above: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`, then `julia --project=. -e 'using Pkg; Pkg.test()'` (50 tests).

## Release

For General registry registration, TagBot, UUID immutability, and the “do not pre-tag” rule, see [RELEASING.md](RELEASING.md). Release-prep PRs must not create git tags or run Registrator.
