# REVIEW.md

> Last updated: 2026-09-05

## Review scope

TemporalFocus.jl owns **spike-native temporal attention only** — reviewers should enforce this boundary on every PR.

See [README Scope](README.md#scope) for the full boundary documentation.

**In scope:**

- Spike events, spike trains, and temporal buffers
- Coincidence-based and temporally decayed spike interaction
- Temporal attention kernels with recency weighting
- Attention normalization (L1, max)
- Synaptic/readout application over spike-derived weights

**Out of scope:**

- STDP (Spike-Timing-Dependent Plasticity), Hebbian learning, reward-modulated plasticity, eligibility traces
- Tokenization, embeddings, transformer attention, gating mechanisms
- Cross-modal projector weights between SNN (Spiking Neural Network) and LLM spaces
- Runtime execution or event-loop scheduling
- LLM-side fusion logic
- Finance/HFT semantics — order books, positions, PnL, market data, trading signals

If a PR introduces features outside this scope, request scope clarification before reviewing the implementation.

## Reviewer checklist

### Correctness

- Do the spike attention primitives produce correct outputs for the defined inputs?
- Are edge cases handled: empty trains, single neuron, out-of-range IDs, zero τ?
- Do return values match expected types (Float32)?

### Boundary enforcement

- Does the change stay within TemporalFocus's documented scope?
- Does it introduce STDP, plasticity, tokenization, embeddings, transformer logic, or
  finance/HFT semantics?
- If so, does it belong in a dedicated package or a downstream domain application such
  as `rmems/Limen-Capital` or `rmems/DendriteTrader.jl` instead?
- A one-time boundary broadening is recorded in
  [ADR 0001](docs/adr/0001-consolidate-neuropulse-and-spikestream.md)
  (spike-stream feature extraction and an activity-routing kernel). **That change is
  not yet in effect** — enforce the scope list above until the migration lands and
  updates it.

### Type stability

- Are return types concrete and predictable for Float32 spike values?
- Are generic abstract types avoided in hot paths?

### Performance

- Are inner loops `@inbounds` where safe?
- Are there unnecessary allocations in hot paths?

### Tests

- Does the PR include tests covering the new behavior?
- Are edge cases tested: empty trains, single neuron, out-of-range IDs, zero τ?
- Run `julia --project=. -e 'using Pkg; Pkg.test()'` locally to verify

### API consistency

- Do new functions follow existing naming conventions:
  - `spike_attention_*` — attention computation variants
  - `normalize_*!` — in-place normalization functions
  - `temporal_weight` — exponential decay weighting
  - `prune!` — in-place buffer pruning
- Internal helpers use `_` prefix

### Documentation

- Are exported functions documented?
- Does the README stay accurate?
- Does the CHANGELOG reflect the change?

## PR labeling

| Label | When to apply |
|---|---|
| `size:S` / `size:M` / `size:L` | Auto-assigned based on diff size |
| `documentation` | Doc-only changes |
| `enhancement` | New feature or API addition |
| `chore` | Maintenance, CI, licensing, tooling |
| `bug` | Fix for incorrect behavior |

## Merge requirements

- All CI checks must pass (Codacy, Kilo, Julia test matrix)
- At least one review approval
- All review threads resolved
- No unresolved `action_required` checks
- Branch follows `<type>/<description>` naming convention
- Commit follows `<type>(<scope>): <description>` format
