# Experiment Gallery

TemporalFocus is characterized with reproducible spike-native experiments covering
recency, kernel behavior, distractor rejection, timing jitter, streaming focus, and
τ/window trade-offs. This page is the human-facing view of those runs: question,
setup, artifact, observed result, reproduction command, and a link back to the
machine-readable evidence.

!!! note "This page is generated"
    `docs/gallery.jl` renders `docs/src/experiments.md` from the artifacts under
    `experiments/results/`, and `docs/make.jl` regenerates it on every
    docs build. Do not edit the Markdown by hand — edit the generator, or re-run the
    experiment.

## What this gallery answers

| Question a reviewer asks | Answered by |
|---|---|
| What does TemporalFocus do? | [Home](index.md) and the [API reference](api.md) |
| How does recency weighting actually behave? | Temporal Lens |
| What differentiates the three attention modes? | Three Regimes |
| How sensitive is focus to noise and stale activity? | Focus Under Fire |
| How much timing jitter can it absorb? | Jitter Test |
| How does focus move over a streaming buffer? | Attention Spotlight |
| How do τ and the hard window interact? | Memory Gate |

## Evidence policy

1. **Figures are generated, never redrawn.** Every image on this page is mirrored
   from a `figure.png` written by an experiment script into
   `docs/src/assets/experiments/<slug>/` at docs-build time. Those mirrors are
   build output and are not tracked in git; the committed artifacts under
   `experiments/results/<slug>/` are the source of truth.
2. **Every quantitative claim is traceable.** Setup values are read from
   `config.toml`, numbers come from `metrics.csv`, and the interpretation is the
   experiment's own `summary.md` embedded verbatim. The generator adds no numbers
   of its own.
3. **Null and negative findings stay.** Summaries are quoted in full rather than
   excerpted, so a result that fails to support its hypothesis is published as-is.
4. **Absent results are shown as absent.** An experiment that has not run is listed
   with its question and hypothesis and explicitly no results.
5. **Provenance identifies the code version.** Experiment scripts should record a
   commit field (`commit`, `git_commit`, `revision`, …) in `config.toml`, alongside
   `generated_at`, `julia_version` and any RNG seeds. When no commit field is
   recorded, the gallery falls back to the commit that last changed the result
   directory and says so; when neither is available it publishes a warning instead
   of an unverifiable claim.
6. **Published results are committed.** To appear in the hosted docs, a result
   directory must be committed to the repository. Keep figures small (≈500 KB or
   less); large or intermediate data stays out of git.

## Reproducing this gallery from a fresh clone

```bash
git clone https://github.com/rmems/TemporalFocus.jl.git
cd TemporalFocus.jl

# 1. set up the isolated experiment environment (no root dependencies are added)
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# 2. regenerate every experiment artifact under experiments/results/
julia --project=experiments experiments/run_all.jl

# 3. rebuild the docs; the gallery is re-rendered from those artifacts
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Individual experiments run standalone with
`julia --project=experiments experiments/<slug>.jl`. The experiment environment
carries its own visualization dependencies; the root package stays dependency-free.

## Index

| Experiment | Tracking issue | Status |
|---|---|---|
| Temporal Lens | [#46](https://github.com/rmems/TemporalFocus.jl/issues/46) | not yet published |
| Three Regimes | [#47](https://github.com/rmems/TemporalFocus.jl/issues/47) | not yet published |
| Focus Under Fire | [#48](https://github.com/rmems/TemporalFocus.jl/issues/48) | not yet published |
| Jitter Test | [#49](https://github.com/rmems/TemporalFocus.jl/issues/49) | not yet published |
| Attention Spotlight | [#50](https://github.com/rmems/TemporalFocus.jl/issues/50) | not yet published |
| Memory Gate | [#51](https://github.com/rmems/TemporalFocus.jl/issues/51) | not yet published |

!!! warning "No results published yet"
    No result directories were found under `experiments/results/`,
    so every entry below is pending. The gallery deliberately shows an empty
    state rather than placeholder numbers: nothing here is claimed until an
    experiment has actually produced it.

## Experiments

### Temporal Lens

**Status** — not yet published · tracking issue [#46](https://github.com/rmems/TemporalFocus.jl/issues/46)

**Question.** How does the temporal attention contribution change jointly with spike separation `Δt` and time constant `τ`?

**Hypothesis.** Attention is governed by the ratio `|Δt| / τ`: small `τ` gives a narrow focus around `Δt = 0`, large `τ` preserves contributions over longer delays, and the field is symmetric in `Δt` and matches the analytic `exp(-|Δt|/τ)` law to `Float32` tolerance.

**Planned primary artifact.** A `Δt × τ` heatmap of the recency field — the temporal focus cone — plus decay curves for representative short-, medium- and long-memory `τ`.

No artifacts exist under `experiments/results/temporal_lens/`, so this entry
publishes no setup values, no figure, no metrics and no result. It will fill in
automatically on the next docs build once the experiment has been run and its
artifacts committed.

**Reproduce (once the experiment lands).**

```bash
julia --project=experiments experiments/temporal_lens.jl
```

### Three Regimes

**Status** — not yet published · tracking issue [#47](https://github.com/rmems/TemporalFocus.jl/issues/47)

**Question.** Given exactly the same synthetic spike scene, how do `spike_attention_discrete`, `spike_attention_temporal` and `spike_attention_continuous` differ in what they preserve, decay and reject?

**Hypothesis.** Discrete attention ignores timing and accumulates every matching-neuron interaction; temporal attention keeps them but exponentially downweights stale ones; continuous attention adds a hard window and rejects pairs outside it.

**Planned primary artifact.** One spike scene shown three ways — the raster plus per-neuron attention mass under each kernel: *same spikes, three notions of focus*.

No artifacts exist under `experiments/results/three_regimes/`, so this entry
publishes no setup values, no figure, no metrics and no result. It will fill in
automatically on the next docs build once the experiment has been run and its
artifacts committed.

**Reproduce (once the experiment lands).**

```bash
julia --project=experiments experiments/three_regimes.jl
```

### Focus Under Fire

**Status** — not yet published · tracking issue [#48](https://github.com/rmems/TemporalFocus.jl/issues/48)

**Question.** How robust is spike-native temporal attention when the context fills up with stale and random same-neuron distractor activity?

**Hypothesis.** Temporal and continuous attention preserve a higher share of attention on the recent target than timing-agnostic discrete attention as distractor load grows, with continuous attention rejecting most strongly once distractors fall outside the active window.

**Planned primary artifact.** Focus-retention and top-1 correctness curves versus distractor load: how much temporal garbage the signal survives before focus flips.

No artifacts exist under `experiments/results/focus_under_fire/`, so this entry
publishes no setup values, no figure, no metrics and no result. It will fill in
automatically on the next docs build once the experiment has been run and its
artifacts committed.

**Reproduce (once the experiment lands).**

```bash
julia --project=experiments experiments/focus_under_fire.jl
```

### Jitter Test

**Status** — not yet published · tracking issue [#49](https://github.com/rmems/TemporalFocus.jl/issues/49)

**Question.** How quickly does TemporalFocus lose target selectivity as spike timestamps are perturbed, and how does that sensitivity depend on `τ` and the continuous window?

**Hypothesis.** Discrete attention is invariant to timestamp jitter; temporal attention degrades smoothly as jitter grows relative to `τ`; continuous attention degrades smoothly until jitter pushes relevant pairs outside the window, where sharper transitions appear.

**Planned primary artifact.** A timing-tolerance envelope: target retention and output drift versus jitter scale, with a regime map over `τ` and window settings.

No artifacts exist under `experiments/results/jitter_test/`, so this entry
publishes no setup values, no figure, no metrics and no result. It will fill in
automatically on the next docs build once the experiment has been run and its
artifacts committed.

**Reproduce (once the experiment lands).**

```bash
julia --project=experiments experiments/jitter_test.jl
```

### Attention Spotlight

**Status** — not yet published · tracking issue [#50](https://github.com/rmems/TemporalFocus.jl/issues/50)

**Question.** How does continuous attention move between neurons as a `TemporalBuffer` is filled, pruned and replayed through simulated time?

**Hypothesis.** Causal ingestion plus repeated `prune!` produces visible focus handoffs: a pattern holds attention while it is recent, then loses it as it decays and ages out of the window.

**Planned primary artifact.** A time × neuron attention heatmap with a top-1 focus trace and buffer occupancy — an attention spotlight sweeping across neurons.

No artifacts exist under `experiments/results/attention_spotlight/`, so this entry
publishes no setup values, no figure, no metrics and no result. It will fill in
automatically on the next docs build once the experiment has been run and its
artifacts committed.

**Reproduce (once the experiment lands).**

```bash
julia --project=experiments experiments/attention_spotlight.jl
```

### Memory Gate

**Status** — not yet published · tracking issue [#51](https://github.com/rmems/TemporalFocus.jl/issues/51)

**Question.** Across the `τ × window` parameter plane, where does continuous attention preserve a recent target while suppressing a stale same-neuron distractor?

**Hypothesis.** The plane separates into a hard-clipped regime (the window clips the target), a selective gate (target in, stale out), a soft-decay regime (both admitted but `τ` suppresses the stale contribution), and an over-retentive regime (wide window and large `τ` let stale mass back in).

**Planned primary artifact.** A `τ × window` phase map of target attention share with a companion stale-leakage map — a memory control panel.

No artifacts exist under `experiments/results/memory_gate/`, so this entry
publishes no setup values, no figure, no metrics and no result. It will fill in
automatically on the next docs build once the experiment has been run and its
artifacts committed.

**Reproduce (once the experiment lands).**

```bash
julia --project=experiments experiments/memory_gate.jl
```

## Downstream boundary

This gallery stays spike-native. Domain-specific experiments — trading, market
microstructure, or any other applied setting — belong in a downstream workspace
that depends on TemporalFocus, not in this repository. See the
[scope section](index.md) for the boundary this package enforces.
