# ADR 0002 — Merge TemporalFocus.jl into NeuroPulse.jl

- **Status:** Accepted (plan) — supersedes ADR 0001 / PR #54
- **Date:** 2026-09-05
- **Owner decision:** Raul — surviving package/repo is `NeuroPulse.jl`
- **Supersedes:** [ADR 0001](0001-consolidate-neuropulse-and-spikestream.md)
  (landed via [TemporalFocus.jl#54](https://github.com/rmems/TemporalFocus.jl/pull/54))
- **Tracking issue:** [TemporalFocus.jl#53](https://github.com/rmems/TemporalFocus.jl/issues/53)
  — title and body still describe the old NP+SpikeStream → TemporalFocus direction;
  **retitle / rewrite needed**
- **Related in-flight:** [NeuroPulse.jl#42](https://github.com/rmems/NeuroPulse.jl/pull/42)
  (Closes NeuroPulse #40) — README public-identity fix. This ADR does not edit NeuroPulse.

This ADR is planning-only. **No source code moves in the change that introduces this
document.** Implementers must not follow ADR 0001's import sequence.

---

## Decision

`rmems/NeuroPulse.jl` is the single canonical repository and Julia package.

TemporalFocus coincidence-attention, buffer, and normalization surface is imported
**into NeuroPulse**. This repository is then redirected and, later, archived by a
human. SpikeStream is a later, optional follow-on and **does not block** TemporalFocus
→ NeuroPulse.

Canonical package identity is the NeuroPulse UUID:

`b7e4c3f2-1d2e-4a5b-8c9d-0e1f2a3b4c5e`

The TemporalFocus UUID `7f3c9f2a-6b2e-4d91-9c4f-1a2b3c4d5e6f` is retired after the
import, not regenerated as a prelude to absorbing NeuroPulse.

---

## Why this supersedes #54

[#54](https://github.com/rmems/TemporalFocus.jl/pull/54) already merged. Its ADR
text planned **NeuroPulse + SpikeStream → TemporalFocus**. That direction is
overridden.

**Do not execute ADR 0001.** In particular, do not:

- import NeuroPulse or SpikeStream into this repository
- regenerate this package's UUID as step 2 of a TF-as-survivor plan
- treat SpikeStream feature extraction or the activity-routing kernel as work
  that lands here
- follow ADR 0001's steps 2–12, upgrade matrix, or archive gates as written

Useful salvage from #54 is kept below (inventory facts, adapter/precision ideas)
with the survivor flipped. The sequence, package identity, and destination repo
are new.

---

## Why NeuroPulse survives

1. **The only live downstream pin already uses the NeuroPulse UUID.**
   `rmems/Limen-Capital` `brain/Project.toml` depends on
   `TemporalFocus = "b7e4c3f2-…"` via a `[sources]` git pin still pointing at
   `Limen-Neural/NeuroPulse.jl`. Absorbing NeuroPulse into TemporalFocus would
   force a UUID + URL + `rev` change on the one consumer that exists. Absorbing
   TemporalFocus into NeuroPulse keeps that UUID.
2. **NeuroPulse already claims the `TemporalFocus` module name** in
   `Project.toml` (`name = "TemporalFocus"`, different UUID). Two packages
   cannot keep that collision. Resolve it by one tree — the tree the consumer
   already pins.
3. **NeuroPulse's public README identity is being restored now**
   ([NeuroPulse.jl#42](https://github.com/rmems/NeuroPulse.jl/pull/42)), so the
   repo name and the recruiter-facing brand agree: NeuroPulse. The loadable
   Julia module may remain `TemporalFocus` until a later rename (open question).

Finance/HFT semantics stay out of both packages. Limen-Capital's pin update
remains a cross-repo follow-up, not a gate on the import PR.

---

## Package identity

| Role | Repository | Declared package name | UUID | Disposition |
|---|---|---|---|---|
| **Survivor** | `rmems/NeuroPulse.jl` | `TemporalFocus` (today) | `b7e4c3f2-1d2e-4a5b-8c9d-0e1f2a3b4c5e` | Canonical repo and UUID |
| Import source | `rmems/TemporalFocus.jl` | `TemporalFocus` | `7f3c9f2a-6b2e-4d91-9c4f-1a2b3c4d5e6f` | Import surface, then redirect / retire UUID |
| Later, optional | `rmems/SpikeStream.jl` | `SpikeStream` | `a3c7f1e2-8b4d-5c6e-9f0a-1b2c3d4e5f6a` | Not on the TF → NP critical path |

Nothing is in the Julia General registry (`General/T/TemporalFocus` 404 as of
ADR 0001's 2026-08-23 audit). There is no registry-level UUID immutability
constraint. TemporalFocus still has no git tags; NeuroPulse has none;
SpikeStream has `v0.1.0`. `RELEASING.md` in *this* repo still flags
`7f3c9f2a-…` as a template-looking UUID — that is now moot for the survivor.

---

## TemporalFocus cleanup is already done

ADR 0001 treated "clean TemporalFocus first" as a future gate. That wave has
landed on `main`. It is **not** a remaining prerequisite.

| Work | PRs | Status |
|---|---|---|
| Isolated experiment harness + artifact contract | [#61](https://github.com/rmems/TemporalFocus.jl/pull/61) | Merged |
| `temporal_lens` Δt × τ | [#58](https://github.com/rmems/TemporalFocus.jl/pull/58) | Merged |
| `three_regimes` | [#57](https://github.com/rmems/TemporalFocus.jl/pull/57) | Merged |
| `focus_under_fire` | [#56](https://github.com/rmems/TemporalFocus.jl/pull/56) | Merged |
| `jitter_test` | [#60](https://github.com/rmems/TemporalFocus.jl/pull/60) | Merged |
| Attention spotlight streaming replay | [#59](https://github.com/rmems/TemporalFocus.jl/pull/59) | Merged |
| Memory-gate τ × window | [#62](https://github.com/rmems/TemporalFocus.jl/pull/62) | Merged |
| Slim experiment gallery | [#55](https://github.com/rmems/TemporalFocus.jl/pull/55) | Merged |
| Julia 1.12 CI | [#63](https://github.com/rmems/TemporalFocus.jl/pull/63) | Merged |

Import NeuroPulse from this cleaned `main`. Do not open a "prep TF first" PR.

---

## Migration sequence

Each implementation step is a separately reviewable PR. Steps 5–6 do not block
step 2.

| # | Step | Status / gate |
|---|---|---|
| 1 | TemporalFocus `main` cleaned (experiments + gallery + 1.12 CI) | **DONE** |
| 2 | In **NeuroPulse**: import TemporalFocus attention / buffer / normalization + parity tests. History-aware import preferred (`git subtree` or unrelated-histories merge); fallback is a file copy plus source SHA recorded in NeuroPulse's changelog. | NeuroPulse's existing suite and the ported TemporalFocus suite both pass. Do not import SpikeStream here. |
| 3 | Upgrade notes in NeuroPulse; TemporalFocus README becomes a successor pointer to `rmems/NeuroPulse.jl`. | Readers of this repo are sent to the survivor. No archive yet. |
| 4 | Migrate or document open TemporalFocus issues / PRs that should live on NeuroPulse. | Every still-open item has a destination. |
| 5 | Optional SpikeStream import into NeuroPulse (adapters + precision policy below). | **Later. Not blocking step 2.** |
| 6 | Human archive of TemporalFocus (and SpikeStream if/when imported). | **Out of this workstream.** Owner action only. Gated on Limen-Capital pin work and published upgrade notes. |

**This repository does not import NeuroPulse or SpikeStream.** The code movement
is TemporalFocus → NeuroPulse, implemented in the NeuroPulse tree.

**Do not edit NeuroPulse from this TemporalFocus PR.** Identity-README work is
already in [NeuroPulse.jl#42](https://github.com/rmems/NeuroPulse.jl/pull/42).
Step 2 is a later NeuroPulse PR.

---

## Inventory (salvaged from ADR 0001, destination flipped)

Facts below are from the 2026-08-23 inspection recorded in ADR 0001, plus the
2026-09-05 TemporalFocus experiment wave. Re-check SHAs and open issues at
import time; do not re-litigate the identity decision.

### `rmems/TemporalFocus.jl` — import into NeuroPulse

Public surface to port (v0.1.0 API, additive on NeuroPulse):

`SpikeEvent`, `SpikeTrain`, `TemporalBuffer`, `prune!`, `temporal_weight`,
`spike_attention_discrete`, `spike_attention_temporal`,
`spike_attention_continuous`, `normalize_l1!`, `normalize_max!`, and the
`Base.==` / `isequal` / `hash` / `show` / `isempty` methods.

Also port the experiment harness and gallery if NeuroPulse wants the same
reproducible figures; that is optional and must not block the kernel import.

ADR 0001 source snapshot: ~370 source LOC, 583 test LOC, empty root `[deps]`,
Documenter site, BenchmarkTools suite, `examples/`. Experiments and the slim
gallery landed after that snapshot (table above).

### `rmems/NeuroPulse.jl` — stays put

`ActivityRegion`, `RegionRouter`, `RoutingConfig`, `update_routing!`,
`routing_diagnostics`, `save_state` / `load_state!`, `default_inhibition_matrix`,
legacy `NERO_*` / `LobeState` / `NeroOrchestrator` / `update_relevance!` /
`nero_diagnostics` aliases, and `adapt_leak!` (still a scope wart; not a
TemporalFocus import concern).

ADR 0001 snapshot: `src/TemporalFocus.jl` + `activity_region.jl` +
`region_router.jl` (~743 LOC), `test/test_nero.jl` (899), three examples,
Documenter site. Open then: #40 (now addressed by #42), #14.

### `rmems/SpikeStream.jl` — later, optional

`spike_count`, `spike_density`, `isi_stats`, `detect_bursts`,
`windowed_spike_features`, `normalized_feature_vector`, frozen
`test/fixtures/spike_vectors.json`. Not part of step 2.

### Downstream (still out of this PR)

| Reference | Disposition |
|---|---|
| Limen-Capital `brain/Project.toml` pin (`TemporalFocus = "b7e4c3f2-…"` → old `Limen-Neural/NeuroPulse.jl` URL + frozen `rev`) | **Out.** Owned by `rmems/Limen-Capital#9`. Archiving TemporalFocus does not break that pin. Updating URL/`rev` to current `rmems/NeuroPulse.jl` is a later human/cross-repo change. |
| Limen-Capital vendored router (`brain/synapse_conductor.jl`) | **Out.** De-vendor onto NeuroPulse under Limen-Capital#9. |
| kinetic-signals SpikeStream boundary docs | Only if/when SpikeStream is imported (step 5). |
| Finance / HFT / order-book / PnL semantics | **Never.** Downstream apps consume the kernel. |

---

## Adapters and precision (salvage; SpikeStream-later only)

These contracts stay useful if SpikeStream later lands **in NeuroPulse**. They
are not a TemporalFocus-repo implementation task and they do not gate step 2.

**Do not silently discard `neuron_id` or `value`.** SpikeStream kernels take
bare timestamp vectors. TemporalFocus events are
`SpikeEvent(neuron_id::Int, t::Float32, value::Float32)`. A whole-train
`spike_times(train) = [e.t for e in train.events]` mixes neurons and drops
amplitude. ADR 0001's adapter shape is the intended later contract:

- `spike_times(train, neuron_id; ignore_values=false)` — required `neuron_id`,
  no whole-train overload
- grouped `*_by_neuron` variants
- `ignore_values` defaults to **error** on any `value != 1.0f0`
- buffer adapters take required `current_time` and use the same window rule as
  `prune!`
- burst adapters return time intervals `(t_first, t_last)`, not indices into a
  sorted copy
- reverse `SpikeTrain(times; neuron_id, …)` requires `neuron_id` only when
  `times` is non-empty

**Precision.** TemporalFocus / NeuroPulse spike data stay `Float32`. SpikeStream
features stay `Float64` (ISI moments degrade under `Float32` accumulation).
Ingest `Float64` → `Float32` is the lossy direction; ADR 0001's
`check_precision` modes (`:collisions` default, `:strict`, `:none`) and the
large-magnitude collision tests remain the right later gate. Unknown symbols
(including `:collision`) must throw `ArgumentError`.

Name collision to remember later: SpikeStream's `spike_density` function vs
`RegionRouter.spike_density` field. Not a Julia collision; still document if
both live in one package.

---

## Still out

These are **not** in this ADR PR and are **not** automatic follow-through:

- Archiving TemporalFocus.jl or SpikeStream.jl
- Changing the Limen-Capital pin, README, or vendored router
- Finance / HFT semantics in either kernel package
- Importing SpikeStream (optional, later)
- Editing NeuroPulse beyond the already-open #42 identity README
- Self-merging this planning PR

---

## Open questions

1. **Module name inside the NeuroPulse repo.** Keep the loadable module
   `TemporalFocus` (matches today's `Project.toml` and the Limen-Capital
   `using TemporalFocus` attempt) or rename the module to `NeuroPulse`?
   [NeuroPulse.jl#42](https://github.com/rmems/NeuroPulse.jl/pull/42) keeps
   the module as `TemporalFocus` and treats NeuroPulse as the public brand.
   Default until Raul says otherwise: **keep the module name**, document the
   lag.
2. **SpikeStream.** Same consolidation program or a later one? Default:
   **after** TemporalFocus → NeuroPulse. Adapters/precision above apply then.

`adapt_leak!` destination, `Printf` vs zero-dep diagnostics, and
`RegionRouter.spike_density` rename remain NeuroPulse-internal questions from
ADR 0001. They do not block importing TemporalFocus.

---

## Status vs issue #53

#53's title, goal, target tree, and "use TemporalFocus.jl's UUID" instruction
describe the superseded direction. This ADR satisfies the *intent* (one
canonical package, inventory, dispositions, adapters/precision recorded,
no archive-before-upgrade-path) with the owner override: **NeuroPulse
survives.**

Ask the owner to retitle #53 to something like "portfolio: consolidate
TemporalFocus.jl into NeuroPulse.jl" so the closed issue does not keep
sending implementers into ADR 0001.

---

## References

- [TemporalFocus.jl#54](https://github.com/rmems/TemporalFocus.jl/pull/54) — landed, superseded plan
- [TemporalFocus.jl#53](https://github.com/rmems/TemporalFocus.jl/issues/53) — tracking issue; retitle needed
- [NeuroPulse.jl#42](https://github.com/rmems/NeuroPulse.jl/pull/42) — README identity (Closes #40)
- [NeuroPulse.jl#40](https://github.com/rmems/NeuroPulse.jl/issues/40) — post-transfer hygiene
- [rmems/.github#3](https://github.com/rmems/.github/issues/3) — portfolio umbrella
- [Limen-Capital#9](https://github.com/rmems/Limen-Capital/issues/9) — downstream pin / de-vendor
- [ADR 0001](0001-consolidate-neuropulse-and-spikestream.md) — historical inventory; do not implement
