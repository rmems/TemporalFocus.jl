# ADR 0001 — Consolidate NeuroPulse.jl and SpikeStream.jl into TemporalFocus.jl

- **Status:** Accepted (plan), not yet implemented
- **Date:** 2026-08-23
- **Tracking issue:** [rmems/TemporalFocus.jl#53](https://github.com/rmems/TemporalFocus.jl/issues/53)
- **Umbrella issue:** [rmems/.github#3](https://github.com/rmems/.github/issues/3)
- **Supersedes:** the README/AGENTS.md exclusion of "routing mechanisms" (see [Decision](#decision))

This ADR is the pre-migration inventory, symbol disposition record, and migration
sequence required by issue #53. **No source code moves in the change that introduces
this document.** Everything below describes work that is still to be done, in the
order it should be done, so each subsequent step is a small reviewable PR.

---

## Context

Three repositories currently split one accomplishment — spike-native temporal
processing — across overlapping boundaries:

| Repository | Julia package | UUID | Package deps | Source LOC | Test LOC |
|---|---|---|---|---|---|
| `rmems/TemporalFocus.jl` | `TemporalFocus` | `7f3c9f2a-6b2e-4d91-9c4f-1a2b3c4d5e6f` | *(none)* | ~370 | 583 |
| `rmems/NeuroPulse.jl` | `TemporalFocus` | `b7e4c3f2-1d2e-4a5b-8c9d-0e1f2a3b4c5e` | `LinearAlgebra`, `Printf` | ~743 | 899 |
| `rmems/SpikeStream.jl` | `SpikeStream` | `a3c7f1e2-8b4d-5c6e-9f0a-1b2c3d4e5f6a` | `Statistics` | ~273 | 245 |

### The identity conflict is real

`rmems/NeuroPulse.jl` declares `name = "TemporalFocus"` in its `Project.toml` with a
*different* UUID from `rmems/TemporalFocus.jl`. Two distinct packages therefore claim
the same module name. `NeuroPulse.jl`'s own `AGENTS.md` acknowledges this and instructs
agents to "verify the active project resolves to this repo's `Project.toml` so the two
packages do not collide" — a workaround, not a fix.

### Facts established by direct inspection (2026-08-23)

These materially change the migration risk profile and are recorded here so later
steps do not re-litigate them.

1. **Nothing is registered.** Neither `TemporalFocus` nor `SpikeStream` exists in the
   Julia [General](https://github.com/JuliaRegistries/General) registry
   (`General/T/TemporalFocus/Package.toml` and `General/S/SpikeStream/Package.toml`
   both 404). There is therefore **no registry-level UUID immutability constraint**
   and no registered downstream to break.
2. **No git tags exist** on `TemporalFocus.jl` or `NeuroPulse.jl`. `SpikeStream.jl`
   has a `v0.1.0` tag.
3. **Exactly one live downstream consumer exists:** `rmems/Limen-Capital`
   (`brain/Project.toml`) depends on `TemporalFocus = "b7e4c3f2-…"` — the **NeuroPulse**
   UUID — via a `[sources]` git pin:

   ```toml
   TemporalFocus = {url = "https://github.com/Limen-Neural/NeuroPulse.jl", rev = "40e39206…"}
   ```

   That URL still points at the **former `Limen-Neural` org**. Because it is a frozen
   `rev` pin, Limen-Capital does not break when the source repo is archived — but it will
   need UUID, URL, **and `rev`** updated as part of `rmems/Limen-Capital#9`.
4. **That consumer's dependency is declared but optional, and its routing code is a
   vendored fork.** `brain/adapters.jl` loads the package opportunistically
   (`@eval using TemporalFocus` inside a `try`, behind a `HAS_TEMPORAL_FOCUS` flag) and
   uses no symbol from it. The routing that actually runs is a **local copy** in
   `brain/synapse_conductor.jl`, which defines `NeroOrchestrator`, `update_relevance!`,
   `nero_diagnostics`, and the `NERO_*` constants itself; `brain/sonar_probe.jl` calls
   those local definitions, not the package's.

   Two consequences. First, **no package-level consumer of the legacy alias API has been
   found** — the aliases are retained because the vendored fork uses exactly those names
   and de-vendoring onto the consolidated package is the intended path, not because a
   live import would break. Second, the vendored copy is an older, 4-lobe `NERO`-named
   variant that has **drifted from NeuroPulse's generalized `ActivityRegion` API**;
   reconciling it is a cross-repo follow-up under `rmems/Limen-Capital#9`, and is an
   additional argument for consolidating rather than maintaining the split.
5. **`SpikeStream.jl`'s GitHub repository description is stale.** It advertises "Hurst
   exponent, Hawkes intensity, GBM surprise Z-score", all of which were removed in
   LIM-47 and now live in `rmems/kinetic-signals` (Rust). The shipped API is
   count/ISI/burst feature extraction only.
6. **`TemporalFocus.jl`'s current UUID is a documented placeholder.**
   [`RELEASING.md`](../../RELEASING.md) flags `7f3c9f2a-…` as a "template-looking"
   UUID and makes regenerating it a **blocker before first registration**. All three
   UUIDs are in fact obviously patterned. See [Package identity](#package-identity-and-uuid).

---

## Decision

`TemporalFocus.jl` becomes the single canonical repository and package for spike-native
temporal processing. Its boundary is deliberately **broadened once**, to add an
activity-routing kernel and spike-stream feature extraction:

### TemporalFocus will own

- spike events, spike trains, and temporal buffers *(today)*
- coincidence-based and temporally decayed spike interaction *(today)*
- temporal attention kernels with recency weighting *(today)*
- attention normalization — L1, max *(today)*
- synaptic/readout application over spike-derived weights *(today)*
- **spike-stream feature extraction** — counts, density, ISI statistics, burst
  detection, windowed and normalized feature vectors *(from SpikeStream.jl)*
- **an activity-routing kernel** — deterministic scoring and normalization over
  caller-provided activity summaries and readouts *(from NeuroPulse.jl)*. Issue #53 calls
  this "pure"; it is deterministic and self-contained but mutates its own router state,
  so the admissible limits are spelled out in
  [Why routing is admissible](#why-routing-is-admissible-but-the-rest-is-not) rather than
  carried by that one word.

### TemporalFocus will still not own

- STDP, Hebbian learning, reward-modulated plasticity, eligibility traces, or any
  weight update
- tokenization, embeddings, dense/transformer attention, gating, LLM fusion
- cross-modal projector weights between SNN and LLM spaces
- encoding or decoding logic
- neuromodulatory signals
- runtime execution, event-loop scheduling, telemetry ingestion, deployment
  supervision, or hardware control
- distillation
- finance/HFT semantics

This ADR therefore **supersedes only the "routing mechanisms" half of the README bullet
"distillation or routing mechanisms"** in the not-owned list. Distillation and the other
exclusions remain unchanged and are strengthened, not weakened, by this ADR.

### Why routing is admissible but the rest is not

The routing kernel is **deterministic and self-contained, but not pure**: `update_routing!`
mutates the router it is given, and the router carries a `readout_ema` across ticks. That
distinction matters, because purity is not what makes the exception acceptable — the
bounded mutation is. State the limits precisely, so this exception cannot be stretched:

The kernel may mutate **only the pre-allocated buffers of the `RegionRouter` passed to
it**. It must not perform I/O, read a clock or any ambient state, touch global state,
allocate on the hot path, or update a synaptic weight. Given the same router state and
the same `ActivityRegion` inputs, it produces the same outputs and the same next state.
It scores and normalizes; the caller owns the loop that feeds it, decides when to tick,
and owns the router's lifetime.

**Any future routing change that needs more than that is out of scope**, and reviewers
should treat the list above as the enforceable form of this exception rather than the
word "pure".

**`readout_ema` is not learning.** The router keeps an exponential moving average of each
region's readout to compute "surprise". Reviewers must not read this as plasticity: there
is no gradient, no reward signal, no synaptic weight, and no learning rule — it is a
first-order smoothing filter over observations the caller supplies. This distinction is
load-bearing for future boundary enforcement and must be stated in the module docstring.

---

## Pre-migration inventory

### `rmems/NeuroPulse.jl`

| Artifact | Detail |
|---|---|
| Source | `src/TemporalFocus.jl` (49), `src/activity_region.jl` (91), `src/region_router.jl` (603) |
| Tests | `test/runtests.jl` (1-line include), `test/test_nero.jl` (899) |
| Examples | `examples/three_region.jl`, `examples/six_region.jl`, `examples/reservoir_integration.jl` |
| Benchmarks | *(none)* |
| Docs | Documenter site under `docs/src/{index,api,overview,interop,roadmap}.md` **plus a duplicated flat copy** at `docs/{api,interop,overview,roadmap,README}.md`; `docs/make.jl`, `docs/Project.toml`, `docs/Manifest.toml`, `docs/logo.png` |
| Fixtures | *(none)* |
| CI | `.github/workflows/{ci,documentation,format}.yml`, `dependabot.yml` |
| Other | `.devcontainer/devcontainer.json`, `.devin/blueprint.yaml`, `LICENSE-MIT`, `LICENSE-APACHE-2.0` (**no top-level `LICENSE`**) |
| Open issues | #40 (post-transfer hygiene), #14 (standardize on common data shapes) |
| Open PRs | #41 (Dependabot: `julia-actions/cache` bump) |

### `rmems/SpikeStream.jl`

| Artifact | Detail |
|---|---|
| Source | `src/SpikeStream.jl` (26), `src/spike_features.jl` (247) |
| Tests | `test/runtests.jl` (245) |
| Examples | *(none)* |
| Benchmarks | `benchmark/{Project.toml,run_benchmarks.jl}` |
| Docs | `README.md` only (no Documenter site); `docs/logo.png` |
| Fixtures | `test/fixtures/spike_vectors.json` — frozen golden outputs + range invariants (LIM-41) |
| CI | `.github/workflows/{ci,codecov}.yml` |
| Other | `AGENTS.md`, `REVIEW.md`, `.markdownlint.json`, `LICENSE-MIT`, `LICENSE-APACHE` |
| Open issues | #27 (post-transfer hygiene) |
| Open PRs | #26 (ImgBot image optimization) |

### Cross-repository references to update or retire

| Reference | Location | Disposition |
|---|---|---|
| `TemporalFocus` UUID + `Limen-Neural/NeuroPulse.jl` source pin | `rmems/Limen-Capital` `brain/Project.toml`, `docs/deps.md`, `README.md` | **Cross-repo follow-up** under `rmems/Limen-Capital#9`. Not changed by this workstream. |
| Vendored fork of the routing kernel (`NeroOrchestrator`, `update_relevance!`, `NERO_*`) | `rmems/Limen-Capital` `brain/synapse_conductor.jl`, consumed by `brain/sonar_probe.jl` | **Cross-repo follow-up** under `rmems/Limen-Capital#9`: de-vendor onto the consolidated package. Drifted from the generalized `ActivityRegion` API; keeping the deprecated aliases makes that a rename-free change. |
| SpikeStream boundary contract | `rmems/kinetic-signals` `AGENTS.md`, `README.md`, `REVIEW.md`, `docs/boundary-matrix.md` | **Cross-repo follow-up in step 9:** repoint "SpikeStream.jl" to "TemporalFocus.jl (spike features)". The no-FFI, fixture-only integration contract is unchanged. Archiving is gated on this update. |
| `spikestream-jl-*` dataset cards, manifests, JSONL | `rmems/operation-prometheus` | **Leave as-is.** These are historical trajectory datasets keyed to the source repos. Archiving (not deleting) keeps their URLs resolvable. |
| `limen-neural.md`, `spikestream-jl.md` source-repo docs | `rmems/operation-prometheus/docs/source-repos/` | Leave as-is (provenance). |

---

## Symbol disposition

Every current public symbol has exactly one disposition. "Migrate" means the behavior
lands in `TemporalFocus` with parity tests ported first.

### `rmems/TemporalFocus.jl` (current package) — all retained, unchanged

`SpikeEvent`, `SpikeTrain`, `TemporalBuffer`, `prune!`, `temporal_weight`,
`spike_attention_discrete`, `spike_attention_temporal`, `spike_attention_continuous`,
`normalize_l1!`, `normalize_max!`, and the `Base.==` / `isequal` / `hash` / `show` /
`isempty` methods.

**The v0.1.0 public API is not broken by this consolidation.** The consolidating
release is additive for existing TemporalFocus consumers.

### `rmems/NeuroPulse.jl`

| Symbol | Disposition | Notes |
|---|---|---|
| `ActivityRegion` | **Migrate** | Pure `Float32` per-region summary (rate + readout vector). |
| `RegionRouter` | **Migrate** | Field `spike_density::Vector{Float32}` conflicts *conceptually* with SpikeStream's `spike_density` function — see [Name collisions](#name-collisions). |
| `RoutingConfig` | **Migrate** | Validation logic retained verbatim. |
| `update_routing!` | **Migrate** | Hot path, zero allocation. Retry-safe staging order (LIM-229/230) must be preserved exactly; it is behavior, not style. |
| `routing_diagnostics` | **Migrate, changed** | Only user of `Printf`. See [Zero-dependency policy](#zero-dependency-policy). |
| `save_state` | **Migrate** | Pure in-memory `NamedTuple` snapshot; no filesystem, no serialization format. Not runtime orchestration. |
| `load_state!` | **Migrate** | |
| `load_state` | **Migrate, deprecated** | Currently `const load_state = load_state!` — a non-bang name bound to a mutating function. Keep as a documented deprecated alias; steer callers to `load_state!`. |
| `adapt_leak!` | **Does not migrate — rehome** | See [Out of scope](#what-does-not-migrate). |
| `default_inhibition_matrix` | **Migrate** | Including the `n ≤ 4` historical-slice rule and the `0.08/\|i-j\|` decay rule for `n > 4`. |
| `LobeState`, `NeroOrchestrator` | **Migrate, deprecated aliases** | Used downstream. |
| `update_relevance!`, `nero_diagnostics` | **Migrate, deprecated aliases** | No package-level consumer found, but `rmems/Limen-Capital`'s **vendored fork** (`brain/synapse_conductor.jl`) uses these exact names. Keeping them is what makes de-vendoring onto this package a rename-free change. |
| `ALPHA`, `BETA`, `GAMMA`, `EMA_DECAY`, `MIN_SCORE`, `EPSILON` | **Migrate, renamed** | Unexported but far too generic for a package that also owns attention. Rename them to `ROUTING_ALPHA`, `ROUTING_BETA`, `ROUTING_GAMMA`, `ROUTING_EMA_DECAY`, `ROUTING_MIN_SCORE`, and `ROUTING_EPSILON`; the deprecated `NERO_*` constants bind to these values. |
| `NERO_ALPHA`, `NERO_BETA`, `NERO_GAMMA`, `NERO_EMA_DECAY`, `NERO_MIN_SCORE`, `NERO_EPSILON` | **Migrate, deprecated aliases** | |
| `DEFAULT_REGION_NAMES`, `NERO_DEFAULT_LOBE_NAMES` | **Migrate** | Latter deprecated. |
| `INHIBIT`, `NERO_INHIBIT` | **Migrate** | The hardcoded asymmetric 4×4 matrix is arbitrary legacy tuning; keep it as the documented default and say so. |
| `Base.setproperty!(::RegionRouter, …)` | **Migrate** | Guards `config` replacement against `min_score` infeasibility. Easy to lose in a port; it is load-bearing. |

#### Deprecation mechanics caveat

`update_relevance!` and `nero_diagnostics` are `const` bindings to functions, not
methods. `Base.depwarn` cannot be attached to a `const` alias. To emit real deprecation
warnings the aliases must become thin wrapper functions:

```julia
update_relevance!(args...; kwargs...) = (Base.depwarn(...); update_routing!(args...; kwargs...))
```

That costs one extra call frame outside the hot loop. **Recommendation:** convert to
wrapper functions and accept the frame; a silent alias that disappears in a later
release is worse. If the wrapper is rejected on performance grounds, the deprecation
must be documentation-only and stated as such in the migration guide.

**The wrapper treatment applies to function aliases only.** The deprecated set is not
homogeneous, and turning all of it into functions would break the compatibility promise
it exists to keep:

| Alias kind | Members | Treatment |
|---|---|---|
| Function aliases | `update_relevance!`, `nero_diagnostics`, `load_state` | **Conditional on open question 5.** If wrappers are selected, wrapper function + `depwarn`; if zero-cost aliases are selected, keep `const` aliases and make deprecation documentation-only. |
| **Type** aliases | `LobeState`, `NeroOrchestrator` | **Stay type aliases.** A wrapper function would break `x::LobeState`, `isa(x, LobeState)`, and dispatch. Deprecation is documentation-only. |
| **Constant** aliases | `NERO_ALPHA`, `NERO_BETA`, `NERO_GAMMA`, `NERO_EMA_DECAY`, `NERO_MIN_SCORE`, `NERO_EPSILON`, `NERO_INHIBIT`, `NERO_DEFAULT_LOBE_NAMES` | **Stay constants.** A wrapper function would break arithmetic (`NERO_ALPHA * x`) and indexing. Deprecation is documentation-only. |

Julia has no general mechanism for warning on the *use* of a deprecated type or constant
binding without changing what that binding is, so for those two rows the honest answer is
a documented deprecation, not a silent pretence that a warning will fire.

### `rmems/SpikeStream.jl`

| Symbol | Disposition | Notes |
|---|---|---|
| `spike_count` | **Migrate** | Signature and `Int` return preserved. |
| `spike_density` | **Migrate** | `Float64` return preserved. |
| `isi_stats` | **Migrate** | `NamedTuple{(:mean,:std,:min,:max,:cv)}` of `Float64` preserved. |
| `detect_bursts` | **Migrate** | Returns index ranges **into the sorted sequence**, not into the caller's input order. This is a latent footgun when adapting from a `SpikeTrain`; it must be documented at the adapter boundary, not just in the function docstring. |
| `windowed_spike_features` | **Migrate** | The `include_right_edge` / `nextfloat` boundary rule is subtle and parity-critical. Port fixture-first. |
| `normalized_feature_vector` | **Migrate** | `[count_norm, density_norm, isi_cv_norm, burst_norm]`, each in `[0,1]`. Documented output ranges are a compatibility contract with `rmems/kinetic-signals`. |
| `isi_stats_sorted`, `detect_bursts_sorted` | **Migrate, renamed** | Not exported. Rename to `_isi_stats_sorted` / `_detect_bursts_sorted` per this repo's `_`-prefix convention. |
| `test/fixtures/spike_vectors.json` | **Migrate verbatim** | Frozen golden values. Do not regenerate; regenerating would silently redefine the parity contract. |

The migrated feature names are an explicit public-API naming exception: parity requires
`spike_count`, `spike_density`, `isi_stats`, `detect_bursts`,
`windowed_spike_features`, and `normalized_feature_vector` to retain their SpikeStream
names. Step 3 extends `AGENTS.md`'s public naming policy with this feature-extraction
family in the same PR; internal helpers remain `_`-prefixed.

### What does not migrate

| Symbol / concern | Why | Where it should live |
|---|---|---|
| `adapt_leak!` | Maps a hardware/thermal **stress signal** onto a neuron **leak rate**. That is telemetry ingestion plus neuron-dynamics mutation plus hardware co-design — excluded by `AGENTS.md` ("runtime execution") and by issue #53's non-goals ("telemetry ingestion, deployment supervision, or hardware-control ownership"). It also has nothing to do with routing: it never touches a `RegionRouter`. | A neuron-dynamics or runtime package. Candidates: `rmems/thalamic-relay` (hardware orchestration relay), the `brainstem-daemon` runtime workspace named in rmems/.github#3, or `rmems/LiquidCortex.jl` if the leak rate belongs to reservoir dynamics. **Owner decision required** — see [Open questions](#open-questions-requiring-a-human-decision). |
| NeuroPulse `docs/{api,interop,overview,roadmap,README}.md` flat copies | Duplicates of the `docs/src/` versions. | Retired. Only the `docs/src/` variants are ported. |
| NeuroPulse `.devin/blueprint.yaml`, `.devcontainer/` | Tooling for a different agent/dev-container setup than this repo uses. | Retired, not migrated. |
| Hurst / Hawkes / GBM surprise | Already removed from SpikeStream in LIM-47. | `rmems/kinetic-signals` (Rust). Must not be reintroduced here. |
| STDP / plasticity | Removed from TemporalFocus in v0.1.0. | A dedicated plasticity package (`plasticity-lab`). |

---

## Target shape

```text
TemporalFocus.jl/
├── src/
│   ├── TemporalFocus.jl        # module, exports, includes
│   ├── types.jl                # SpikeEvent, SpikeTrain, TemporalBuffer  (existing)
│   ├── discrete.jl             # spike_attention_discrete               (existing)
│   ├── temporal.jl             # temporal_weight, ..._temporal          (existing)
│   ├── continuous.jl           # spike_attention_continuous             (existing)
│   ├── normalization.jl        # normalize_l1!, normalize_max!          (existing)
│   ├── spike_features.jl       # from SpikeStream.jl                    (new)
│   ├── adapters.jl             # SpikeTrain <-> timestamp vectors       (new)
│   ├── activity_region.jl      # from NeuroPulse.jl                     (new)
│   └── activity_routing.jl     # from NeuroPulse.jl region_router.jl    (new)
├── test/                       # merged suites + ported fixtures
├── examples/                   # 4 existing + 3 ported routing examples
├── benchmark/                  # existing attention suite + ported feature suite
└── docs/
    ├── adr/                    # this document
    └── src/                    # index, api, migration guide
```

Single flat module (`TemporalFocus`), file-level separation, no submodules. Submodules
would force `TemporalFocus.SpikeFeatures.spike_count` on callers for no isolation
benefit, since the whole point is one importable namespace.

---

## Package identity and UUID

Issue #53 says to use `TemporalFocus.jl`'s UUID "unless a downstream dependency audit
proves another choice is safer". The audit is done, and it surfaces a conflict the issue
did not anticipate.

**Audit result.** The only live consumer (`Limen-Capital`) depends on the **NeuroPulse**
UUID `b7e4c3f2-…`, not on `7f3c9f2a-…`. But it does so through a frozen git `rev` pin,
so it cannot break spontaneously; migrating it is a two-line edit whenever its owner
chooses.

**Conflict.** `RELEASING.md` designates `7f3c9f2a-…` a template placeholder and makes
regenerating it a **hard blocker before first registration**. Issue #53 says to *retain*
it. Both cannot hold at registration time.

### Recommendation

Regenerate **before** the consolidated release, not after it.

1. Perform the `RELEASING.md`-mandated `uuid4()` regeneration **first**, as a single
   dedicated change (step 2 of the [migration sequence](#migration-sequence)), before any
   source is imported. `7f3c9f2a-…` is retired along with the other two.
2. Retire `b7e4c3f2-…` (NeuroPulse) and `a3c7f1e2-…` (SpikeStream). None of the three is
   registered; none can be squatted.
3. Ship `v0.2.0` — the consolidated release — already carrying the final identity.

**Why this ordering.** The obvious alternative is to hold `7f3c9f2a-…` through
consolidation and regenerate later, just before registration. That is wrong: a consumer
who adopts `v0.2.0` would then have to migrate a *second* time when the UUID is
regenerated. Regenerating first means **every consumer sees exactly one identity change,
ever**. The consumer audit — including package declarations, Git source URLs, and fixed
`rev` pins — found no current consumer pinned to `7f3c9f2a-…`. The absence of a tag or
registry entry is supporting context, not evidence that a Git consumer could not be
pinned to the UUID.

**This deviates from the literal wording of issue #53**, which says to retain
TemporalFocus's UUID. It honors the intent — one canonical identity, chosen by
downstream audit — while also clearing `RELEASING.md`'s registration blocker instead of
deferring it. Owner sign-off required.

---

## Adapters between the two data models

`SpikeStream` consumes bare timestamp vectors (`AbstractVector{<:Real}`). `TemporalFocus`
carries `SpikeEvent(neuron_id::Int, t::Float32, value::Float32)`. A naive
`spike_times(train) = [e.t for e in train.events]` would silently discard **both**
`neuron_id` and `value` and mix neurons into one ISI sequence — producing plausible,
wrong numbers. Issue #53 forbids this, and it is the single highest-risk part of the
migration.

### Contract

| Adapter | Behavior |
|---|---|
| `spike_times(train::SpikeTrain, neuron_id::Integer) -> Vector{Float64}` | Explicit single-neuron projection. `neuron_id` is required — there is no whole-train overload. |
| `spike_times_by_neuron(train::SpikeTrain) -> Dict{Int,Vector{Float64}}` | Groups by neuron; IDs are preserved as keys. |
| `spike_features_by_neuron(train; …)` | Per-neuron feature extraction; returns results keyed by neuron ID. |
| `SpikeTrain(times::AbstractVector{<:Real}; neuron_id, value=1.0f0)` | Reverse direction. `neuron_id` is a required keyword. |
| Same set for `TemporalBuffer`, **plus a required `current_time`** | See below. |

Every projection and feature adapter validates that its selected events have finite
timestamps **before** widening `Float32` to `Float64`. This applies to pre-existing
`SpikeTrain` and `TemporalBuffer` values as well as to the reverse constructor: a caller
can construct `SpikeEvent(..., NaN32, ...)` or `SpikeEvent(..., Inf32, ...)` directly, so
ingest-only checks are insufficient.

**`TemporalBuffer` needs a reference time.** A `TemporalBuffer` stores only `window` and
`events`; it does **not** store the reference time that decides which events are currently
inside the window — which is why the existing [`prune!`](../../src/types.jl) takes
`current_time` explicitly. So a buffer adapter cannot "honor `buffer.window`" from the
buffer alone: without a cutoff it would let stale events silently into feature
calculations. Buffer adapters therefore take `current_time` as a required argument and
reject it unless it is finite **before** selecting events. They then apply the same
`(current_time - event.t) <= buffer.window` rule `prune!` uses, so an
adapted buffer and a pruned one agree exactly. (Requiring callers to `prune!` first was
considered and rejected: it mutates the caller's buffer and fails silently if skipped.)

**Handling `value`.** Every SpikeStream feature is count- and timing-based; none consume
spike amplitude. Rather than discard `value` quietly, adapters take an explicit
`ignore_values::Bool` keyword that **defaults to erroring** when a train contains any
`value != 1.0f0`. This includes uniform non-unit-amplitude trains: every non-unit value
requires `ignore_values=true`, forcing the caller to acknowledge what is being dropped.

**`detect_bursts` index provenance — decided: adapters return time intervals.** The
migrated `detect_bursts` returns ranges into the *sorted* timestamp sequence, so after adapting
from a `SpikeTrain` those indices no longer address `train.events`. Leaving the choice
open would let two conforming implementations produce incompatible public APIs, so it is
settled here:

- **The migrated `detect_bursts(times; …)` keeps its `Vector{UnitRange{Int}}` return
  unchanged**, for parity with SpikeStream v0.1.0 and the frozen fixtures.
- **Adapter-level burst results are time intervals**, `Vector{Tuple{Float64,Float64}}`
  of `(t_first, t_last)`. Indices into a sorted internal copy are meaningless to a caller
  holding a `SpikeTrain`; times are meaningful in the caller's own frame and survive
  re-sorting, filtering, and per-neuron grouping.

This is an explicit gate on step 4: the adapter PR implements the interval form, not the
permutation form.

---

## Precision policy

`TemporalFocus` is `Float32`-native. `SpikeStream` computes and returns `Float64`
throughout (`sort(Float64.(spike_times))`).

### The governing Float32 rule must be scoped, not ignored

`AGENTS.md`'s code-style rule reads "Float32 for all spike values and temporal
quantities". Taken literally that forbids the `Float64` feature API below, which would
leave every later migration PR unable to implement this ADR while staying
repository-compliant. **This ADR resolves that conflict explicitly rather than deferring
it**, by scoping the rule instead of weakening it:

> `Float32` governs the **spike data model and the kernels over it** — `SpikeEvent.t`,
> spike `value`, attention weights, readout vectors, routing scores and summaries.
> **Derived statistical features** (ISI moments, densities, normalized feature vectors)
> are computed and returned in `Float64`.

Nothing that was `Float32` becomes `Float64`; the rule's existing surface is untouched.
`AGENTS.md` is amended to this wording **in step 3, the same PR that lands the first
`Float64` kernel** — not at step 7. Deferring it would leave `main` shipping derived
`Float64` quantities against a still-active `Float32` rule for several PRs.

### Policy

1. **Feature kernels keep `Float64` internals and `Float64` return types**, unchanged
   from SpikeStream v0.1.0. ISI variance and coefficient of variation are exactly the
   places where `Float32` accumulation degrades; there is no benefit in narrowing them.
   Frozen fixtures continue to match bit-for-bit.
2. **`SpikeTrain` → features is exact.** Widening the stored `Float32` `t` to `Float64`
   is lossless, so features computed from a `SpikeTrain` are reproducible.
3. **The lossy direction is ingest.** It cannot be made lossless — `Float32` storage is
   the data model — but the failures that change results *qualitatively* must not be
   silent. See the guard modes below for exactly what is and is not promised.

### The narrowing hazard, concretely

`Float32` carries ~7 significant decimal digits. At timestamps around `1e6` seconds the
`Float32` ulp is ≈ `0.0625 s`, so two spikes 1 ms apart collapse to the *same* `t` when
stored in a `SpikeEvent`. Downstream that is not a rounding nuisance, it changes results:

- `isi_stats` yields `min = 0`, deflated `mean`, and a materially different `cv`;
- `detect_bursts` sees zero-length ISIs, which are `≤ max_isi` by definition, so burst
  counts **inflate**;
- `normalized_feature_vector`'s `burst_norm` and `isi_cv_norm` components move with them.

**Mitigation, and the limit of what it promises.** `SpikeTrain(times; …)` takes
`check_precision::Symbol = :collisions`:

| Mode | Rejects | Use when |
|---|---|---|
| `:collisions` *(default)* | narrowing that **merges timestamps that were distinct**, or that exceeds `abs(t) > 2^24` | normal ingest |
| `:strict` | **any** timestamp that does not round-trip `Float64 → Float32 → Float64` | exact parity with a `Float64` pipeline is required |
| `:none` | non-finite values only (see below) | the caller has accepted lossy ingest |

**Non-finite timestamps are rejected in every checked mode.** A `NaN` neither merges two
distinct timestamps nor exceeds `2^24`, so the collision and magnitude rules alone would
wave it through — and it then poisons every ISI statistic downstream. `:collisions` and
`:strict` therefore also reject any non-finite `t`, and even `:none` rejects it: accepting
lossy ingest is not the same as accepting a value that is not a time.

The default deliberately does **not** promise that ingest is lossless. Ordinary decimal
values — `0.1`, `0.2` — are altered by narrowing without merging or exceeding `2^24`, so
`:collisions` accepts them and their ISIs shift in the last few digits. That is inherent
to a `Float32` data model, not a bug the guard can remove: `:strict` would reject almost
every real decimal timestamp. So the guarantee is scoped precisely — **the default
prevents the failures that change results qualitatively** (zero ISIs, inflated burst
counts, reordering), and `:strict` is available when bit-level parity matters. Sub-ulp
rounding under `:collisions` is documented, not prevented.

**Convenience.** Provide `Float32` wrappers (e.g. `normalized_feature_vector_f32`)
rather than changing any existing return type.

### Required edge-case tests

Non-negotiable before the SpikeStream source is retired:

- empty input; single spike; two spikes
- unsorted input; duplicated timestamps; negative timestamps
- `NaN` / `Inf` timestamps
- **large-magnitude timestamps that collide under `Float32` narrowing** (the hazard above)
- a value that narrows lossily **without** colliding (e.g. `0.1`): accepted under
  `:collisions`, rejected under `:strict` — pins down exactly what the default promises
- `abs(t) > 2^24`
- multi-neuron trains adapted per-neuron, asserting no cross-neuron ISI leakage
- non-uniform `value` trains hitting the `ignore_values` guard
- `TemporalBuffer` adapters: events outside `window` relative to `current_time` are
  excluded, and the adapted result equals `prune!`-then-adapt exactly
- `windowed_spike_features` right-edge inclusion at the `nextfloat` boundary
- zero-duration and negative-duration windows
- `detect_bursts` index provenance after adaptation

---

## Name collisions

| Collision | Resolution |
|---|---|
| `spike_density` — SpikeStream free function (`Float64`, spikes per unit time) vs `RegionRouter.spike_density::Vector{Float32}` field (per-region normalized rate in `[0,1]`) | Not a Julia-level collision (the field is namespaced by the struct) but a genuine conceptual one in a single package. **Rename the field** to `region_spike_rate` and keep the free function's name. The rename is a breaking read-access change for anyone touching `router.spike_density` — flagged in [Open questions](#open-questions-requiring-a-human-decision). |
| `ALPHA` / `BETA` / `GAMMA` / `EPSILON` / `MIN_SCORE` / `EMA_DECAY` | **Rename with a `ROUTING_` prefix.** This preserves named values for the deprecated `NERO_*` aliases while removing the generic unqualified bindings. `RoutingConfig()` continues to use the same values as defaults. |
| `normalize_l1!` / `normalize_max!` vs the router's inline floored softmax | **Keep both; do not unify.** The router performs softmax, then a `min_score` floor, then a renormalization that preserves the floor — different semantics from L1 or max normalization. Document why they coexist so a future cleanup PR does not "simplify" them together. |
| `save_state` / `load_state!` | Generic names, but they dispatch on `RegionRouter` and are router-only. Acceptable; document the restriction. |
| `ActivityRegion.last_spike_rate` vs `spike_density(times)` | Different units (`[0,1]` normalized vs spikes/second). Document explicitly; the adapter must never feed one into the other unscaled. |

---

## Zero-dependency policy

`TemporalFocus.jl`'s root `Project.toml` has an **empty `[deps]`**. The recommended path
keeps it empty. Naively merging both packages would add `LinearAlgebra`, `Printf`, and
`Statistics`; all three can be avoided, subject to the diagnostics-format decision:

| Dependency | Actual usage | Disposition |
|---|---|---|
| `LinearAlgebra` (NeuroPulse) | **Dead import.** `region_router.jl:16` has `using LinearAlgebra: norm`, but `norm(` appears only inside a docstring at line 277 — the implementation computes norms manually in `Float64` for overflow safety. | **Drop.** No behavior change. |
| `Printf` (NeuroPulse) | Three `@sprintf` calls in `routing_diagnostics` only. | **Conditional on open question 3.** If changed formatting is accepted, drop it and use `round(x; digits=n)` plus interpolation, test the string structure, and document the intentional change. If byte-identical output is required, retain `Printf` and its golden-string coverage. |
| `Statistics` (SpikeStream) | `mean(isis)` and `std(isis)` at `spike_features.jl:76-77` only. | **Drop**; inline both. `std` must replicate the **corrected (n−1) denominator** exactly, or the frozen fixtures will fail. That fixture failure is the intended safety net — verify it fires before trusting the reimplementation. |

**Recommended result: the consolidated package still ships with an empty `[deps]`.** It
is worth preserving because it minimizes resolution risk, but step 5 may not claim that
result or remove `Printf` until open question 3 is answered.

---

## Migration sequence

Each step is a separately reviewable PR. Steps 1–8 land before any repository is
archived. Steps 9–11 are human/cross-repo actions outside this repository.

**Every implementation step carries its own boundary update.** Because each step lands
on `main` independently, deferring the whole scope rewrite to one late PR would leave
`main` shipping code the enforced `AGENTS.md`/`README.md`/`REVIEW.md` scope says does not
belong here — for several PRs. So steps 3 and 5 each extend the Scope list with exactly
the surface they land, in the same PR. Step 7 is then the consolidated rewrite (the
one-sentence README lead, `REVIEW.md`'s checklist, verification of the scoped `Float32`
rule already updated in step 3, and the scope tests), not the first time the boundary
text moves.

| # | Step | Gate before proceeding |
|---|---|---|
| 1 | **This ADR** — inventory, dispositions, boundary decision | Owner accepts the boundary broadening and the UUID recommendation |
| 2 | **Regenerate the canonical UUID** per `RELEASING.md` (`Project.toml`, `docs/Project.toml`, `benchmark/Project.toml`, and `Manifest.toml` via `Pkg.resolve()`); atomically update `RELEASING.md`'s UUID-hygiene section so it records the new UUID and no longer instructs a second regeneration | The audit found nothing dependent on `7f3c9f2a-…`; done **before** any import so `v0.2.0` ships the final identity and no consumer migrates twice |
| 3 | Port SpikeStream feature kernels + `test/fixtures/spike_vectors.json`, `Statistics` inlined; **add spike-stream feature extraction to the Scope lists and naming policy in the same PR** | Frozen fixtures pass unmodified; `[deps]` still empty; boundary and naming text match what `main` now ships |
| 4 | Add `SpikeTrain` ↔ timestamp adapters + the full edge-case suite from [Precision policy](#precision-policy), including pre-built events with non-finite timestamps, non-finite buffer `current_time` values, and uniform non-unit values | Narrowing-collision, non-finite projection, and non-finite reference-time tests fail loudly without their guards and pass with them; every non-unit value requires `ignore_values=true` |
| 5 | Port `ActivityRegion` + routing kernel, `LinearAlgebra` dropped, generic constants renamed to the decided `ROUTING_*` family; exclude `adapt_leak!` only when its destination or explicit retirement is decided, otherwise carry it temporarily as deprecated; rename `spike_density` only if accepted, otherwise retain the existing field name; drop `Printf` only if changed diagnostics formatting is accepted, otherwise preserve it; **add the routing kernel to the Scope lists in the same PR**, with the admissible-mutation limits | NeuroPulse's suite ports and passes; `adapt_leak!` tests are omitted only if its removal is decided, otherwise they remain; boundary text matches what `main` now ships; diagnostics formatting is decided and tested before `Printf` is removed; unresolved `adapt_leak!` and field-name decisions use the documented compatibility fallbacks below |
| 6 | Port deprecated aliases, **preserving each binding's kind** — see below | Function aliases follow the mechanism selected in open question 5 and warn only if wrappers were selected; `LobeState` remains usable in `::`/`isa`; `NERO_ALPHA` remains arithmetic and binds to `ROUTING_ALPHA` |
| 7 | Consolidated docs pass: README one-sentence lead, `REVIEW.md` checklist, verification of the scoped `Float32` rule landed in step 3, `docs/src/`, and scope tests asserting the exported symbol set | Stated boundary matches shipped code exactly |
| 8 | Port examples and benchmarks; write `docs/src/migration.md` with the [upgrade matrix](#upgrade-matrix); bump to `0.2.0`, update `CHANGELOG.md`, **and update `RELEASING.md`'s first-registration procedure** from v0.1.0 to v0.2.0 (the UUID-hygiene section was already corrected atomically in step 2) | Fresh-clone `Pkg.instantiate()` + `Pkg.test()` green on the full CI matrix |
| 9 | Add redirect READMEs to `NeuroPulse.jl` and `SpikeStream.jl`; update `rmems/kinetic-signals` boundary docs to name `TemporalFocus.jl (spike features)` | *Cross-repo; requires write access to those repos; all three documentation updates must land before archiving* |
| 10 | Migrate/triage open issues (NeuroPulse #40, #14; SpikeStream #27) and open PRs (NeuroPulse #41, SpikeStream #26) per rmems/.github#4 | Every open item has a documented destination |
| 11 | **Archive** `NeuroPulse.jl` and `SpikeStream.jl` | Steps 8–10 complete **and** the v0.2.0 upgrade path is published |

**Archiving is the last step and is a deliberate human action.** No automation in this
workstream may archive, transfer, or delete a repository. Source repositories remain as
provenance and must never be deleted (rmems/.github#3, migration policy 9).

### Gate: `adapt_leak!` may not be removed into a void

`adapt_leak!` is a working public function, and issue #53 promises that no repository is
archived before its supported upgrade path is published. Removing it while its
destination is still an open question would hand a consumer a migration guide that points
nowhere. Step 5 therefore has a compatibility fallback, while release and archive remain
hard gates:

- **Step 5** (conditional removal) — remove it only when the destination or an explicit
  no-successor retirement is decided and recorded; otherwise carry it temporarily as a
  deprecated, documented-as-out-of-scope compatibility function.
- **Step 8** (`v0.2.0`) — the migration guide either names where to get it, with a working
  reference (a repository and, if it has landed, a version or commit), **or** plainly
  documents the owner's decision to retire it without a successor.
- **Step 11** (archive) — the replacement is published, **or** the owner has explicitly
  retired the function with no successor and the guide says so plainly.

"Decided" means a named destination, not an intention. If the decision is still open when
step 5 comes up, the correct move is to carry `adapt_leak!` forward temporarily as a
deprecated, documented-as-out-of-scope function rather than to drop it — an acknowledged
scope wart beats a broken upgrade path.

### Gate: the `spike_density` field name must be settled before the router lands

[Open question 4](#open-questions-requiring-a-human-decision) — whether
`RegionRouter.spike_density` is renamed to `region_spike_rate` — decides a **public struct
layout**. An implementer starting step 5 without an answer cannot know which field to
port, which assertions to write, or which name the migration guide should teach; and
changing it after the router lands is a second breaking change for anyone reading router
state.

So step 5 is gated on it, with the same shape of fallback as `adapt_leak!`: **if the
decision is still open, port the field under its existing name `spike_density`** and
leave the rename to a later, deliberate breaking change. Landing the old name is
reversible; landing the wrong new one is not.

### History preservation

`git subtree add --prefix=… <remote> main` (or `git merge --allow-unrelated-histories`
over a filtered branch) preserves the source commit history in this repository at import
time. If subtree import proves noisy against nine concurrent branches, the fallback is a
plain file copy plus a permanent provenance record: source repo URL, imported commit SHA,
and import date, recorded in this ADR and in `CHANGELOG.md`. **The archived source
repositories are themselves the authoritative history** — history preservation here is a
convenience, not the safety net.

---

## Upgrade matrix

Published as `docs/src/migration.md` at step 8. Recorded here so the plan is reviewable
before the code exists.

`v0.2.0` ships the **regenerated canonical UUID** from step 2, written below as
`<canonical>`. Because regeneration happens before the release, every consumer migrates
its identity exactly once; there is no second transition.

| Consumer | Today | After v0.2.0 | Break? |
|---|---|---|---|
| **TemporalFocus v0.1.0 consumers** | `TemporalFocus` @ `7f3c9f2a-…`, attention API | `TemporalFocus` @ `<canonical>`, same API, plus features + routing | **API: no**, purely additive. **UUID: yes.** The consumer audit, including Git URLs and fixed revisions, found no current consumer pinned to this identity; lack of tags or registration alone would not make a Git package unpinnable. |
| **NeuroPulse consumers** (`rmems/Limen-Capital`) | `TemporalFocus` @ `b7e4c3f2-…`, `[sources]` → `Limen-Neural/NeuroPulse.jl` @ `40e39206…`; loaded optionally, routing actually vendored | `TemporalFocus` @ `<canonical>`, `[sources]` → `rmems/TemporalFocus.jl` **with `rev` replaced by a consolidated commit (or dropped in favour of the `v0.2.0` tag)**; `update_relevance!` etc. keep working and warn only if wrappers are selected in open question 5, otherwise their deprecation is documentation-only; `adapt_leak!` follows the decided replacement/retirement path or remains temporarily deprecated while that decision is open | **Yes, three:** UUID, source URL, **and the `rev` pin** must change — leaving `rev = "40e39206…"` in place would either fail to resolve or, if history is imported, check out the old NeuroPulse tree and its `b7e4c3f2-…` identity instead of consolidated v0.2.0. Any decided `adapt_leak!` removal adds its documented replacement or explicit retirement; otherwise compatibility is retained temporarily. Low urgency in practice — the `rev` pin is frozen and the load is behind a `try`, so nothing breaks until they choose to de-vendor. |
| **SpikeStream consumers** | `SpikeStream` @ `a3c7f1e2-…`, tag `v0.1.0` | `using TemporalFocus` @ `<canonical>`; function names, signatures, and `Float64` return types unchanged | **Yes, one:** package name + UUID. No API change. |
| **`rmems/kinetic-signals`** (Rust) | Fixture-parity contract with `SpikeStream.jl`, no FFI | Same fixtures, now under `TemporalFocus.jl` | Docs-only. Output ranges are unchanged and remain a contract. |

No repository is archived before its supported upgrade path is published.

---

## Risks

| Risk | Mitigation |
|---|---|
| **Boundary dilution.** Three concerns in one package makes future scope arguments harder to win. | Keep file-level module boundaries; add scope tests asserting the exact exported symbol set; keep `AGENTS.md` / `REVIEW.md` exclusions verbatim and strengthened. |
| **`readout_ema` misread as learning state**, inviting a future STDP PR. | Stated explicitly in the module docstring and in `REVIEW.md`'s boundary checklist. |
| **Silent `Float32` corruption of features.** | The `check_precision` guard plus the dedicated collision test in step 4. |
| **`windowed_spike_features` boundary semantics** are subtle (`include_right_edge`, `nextfloat`) and easy to "clean up" wrongly. | Port fixture-first; never regenerate `spike_vectors.json`. |
| **Julia compat floor.** TemporalFocus targets 1.9–1.12 (+ macOS/Windows on 1.11); NeuroPulse's practice was **1.12-only**; SpikeStream tests `min`/`1`/`pre`. Imported code may not actually run on 1.9. | Validate the ported routing kernel on 1.9 in step 5 **before** claiming the compat range. Raise the floor deliberately if needed — do not discover it in a release. |
| **Test-suite merge.** 583 + 899 + 245 = 1727 lines of tests across three conventions. | Merge as separate top-level `@testset`s per step; never rewrite an assertion while moving it. |
| **Concurrent branches.** Eight sibling workstreams are touching this repository. | Keep each step's diff narrow; expect `README.md` / `CHANGELOG.md` conflicts and resolve additively. |
| **Premature archiving** strands Limen-Capital's pin or open issues. | Step 11 is gated on steps 8–10 and is a human action. |

---

## Open questions requiring a human decision

1. **UUID.** Accept the [recommendation](#package-identity-and-uuid) — regenerate
   `uuid4()` at step 2, before the consolidated release, so no consumer migrates its
   identity twice — even though it deviates from the literal wording of #53?
2. **Where does `adapt_leak!` go?** `thalamic-relay`, the `brainstem-daemon` workspace,
   or `LiquidCortex.jl`? This blocks step 5 under the
   [removal gate](#gate-adapt_leak-may-not-be-removed-into-a-void); until it is answered,
   the fallback is to carry the function forward as deprecated rather than drop it.
3. **`Printf` vs. zero-dep diagnostics.** Accept a changed `routing_diagnostics` output
   format to keep `[deps]` empty, or retain `Printf` and keep byte-identical output? Step 5
   is gated on this decision; until then it must preserve the current formatting path.
4. **Rename `RegionRouter.spike_density` → `region_spike_rate`?** It is a breaking
   read-access change for any consumer inspecting router state. This blocks step 5 under
   the [field-name gate](#gate-the-spike_density-field-name-must-be-settled-before-the-router-lands);
   the fallback is to port the existing name.
5. **The three *function* aliases as wrapper functions** (real `depwarn`, one extra
   frame) or documentation-only deprecation (zero cost, silent)? The type and constant
   aliases are already settled — they keep their binding kind and are documented, not
   wrapped.
6. **Confirm** that Limen-Capital's dependency migration is owned by
   `rmems/Limen-Capital#9` and is not blocking on this repository.

---

## Status against issue #53's acceptance criteria

| Criterion | Status |
|---|---|
| Boundary documents the deterministic, self-contained routing kernel with bounded mutation and excludes runtime/learning/dense/finance | **Decided here**; the code and README/AGENTS text land together at step 5 and are consolidated at step 7 |
| Fresh-clone `Pkg.instantiate()` + `Pkg.test()` succeed | Passing today; re-gated at step 8 |
| Every inventoried artifact retained, migrated, or explicitly retired with rationale | **Planned** — source symbols and major artifacts are dispositioned here; detailed retention/migration/retirement rationales for CI workflows, Dependabot configuration, licenses, and documentation still land in the implementation steps |
| Imported behavior has parity tests or documented intentional changes | **Planned** (steps 3, 5); intentional changes recorded here |
| Timestamp adaptation and precision edge-case tests | **Specified** here; land at step 4 |
| Only one active package UUID/name pair | **Decided** here; identity set at step 2, shipped at step 8 |
| Versioned migration guide and upgrade matrix | **Drafted** here as [Upgrade matrix](#upgrade-matrix); published at step 8 |
| No active docs instruct users to depend on superseded repositories | Steps 7–9 |
| `NeuroPulse.jl` / `SpikeStream.jl` archived with successor links | Step 11, human action, explicitly **not** automated |
| README explains the whole accomplishment in one sentence before modules | Step 7 |

---

## References

- [rmems/TemporalFocus.jl#53](https://github.com/rmems/TemporalFocus.jl/issues/53) — this workstream
- [rmems/.github#3](https://github.com/rmems/.github/issues/3) — portfolio consolidation umbrella
- [rmems/.github#4](https://github.com/rmems/.github/issues/4) — issue/Linear migration
- [rmems/Limen-Capital#9](https://github.com/rmems/Limen-Capital/issues/9) — downstream consumer workstream
- [`AGENTS.md`](../../AGENTS.md) — scope boundary and code style
- [`REVIEW.md`](../../REVIEW.md) — reviewer boundary checklist
- [`RELEASING.md`](../../RELEASING.md) — UUID hygiene blocker and registration process
