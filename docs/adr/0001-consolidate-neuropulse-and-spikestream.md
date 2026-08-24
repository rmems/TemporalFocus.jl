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
   `rev` pin, Limen-Capital does not break when the source repo is archived — but the
   pin will need one edit (UUID + URL) as part of `rmems/Limen-Capital#9`.
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
temporal processing. Its boundary is deliberately **broadened once**, to add a pure
activity-routing kernel and spike-stream feature extraction:

### TemporalFocus will own

- spike events, spike trains, and temporal buffers *(today)*
- coincidence-based and temporally decayed spike interaction *(today)*
- temporal attention kernels with recency weighting *(today)*
- attention normalization — L1, max *(today)*
- synaptic/readout application over spike-derived weights *(today)*
- **spike-stream feature extraction** — counts, density, ISI statistics, burst
  detection, windowed and normalized feature vectors *(from SpikeStream.jl)*
- **a pure activity-routing kernel** — deterministic scoring and normalization over
  caller-provided activity summaries and readouts *(from NeuroPulse.jl)*

### TemporalFocus will still not own

- STDP, Hebbian learning, reward-modulated plasticity, eligibility traces, or any
  weight update
- tokenization, embeddings, dense/transformer attention, gating, LLM fusion
- cross-modal projector weights between SNN and LLM spaces
- runtime execution, event-loop scheduling, telemetry ingestion, deployment
  supervision, or hardware control
- finance/HFT semantics

This ADR therefore **supersedes the README bullet "distillation or routing mechanisms"**
in the not-owned list. The remaining exclusions are unchanged and are strengthened, not
weakened, by this ADR.

### Why routing is admissible but the rest is not

The routing kernel imported from NeuroPulse is a pure function of caller-supplied
`Float32` summaries: it allocates nothing on the hot path, performs no I/O, reads no
clock, and updates no synaptic weight. It scores and normalizes; the caller owns the
loop that feeds it. That is the same shape as the existing attention kernels.

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
| SpikeStream boundary contract | `rmems/kinetic-signals` `AGENTS.md`, `README.md`, `REVIEW.md`, `docs/boundary-matrix.md` | Cross-repo follow-up: repoint "SpikeStream.jl" to "TemporalFocus.jl (spike features)". The no-FFI, fixture-only integration contract is unchanged. |
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
| `default_inhibition_matrix` | **Migrate** | Including the `n ≤ 4` historical-slice rule and the `0.08/|i-j|` decay rule for `n > 4`. |
| `LobeState`, `NeroOrchestrator` | **Migrate, deprecated aliases** | Used downstream. |
| `update_relevance!`, `nero_diagnostics` | **Migrate, deprecated aliases** | No package-level consumer found, but `rmems/Limen-Capital`'s **vendored fork** (`brain/synapse_conductor.jl`) uses these exact names. Keeping them is what makes de-vendoring onto this package a rename-free change. |
| `ALPHA`, `BETA`, `GAMMA`, `EMA_DECAY`, `MIN_SCORE`, `EPSILON` | **Migrate, renamed** | Unexported but far too generic for a package that also owns attention. Namespace as `ROUTING_ALPHA`, … or expose only through `RoutingConfig()` defaults. |
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

1. Keep `7f3c9f2a-6b2e-4d91-9c4f-1a2b3c4d5e6f` as the canonical identity **through the
   consolidation**, so consolidation and identity churn are never in the same PR.
2. Retire `b7e4c3f2-…` (NeuroPulse) and `a3c7f1e2-…` (SpikeStream). Neither is
   registered; neither can be squatted.
3. Perform the `RELEASING.md`-mandated `uuid4()` regeneration as a **single, separate,
   pre-registration change**, after consolidation lands and before the first
   `@JuliaRegistrator register`. That way exactly one identity change ever reaches a
   consumer, and it is announced.
4. Both identity events go in the migration guide's upgrade matrix.

This requires owner sign-off because it deviates from the literal wording of #53.

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
| Same set for `TemporalBuffer`, honoring `buffer.window` | |

**Handling `value`.** Every SpikeStream feature is count- and timing-based; none consume
spike amplitude. Rather than discard `value` quietly, adapters take an explicit
`ignore_values::Bool` keyword that **defaults to erroring** when a train contains any
`value != 1.0f0`. Uniform-amplitude trains adapt without ceremony; weighted trains force
the caller to acknowledge what is being dropped.

**`detect_bursts` index provenance.** `detect_bursts` returns ranges into the *sorted*
timestamp sequence. After adapting from a `SpikeTrain` those indices no longer address
`train.events`. Adapters must either return the sorting permutation alongside the
result, or return burst *time intervals* rather than indices. Decide before porting;
do not leave it to the caller to discover.

---

## Precision policy

`TemporalFocus` is `Float32`-native. `SpikeStream` computes and returns `Float64`
throughout (`sort(Float64.(spike_times))`).

### Policy

1. **Feature kernels keep `Float64` internals and `Float64` return types**, unchanged
   from SpikeStream v0.1.0. ISI variance and coefficient of variation are exactly the
   places where `Float32` accumulation degrades; there is no benefit in narrowing them.
   Frozen fixtures continue to match bit-for-bit.
2. **`SpikeTrain` → features is exact.** Widening the stored `Float32` `t` to `Float64`
   is lossless, so features computed from a `SpikeTrain` are reproducible.
3. **The lossy direction is ingest**, and it must not be silent.

### The narrowing hazard, concretely

`Float32` carries ~7 significant decimal digits. At timestamps around `1e6` seconds the
`Float32` ulp is ≈ `0.0625 s`, so two spikes 1 ms apart collapse to the *same* `t` when
stored in a `SpikeEvent`. Downstream that is not a rounding nuisance, it changes results:

- `isi_stats` yields `min = 0`, deflated `mean`, and a materially different `cv`;
- `detect_bursts` sees zero-length ISIs, which are `≤ max_isi` by definition, so burst
  counts **inflate**;
- `normalized_feature_vector`'s `burst_norm` and `isi_cv_norm` components move with them.

**Mitigation.** `SpikeTrain(times; …)` takes `check_precision::Bool = true` and throws
when `Float32` narrowing merges timestamps that were distinct in the input, or when any
`|t| > 2^24` (past the exactly-representable-integer range). Callers who genuinely want
lossy ingest pass `check_precision = false`.

**Convenience.** Provide `Float32` wrappers (e.g. `normalized_feature_vector_f32`)
rather than changing any existing return type.

### Required edge-case tests

Non-negotiable before the SpikeStream source is retired:

- empty input; single spike; two spikes
- unsorted input; duplicated timestamps; negative timestamps
- `NaN` / `Inf` timestamps
- **large-magnitude timestamps that collide under `Float32` narrowing** (the hazard above)
- multi-neuron trains adapted per-neuron, asserting no cross-neuron ISI leakage
- non-uniform `value` trains hitting the `ignore_values` guard
- `windowed_spike_features` right-edge inclusion at the `nextfloat` boundary
- zero-duration and negative-duration windows
- `detect_bursts` index provenance after adaptation

---

## Name collisions

| Collision | Resolution |
|---|---|
| `spike_density` — SpikeStream free function (`Float64`, spikes per unit time) vs `RegionRouter.spike_density::Vector{Float32}` field (per-region normalized rate in `[0,1]`) | Not a Julia-level collision (the field is namespaced by the struct) but a genuine conceptual one in a single package. **Rename the field** to `region_spike_rate` and keep the free function's name. The rename is a breaking read-access change for anyone touching `router.spike_density` — flagged in [Open questions](#open-questions-requiring-a-human-decision). |
| `ALPHA` / `BETA` / `GAMMA` / `EPSILON` / `MIN_SCORE` / `EMA_DECAY` | Namespace with a `ROUTING_` prefix, or expose only via `RoutingConfig()` defaults. Unqualified single-word constants of that generality do not belong in a package that also owns attention kernels. |
| `normalize_l1!` / `normalize_max!` vs the router's inline floored softmax | **Keep both; do not unify.** The router performs softmax, then a `min_score` floor, then a renormalization that preserves the floor — different semantics from L1 or max normalization. Document why they coexist so a future cleanup PR does not "simplify" them together. |
| `save_state` / `load_state!` | Generic names, but they dispatch on `RegionRouter` and are router-only. Acceptable; document the restriction. |
| `ActivityRegion.last_spike_rate` vs `spike_density(times)` | Different units (`[0,1]` normalized vs spikes/second). Document explicitly; the adapter must never feed one into the other unscaled. |

---

## Zero-dependency policy

`TemporalFocus.jl`'s root `Project.toml` has an **empty `[deps]`**, and keeping it empty
is a repository invariant. Naively merging both packages would add `LinearAlgebra`,
`Printf`, and `Statistics`. All three are avoidable:

| Dependency | Actual usage | Disposition |
|---|---|---|
| `LinearAlgebra` (NeuroPulse) | **Dead import.** `region_router.jl:16` has `using LinearAlgebra: norm`, but `norm(` appears only inside a docstring at line 277 — the implementation computes norms manually in `Float64` for overflow safety. | **Drop.** No behavior change. |
| `Printf` (NeuroPulse) | Three `@sprintf` calls in `routing_diagnostics` only. | **Drop**; reimplement formatting with `round(x; digits=n)` and interpolation. Exact digit rendering may differ; test the *structure* of the diagnostic string, not a golden string, and note the change in the migration guide. |
| `Statistics` (SpikeStream) | `mean(isis)` and `std(isis)` at `spike_features.jl:76-77` only. | **Drop**; inline both. `std` must replicate the **corrected (n−1) denominator** exactly, or the frozen fixtures will fail. That fixture failure is the intended safety net — verify it fires before trusting the reimplementation. |

**Result: the consolidated package still ships with an empty `[deps]`.** This is worth
preserving; it is the reason the package installs anywhere with no resolution risk.

---

## Migration sequence

Each step is a separately reviewable PR. Steps 1–7 land before any repository is
archived. Steps 8–10 are human/cross-repo actions outside this repository.

| # | Step | Gate before proceeding |
|---|---|---|
| 1 | **This ADR** — inventory, dispositions, boundary decision | Owner accepts the boundary broadening and the UUID recommendation |
| 2 | Port SpikeStream feature kernels + `test/fixtures/spike_vectors.json`, `Statistics` inlined | Frozen fixtures pass unmodified; `[deps]` still empty |
| 3 | Add `SpikeTrain` ↔ timestamp adapters + the full edge-case suite from [Precision policy](#precision-policy) | Narrowing-collision test fails loudly without the guard, passes with it |
| 4 | Port `ActivityRegion` + routing kernel, `Printf` dropped, `LinearAlgebra` dropped, constants renamed, `adapt_leak!` **excluded** | NeuroPulse's 899-line suite ports and passes (minus `adapt_leak!` tests) |
| 5 | Port deprecated aliases as wrapper functions with `depwarn` | `update_relevance!` etc. resolve and warn |
| 6 | Update `README.md`, `AGENTS.md`, `REVIEW.md`, `docs/src/`, and add scope tests asserting the exported symbol set | Stated boundary matches shipped code exactly |
| 7 | Port examples and benchmarks; write `docs/src/migration.md` with the [upgrade matrix](#upgrade-matrix); bump to `0.2.0` and update `CHANGELOG.md` | Fresh-clone `Pkg.instantiate()` + `Pkg.test()` green on the full CI matrix |
| 8 | Add redirect READMEs to `NeuroPulse.jl` and `SpikeStream.jl` | *Cross-repo; requires write access to those repos* |
| 9 | Migrate/triage open issues (NeuroPulse #40, #14; SpikeStream #27) and open PRs (NeuroPulse #41, SpikeStream #26) per rmems/.github#4 | Every open item has a documented destination |
| 10 | **Archive** `NeuroPulse.jl` and `SpikeStream.jl` | Steps 7–9 complete **and** the v0.2.0 upgrade path is published |

**Archiving is the last step and is a deliberate human action.** No automation in this
workstream may archive, transfer, or delete a repository. Source repositories remain as
provenance and must never be deleted (rmems/.github#3, migration policy 9).

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

Published as `docs/src/migration.md` at step 7. Recorded here so the plan is reviewable
before the code exists.

| Consumer | Today | After v0.2.0 | Break? |
|---|---|---|---|
| **TemporalFocus v0.1.0 consumers** | `TemporalFocus` @ `7f3c9f2a-…`, attention API | Same name, same UUID, same API, plus features + routing | **No.** Purely additive. |
| **NeuroPulse consumers** (`rmems/Limen-Capital`) | `TemporalFocus` @ `b7e4c3f2-…`, `[sources]` → `Limen-Neural/NeuroPulse.jl` @ `40e39206…`; loaded optionally, routing actually vendored | `TemporalFocus` @ `7f3c9f2a-…`, `[sources]` → `rmems/TemporalFocus.jl`; `update_relevance!` etc. keep working with a deprecation warning | **Yes, two:** UUID + source URL must change; **`adapt_leak!` is removed** and must be re-sourced from wherever it is rehomed. Low urgency in practice — the `rev` pin is frozen and the load is behind a `try`, so nothing breaks until they choose to de-vendor. |
| **SpikeStream consumers** | `SpikeStream` @ `a3c7f1e2-…`, tag `v0.1.0` | `using TemporalFocus`; function names, signatures, and `Float64` return types unchanged | **Yes, one:** package name + UUID. No API change. |
| **`rmems/kinetic-signals`** (Rust) | Fixture-parity contract with `SpikeStream.jl`, no FFI | Same fixtures, now under `TemporalFocus.jl` | Docs-only. Output ranges are unchanged and remain a contract. |

No repository is archived before its supported upgrade path is published.

---

## Risks

| Risk | Mitigation |
|---|---|
| **Boundary dilution.** Three concerns in one package makes future scope arguments harder to win. | Keep file-level module boundaries; add scope tests asserting the exact exported symbol set; keep `AGENTS.md` / `REVIEW.md` exclusions verbatim and strengthened. |
| **`readout_ema` misread as learning state**, inviting a future STDP PR. | Stated explicitly in the module docstring and in `REVIEW.md`'s boundary checklist. |
| **Silent `Float32` corruption of features.** | The `check_precision` guard plus the dedicated collision test in step 3. |
| **`windowed_spike_features` boundary semantics** are subtle (`include_right_edge`, `nextfloat`) and easy to "clean up" wrongly. | Port fixture-first; never regenerate `spike_vectors.json`. |
| **Julia compat floor.** TemporalFocus targets 1.9–1.12 (+ macOS/Windows on 1.11); NeuroPulse's practice was **1.12-only**; SpikeStream tests `min`/`1`/`pre`. Imported code may not actually run on 1.9. | Validate the ported routing kernel on 1.9 in step 4 **before** claiming the compat range. Raise the floor deliberately if needed — do not discover it in a release. |
| **Test-suite merge.** 583 + 899 + 245 = 1727 lines of tests across three conventions. | Merge as separate top-level `@testset`s per step; never rewrite an assertion while moving it. |
| **Concurrent branches.** Eight sibling workstreams are touching this repository. | Keep each step's diff narrow; expect `README.md` / `CHANGELOG.md` conflicts and resolve additively. |
| **Premature archiving** strands Limen-Capital's pin or open issues. | Step 10 is gated on steps 7–9 and is a human action. |

---

## Open questions requiring a human decision

1. **UUID at registration.** Accept the [recommendation](#package-identity-and-uuid)
   (retain `7f3c9f2a-…` through consolidation, regenerate once pre-registration), or
   override `RELEASING.md`?
2. **Where does `adapt_leak!` go?** `thalamic-relay`, the `brainstem-daemon` workspace,
   or `LiquidCortex.jl`? Until this is answered, step 4 removes a working function with
   no published destination.
3. **`Printf` vs. zero-dep diagnostics.** Accept a changed `routing_diagnostics` output
   format to keep `[deps]` empty, or add `Printf` and keep byte-identical output?
4. **Rename `RegionRouter.spike_density` → `region_spike_rate`?** It is a breaking
   read-access change for any consumer inspecting router state.
5. **Deprecated aliases as wrapper functions** (real `depwarn`, one extra frame) or
   documentation-only deprecation (zero cost, silent)?
6. **Confirm** that Limen-Capital's dependency migration is owned by
   `rmems/Limen-Capital#9` and is not blocking on this repository.

---

## Status against issue #53's acceptance criteria

| Criterion | Status |
|---|---|
| Boundary documents the pure routing kernel and excludes runtime/learning/dense/finance | **Decided here**; the code and README/AGENTS text land together at step 6 |
| Fresh-clone `Pkg.instantiate()` + `Pkg.test()` succeed | Passing today; re-gated at step 7 |
| Every inventoried artifact retained, migrated, or explicitly retired with rationale | **Done** — see [Pre-migration inventory](#pre-migration-inventory) and [Symbol disposition](#symbol-disposition) |
| Imported behavior has parity tests or documented intentional changes | **Planned** (steps 2, 4); intentional changes recorded here |
| Timestamp adaptation and precision edge-case tests | **Specified** here; land at step 3 |
| Only one active package UUID/name pair | **Decided** here; effective at step 7 |
| Versioned migration guide and upgrade matrix | **Drafted** here as [Upgrade matrix](#upgrade-matrix); published at step 7 |
| No active docs instruct users to depend on superseded repositories | Steps 6–8 |
| `NeuroPulse.jl` / `SpikeStream.jl` archived with successor links | Step 10, human action, explicitly **not** automated |
| README explains the whole accomplishment in one sentence before modules | Step 6 |

---

## References

- [rmems/TemporalFocus.jl#53](https://github.com/rmems/TemporalFocus.jl/issues/53) — this workstream
- [rmems/.github#3](https://github.com/rmems/.github/issues/3) — portfolio consolidation umbrella
- [rmems/.github#4](https://github.com/rmems/.github/issues/4) — issue/Linear migration
- [rmems/Limen-Capital#9](https://github.com/rmems/Limen-Capital/issues/9) — downstream consumer workstream
- [`AGENTS.md`](../../AGENTS.md) — scope boundary and code style
- [`REVIEW.md`](../../REVIEW.md) — reviewer boundary checklist
- [`RELEASING.md`](../../RELEASING.md) — UUID hygiene blocker and registration process
