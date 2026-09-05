# ADR 0001 — Consolidate NeuroPulse.jl and SpikeStream.jl into TemporalFocus.jl

> **SUPERSEDED.** Owner override: TemporalFocus consolidates **into** NeuroPulse,
> not the other way around. Do not implement this document's import sequence
> (NP+SpikeStream → TemporalFocus). See
> [ADR 0002](0002-merge-temporalfocus-into-neuropulse.md).
> Inventory, adapter, and precision notes below remain historical salvage.

- **Status:** Superseded by [ADR 0002](0002-merge-temporalfocus-into-neuropulse.md)
- **Date:** 2026-08-23
- **Tracking issue:** [rmems/TemporalFocus.jl#53](https://github.com/rmems/TemporalFocus.jl/issues/53)
- **Umbrella issue:** [rmems/.github#3](https://github.com/rmems/.github/issues/3)
- **Supersedes:** the README/AGENTS.md exclusion of "routing mechanisms" (see [Decision](#decision))
- **Superseded by:** [ADR 0002](0002-merge-temporalfocus-into-neuropulse.md) (2026-09-05)

This ADR is the pre-migration inventory, symbol disposition record, and migration
sequence required by issue #53 **as originally written**. It landed via PR #54.
**Do not execute the sequence below.** ADR 0002 is the authoritative plan.

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
7. **NeuroPulse.jl's user-facing README still presents this package's name as its own**
   (inspected 2026-09-03, `main`). Distinct from the `Project.toml` `name = "TemporalFocus"`
   collision already recorded above: `rmems/NeuroPulse.jl` `README.md` uses
   `<h1>TemporalFocus.jl</h1>`, documents `Pkg.add("TemporalFocus")` and
   `using TemporalFocus`, and has a Migration section that treats the rename
   NeuroPulse→TemporalFocus as the intended identity. **This planning PR does not
   edit NeuroPulse.** Disposition: **step 9's redirect README** replaces that file
   and is the planned fix. An earlier NeuroPulse-only hygiene PR (issue #40) may
   correct the live title/install/Migration text before then; it is optional and
   owned by that repository, not this workstream.

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
| `TemporalFocus` UUID + `Limen-Neural/NeuroPulse.jl` source pin | `rmems/Limen-Capital` `brain/Project.toml`, `docs/deps.md`, `README.md` | **Cross-repo follow-up in step 9** under `rmems/Limen-Capital#9`: update the UUID, source URL, revision, README, and dependency guide together. Archiving is gated on that issue landing. |
| Vendored fork of the routing kernel (`NeroOrchestrator`, `update_relevance!`, `NERO_*`) | `rmems/Limen-Capital` `brain/synapse_conductor.jl`, consumed by `brain/sonar_probe.jl` | **Cross-repo follow-up** under `rmems/Limen-Capital#9`: de-vendor onto the consolidated package. Drifted from the generalized `ActivityRegion` API; keeping the deprecated aliases makes that a rename-free change. |
| SpikeStream boundary contract | `rmems/kinetic-signals` `AGENTS.md`, `README.md`, `REVIEW.md`, `docs/boundary-matrix.md` | **Cross-repo follow-up in step 9:** repoint "SpikeStream.jl" to "TemporalFocus.jl (spike features)". The no-FFI, fixture-only integration contract is unchanged. Archiving is gated on this update. |
| NeuroPulse README identity mixup (`<h1>TemporalFocus.jl</h1>`, `Pkg.add("TemporalFocus")`, `using TemporalFocus`, Migration section treating NeuroPulse→TemporalFocus as intended) | `rmems/NeuroPulse.jl` `README.md` on `main` | **Step 9:** replace the file with the successor redirect. Optional earlier hygiene is NeuroPulse #40, not this PR. Do not edit NeuroPulse from this workstream. |
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
| `load_state` | **Migrate, deprecated** | Currently `const load_state = load_state!` — a non-bang name bound to a mutating function. Keep as a `Base.@deprecate_binding` alias; steer callers to `load_state!`. |
| `adapt_leak!` | **Conditional:** rehome/remove once decided; otherwise **migrate temporarily, deprecated** | The step-5 compatibility fallback is authoritative: carry the function until the owner names a working successor or explicitly retires it without one. While carried, it stays callable under its own name and warns via inline `Base.depwarn` or `Base.@deprecate` — see [Deprecation mechanics](#deprecation-mechanics). Its final destination remains [out of scope](#what-does-not-migrate). |
| `default_inhibition_matrix` | **Migrate** | Including the `n ≤ 4` historical-slice rule and the `0.08/\|i-j\|` decay rule for `n > 4`. |
| `LobeState`, `NeroOrchestrator` | **Migrate, deprecated aliases** | Used downstream. |
| `update_relevance!`, `nero_diagnostics` | **Migrate, deprecated aliases** | No package-level consumer found, but `rmems/Limen-Capital`'s **vendored fork** (`brain/synapse_conductor.jl`) uses these exact names. Keeping them is what makes de-vendoring onto this package a rename-free change. |
| `ALPHA`, `BETA`, `GAMMA`, `EMA_DECAY`, `MIN_SCORE`, `EPSILON` | **Migrate, renamed** | Unexported but far too generic for a package that also owns attention. Rename them to `ROUTING_ALPHA`, `ROUTING_BETA`, `ROUTING_GAMMA`, `ROUTING_EMA_DECAY`, `ROUTING_MIN_SCORE`, and `ROUTING_EPSILON`; the deprecated `NERO_*` constants bind to these values. |
| `NERO_ALPHA`, `NERO_BETA`, `NERO_GAMMA`, `NERO_EMA_DECAY`, `NERO_MIN_SCORE`, `NERO_EPSILON` | **Migrate, deprecated aliases** | |
| `DEFAULT_REGION_NAMES` | **Migrate** | Canonical region-name tuple. |
| `NERO_DEFAULT_LOBE_NAMES` | **Migrate, deprecated** | `Base.@deprecate_binding` to `DEFAULT_REGION_NAMES`. |
| `INHIBIT` | **Migrate** | The hardcoded asymmetric 4×4 matrix is arbitrary legacy tuning; keep it as the documented default and say so. |
| `NERO_INHIBIT` | **Migrate, deprecated** | `Base.@deprecate_binding` to `INHIBIT`. |
| `Base.setproperty!(::RegionRouter, …)` | **Migrate** | Guards `config` replacement against `min_score` infeasibility. Easy to lose in a port; it is load-bearing. |

#### Deprecation mechanics

On the supported Julia 1.9+ range, `Base.@deprecate_binding` deprecates a type, constant,
or function alias **without changing the binding's kind**. It is incorrect to claim that
warnings require turning types or constants into wrapper functions, and it is incorrect
to leave those aliases silent. Step 6 uses binding deprecation for every deprecated
*alias*; documentation-only silent aliases are not used.

**Same-name function deprecation is a different case.** `adapt_leak!` is not an alias —
when step 5 carries it, the public name stays `adapt_leak!` and the method remains
callable with its current signature. Binding deprecation cannot emit a warning for a
function under its own name. The carried method therefore warns via one of:

- **Preferred:** an inline `Base.depwarn("adapt_leak! is deprecated; …", :adapt_leak!)`
  at the start of the existing method body. The binding stays a function of the same
  name; no extra public name is introduced.
- **Allowed:** `Base.@deprecate adapt_leak!(args...) _adapt_leak!(args...)` if the
  implementation is moved to an `_`-prefixed internal. That form exists only to emit
  the warning for a same-name carry; it is not forbidden. The alias rule above — do
  not rewrite *type or constant aliases* as wrapper functions — does not apply here.

Step 6 tests the carried `adapt_leak!` the same way it tests aliases: a call emits a
deprecation warning (`Test.@test_deprecated` or equivalent) and the method remains
callable with the current behavior. If step 5 removed the function because a destination
or retirement was decided, this test is omitted.

```julia
Base.@deprecate_binding LobeState ActivityRegion
Base.@deprecate_binding NeroOrchestrator RegionRouter
Base.@deprecate_binding NERO_ALPHA ROUTING_ALPHA
Base.@deprecate_binding NERO_BETA ROUTING_BETA
Base.@deprecate_binding NERO_GAMMA ROUTING_GAMMA
Base.@deprecate_binding NERO_EMA_DECAY ROUTING_EMA_DECAY
Base.@deprecate_binding NERO_MIN_SCORE ROUTING_MIN_SCORE
Base.@deprecate_binding NERO_EPSILON ROUTING_EPSILON
Base.@deprecate_binding NERO_INHIBIT INHIBIT
Base.@deprecate_binding NERO_DEFAULT_LOBE_NAMES DEFAULT_REGION_NAMES
Base.@deprecate_binding update_relevance! update_routing!
Base.@deprecate_binding nero_diagnostics routing_diagnostics
Base.@deprecate_binding load_state load_state!
```

The deprecated set is not homogeneous; the binding kind of each row is load-bearing:

| Alias kind | Members | Treatment |
|---|---|---|
| Function aliases | `update_relevance!`, `nero_diagnostics`, `load_state` | `Base.@deprecate_binding` to the canonical function. Remain functions (callable, no extra wrapper frame). Accessing the old name warns. |
| **Type** aliases | `LobeState`, `NeroOrchestrator` | `Base.@deprecate_binding` to the canonical type. Remain types: `x::LobeState`, `isa(x, LobeState)`, and dispatch still work. Accessing the old name warns. |
| **Constant** aliases | `NERO_ALPHA`, `NERO_BETA`, `NERO_GAMMA`, `NERO_EMA_DECAY`, `NERO_MIN_SCORE`, `NERO_EPSILON`, `NERO_INHIBIT`, `NERO_DEFAULT_LOBE_NAMES` | `Base.@deprecate_binding` to the canonical constant. Remain constants: arithmetic (`NERO_ALPHA * x`) and indexing still work. Accessing the old name warns. |

Step 6 tests both halves of that contract: each listed old name emits a deprecation
warning on use (`Test.@test_deprecated` or equivalent), and the binding kind is
preserved (`LobeState === ActivityRegion` and usable in `::`/`isa`; `NERO_ALPHA ===
ROUTING_ALPHA` and usable in arithmetic; each function alias `isa Function` and
forwards to the canonical method). When `adapt_leak!` is carried, the same step also
asserts that a call under that name warns and still runs.

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
| `adapt_leak!` *(final disposition)* | Maps a hardware/thermal **stress signal** onto a neuron **leak rate**. That is telemetry ingestion plus neuron-dynamics mutation plus hardware co-design — excluded by `AGENTS.md` ("runtime execution") and by issue #53's non-goals ("telemetry ingestion, deployment supervision, or hardware-control ownership"). It also has nothing to do with routing: it never touches a `RegionRouter`. **Until the final disposition is decided, step 5 carries it temporarily as a deprecated compatibility function rather than treating this row as permission to drop it.** | A neuron-dynamics or runtime package. Candidates: `rmems/thalamic-relay` (hardware orchestration relay), the `brainstem-daemon` runtime workspace named in rmems/.github#3, or `rmems/LiquidCortex.jl` if the leak rate belongs to reservoir dynamics; alternatively, the owner may explicitly retire it without a successor. **Owner decision required** — see [Open questions](#open-questions-requiring-a-human-decision). |
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
| `spike_times(train::SpikeTrain, neuron_id::Integer; ignore_values::Bool=false) -> Vector{Float64}` | Explicit single-neuron projection. `neuron_id` is required — there is no whole-train overload. |
| `spike_times_by_neuron(train::SpikeTrain; ignore_values::Bool=false) -> Dict{Int,Vector{Float64}}` | Groups by neuron; IDs are preserved as keys. |
| `spike_features_by_neuron(train::SpikeTrain; t_start=nothing, t_end=nothing, max_density::Real=1000.0, ignore_values::Bool=false) -> Dict{Int,NamedTuple}` | For each neuron returns exactly `(count::Int, density::Float64, isi::NamedTuple, normalized::Vector{Float64})`; `isi` has the `(:mean, :std, :min, :max, :cv)` `Float64` fields and `normalized` is the four-element vector defined above. The window keywords are forwarded identically to the timestamp kernels. |
| `burst_intervals_by_neuron(train::SpikeTrain; max_isi::Real=0.02, min_spikes::Int=3, ignore_values::Bool=false) -> Dict{Int,Vector{Tuple{Float64,Float64}}}` | The exported adapter named by the burst-interval decision below. Each tuple is `(t_first, t_last)` in the sorted per-neuron sequence. |
| `spike_times(buffer::TemporalBuffer, neuron_id::Integer, current_time::Real; ignore_values::Bool=false) -> Vector{Float64}` | Buffer overload of the single-neuron projection. `current_time` is positional and required. |
| `spike_times_by_neuron(buffer::TemporalBuffer, current_time::Real; ignore_values::Bool=false) -> Dict{Int,Vector{Float64}}` | Buffer overload of the grouped projection. |
| `spike_features_by_neuron(buffer::TemporalBuffer, current_time::Real; t_start=nothing, t_end=nothing, max_density::Real=1000.0, ignore_values::Bool=false) -> Dict{Int,NamedTuple}` | Same result schema as the train overload, after applying the buffer window; `t_start`/`t_end` further restrict that selected set. |
| `burst_intervals_by_neuron(buffer::TemporalBuffer, current_time::Real; max_isi::Real=0.02, min_spikes::Int=3, ignore_values::Bool=false) -> Dict{Int,Vector{Tuple{Float64,Float64}}}` | Buffer overload of the interval adapter, after applying the buffer window. |
| `normalized_feature_vector_f32(train::SpikeTrain, neuron_id::Integer; t_start=nothing, t_end=nothing, max_density::Real=1000.0, ignore_values::Bool=false) -> Vector{Float32}` | Single-neuron Float32 convenience path. Same projection and `ignore_values` contract as `spike_times`; then the step-3 timestamp wrapper. No whole-train or grouped overload. Result is exactly four `Float32` values, not the `normalized::Vector{Float64}` field of `spike_features_by_neuron`. |
| `normalized_feature_vector_f32(buffer::TemporalBuffer, neuron_id::Integer, current_time::Real; t_start=nothing, t_end=nothing, max_density::Real=1000.0, ignore_values::Bool=false) -> Vector{Float32}` | Buffer overload of the same name. `neuron_id` then positional required `current_time`; same window/`current_time` contract as the other buffer adapters. |
| `SpikeTrain(times::AbstractVector{<:Real}; neuron_id=nothing, value=1.0f0, check_precision::Symbol=:collisions)` | Reverse direction. For **non-empty** `times`, `neuron_id` is a required keyword and the precision modes are fixed below. **Empty** `times` (`Float64[]`, `Int[]`, and other empty `AbstractVector{<:Real}`) does **not** require `neuron_id`: it constructs an empty train, the same as `SpikeTrain()` / `SpikeTrain(SpikeEvent[])`. |

**Empty reverse-constructor input must not require `neuron_id`.** The planned
`SpikeTrain(times::AbstractVector{<:Real}; neuron_id=nothing, …)` overload defaults
`neuron_id` so `SpikeTrain(Float64[])` and `SpikeTrain(Int[])` reach the empty-input
path instead of throwing `UndefKeywordError`.
Those calls must keep constructing an empty `SpikeTrain` without keywords. The
empty-`times` path is keyword-optional and falls through to the inner
`SpikeTrain(SpikeEvent[])` constructor; `neuron_id` remains required only when
`times` is non-empty. Supplying `neuron_id` on empty input is allowed and still
yields an empty train. Step 4 pins both `SpikeTrain(Float64[])` and `SpikeTrain(Int[])`
as constructor-level empty-train regressions, not only adapter-level empty-input tests.

Every projection and feature adapter validates that its candidate events have finite
timestamps **before any window/feature filtering that could discard them** and before
widening `Float32` to `Float64`. For the single-neuron overload, “candidate” means all
stored events with the requested neuron ID; for the grouped overloads it means every
stored event. This applies to pre-existing `SpikeTrain` and `TemporalBuffer` values as
well as to the reverse constructor: a caller
can construct `SpikeEvent(..., NaN32, ...)` or `SpikeEvent(..., Inf32, ...)` directly, so
ingest-only checks are insufficient.

**`TemporalBuffer` needs a reference time.** A `TemporalBuffer` stores only `window` and
`events`; it does **not** store the reference time that decides which events are currently
inside the window — which is why the existing [`prune!`](../../src/types.jl) takes
`current_time` explicitly. So a buffer adapter cannot "honor `buffer.window`" from the
buffer alone: without a cutoff it would let stale events silently into feature
calculations. Buffer adapters therefore take `current_time` as a required argument and
reject both the supplied value and `current_time_f32 = Float32(current_time)` unless
they are finite. They also reject a non-finite `buffer.window` before inspecting any
event. They validate all candidate event timestamps **before** applying the window
predicate, so `NaN32`/`Inf32` cannot disappear as an ordinary out-of-window event. Only
then do they apply the exact rule `prune!` executes,
`(current_time_f32 - event.t) <= buffer.window`; narrowing the reference time before
subtraction is required for boundary parity. Thus an adapted buffer and a
`prune!`-then-adapt path agree exactly for valid inputs. (Requiring callers to `prune!`
first was considered and rejected: it mutates the caller's buffer and fails silently if
skipped.)

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

This is an explicit gate on step 4: the adapter PR exports
`burst_intervals_by_neuron` with the signatures above and implements the interval form,
not the permutation form.

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
| `:collisions` *(default)* | non-finite `t` (including a finite value that narrows to non-finite `Float32`); narrowing that **merges timestamps that were distinct**; or `abs(t) > 2^24` | normal ingest |
| `:strict` | non-finite `t` (including overflow to non-finite `Float32`); **or any** timestamp that does not round-trip `Float64 → Float32 → Float64` | exact parity with a `Float64` pipeline is required |
| `:none` | non-finite inputs **or a finite input that narrows to non-finite `Float32`** | the caller has accepted lossy but still finite ingest |

**Unknown `check_precision` symbols throw `ArgumentError`.** The only valid values are
`:collisions`, `:strict`, and `:none`. Any other symbol — including the singular typo
`:collision` — is rejected. Implementations must not fall through to `:none` or any
other mode. Step 4 includes a regression test that
`SpikeTrain([0.0, 0.1]; neuron_id=1, check_precision=:collision)` throws
`ArgumentError` and constructs no events.

**Non-finite timestamps are rejected in every checked mode.** A `NaN` neither merges two
distinct timestamps nor exceeds `2^24`, so the collision and magnitude rules alone would
wave it through — and it then poisons every ISI statistic downstream. `:collisions` and
`:strict` therefore also reject any non-finite `t`, and even `:none` rejects it. Every
mode checks the narrowed value too: a finite `Float64` such as `1e40` overflows to
`Inf32` and must be rejected before constructing a `SpikeEvent`. Accepting lossy ingest
is not the same as accepting a value that is not a finite representable time.

The default deliberately does **not** promise that ingest is lossless. Ordinary decimal
values — `0.1`, `0.2` — are altered by narrowing without merging or exceeding `2^24`, so
`:collisions` accepts them and their ISIs shift in the last few digits. That is inherent
to a `Float32` data model, not a bug the guard can remove: `:strict` would reject almost
every real decimal timestamp. So the guarantee is scoped precisely: the default prevents
distinct timestamps from collapsing into zero ISIs (and preserves their strict order),
but ordinary narrowing can still move an ISI across a caller-selected threshold and
therefore change burst classification or counts. Only `:strict` promises parity with the
`Float64` pipeline. Sub-ulp rounding and threshold crossings under `:collisions` are
documented, not prevented.

**Convenience surface — exact, not an open-ended family.** Step 3 exports exactly one
wrapper:

```julia
normalized_feature_vector_f32(
    times::AbstractVector{<:Real};
    t_start=nothing,
    t_end=nothing,
    max_density::Real=1000.0,
)::Vector{Float32}
```

It delegates to `normalized_feature_vector` and converts the final four-element vector;
the six parity APIs above keep their existing `Float64` returns. Step 3 adds this name to
`AGENTS.md`'s feature-extraction naming exception and tests its export, signature, return
type, keyword forwarding, and elementwise conversion.

Step 4 adds exactly two container overloads of that **same** name — no additional `_f32`
public name, and no grouped `_f32` API. `spike_features_by_neuron`'s
`normalized::Vector{Float64}` field is not a substitute for these overloads:

```julia
normalized_feature_vector_f32(
    train::SpikeTrain,
    neuron_id::Integer;
    t_start=nothing,
    t_end=nothing,
    max_density::Real=1000.0,
    ignore_values::Bool=false,
)::Vector{Float32}

normalized_feature_vector_f32(
    buffer::TemporalBuffer,
    neuron_id::Integer,
    current_time::Real;
    t_start=nothing,
    t_end=nothing,
    max_density::Real=1000.0,
    ignore_values::Bool=false,
)::Vector{Float32}
```

Both are single-neuron only (`neuron_id` is required; there is no whole-train overload).
The buffer form takes `current_time` positionally after `neuron_id`. They apply the
`spike_times` / buffer-window contracts above, then delegate to the timestamp wrapper.
The result is exactly
`normalized_feature_vector_f32(projected_times; t_start, t_end, max_density)` — a
four-element `Vector{Float32}`. A neuron with no candidate events forwards the empty
timestamp vector; an invalid ID follows `spike_times`. Step 4 tests both overloads'
argument order, keywords, `Vector{Float32}` length-4 result, and elementwise parity
with project-then-wrapper, plus `ignore_values`, non-finite rejection, and the empty-neuron
case. The step cannot pass without this Float32 container API.

### Required edge-case tests

Non-negotiable before the SpikeStream source is retired:

- empty input; single spike; two spikes
- constructor-level empty Real vectors: `SpikeTrain(Float64[])` and `SpikeTrain(Int[])`
  construct an empty train without `neuron_id` (no `UndefKeywordError`); non-empty
  `times` still requires `neuron_id`
- unsorted input; duplicated timestamps; negative timestamps
- `NaN` / `Inf` timestamps
- a finite timestamp that overflows to `Inf32` during narrowing (including under `:none`)
- **large-magnitude timestamps that collide under `Float32` narrowing** (the hazard above)
- a value that narrows lossily **without** colliding (e.g. `0.1`): accepted under
  `:collisions`, rejected under `:strict` — pins down exactly what the default promises
- `abs(t) > 2^24`
- multi-neuron trains adapted per-neuron, asserting no cross-neuron ISI leakage
- non-uniform `value` trains hitting the `ignore_values` guard
- `TemporalBuffer` adapters: events outside `window` relative to `current_time` are
  excluded; a non-finite candidate event is rejected even when the window predicate
  would otherwise discard it; `NaN`/`Inf` buffer windows and reference times (including
  a finite reference time that narrows to non-finite `Float32`) are rejected; a
  non-`Float32` boundary case pins the pre-subtraction narrowing; and the adapted result
  equals `prune!`-then-adapt exactly
- `windowed_spike_features` right-edge inclusion at the `nextfloat` boundary
- zero-duration and negative-duration windows
- `detect_bursts` index provenance after adaptation
- unsupported `check_precision` (e.g. `:collision`) throws `ArgumentError` and does not ingest
- `normalized_feature_vector_f32` `SpikeTrain`/`TemporalBuffer` overloads: signatures,
  keyword forwarding, four-element `Vector{Float32}` result, and parity with
  project-then-wrapper

---

## Name collisions

| Collision | Resolution |
|---|---|
| `spike_density` — SpikeStream free function (`Float64`, spikes per unit time) vs `RegionRouter.spike_density::Vector{Float32}` field (per-region normalized rate in `[0,1]`) | Not a Julia-level collision (the field is namespaced by the struct) but a genuine conceptual one in a single package. **If the owner accepts open question 4, rename the field** to `region_spike_rate` and keep the free function's name. If the answer is absent or rejects the break, step 5 retains `RegionRouter.spike_density`; this table does not override that fallback. |
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

Each implementation step is a separately reviewable PR. Steps 1–8 land before any
repository is archived. Steps 9–12 are human/cross-repo/release actions outside the
release-preparation PR.

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
| 2 | **Immediately before editing identity, rerun and record the repository-wide consumer audit** for package name, old UUID, Git URL, and fixed `rev` pins; then regenerate the canonical UUID per `RELEASING.md` (`Project.toml`, `docs/Project.toml`, `benchmark/Project.toml`, and `Manifest.toml` via `Pkg.resolve()`); atomically update `RELEASING.md`'s UUID-hygiene section so it records the new UUID and no longer instructs a second regeneration | The fresh audit, not only the 2026-08-23 snapshot, finds no unplanned consumer dependent on `7f3c9f2a-…`; any new consumer is migrated or explicitly blocks the edit; identity changes **before** any import so `v0.2.0` ships the final UUID and no consumer migrates twice |
| 3 | Port SpikeStream feature kernels + `test/fixtures/spike_vectors.json`, `Statistics` inlined; **add spike-stream feature extraction to the Scope lists and naming policy, amend `AGENTS.md`'s Float32 rule to the scoped wording in [Precision policy](#precision-policy), and scope `REVIEW.md`'s return-type rule to permit derived `Float64` features in the same PR** | Frozen fixtures pass unmodified; `[deps]` still empty; boundary, naming, scoped Float32 rule, and reviewer type guidance match what `main` now ships |
| 4 | Add the exact `SpikeTrain`/`TemporalBuffer` adapter surface in [Adapters between the two data models](#adapters-between-the-two-data-models) + the full edge-case suite from [Precision policy](#precision-policy), including pre-built events with non-finite timestamps, non-finite buffer windows/reference times, Float32 reference-time boundary parity, uniform non-unit values, constructor-level empty `Float64[]`/`Int[]` ingest without `neuron_id`, the `normalized_feature_vector_f32` container overloads, and the unknown-`check_precision` rejection; **extend `AGENTS.md`'s naming policy in this same PR with the exact adapter family** `spike_times`, `spike_times_by_neuron`, `spike_features_by_neuron`, `burst_intervals_by_neuron`, and the container overloads of `normalized_feature_vector_f32` | Signatures, keyword forwarding, result schemas, interval returns, and the two `normalized_feature_vector_f32` container overloads (`Vector{Float32}`, project-then-wrapper parity) match the contract; narrowing-collision, unknown-symbol `ArgumentError`, empty `SpikeTrain(Float64[])`/`SpikeTrain(Int[])` construction, pre-filter non-finite projection/window/reference checks and the `prune!` boundary-parity test fail loudly without their guards and pass with them; every non-unit value requires `ignore_values=true`; naming guidance matches the exported adapter surface |
| 5 | Port `ActivityRegion` + routing kernel, `LinearAlgebra` dropped, generic constants renamed to the decided `ROUTING_*` family; exclude `adapt_leak!` only when its destination or explicit retirement is decided, otherwise carry it temporarily as deprecated; rename `spike_density` only if accepted, otherwise retain the existing field name; drop `Printf` only if changed diagnostics formatting is accepted, otherwise preserve it; **add the routing kernel to the Scope lists and extend the naming policy with the exact routing surface** `update_routing!`, `routing_diagnostics`, `save_state`, `load_state!`, and `default_inhibition_matrix` in the same PR (plus an explicit compatibility-only exception for `adapt_leak!` if it is carried), with the admissible-mutation limits | NeuroPulse's suite ports and passes; `adapt_leak!` tests are omitted only if its removal is decided, otherwise they remain; boundary and naming text match what `main` now ships; diagnostics formatting is decided and tested before `Printf` is removed; unresolved `adapt_leak!` and field-name decisions use the documented compatibility fallbacks below |
| 6 | Port deprecated aliases with `Base.@deprecate_binding`, **preserving each binding's kind**, and install the same-name `adapt_leak!` warning if step 5 carried the function — see [Deprecation mechanics](#deprecation-mechanics) | Every listed old alias warns on use; `LobeState` remains a type usable in `::`/`isa` and `=== ActivityRegion`; `NERO_ALPHA` remains arithmetic and `=== ROUTING_ALPHA`; function aliases remain functions and forward to the canonical methods with no extra wrapper frame; if `adapt_leak!` was carried, a call warns (`Test.@test_deprecated` or equivalent) and remains callable |
| 7 | Consolidated docs pass: update the README one-sentence lead **and its Interface Contract and Current API lists** for bare timestamp features, `SpikeTrain`/`TemporalBuffer` adapters, `ActivityRegion` routing inputs, statistical outputs, and every migrated export; perform the full `REVIEW.md` checklist pass (preserving the derived-feature type exception landed in step 3), verify the scoped `Float32` rule, update `docs/src/`, and add scope tests asserting the exported symbol set | README and site describe every shipped input/output shape and public export; stated boundary matches shipped code exactly |
| 8 | Port examples and benchmarks; write `docs/src/migration.md` with the [upgrade matrix](#upgrade-matrix); complete every non-source [artifact disposition ledger](#non-source-artifact-disposition-ledger) row assigned through step 8 and record the step-9/10 rows as pending with their owner/destination; bump to `0.2.0`, update `CHANGELOG.md`, **and update `RELEASING.md`'s first-registration procedure** from v0.1.0 to v0.2.0 (the UUID-hygiene section was already corrected atomically in step 2) | Fresh-clone `Pkg.instantiate()` + `Pkg.test()` green on the full CI matrix; every ledger row assigned through step 8 is complete and later rows have an explicit pending owner/destination |
| 9 | Add redirect READMEs to `NeuroPulse.jl` and `SpikeStream.jl`; update `rmems/kinetic-signals` boundary docs to name `TemporalFocus.jl (spike features)`; land `rmems/Limen-Capital#9`, including its `brain/Project.toml`, README, and `docs/deps.md` updates **and replacement of the vendored router in `brain/synapse_conductor.jl`/`brain/sonar_probe.jl` with the consolidated package API**. Because v0.2.0 is not published until step 11, step 9 pins an exact consolidated commit, never the not-yet-existing tag; a post-step-11 follow-up may switch to the registry/tag. | *Cross-repo; requires write access to those repos; all source redirects, kinetic-signals boundary updates, and the Limen-Capital dependency/docs migration plus de-vendoring tests must land before archiving; no active routing path may still instantiate or call the vendored implementation* |
| 10 | Migrate/triage open issues (NeuroPulse #40, #14; SpikeStream #27) and open PRs (NeuroPulse #41, SpikeStream #26) per rmems/.github#4 | Every open item has a documented destination |
| 11 | **Register and publish TemporalFocus v0.2.0 after the step-8 release-preparation PR has merged**; invoke Registrator only as this separate post-merge action, wait for the General-registry PR and TagBot, and verify the published package under the regenerated UUID | General resolves the regenerated UUID at `0.2.0`, the `v0.2.0` tag exists and identifies the registered source tree, a fresh registry-based install succeeds, and the hosted v0.2.0 upgrade guide is live |
| 12 | **Archive** `NeuroPulse.jl` and `SpikeStream.jl` | Steps 8–11 complete, every artifact-ledger row is complete, `rmems/Limen-Capital#9` is closed by the landed dependency/docs migration **and tested `synapse_conductor.jl`/`sonar_probe.jl` de-vendoring**, no active path uses the fork, **and** the registered/tagged v0.2.0 release plus its upgrade path are publicly available |

**Archiving is the last step and is a deliberate human action.** No automation in this
workstream may archive, transfer, or delete a repository. Source repositories remain as
provenance and must never be deleted (rmems/.github#3, migration policy 9).

### Non-source artifact disposition ledger

This ledger assigns every inventoried non-source artifact to a numbered step. “Retire”
means do not copy it into TemporalFocus; it remains in the source repository's immutable
history and may disappear from active use only when step 12 archives that repository.

| Inventoried artifacts | Numbered disposition |
|---|---|
| NeuroPulse tests; SpikeStream tests and frozen fixture | **Steps 3 and 5:** port with their owning kernels; copy `spike_vectors.json` verbatim in step 3. |
| NeuroPulse examples; SpikeStream benchmark project and runner | **Step 8:** port into the consolidated examples/benchmark trees and validate against the final package identity. |
| NeuroPulse `docs/src/`, `docs/make.jl`, `docs/Project.toml`, manifest, and logo | **Steps 7–8:** migrate still-applicable prose/API material into the existing TemporalFocus site; resolve docs dependencies under the canonical UUID. The source logo is retained only if the consolidated site uses it, otherwise explicitly retire it in the step-8 ledger check. |
| NeuroPulse duplicated flat docs; `.devcontainer/`; `.devin/blueprint.yaml` | **Step 8:** explicitly retire for the rationales in [What does not migrate](#what-does-not-migrate); no copy is permitted. |
| NeuroPulse `ci`, `documentation`, and `format` workflows; Dependabot configuration; SpikeStream `ci` and `codecov` workflows and `.markdownlint.json` | **Steps 3, 5, and 8:** exercise imported code through TemporalFocus's existing matrix as each kernel lands; in step 8 record these source-repository configurations as retired because the consolidated repository's CI, coverage, formatting, dependency-update, and Markdown policies are authoritative. Do not copy competing automation. |
| NeuroPulse and SpikeStream license files | **Step 8:** verify imported files retain compatible provenance and are covered by TemporalFocus's top-level `LICENSE`, `LICENSE-MIT`, and `LICENSE-APACHE`; retain the source license files in source history, but do not create duplicate active license sets. |
| Source-repository READMEs, AGENTS/REVIEW guidance, and remaining logos | **Step 9:** replace each active source README with the successor redirect; leave historical guidance/assets in the archived repository for provenance, not as consolidated policy. |
| Open issues and PRs, including NeuroPulse Dependabot #41 and SpikeStream ImgBot #26 | **Step 10:** migrate, close, or supersede each with a documented destination before archiving. |

Step 8's gate requires a copy of this ledger in the migration PR description with every
row assigned through step 8 checked off. Rows assigned to steps 9 and 10 remain explicitly
pending there, with their owner and destination recorded; they cannot be called complete
before those steps run. Step 12 cannot proceed until **every** row, including the step-9
and step-10 rows, has its completed retain/migrate/retire result recorded.

### Gate: `adapt_leak!` may not be removed into a void

`adapt_leak!` is a working public function, and issue #53 promises that no repository is
archived before its supported upgrade path is published. Removing it while its
destination is still an open question would hand a consumer a migration guide that points
nowhere. Step 5 therefore has a compatibility fallback, while release and archive remain
hard gates:

- **Step 5** (conditional removal) — remove it only when the destination or an explicit
  no-successor retirement is decided and recorded; otherwise carry it temporarily as a
  deprecated, documented-as-out-of-scope compatibility function that warns under its
  own name (inline `Base.depwarn` or `Base.@deprecate`; see [Deprecation mechanics](#deprecation-mechanics)).
- **Step 8** (`v0.2.0`) — the migration guide either names where to get it, with a working
  reference (a repository and, if it has landed, a version or commit), **or** plainly
  documents the owner's decision to retire it without a successor.
- **Step 12** (archive) — the replacement is published, **or** the owner has explicitly
  retired the function with no successor and the guide says so plainly.

"Decided" means a named destination, not an intention. If the decision is still open when
step 5 comes up, the correct move is to carry `adapt_leak!` forward temporarily as a
deprecated, documented-as-out-of-scope function that warns under its own name rather
than to drop it — an acknowledged scope wart beats a broken upgrade path.

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
| **NeuroPulse consumers** (`rmems/Limen-Capital`) | `TemporalFocus` @ `b7e4c3f2-…`, `[sources]` → `Limen-Neural/NeuroPulse.jl` @ `40e39206…`; loaded optionally, routing actually vendored | At step 9, `TemporalFocus` @ `<canonical>`, `[sources]` → `rmems/TemporalFocus.jl` **with `rev` replaced by an exact consolidated commit**, and `synapse_conductor.jl`/`sonar_probe.jl` use that package instead of the vendored router. Only after step 11 publishes v0.2.0 may a follow-up drop `[sources]` for the registry or replace the commit with the tag. `update_relevance!`, `LobeState`, `NeroOrchestrator`, and the `NERO_*` constants keep working and emit deprecation warnings via `Base.@deprecate_binding`; `adapt_leak!` follows the decided replacement/retirement path or remains temporarily deprecated while that decision is open | **Always three:** UUID, source URL, **and the `rev` pin** must change — leaving `rev = "40e39206…"` in place would either fail to resolve or, if history is imported, check out the old NeuroPulse tree. The vendored implementation must also be removed from the active routing path and covered by the step-9 integration tests. **Conditional breaks:** if question 4 accepts `spike_density → region_spike_rate`, field readers must rename that access; if question 3 accepts changed diagnostics formatting, consumers comparing/parsing `routing_diagnostics` strings must adopt the documented new format. Step 8 records the final decisions and concrete before/after examples. Any decided `adapt_leak!` removal also names its replacement or explicit retirement; otherwise compatibility is retained temporarily. |
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
| **Premature archiving** strands Limen-Capital's pin or vendored router, active dependency docs, open issues, or consumers waiting for the successor release. | Step 12 is gated on steps 8–11, completion of the artifact ledger, the landed and tested `rmems/Limen-Capital#9` dependency/docs/de-vendoring migration, and a registry-resolvable/tagged v0.2.0; it remains a human action. |

---

## Open questions requiring a human decision

1. **UUID.** Accept the [recommendation](#package-identity-and-uuid) — regenerate
   `uuid4()` at step 2, before the consolidated release, so no consumer migrates its
   identity twice — even though it deviates from the literal wording of #53?
2. **Where does `adapt_leak!` go?** `thalamic-relay`, the `brainstem-daemon` workspace,
   or `LiquidCortex.jl`? This blocks **removal**, not step 5: under the
   [removal gate](#gate-adapt_leak-may-not-be-removed-into-a-void), an unanswered
   decision requires step 5 to carry the function forward as deprecated.
3. **`Printf` vs. zero-dep diagnostics.** Accept a changed `routing_diagnostics` output
   format to keep `[deps]` empty, or retain `Printf` and keep byte-identical output? This
   blocks changing the format/removing `Printf`, not step 5; without a decision, step 5
   preserves the current formatting path and dependency.
4. **Rename `RegionRouter.spike_density` → `region_spike_rate`?** It is a breaking
   read-access change for any consumer inspecting router state. This blocks the rename,
   not step 5: under the
   [field-name gate](#gate-the-spike_density-field-name-must-be-settled-before-the-router-lands),
   an unanswered decision requires step 5 to port the existing name.
5. **Alias deprecation mechanism — decided:** `Base.@deprecate_binding` for every
   deprecated alias (types, constants, and functions). Those aliases are not rewritten
   as wrapper functions, and documentation-only silent aliases are not used. A
   same-name carried function (`adapt_leak!`) is not an alias: it warns via inline
   `Base.depwarn` or `Base.@deprecate` to an `_`-prefixed implementation. Step 6 tests
   binding-level warnings and binding-kind preservation for aliases, plus the
   same-name warning when `adapt_leak!` is carried.
6. **Confirm** that Limen-Capital's dependency migration is owned by
   `rmems/Limen-Capital#9`. It does not block implementation in this repository, but its
   UUID/source/revision plus README and `docs/deps.md` updates are a hard step-12 archive
   gate.

---

## Status against issue #53's acceptance criteria

| Criterion | Status |
|---|---|
| Boundary documents the deterministic, self-contained routing kernel with bounded mutation and excludes runtime/learning/dense/finance | **Decided here**; the code and README/AGENTS text land together at step 5 and are consolidated at step 7 |
| Fresh-clone `Pkg.instantiate()` + `Pkg.test()` succeed | Passing today; re-gated at step 8 |
| Every inventoried artifact retained, migrated, or explicitly retired with rationale | **Dispositioned here** in the numbered [artifact ledger](#non-source-artifact-disposition-ledger); steps 3, 5, and 8 execute it, and steps 8/12 fail closed on any incomplete row |
| Imported behavior has parity tests or documented intentional changes | **Planned** (steps 3, 5); intentional changes recorded here |
| Timestamp adaptation and precision edge-case tests | **Specified** here; land at step 4 |
| Only one active package UUID/name pair | **Decided** here; identity set at step 2, shipped at step 8 |
| Versioned migration guide and upgrade matrix | **Drafted** here as [Upgrade matrix](#upgrade-matrix); published at step 8 |
| No active docs instruct users to depend on superseded repositories | Steps 7–9, including Limen-Capital README/`docs/deps.md`; hard archive gate |
| `NeuroPulse.jl` / `SpikeStream.jl` archived with successor links | Step 12, human action, explicitly **not** automated |
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
