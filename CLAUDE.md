@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Scope, code style, PR rules, and the consolidation notice are in `AGENTS.md`
(imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Julia 1.12.6 (Linux, CPU only),
the version recorded in the committed `Manifest.toml`.

| Command | Status |
|---|---|
| `julia --project=. -e 'using Pkg; Pkg.instantiate()'` | verified |
| `julia --project=. -e 'using Pkg; Pkg.test()'` | verified (2079 library tests plus 42 experiment-harness tests, all pass) |
| `julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'` | verified (creates an untracked `docs/Manifest.toml`; delete it) |
| `DOCUMENTER_BUILD_ONLY=true julia --project=docs docs/make.jl` | verified (the env var is what CI sets for the build job) |
| `julia --project=experiments experiments/run_all.jl` (after `Pkg.develop`/`instantiate`, see `experiments/README.md`) | not run (pulls CairoMakie; not in CI) |
| `julia --project=benchmark benchmark/run_benchmarks.jl` (after `Pkg.develop`/`instantiate`) | not run (pulls BenchmarkTools; not in CI) |

CI (`.github/workflows/ci.yml`) runs `julia-runtest` on Julia 1.12 only; `Documentation.yml` builds the docs and deploys on push and tags.
`AGENTS.md` says CI covers Julia 1.9 to 1.12 plus macOS and Windows, and "50 tests"; neither matches the workflow or the current suite.
No Julia formatter or linter is configured. Markdown is linted by Codacy's markdownlint (`.codacy.yml`, `.markdownlint.yaml`; MD013, MD033 and MD041 are off), so keep `CLAUDE.md` clean under those rules.

## Architecture map

- `src/TemporalFocus.jl`: module, exports, and `include` order (`types`, `discrete`, `temporal`, `continuous`, `normalization`).
- `src/types.jl`: `SpikeEvent`, `SpikeTrain`, `TemporalBuffer`, `prune!`. `src/temporal.jl`: `temporal_weight` and `spike_attention_temporal`.
- `src/discrete.jl`, `src/continuous.jl`: `spike_attention_discrete`, `spike_attention_continuous`. `src/normalization.jl`: `normalize_l1!`, `normalize_max!`.
- `test/runtests.jl`: single test file; it also runs the experiment-harness tests.
- `examples/*.jl`: runnable usage scripts. `experiments/`: separate project (`ExperimentUtils` harness plus scripts); results go to the gitignored `experiments/results/`.
- `benchmark/`: BenchmarkTools suite with its own `Project.toml`. `docs/`: Documenter site (`docs/make.jl`, `docs/src/`) and `docs/adr/`.
- `RELEASING.md` is the registration gate; `REVIEW.md` is the reviewer checklist.

## Invariants and non-goals

- Root `Project.toml` has an empty `[deps]`; Documenter, CairoMakie, and BenchmarkTools live only in `docs/`, `experiments/`, and `benchmark/`.
- Scope is spike-native temporal attention only. No STDP or plasticity, tokenization or embeddings, runtime scheduling, or finance/HFT semantics (`AGENTS.md`, `REVIEW.md`).
- Do not import NeuroPulse or SpikeStream here; ADR 0002 consolidates this package into NeuroPulse.jl.
- Spike values and temporal quantities are `Float32`; internal helpers are `_`-prefixed.
- Docs are Markdown only; `.markdownlint.yaml` allows inline HTML, but the README currently uses none.
- Source files carry `# SPDX-License-Identifier: MIT OR Apache-2.0` (every `.jl` file in `src/`, `test/`, `examples/`, `experiments/`, `benchmark/`, and `.agents/setup`). Keep it on new files; `docs/make.jl` currently lacks it.

## Gotchas

- Never tag, register, or publish from an agent session. Registration (`@JuliaRegistrator register`) is a human post-merge step in `RELEASING.md`, not a task for agents. `RELEASING.md` flags the template-looking UUID in `Project.toml` as a blocker before first registration.
- `Manifest.toml` is tracked even though `.gitignore` lists `/Manifest.toml`; do not commit changes to it as a side effect.
- `Project.toml` has no `exclude` list, so adding this file needs no manifest change.
- `.claude/` is not gitignored; do not commit local agent state there.
- The first `Pkg.test()` precompiles the package and test extras; allow a minute.
- Edit only what the task needs.
