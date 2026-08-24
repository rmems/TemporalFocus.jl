# TemporalFocus.jl experiments

Reproducible spike-native experiments that make temporal attention observable.

This directory is an **isolated Julia environment**. Visualization and data
dependencies (CairoMakie and friends) live in `experiments/Project.toml` only —
the root `Project.toml` stays dependency-free, and `julia --project=. -e 'using
Pkg; Pkg.test()'` never instantiates this environment.

## Setup

One-time, from the repository root:

```bash
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

`Pkg.develop(path=".")` points the experiment environment at the working copy of
TemporalFocus, so experiments always run against the checked-out source.

The experiment environment needs Julia 1.10+ (CairoMakie); the package itself
still supports Julia 1.9.

## Running

```bash
julia --project=experiments experiments/run_all.jl              # everything present
julia --project=experiments experiments/run_all.jl --list       # what would run
julia --project=experiments experiments/run_all.jl jitter_test  # one experiment
julia --project=experiments experiments/harness_smoke.jl        # or call a script directly
```

`run_all.jl` runs each script in its own Julia process, in a fixed narrative
order, and skips scripts that do not exist yet — the experiment wave lands
incrementally, so a missing `temporal_lens.jl` is reported as *pending*, not as
an error. Unknown `*.jl` files directly under `experiments/` are discovered
automatically and run afterwards, sorted by name. The runner exits non-zero if
any script fails, and one failing script never stops the others.

## Artifact contract

Every experiment writes to `experiments/results/<slug>/`:

| File | Written by | Contents |
|------|------------|----------|
| `config.toml` | `write_config` | the configuration actually used, plus a generated `[provenance]` table |
| `metrics.csv` | `write_metrics` | machine-readable rows (one header line, one row per measurement) |
| `figure.png` | the script, saved at `figure_path(slug)` | the human-readable artifact |
| `summary.md` | `write_summary` | hypothesis, observation, and whether the result supports it |

Every experiment must:

1. run from a single documented command;
2. seed its randomness deterministically;
3. record the configuration it used;
4. emit machine-readable metrics;
5. emit at least one human-readable artifact;
6. state its hypothesis and whether the observation supports it;
7. derive every path from the repository root, never from a local absolute path.

## Harness API

`experiments/src/ExperimentUtils.jl` provides the shared contract. It depends
only on Julia standard libraries, so it loads either as the environment's
package or as a plain include:

```julia
using ExperimentUtils                                     # experiments environment
# or, standalone:
include(joinpath(@__DIR__, "src", "ExperimentUtils.jl"))
using .ExperimentUtils
```

| Function | Returns |
|----------|---------|
| `repo_root()` | repository root, derived from the harness file's location |
| `result_dir(slug)` | `experiments/results/<slug>`, created if missing |
| `figure_path(slug, name="figure.png")` | path to write a figure into that directory |
| `write_config(slug, cfg::AbstractDict)` | path of the written `config.toml` |
| `write_metrics(slug, rows::Vector{<:NamedTuple})` | path of the written `metrics.csv` |
| `write_summary(slug, md::AbstractString)` | path of the written `summary.md` |

Notes:

- `Float32` values are written in their shortest decimal form, so `0.2f0`
  becomes `0.2` in both TOML and CSV — no `f0` suffix leaks into artifacts.
  `metrics.csv` keeps each value's own precision; `config.toml` records floats
  as `Float64`, which is what the TOML specification allows.
- `write_metrics` requires every row to share the same field names, in the same
  order; field names become the CSV header.
- `write_config` appends a `[provenance]` table (git commit, dirty flag, Julia
  version, UTC timestamp) unless the configuration already defines one. It is
  the only intentionally non-reproducible part of the artifacts.
- Setting `TEMPORALFOCUS_RESULTS_DIR` redirects the results root (a relative
  value is resolved against the working directory). Leave it unset for normal
  runs; the package test suite uses it to exercise the harness in a temporary
  directory.

## Writing a new experiment

Add `experiments/<name>.jl` and, if it belongs to the gallery narrative, its
file name to `ORDERED_EXPERIMENTS` in `run_all.jl`:

```julia
# SPDX-License-Identifier: MIT OR Apache-2.0
using CairoMakie
using ExperimentUtils
using Random
using TemporalFocus

const SLUG = "my-experiment"
const RNG_SEED = 42

function main()
    rng = MersenneTwister(RNG_SEED)
    # ... build a spike scene, measure something ...

    write_config(SLUG, Dict("seed" => RNG_SEED, "tau" => 0.2f0))
    write_metrics(SLUG, [(tau = 0.2f0, value = 1.0f0)])
    save(figure_path(SLUG), fig)
    write_summary(SLUG, "# My experiment\n\n## Hypothesis\n...\n\n## Verdict\n...")
    return nothing
end

main()
```

Determinism rules: seed every RNG explicitly (`MersenneTwister(seed)` rather
than the global RNG), keep sweeps as explicit grids, and record every parameter
that changes a number in `config.toml`. Identical inputs on the same Julia
version reproduce byte-identical `metrics.csv`.

If an experiment needs another package, add it to `experiments/Project.toml`
with a compat bound — never to the root `Project.toml`. `Statistics`, `Printf`,
`Random`, `Dates`, and `TOML` are pre-declared so common standard-library use
does not require editing that file.

## Results policy

`experiments/results/` and `experiments/Manifest.toml` are **git-ignored**:

- results are regenerated by one command, and six experiments' worth of PNGs
  would churn the repository (and conflict across parallel PRs) for no gain;
- the resolved manifest for CairoMakie is large and platform/version-specific,
  and pinning it would break the Julia versions this repository supports.
  Reproducibility comes from compat bounds plus the recorded `[provenance]`
  table instead.

Figures that are *published* (the docs experiment gallery) are copied into the
documentation tree and committed there, kept small (well under 1 MB each) and
regenerable from the script that produced them.

## Boundary

Experiments stay pure spike-native characterizations of TemporalFocus: spike
trains, temporal buffers, attention kernels, normalization, readouts. No market
or exchange data, no token/embedding/LLM integration, no plasticity rules, no
runtime orchestration — those belong in downstream workspaces that consume
TemporalFocus through a narrow interface.
