# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    ExperimentUtils

Shared artifact contract for TemporalFocus.jl experiments.

Every experiment writes into `experiments/results/<slug>/`:

| File | Written by | Contents |
|------|------------|----------|
| `config.toml` | [`write_config`](@ref) | configuration actually used, plus a generated `[provenance]` table |
| `metrics.csv` | [`write_metrics`](@ref) | machine-readable metric rows |
| `figure.png` | the experiment script, at [`figure_path`](@ref) | human-readable artifact |
| `summary.md` | [`write_summary`](@ref) | hypothesis, observation, and whether the result supports it |

All paths are derived from [`repo_root`](@ref), which is computed from this
file's own location, so experiments run correctly from a fresh clone and never
depend on a local absolute path.

Setting the `TEMPORALFOCUS_RESULTS_DIR` environment variable redirects the
results root; leave it unset for normal runs. The package test suite uses it to
exercise the harness in a temporary directory.

This module depends only on Julia standard libraries. It does not load
TemporalFocus, CairoMakie, or any other experiment dependency, so it can be
`include`d from anywhere:

```julia
using ExperimentUtils                                        # experiments/ environment
include(joinpath(@__DIR__, "src", "ExperimentUtils.jl"))     # or standalone
using .ExperimentUtils
```
"""
module ExperimentUtils

using Dates
using TOML

export repo_root, result_dir, figure_path, write_config, write_metrics, write_summary

const _CONFIG_FILE = "config.toml"
const _METRICS_FILE = "metrics.csv"
const _SUMMARY_FILE = "summary.md"
const _RESULTS_ENV = "TEMPORALFOCUS_RESULTS_DIR"
const _SLUG_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

"""
    repo_root() -> String

Absolute path of the TemporalFocus.jl repository root.

Derived from this file's location (`experiments/src/`), never from a hardcoded
absolute path, so it resolves correctly in any clone or worktree.
"""
repo_root() = dirname(dirname(normpath(@__DIR__)))

# Directory holding every experiment's result folder. Defaults to
# `<repo_root>/experiments/results`; `TEMPORALFOCUS_RESULTS_DIR` overrides it.
_results_root() = get(ENV, _RESULTS_ENV, joinpath(repo_root(), "experiments", "results"))

@inline function _check_slug(slug::AbstractString)
    occursin(_SLUG_PATTERN, slug) && slug != ".." ||
        throw(ArgumentError("invalid experiment slug $(repr(slug)): expected a path-free name matching $(_SLUG_PATTERN.pattern)"))
    return String(slug)
end

@inline function _check_filename(name::AbstractString)
    occursin(_SLUG_PATTERN, name) && name != ".." ||
        throw(ArgumentError("invalid artifact file name $(repr(name)): expected a path-free name matching $(_SLUG_PATTERN.pattern)"))
    return String(name)
end

"""
    result_dir(slug) -> String

Absolute path of `experiments/results/<slug>`, creating it if missing.

# Arguments
- `slug::AbstractString`: experiment slug, e.g. `"temporal-lens"`. Must be a
  path-free name (letters, digits, `.`, `-`, `_`).

# Returns
- the directory path, guaranteed to exist

# Throws
- `ArgumentError` if `slug` contains a path separator or is otherwise unsafe
"""
function result_dir(slug::AbstractString)
    dir = joinpath(_results_root(), _check_slug(slug))
    mkpath(dir)
    return dir
end

"""
    figure_path(slug, name="figure.png") -> String

Path to write a figure for experiment `slug`, inside [`result_dir`](@ref).

The directory is created if missing, so the returned path is always writable:

```julia
save(figure_path("temporal-lens"), fig)
save(figure_path("temporal-lens", "decay_curves.png"), fig2)
```

# Arguments
- `slug::AbstractString`: experiment slug
- `name::AbstractString`: file name (default `"figure.png"`); must be path-free

# Returns
- absolute path of the figure file (the file itself is not created)
"""
function figure_path(slug::AbstractString, name::AbstractString = "figure.png")
    return joinpath(result_dir(slug), _check_filename(name))
end

_toml_value(x::AbstractString) = String(x)
_toml_value(x::Bool) = x
_toml_value(x::Integer) = Int(x)
_toml_value(x::Symbol) = String(x)
_toml_value(x::Char) = string(x)
_toml_value(x::Union{Dates.DateTime,Dates.Date,Dates.Time}) = x
# Round-trip through the shortest decimal form so Float32 inputs stay readable
# (0.2f0 becomes 0.2, not 0.20000000298023224).
_toml_value(x::AbstractFloat) = isfinite(x) ? parse(Float64, _format_float(x)) : Float64(x)
_toml_value(x::AbstractDict) = _toml_table(x)
_toml_value(x::NamedTuple) = _toml_table(x)
_toml_value(x::Union{AbstractVector,Tuple}) = Any[_toml_value(v) for v in x]
_toml_value(x) = string(x)

function _toml_table(cfg)
    table = Dict{String,Any}()
    for (key, value) in pairs(cfg)
        table[string(key)] = _toml_value(value)
    end
    return table
end

function _git_output(args::Cmd)
    try
        return strip(read(pipeline(Cmd(`git -C $(repo_root()) $args`); stderr = devnull), String))
    catch
        return ""
    end
end

function _provenance()
    commit = _git_output(`rev-parse HEAD`)
    return Dict{String,Any}(
        "git_commit" => isempty(commit) ? "unknown" : commit,
        "git_dirty" => !isempty(_git_output(`status --porcelain`)),
        "julia_version" => string(VERSION),
        "generated_utc" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ"),
    )
end

"""
    write_config(slug, cfg) -> String

Write the configuration a run actually used to `experiments/results/<slug>/config.toml`.

Keys are emitted sorted, so re-running with the same configuration reproduces the
same file. Values are normalized for TOML: `Float32` is widened through its
shortest decimal form (`0.2f0` → `0.2`), `Symbol` becomes a string, and nested
dictionaries/named tuples become sub-tables.

A `[provenance]` table (git commit, dirty flag, Julia version, UTC timestamp) is
appended automatically unless `cfg` already defines `provenance`.

# Arguments
- `slug::AbstractString`: experiment slug
- `cfg::AbstractDict`: configuration values (a `NamedTuple` is also accepted)

# Returns
- path of the written `config.toml`
"""
function write_config(slug::AbstractString, cfg::AbstractDict)
    table = _toml_table(cfg)
    get!(table, "provenance", _provenance())
    path = joinpath(result_dir(slug), _CONFIG_FILE)
    open(path, "w") do io
        TOML.print(io, table; sorted = true)
    end
    return path
end

write_config(slug::AbstractString, cfg::NamedTuple) = write_config(slug, Dict(pairs(cfg)))

_format_float(x::AbstractFloat) = _format_float(Float32(x))
function _format_float(x::Union{Float32,Float64})
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    s = string(x)
    # Julia < 1.12 prints Float32 with an `f` exponent marker ("0.2f0", "1.0f-5").
    if occursin('f', s)
        s = replace(s, 'f' => 'e')
        endswith(s, "e0") && (s = s[1:(end - 2)])
    end
    return s
end

_csv_cell(::Nothing) = ""
_csv_cell(::Missing) = ""
_csv_cell(x::Bool) = x ? "true" : "false"
_csv_cell(x::Integer) = string(x)
_csv_cell(x::AbstractFloat) = _format_float(x)
_csv_cell(x) = _csv_escape(string(x))

function _csv_escape(s::AbstractString)
    if any(c -> c == ',' || c == '"' || c == '\n' || c == '\r', s)
        return string('"', replace(s, '"' => "\"\""), '"')
    end
    return String(s)
end

"""
    write_metrics(slug, rows) -> String

Write machine-readable metrics to `experiments/results/<slug>/metrics.csv`.

# Arguments
- `slug::AbstractString`: experiment slug
- `rows`: non-empty collection of `NamedTuple`s sharing the same field names,
  e.g. `[(dt = 0.1f0, tau = 0.2f0, weight = 0.6f0), ...]`. Field names become
  the header row, in declaration order.

Floats are written in their shortest round-tripping decimal form (`Float32`
never leaks a `f0` suffix), `nothing`/`missing` become empty cells, and string
cells are quoted only when they contain a comma, quote, or newline.

# Returns
- path of the written `metrics.csv`

# Throws
- `ArgumentError` if `rows` is empty, holds anything other than `NamedTuple`s,
  or the rows disagree on field names
"""
function write_metrics(slug::AbstractString, rows)
    collected = collect(rows)
    isempty(collected) && throw(ArgumentError("metrics rows must not be empty"))
    for (i, row) in enumerate(collected)
        row isa NamedTuple ||
            throw(ArgumentError("metrics row $(i) is a $(typeof(row)); expected a NamedTuple"))
    end
    header = keys(first(collected))
    for (i, row) in enumerate(collected)
        keys(row) == header ||
            throw(ArgumentError("metrics row $(i) has columns $(keys(row)); expected $(header)"))
    end

    path = joinpath(result_dir(slug), _METRICS_FILE)
    open(path, "w") do io
        println(io, join((_csv_escape(string(name)) for name in header), ","))
        for row in collected
            println(io, join((_csv_cell(value) for value in values(row)), ","))
        end
    end
    return path
end

"""
    write_summary(slug, md) -> String

Write a human-readable summary to `experiments/results/<slug>/summary.md`.

The summary is where an experiment states its hypothesis and whether the
observed result supports it. A trailing newline is added if `md` lacks one.

# Arguments
- `slug::AbstractString`: experiment slug
- `md::AbstractString`: Markdown body

# Returns
- path of the written `summary.md`
"""
function write_summary(slug::AbstractString, md::AbstractString)
    path = joinpath(result_dir(slug), _SUMMARY_FILE)
    open(path, "w") do io
        write(io, md)
        endswith(md, "\n") || write(io, "\n")
    end
    return path
end

end
