# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    ExperimentUtils

Minimal shared plumbing for the scripts in `experiments/`.

Every experiment writes its artifacts under `experiments/results/<slug>/`:

- `config.toml` — the exact swept parameters and seeds ([`write_config`](@ref))
- `metrics.csv` — one row per measured condition ([`write_metrics`](@ref))
- `summary.md` — generated interpretation ([`write_summary`](@ref))
- `figure.png` — the primary figure ([`figure_path`](@ref))

Paths are always derived from `@__DIR__`, never hardcoded, so a checkout can
live anywhere.
"""
module ExperimentUtils

export repo_root, result_dir, figure_path, write_config, write_metrics, write_summary

"""
    repo_root() -> String

Absolute path to the repository root, derived from this file's location.
"""
repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

"""
    result_dir(slug) -> String

Absolute path to `experiments/results/<slug>`, creating it if missing.
"""
function result_dir(slug::AbstractString)
    dir = joinpath(repo_root(), "experiments", "results", String(slug))
    mkpath(dir)
    return dir
end

"""
    figure_path(slug, name="figure.png") -> String

Absolute path for a figure inside the experiment's result directory.
"""
figure_path(slug::AbstractString, name::AbstractString = "figure.png") =
    joinpath(result_dir(slug), String(name))

_toml_scalar(value::AbstractString) = string('"', replace(String(value), '\\' => "\\\\", '"' => "\\\""), '"')
_toml_scalar(value::Bool) = value ? "true" : "false"
_toml_scalar(value::Integer) = string(value)
# `string` gives the shortest round-tripping decimal (and drops the `f0` suffix
# for `Float32`), which keeps `config.toml` readable and diff-stable.
_toml_scalar(value::AbstractFloat) = isfinite(value) ? string(value) : string('"', value, '"')
_toml_scalar(value) = _toml_scalar(string(value))
_toml_value(value::AbstractVector) = string('[', join(_toml_scalar.(value), ", "), ']')
_toml_value(value) = _toml_scalar(value)

"""
    write_config(slug, cfg::AbstractDict) -> String

Write `cfg` to `experiments/results/<slug>/config.toml` as a flat TOML table
with keys sorted for a stable diff. Values may be scalars or vectors of
scalars. Returns the path written.
"""
function write_config(slug::AbstractString, cfg::AbstractDict)
    path = joinpath(result_dir(slug), "config.toml")
    open(path, "w") do io
        for key in sort!(collect(string.(keys(cfg))))
            println(io, key, " = ", _toml_value(cfg[key]))
        end
    end
    return path
end

_csv_field(value::AbstractString) =
    any(c -> c in (',', '"', '\n'), value) ? string('"', replace(String(value), '"' => "\"\""), '"') : String(value)
_csv_field(value) = _csv_field(string(value))

"""
    write_metrics(slug, rows) -> String

Write `rows::Vector{<:NamedTuple}` to `experiments/results/<slug>/metrics.csv`.
The header comes from the first row's field names; every row must share them.
Returns the path written.
"""
function write_metrics(slug::AbstractString, rows)
    path = joinpath(result_dir(slug), "metrics.csv")
    open(path, "w") do io
        isempty(rows) && return nothing
        columns = keys(first(rows))
        println(io, join(string.(columns), ","))
        for row in rows
            println(io, join((_csv_field(getproperty(row, c)) for c in columns), ","))
        end
        return nothing
    end
    return path
end

"""
    write_summary(slug, md) -> String

Write `md` to `experiments/results/<slug>/summary.md`. Returns the path written.
"""
function write_summary(slug::AbstractString, md::AbstractString)
    path = joinpath(result_dir(slug), "summary.md")
    write(path, md)
    return path
end

end
