# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    ExperimentUtils

Shared filesystem helpers for the scripts under `experiments/`.

Every experiment writes its artifacts to `experiments/results/<slug>/`:

- `config.toml` — the exact parameters the run used
- `metrics.csv` — one row per recorded step
- `summary.md` — human-readable summary
- `figure.png` (and friends) — figures

Nothing here depends on the TemporalFocus package; the experiments environment
is deliberately separate so that plotting and data dependencies never leak into
the root `Project.toml`.
"""
module ExperimentUtils

using TOML

export repo_root, result_dir, figure_path, write_config, write_metrics, write_summary

"""
    repo_root() -> String

Absolute path to the repository root, derived from this file's location.
"""
repo_root()::String = normpath(joinpath(@__DIR__, "..", ".."))

"""
    result_dir(slug) -> String

Absolute path to `experiments/results/<slug>`, creating it if missing.
"""
function result_dir(slug::AbstractString)::String
    dir = joinpath(repo_root(), "experiments", "results", slug)
    isdir(dir) || mkpath(dir)
    return dir
end

"""
    figure_path(slug, name="figure.png") -> String

Absolute path for a figure inside the result directory of `slug`.
"""
figure_path(slug::AbstractString, name::AbstractString = "figure.png")::String =
    joinpath(result_dir(slug), name)

"""
    write_config(slug, cfg) -> String

Write `cfg` as `config.toml` in the result directory of `slug` (keys sorted so
repeated runs produce identical files). Returns the written path.
"""
function write_config(slug::AbstractString, cfg::AbstractDict)::String
    path = joinpath(result_dir(slug), "config.toml")
    open(path, "w") do io
        TOML.print(io, Dict{String,Any}(String(k) => v for (k, v) in cfg); sorted = true)
    end
    return path
end

_csv_field(x::Integer) = string(x)
_csv_field(x::Bool) = x ? "true" : "false"
_csv_field(x::AbstractFloat) = string(x)
function _csv_field(x)
    s = string(x)
    return any(c -> c in (',', '"', '\n'), s) ? '"' * replace(s, '"' => "\"\"") * '"' : s
end

"""
    write_metrics(slug, rows) -> String

Write `rows` (a vector of `NamedTuple`s that all share the same field names) as
`metrics.csv` in the result directory of `slug`. Returns the written path.
"""
function write_metrics(slug::AbstractString, rows)::String
    path = joinpath(result_dir(slug), "metrics.csv")
    open(path, "w") do io
        isempty(rows) && return nothing
        header = keys(first(rows))
        println(io, join(String.(header), ","))
        for row in rows
            keys(row) == header ||
                throw(ArgumentError("all metric rows must share the same field names"))
            println(io, join((_csv_field(v) for v in values(row)), ","))
        end
        return nothing
    end
    return path
end

"""
    write_summary(slug, md) -> String

Write `md` as `summary.md` in the result directory of `slug`. Returns the
written path.
"""
function write_summary(slug::AbstractString, md::AbstractString)::String
    path = joinpath(result_dir(slug), "summary.md")
    write(path, endswith(md, "\n") ? md : md * "\n")
    return path
end

end # module
