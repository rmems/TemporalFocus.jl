# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    ExperimentUtils

Shared artifact contract for TemporalFocus experiments.

Every experiment writes into `experiments/results/<slug>/`:

- `config.toml` — the configuration actually used for the run
- `metrics.csv` — machine-readable per-condition metrics
- `figure.png` — the human-readable figure
- `summary.md` — the written interpretation

All paths are derived from `@__DIR__`, so scripts work from a fresh clone and
never depend on a local absolute path.
"""
module ExperimentUtils

using TOML

export repo_root, result_dir, figure_path, write_config, write_metrics, write_summary

"""
    repo_root() -> String

Absolute path to the repository root, derived from this file's location
(`experiments/src/ExperimentUtils.jl`). Never a hardcoded absolute path.
"""
repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

"""
    result_dir(slug) -> String

Absolute path to `experiments/results/<slug>`, created if it does not exist.
"""
function result_dir(slug::AbstractString)
    dir = joinpath(repo_root(), "experiments", "results", slug)
    mkpath(dir)
    return dir
end

"""
    figure_path(slug, name="figure.png") -> String

Absolute path for a figure inside the result directory of `slug`.
"""
figure_path(slug::AbstractString, name::AbstractString = "figure.png") =
    joinpath(result_dir(slug), name)

"""
    write_config(slug, cfg) -> String

Write `cfg` to `experiments/results/<slug>/config.toml` and return the path.
Keys are sorted so the file is byte-reproducible across runs.
"""
function write_config(slug::AbstractString, cfg::AbstractDict)
    path = joinpath(result_dir(slug), "config.toml")
    open(path, "w") do io
        TOML.print(io, cfg; sorted = true)
    end
    return path
end

_csv_cell(x::AbstractString) =
    any(c -> c in (',', '"', '\n'), x) ? string('"', replace(x, '"' => "\"\""), '"') : String(x)
_csv_cell(x::Bool) = x ? "true" : "false"
_csv_cell(x::Symbol) = _csv_cell(String(x))
_csv_cell(x::Nothing) = ""
_csv_cell(x::Missing) = ""
_csv_cell(x) = _csv_cell(string(x))

"""
    write_metrics(slug, rows) -> String

Write `rows::Vector{<:NamedTuple}` to `experiments/results/<slug>/metrics.csv`
and return the path. The header comes from the first row's field names; every
row must share those field names.

Non-finite floats are written as `Inf`, `-Inf`, and `NaN` (both Julia's
`parse`/`CSV.jl` and pandas read these back).
"""
function write_metrics(slug::AbstractString, rows)
    isempty(rows) && throw(ArgumentError("write_metrics needs at least one row"))
    path = joinpath(result_dir(slug), "metrics.csv")
    header = keys(first(rows))
    open(path, "w") do io
        println(io, join((_csv_cell(String(k)) for k in header), ','))
        for row in rows
            keys(row) == header ||
                throw(ArgumentError("all metric rows must share the same field names"))
            println(io, join((_csv_cell(v) for v in values(row)), ','))
        end
    end
    return path
end

"""
    write_summary(slug, md) -> String

Write `md` to `experiments/results/<slug>/summary.md` and return the path.
"""
function write_summary(slug::AbstractString, md::AbstractString)
    path = joinpath(result_dir(slug), "summary.md")
    open(path, "w") do io
        write(io, md)
        endswith(md, '\n') || write(io, '\n')
    end
    return path
end

end # module
