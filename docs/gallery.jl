# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    Gallery

Generator for the TemporalFocus **Experiment Gallery** (`docs/src/experiments.md`).

The gallery is never written by hand. It is rendered from the reproducible
artifacts that experiment scripts emit under

    experiments/results/<slug>/{config.toml, metrics.csv, figure.png, summary.md}

Design rules:

* **Nothing is invented.** Setup values come from `config.toml`, quantitative
  values come from `metrics.csv`, and the observed result is the experiment's own
  generated `summary.md` embedded verbatim (so null and negative findings survive).
* **Degrades gracefully.** Any subset of the experiments — including none of them —
  renders a correct, buildable page. Missing experiments are listed as *pending*
  with their question and hypothesis, and explicitly no results.
* **Stdlib only.** Uses `TOML` and plain string handling so it can run inside the
  docs environment and inside `Pkg.test()` without new dependencies.

Entry point: [`build_gallery`](@ref).
"""
module Gallery

using TOML

export build_gallery

"""
    GalleryEntry

Editorial metadata for one expected experiment. Question, hypothesis and the
description of the primary artifact are prose taken from the tracking issue; they
contain no quantitative claims. Every number on the rendered page comes from a
generated artifact instead.
"""
struct GalleryEntry
    slug::String
    title::String
    issue::Int
    question::String
    hypothesis::String
    artifact::String
end

"""
Expected experiments, in narrative order. A slug may be absent from
`experiments/results/`; it is then rendered as a pending entry. Result
directories whose slug is not listed here are rendered too, in a trailing
section, so a new experiment shows up without editing this file.
"""
const ENTRIES = GalleryEntry[
    GalleryEntry(
        "temporal_lens",
        "Temporal Lens",
        46,
        "How does the temporal attention contribution change jointly with spike separation `Δt` and time constant `τ`?",
        "Attention is governed by the ratio `|Δt| / τ`: small `τ` gives a narrow focus around `Δt = 0`, large `τ` preserves contributions over longer delays, and the field is symmetric in `Δt` and matches the analytic `exp(-|Δt|/τ)` law to `Float32` tolerance.",
        "A `Δt × τ` heatmap of the recency field — the temporal focus cone — plus decay curves for representative short-, medium- and long-memory `τ`.",
    ),
    GalleryEntry(
        "three_regimes",
        "Three Regimes",
        47,
        "Given exactly the same synthetic spike scene, how do `spike_attention_discrete`, `spike_attention_temporal` and `spike_attention_continuous` differ in what they preserve, decay and reject?",
        "Discrete attention ignores timing and accumulates every matching-neuron interaction; temporal attention keeps them but exponentially downweights stale ones; continuous attention adds a hard window and rejects pairs outside it.",
        "One spike scene shown three ways — the raster plus per-neuron attention mass under each kernel: *same spikes, three notions of focus*.",
    ),
    GalleryEntry(
        "focus_under_fire",
        "Focus Under Fire",
        48,
        "How robust is spike-native temporal attention when the context fills up with stale and random same-neuron distractor activity?",
        "Temporal and continuous attention preserve a higher share of attention on the recent target than timing-agnostic discrete attention as distractor load grows, with continuous attention rejecting most strongly once distractors fall outside the active window.",
        "Focus-retention and top-1 correctness curves versus distractor load: how much temporal garbage the signal survives before focus flips.",
    ),
    GalleryEntry(
        "jitter_test",
        "Jitter Test",
        49,
        "How quickly does TemporalFocus lose target selectivity as spike timestamps are perturbed, and how does that sensitivity depend on `τ` and the continuous window?",
        "Discrete attention is invariant to timestamp jitter; temporal attention degrades smoothly as jitter grows relative to `τ`; continuous attention degrades smoothly until jitter pushes relevant pairs outside the window, where sharper transitions appear.",
        "A timing-tolerance envelope: target retention and output drift versus jitter scale, with a regime map over `τ` and window settings.",
    ),
    GalleryEntry(
        "attention_spotlight",
        "Attention Spotlight",
        50,
        "How does continuous attention move between neurons as a `TemporalBuffer` is filled, pruned and replayed through simulated time?",
        "Causal ingestion plus repeated `prune!` produces visible focus handoffs: a pattern holds attention while it is recent, then loses it as it decays and ages out of the window.",
        "A time × neuron attention heatmap with a top-1 focus trace and buffer occupancy — an attention spotlight sweeping across neurons.",
    ),
    GalleryEntry(
        "memory_gate",
        "Memory Gate",
        51,
        "Across the `τ × window` parameter plane, where does continuous attention preserve a recent target while suppressing a stale same-neuron distractor?",
        "The plane separates into a hard-clipped regime (the window clips the target), a selective gate (target in, stale out), a soft-decay regime (both admitted but `τ` suppresses the stale contribution), and an over-retentive regime (wide window and large `τ` let stale mass back in).",
        "A `τ × window` phase map of target attention share with a companion stale-leakage map — a memory control panel.",
    ),
]

"Config keys that describe *provenance* rather than experiment setup."
const PROVENANCE_KEYS = Set([
    "commit", "git_commit", "commit_sha", "sha", "revision", "rev", "git_rev",
    "generated_at", "generated", "timestamp", "date", "run_at",
    "julia_version", "package_version", "temporalfocus_version", "version",
    "script", "hostname", "os",
])

"Provenance keys that identify the code version a result was produced from."
const COMMIT_KEYS = ["commit", "git_commit", "commit_sha", "sha", "revision", "rev", "git_rev"]

"""
    leaf_key(flattened) -> String

Last component of a flattened config key, lowercased. Provenance is classified on
the leaf so a script that groups its fields (`[metadata] commit = ...` flattening
to `metadata.commit`) is still recognized as recording a code version.
"""
leaf_key(flattened::AbstractString) = lowercase(String(last(split(flattened, '.'))))

is_provenance_key(flattened::AbstractString) = leaf_key(flattened) in PROVENANCE_KEYS
is_commit_key(flattened::AbstractString) = leaf_key(flattened) in COMMIT_KEYS

const REPO_URL = "https://github.com/rmems/TemporalFocus.jl"

"Repository root, inferred from this file's location (`<root>/docs/gallery.jl`)."
default_repo_root() = normpath(joinpath(@__DIR__, ".."))

# ---------------------------------------------------------------------------
# artifact discovery
# ---------------------------------------------------------------------------

"""
    ResultSet

Everything found on disk for one result slug. Every field is optional: an entry
with a directory but no artifacts is still rendered, and reports what is
missing.
"""
struct ResultSet
    slug::String
    dir::String
    config::Union{Nothing,Dict{String,Any}}
    config_path::Union{Nothing,String}
    metrics_path::Union{Nothing,String}
    figure_path::Union{Nothing,String}
    summary_path::Union{Nothing,String}
    extra_figures::Vector{String}
end

has_artifacts(r::ResultSet) =
    r.config_path !== nothing || r.metrics_path !== nothing ||
    r.figure_path !== nothing || r.summary_path !== nothing ||
    !isempty(r.extra_figures)

_existing(path) = isfile(path) ? path : nothing

function _read_config(path)
    path === nothing && return nothing
    try
        return TOML.parsefile(path)
    catch err
        @warn "gallery: could not parse config" path exception = err
        return nothing
    end
end

"""
    collect_result(results_dir, slug) -> ResultSet
"""
function collect_result(results_dir::AbstractString, slug::AbstractString)
    dir = joinpath(results_dir, slug)
    config_path = _existing(joinpath(dir, "config.toml"))
    figure = _existing(joinpath(dir, "figure.png"))
    extras = String[]
    if isdir(dir)
        for name in sort(readdir(dir))
            name == "figure.png" && continue
            if endswith(lowercase(name), ".png") ||
               endswith(lowercase(name), ".svg") ||
               endswith(lowercase(name), ".gif")
                push!(extras, name)
            end
        end
    end
    return ResultSet(
        slug,
        dir,
        _read_config(config_path),
        config_path,
        _existing(joinpath(dir, "metrics.csv")),
        figure,
        _existing(joinpath(dir, "summary.md")),
        extras,
    )
end

"""
    discovered_slugs(results_dir) -> Vector{String}

Every subdirectory of `results_dir`, sorted. Returns an empty vector when the
directory does not exist (the common case before the experiment wave lands).
"""
function discovered_slugs(results_dir::AbstractString)
    isdir(results_dir) || return String[]
    return sort([name for name in readdir(results_dir) if isdir(joinpath(results_dir, name))])
end

# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

"Escape a value so it is safe inside a Markdown table cell."
function cell(value::AbstractString)
    text = replace(String(value), "\r\n" => " ", '\n' => " ", '\r' => " ")
    text = replace(text, "|" => "\\|")
    return strip(text)
end

format_value(v::AbstractString) = String(v)
format_value(v::Bool) = string(v)
format_value(v::Real) = string(v)
format_value(v::AbstractVector) = join(map(format_value, v), ", ")
format_value(v) = string(v)

"""
    flatten_config(cfg) -> Vector{Pair{String,String}}

Flatten nested TOML tables into `parent.child` keys, sorted by key.
"""
function flatten_config(cfg::AbstractDict, prefix::AbstractString = "")
    out = Pair{String,String}[]
    for key in sort(collect(keys(cfg)); by = string)
        value = cfg[key]
        name = isempty(prefix) ? string(key) : string(prefix, ".", key)
        if value isa AbstractDict
            append!(out, flatten_config(value, name))
        else
            push!(out, name => format_value(value))
        end
    end
    return out
end

"Split one CSV record, honoring double-quoted fields."
function split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quotes = false
    chars = collect(line)
    i = 1
    while i <= length(chars)
        c = chars[i]
        if in_quotes
            if c == '"'
                if i < length(chars) && chars[i+1] == '"'
                    print(buf, '"')
                    i += 1
                else
                    in_quotes = false
                end
            else
                print(buf, c)
            end
        elseif c == '"'
            in_quotes = true
        elseif c == ','
            push!(fields, String(take!(buf)))
        else
            print(buf, c)
        end
        i += 1
    end
    push!(fields, String(take!(buf)))
    return fields
end

"""
    csv_preview(path; maxrows=8) -> (header, rows, total_rows)

Read the header and at most `maxrows` data rows, and count the remaining data
rows without holding them in memory.
"""
function csv_preview(path::AbstractString; maxrows::Int = 8)
    header = String[]
    rows = Vector{Vector{String}}()
    total = 0
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            if isempty(header)
                header = split_csv_line(line)
                continue
            end
            total += 1
            if length(rows) < maxrows
                push!(rows, split_csv_line(line))
            end
        end
    end
    return header, rows, total
end

"""
    shift_headings(md, shift) -> String

Push every ATX heading down by `shift` levels so an embedded `summary.md` nests
under the gallery's own headings, leaving fenced code blocks untouched. Fenced
blocks tagged with a Documenter directive (```@example`, ```@docs`, …) are
neutralized to plain text: embedded summaries are generated content and must not
be able to execute code during the docs build.
"""
function shift_headings(md::AbstractString, shift::Integer)
    out = IOBuffer()
    fence = nothing  # currently open fence marker, or nothing
    for line in split(md, '\n'; keepempty = true)
        m = match(r"^(\s*)(`{3,}|~{3,})(.*)$", line)
        if m !== nothing
            indent, marker, info = m.captures[1], m.captures[2], m.captures[3]
            if fence === nothing
                fence = marker[1:1]
                tag = strip(info)
                startswith(tag, "@") && (info = "text")
                println(out, indent, marker, info)
            else
                marker[1:1] == fence && (fence = nothing)
                println(out, line)
            end
            continue
        end
        if fence === nothing
            h = match(r"^(#{1,6})(\s.*)?$", line)
            if h !== nothing
                level = min(6, length(h.captures[1]) + shift)
                println(out, "#"^level, h.captures[2] === nothing ? "" : h.captures[2])
                continue
            end
        end
        println(out, line)
    end
    text = String(take!(out))
    return endswith(text, "\n") ? text[1:prevind(text, lastindex(text))] : text
end

"Repo-relative path with forward slashes, for display and for GitHub links."
function rel_path(root::AbstractString, path::AbstractString)
    rel = relpath(path, root)
    return replace(rel, '\\' => '/')
end

blob_url(root, path) = string(REPO_URL, "/blob/main/", rel_path(root, path))

"""
    result_commit(root, dir) -> Union{Nothing,String}

Best-effort provenance fallback: the commit that last touched a result
directory. Returns `nothing` when git is unavailable or the artifacts are not
committed yet.
"""
function result_commit(root::AbstractString, dir::AbstractString)
    isdir(joinpath(root, ".git")) || isfile(joinpath(root, ".git")) || return nothing
    try
        sha = readchomp(`git -C $root log -1 --format=%H -- $dir`)
        return isempty(sha) ? nothing : sha
    catch
        return nothing
    end
end

# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

function render_setup(io::IO, result::ResultSet, root::AbstractString)
    println(io, "#### Setup")
    println(io)
    if result.config === nothing
        println(io, "No `config.toml` was recorded for this run, so the setup cannot be")
        println(io, "reported from generated evidence.")
        println(io)
        return
    end
    pairs = filter(p -> !is_provenance_key(first(p)), flatten_config(result.config))
    if isempty(pairs)
        println(io, "`config.toml` records no setup parameters beyond provenance fields.")
        println(io)
        return
    end
    println(io, "Values below are read from ",
        "[`", rel_path(root, result.config_path), "`](", blob_url(root, result.config_path), ").")
    println(io)
    println(io, "| Parameter | Value |")
    println(io, "|---|---|")
    for (key, value) in pairs
        println(io, "| `", cell(key), "` | `", cell(value), "` |")
    end
    println(io)
end

function render_artifact(io::IO, result::ResultSet, title::AbstractString, root::AbstractString,
                         assets_rel::AbstractString)
    println(io, "#### Primary artifact")
    println(io)
    if result.figure_path === nothing
        println(io, "No `figure.png` was emitted for this run.")
        if !isempty(result.extra_figures)
            println(io)
            println(io, "Other generated graphics in this result directory:")
            println(io)
            for name in result.extra_figures
                path = joinpath(result.dir, name)
                println(io, "- [`", rel_path(root, path), "`](", blob_url(root, path), ")")
            end
        end
        println(io)
        return
    end
    println(io, "![", title, " — generated figure](", assets_rel, "/", result.slug, "/figure.png)")
    println(io)
    println(io, "*Generated by the experiment script; source of truth is ",
        "[`", rel_path(root, result.figure_path), "`](", blob_url(root, result.figure_path), ").*")
    println(io)
    if !isempty(result.extra_figures)
        println(io, "Companion graphics: ",
            join(["[`$(name)`]($(blob_url(root, joinpath(result.dir, name))))"
                  for name in result.extra_figures], ", "), ".")
        println(io)
    end
end

function render_result(io::IO, result::ResultSet, root::AbstractString)
    println(io, "#### Observed result")
    println(io)
    if result.summary_path === nothing
        println(io, "No `summary.md` was emitted for this run, so no interpretation is")
        println(io, "published here. The metrics below remain the primary evidence.")
        println(io)
        return
    end
    text = strip(read(result.summary_path, String))
    if isempty(text)
        println(io, "The generated `summary.md` is empty.")
        println(io)
        return
    end
    println(io, "Quoted verbatim from the generated ",
        "[`", rel_path(root, result.summary_path), "`](", blob_url(root, result.summary_path), ") ",
        "— including any null or negative finding.")
    println(io)
    println(io, shift_headings(text, 4))
    println(io)
end

function render_metrics(io::IO, result::ResultSet, root::AbstractString)
    println(io, "#### Metrics")
    println(io)
    if result.metrics_path === nothing
        println(io, "No `metrics.csv` was emitted for this run.")
        println(io)
        return
    end
    header, rows, total = csv_preview(result.metrics_path)
    link = string("[`", rel_path(root, result.metrics_path), "`](",
        blob_url(root, result.metrics_path), ")")
    if isempty(header)
        println(io, "The generated metrics file ", link, " is empty.")
        println(io)
        return
    end
    println(io, total, " recorded row", total == 1 ? "" : "s", " over ", length(header),
        " column", length(header) == 1 ? "" : "s", " in ", link, ".")
    println(io)
    println(io, "| ", join([cell(h) for h in header], " | "), " |")
    println(io, "|", repeat("---|", length(header)))
    for row in rows
        padded = length(row) < length(header) ?
                 vcat(row, fill("", length(header) - length(row))) : row[1:length(header)]
        println(io, "| ", join([cell(c) for c in padded], " | "), " |")
    end
    println(io)
    if total > length(rows)
        println(io, "*Preview of the first ", length(rows), " of ", total,
            " rows; the full machine-readable table is in ", link, ".*")
        println(io)
    end
end

function render_provenance(io::IO, result::ResultSet, root::AbstractString)
    println(io, "#### Evidence and provenance")
    println(io)
    println(io, "| Artifact | Path |")
    println(io, "|---|---|")
    for (label, path) in ("Configuration" => result.config_path,
                          "Metrics" => result.metrics_path,
                          "Figure" => result.figure_path,
                          "Summary" => result.summary_path)
        if path === nothing
            println(io, "| ", label, " | *not emitted* |")
        else
            println(io, "| ", label, " | [`", rel_path(root, path), "`](", blob_url(root, path), ") |")
        end
    end
    println(io)

    prov = result.config === nothing ? Pair{String,String}[] :
           filter(p -> is_provenance_key(first(p)), flatten_config(result.config))
    recorded_commit = nothing
    for (key, value) in prov
        if is_commit_key(key) && !isempty(strip(value))
            recorded_commit = value
            break
        end
    end

    if !isempty(prov)
        println(io, "Provenance fields recorded by the experiment script:")
        println(io)
        println(io, "| Field | Value |")
        println(io, "|---|---|")
        for (key, value) in prov
            println(io, "| `", cell(key), "` | `", cell(value), "` |")
        end
        println(io)
    end

    if recorded_commit === nothing
        commit = result_commit(root, result.dir)
        if commit === nothing
            println(io, "!!! warning \"Provenance incomplete\"")
            println(io, "    This run recorded no code-version field in `config.toml`, and the")
            println(io, "    artifacts are not committed yet, so the exact code version behind")
            println(io, "    these numbers cannot be identified. See the provenance policy above.")
        else
            println(io, "Code version: the experiment script did not record a commit field, so")
            println(io, "provenance falls back to the commit that last changed these artifacts, ",
                "[`", commit[1:min(end, 12)], "`](", REPO_URL, "/commit/", commit, ").")
        end
        println(io)
    end
end

function render_published(io::IO, result::ResultSet, entry::Union{Nothing,GalleryEntry},
                          root::AbstractString, assets_rel::AbstractString)
    title = entry === nothing ? replace(result.slug, '_' => ' ') |> titlecase : entry.title
    println(io, "### ", title)
    println(io)
    if entry === nothing
        println(io, "**Status** — published · not part of the planned experiment set")
        println(io)
        println(io, "Discovered at `", rel_path(root, result.dir), "/`. This slug carries no")
        println(io, "editorial question or hypothesis in the gallery, so its own generated")
        println(io, "artifacts speak for it.")
    else
        println(io, "**Status** — published · tracking issue [#", entry.issue, "](",
            REPO_URL, "/issues/", entry.issue, ")")
        println(io)
        println(io, "**Question.** ", entry.question)
        println(io)
        println(io, "**Hypothesis.** ", entry.hypothesis)
    end
    println(io)

    render_setup(io, result, root)
    render_artifact(io, result, title, root, assets_rel)
    render_result(io, result, root)
    render_metrics(io, result, root)

    println(io, "#### Reproduce")
    println(io)
    script = joinpath(root, "experiments", string(result.slug, ".jl"))
    println(io, "```bash")
    if entry !== nothing || isfile(script)
        println(io, "julia --project=experiments experiments/", result.slug, ".jl")
    else
        println(io, "julia --project=experiments experiments/run_all.jl")
    end
    println(io, "```")
    println(io)

    render_provenance(io, result, root)
end

function render_pending(io::IO, entry::GalleryEntry, root::AbstractString)
    println(io, "### ", entry.title)
    println(io)
    println(io, "**Status** — not yet published · tracking issue [#", entry.issue, "](",
        REPO_URL, "/issues/", entry.issue, ")")
    println(io)
    println(io, "**Question.** ", entry.question)
    println(io)
    println(io, "**Hypothesis.** ", entry.hypothesis)
    println(io)
    println(io, "**Planned primary artifact.** ", entry.artifact)
    println(io)
    println(io, "No artifacts exist under `experiments/results/", entry.slug, "/`, so this entry")
    println(io, "publishes no setup values, no figure, no metrics and no result. It will fill in")
    println(io, "automatically on the next docs build once the experiment has been run and its")
    println(io, "artifacts committed.")
    println(io)
    println(io, "**Reproduce (once the experiment lands).**")
    println(io)
    println(io, "```bash")
    println(io, "julia --project=experiments experiments/", entry.slug, ".jl")
    println(io, "```")
    println(io)
end

function render_header(io::IO, published::Vector{String}, pending::Vector{String},
                       results_dir::AbstractString, root::AbstractString)
    println(io, "# Experiment Gallery")
    println(io)
    println(io, "TemporalFocus is characterized with reproducible spike-native experiments covering")
    println(io, "recency, kernel behavior, distractor rejection, timing jitter, streaming focus, and")
    println(io, "τ/window trade-offs. This page is the human-facing view of those runs: question,")
    println(io, "setup, artifact, observed result, reproduction command, and a link back to the")
    println(io, "machine-readable evidence.")
    println(io)
    println(io, "!!! note \"This page is generated\"")
    println(io, "    `docs/gallery.jl` renders `docs/src/experiments.md` from the artifacts under")
    println(io, "    `", rel_path(root, results_dir), "/`, and `docs/make.jl` regenerates it on every")
    println(io, "    docs build. Do not edit the Markdown by hand — edit the generator, or re-run the")
    println(io, "    experiment.")
    println(io)

    println(io, "## What this gallery answers")
    println(io)
    println(io, "| Question a reviewer asks | Answered by |")
    println(io, "|---|---|")
    println(io, "| What does TemporalFocus do? | [Home](index.md) and the [API reference](api.md) |")
    println(io, "| How does recency weighting actually behave? | Temporal Lens |")
    println(io, "| What differentiates the three attention modes? | Three Regimes |")
    println(io, "| How sensitive is focus to noise and stale activity? | Focus Under Fire |")
    println(io, "| How much timing jitter can it absorb? | Jitter Test |")
    println(io, "| How does focus move over a streaming buffer? | Attention Spotlight |")
    println(io, "| How do τ and the hard window interact? | Memory Gate |")
    println(io)

    println(io, "## Evidence policy")
    println(io)
    println(io, "1. **Figures are generated, never redrawn.** Every image on this page is mirrored")
    println(io, "   from a `figure.png` written by an experiment script into")
    println(io, "   `docs/src/assets/experiments/<slug>/` at docs-build time. Those mirrors are")
    println(io, "   build output and are not tracked in git; the committed artifacts under")
    println(io, "   `experiments/results/<slug>/` are the source of truth.")
    println(io, "2. **Every quantitative claim is traceable.** Setup values are read from")
    println(io, "   `config.toml`, numbers come from `metrics.csv`, and the interpretation is the")
    println(io, "   experiment's own `summary.md` embedded verbatim. The generator adds no numbers")
    println(io, "   of its own.")
    println(io, "3. **Null and negative findings stay.** Summaries are quoted in full rather than")
    println(io, "   excerpted, so a result that fails to support its hypothesis is published as-is.")
    println(io, "4. **Absent results are shown as absent.** An experiment that has not run is listed")
    println(io, "   with its question and hypothesis and explicitly no results.")
    println(io, "5. **Provenance identifies the code version.** Experiment scripts should record a")
    println(io, "   commit field (`commit`, `git_commit`, `revision`, …) in `config.toml`, alongside")
    println(io, "   `generated_at`, `julia_version` and any RNG seeds. When no commit field is")
    println(io, "   recorded, the gallery falls back to the commit that last changed the result")
    println(io, "   directory and says so; when neither is available it publishes a warning instead")
    println(io, "   of an unverifiable claim.")
    println(io, "6. **Published results are committed.** To appear in the hosted docs, a result")
    println(io, "   directory must be committed to the repository. Keep figures small (≈500 KB or")
    println(io, "   less); large or intermediate data stays out of git.")
    println(io)

    println(io, "## Reproducing this gallery from a fresh clone")
    println(io)
    println(io, "```bash")
    println(io, "git clone ", REPO_URL, ".git")
    println(io, "cd TemporalFocus.jl")
    println(io)
    println(io, "# 1. set up the isolated experiment environment (no root dependencies are added)")
    println(io, "julia --project=experiments -e 'using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()'")
    println(io)
    println(io, "# 2. regenerate every experiment artifact under experiments/results/")
    println(io, "julia --project=experiments experiments/run_all.jl")
    println(io)
    println(io, "# 3. rebuild the docs; the gallery is re-rendered from those artifacts")
    println(io, "julia --project=docs -e 'using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()'")
    println(io, "julia --project=docs docs/make.jl")
    println(io, "```")
    println(io)
    println(io, "Individual experiments run standalone with")
    println(io, "`julia --project=experiments experiments/<slug>.jl`. The experiment environment")
    println(io, "carries its own visualization dependencies; the root package stays dependency-free.")
    println(io)

    println(io, "## Index")
    println(io)
    println(io, "| Experiment | Tracking issue | Status |")
    println(io, "|---|---|---|")
    for entry in ENTRIES
        status = entry.slug in published ? "published" : "not yet published"
        println(io, "| ", entry.title, " | [#", entry.issue, "](", REPO_URL, "/issues/",
            entry.issue, ") | ", status, " |")
    end
    for slug in published
        any(e -> e.slug == slug, ENTRIES) && continue
        println(io, "| `", slug, "` | — | published |")
    end
    println(io)
    if isempty(published)
        println(io, "!!! warning \"No results published yet\"")
        println(io, "    No result directories were found under `", rel_path(root, results_dir), "/`,")
        println(io, "    so every entry below is pending. The gallery deliberately shows an empty")
        println(io, "    state rather than placeholder numbers: nothing here is claimed until an")
        println(io, "    experiment has actually produced it.")
        println(io)
    elseif !isempty(pending)
        done = length(ENTRIES) - length(pending)
        println(io, "!!! note \"Partial gallery\"")
        println(io, "    ", done, " of ", length(ENTRIES), " planned experiments have published")
        println(io, "    artifacts. Pending entries below show their question and hypothesis but")
        println(io, "    no results.")
        println(io)
    end
end

# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

"""
    build_gallery(; repo_root, results_dir, out_path, assets_dir, assets_rel)

Render the Experiment Gallery page from whatever experiment artifacts exist.

Returns a named tuple `(; out_path, published, pending, copied)` where `published`
and `pending` are slug lists and `copied` are the figure files mirrored into the
docs asset directory.

Safe to call with no experiments present: the result is a valid page in which
every expected experiment is listed as pending.
"""
function build_gallery(;
    repo_root::AbstractString = default_repo_root(),
    results_dir::AbstractString = joinpath(repo_root, "experiments", "results"),
    out_path::AbstractString = joinpath(repo_root, "docs", "src", "experiments.md"),
    assets_dir::AbstractString = joinpath(repo_root, "docs", "src", "assets", "experiments"),
    assets_rel::AbstractString = "assets/experiments",
)
    slugs = discovered_slugs(results_dir)
    results = Dict{String,ResultSet}()
    published = String[]
    for slug in slugs
        result = collect_result(results_dir, slug)
        has_artifacts(result) || continue
        results[slug] = result
        push!(published, slug)
    end
    pending = [e.slug for e in ENTRIES if !(e.slug in published)]

    # Mirror generated figures into the docs asset tree (build output, untracked).
    copied = String[]
    if isdir(assets_dir)
        rm(assets_dir; recursive = true)
    end
    for slug in published
        result = results[slug]
        sources = String[]
        result.figure_path === nothing || push!(sources, result.figure_path)
        isempty(sources) && continue
        target_dir = joinpath(assets_dir, slug)
        mkpath(target_dir)
        for src in sources
            dest = joinpath(target_dir, basename(src))
            cp(src, dest; force = true)
            push!(copied, dest)
        end
    end

    io = IOBuffer()
    render_header(io, published, pending, results_dir, repo_root)

    println(io, "## Experiments")
    println(io)
    for entry in ENTRIES
        if entry.slug in published
            render_published(io, results[entry.slug], entry, repo_root, assets_rel)
        else
            render_pending(io, entry, repo_root)
        end
    end

    extra = [slug for slug in published if !any(e -> e.slug == slug, ENTRIES)]
    if !isempty(extra)
        println(io, "## Additional published results")
        println(io)
        println(io, "Result directories that are not part of the planned experiment set — for")
        println(io, "example harness smoke tests — rendered from their own artifacts.")
        println(io)
        for slug in extra
            render_published(io, results[slug], nothing, repo_root, assets_rel)
        end
    end

    println(io, "## Downstream boundary")
    println(io)
    println(io, "This gallery stays spike-native. Domain-specific experiments — trading, market")
    println(io, "microstructure, or any other applied setting — belong in a downstream workspace")
    println(io, "that depends on TemporalFocus, not in this repository. See the")
    println(io, "[scope section](index.md) for the boundary this package enforces.")

    mkpath(dirname(out_path))
    write(out_path, String(take!(io)))
    return (; out_path, published, pending, copied)
end

end # module
