# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Three Regimes — same spikes, three notions of focus.

Runs `spike_attention_discrete`, `spike_attention_temporal`, and
`spike_attention_continuous` over **one** deterministic spike scene and an
identity readout, so the readout vector is literally the per-neuron attention
mass.

Research question: given identical input, what does each regime preserve,
decay, and reject?

Run from the repository root:

    julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
    julia --project=experiments experiments/three_regimes.jl

Artifacts land in `experiments/results/three_regimes/`.
"""

using CairoMakie
using ExperimentUtils
using Printf
using TemporalFocus

const SLUG = "three_regimes"

# --- Configuration --------------------------------------------------------

const N_NEURONS = 5
const τ = 0.20f0            # decay time constant, shared by temporal + continuous
const WINDOW = 0.25f0       # continuous-kernel temporal window (both buffers)
const SPIKE_VALUE = 1.0f0   # every spike has unit amplitude: mass == pair count
const T_REF = 0.980f0       # reference source time for the coincident neurons

# Neuron roles, indexed by neuron_id.
const NEURON_ROLES = (
    "aligned-recent",
    "moderate-lag",
    "stale-distractor",
    "unrelated",
    "window-boundary",
)

# --- The one shared scene -------------------------------------------------
#
# Each entry is a labeled source/context pairing on a single neuron.
# `t_context === nothing` means the source spike has no partner at all.
# The spike trains and buffers below are derived from this table, so the
# narrative labels and the data handed to the kernels cannot drift apart.

const PAIRS = (
    (label = "aligned_recent",   neuron_id = 1, t_source = T_REF,   t_context = 0.970f0,
     note = "strongly aligned recent pair"),
    (label = "moderate_lag",     neuron_id = 2, t_source = T_REF,   t_context = 0.800f0,
     note = "same neuron, moderate temporal lag"),
    (label = "stale_distractor", neuron_id = 3, t_source = T_REF,   t_context = 0.100f0,
     note = "same neuron, stale — far outside the window"),
    (label = "unrelated_neuron", neuron_id = 4, t_source = 0.520f0, t_context = nothing,
     note = "source spike with no context partner on this neuron"),
    (label = "boundary_inside",  neuron_id = 5, t_source = T_REF,   t_context = 0.740f0,
     note = "just inside the continuous window"),
    (label = "boundary_outside", neuron_id = 5, t_source = T_REF,   t_context = 0.720f0,
     note = "just outside the continuous window"),
)

const REGIMES = ("discrete", "temporal", "continuous")

"""
    _identity_readout(n) -> Matrix{Float32}

`n×n` identity readout. With this readout the kernel output *is* the
per-neuron attention vector, so the comparison stays transparent.
"""
_identity_readout(n::Int) = Float32[i == j ? 1.0f0 : 0.0f0 for i in 1:n, j in 1:n]

"""
    _build_scene() -> (source_train, context_train)

Derive the shared spike trains from [`PAIRS`](@ref). Source spikes are
de-duplicated so a neuron with two labeled pairs still emits one source spike.
"""
function _build_scene()
    source_events = SpikeEvent[]
    context_events = SpikeEvent[]
    seen_sources = Set{Tuple{Int,Float32}}()
    for p in PAIRS
        key = (p.neuron_id, p.t_source)
        if !(key in seen_sources)
            push!(seen_sources, key)
            push!(source_events, SpikeEvent(p.neuron_id, p.t_source, SPIKE_VALUE))
        end
        p.t_context === nothing && continue
        push!(context_events, SpikeEvent(p.neuron_id, p.t_context, SPIKE_VALUE))
    end
    return SpikeTrain(source_events), SpikeTrain(context_events)
end

"""
    _pair_delta(p) -> Float32

Signed `t_source - t_context` for a labeled pair, or `NaN32` when the pair has
no context partner.
"""
_pair_delta(p) = p.t_context === nothing ? NaN32 : p.t_source - p.t_context

"""
    _source_time(neuron_id) -> Float32

The source spike time on `neuron_id`. Every labeled pair on a neuron shares one
source spike, so this is well defined by construction.
"""
function _source_time(neuron_id::Int)
    for p in PAIRS
        p.neuron_id == neuron_id && return p.t_source
    end
    error("no source spike defined for neuron $(neuron_id)")
end

"""
    _pair_outcome(regime, p) -> (status, weight)

What a single labeled pair contributes under `regime`. This mirrors the kernel
bodies in `src/`; the run cross-checks the totals against the kernels
themselves before writing any artifact.
"""
function _pair_outcome(regime::AbstractString, p)
    p.t_context === nothing && return ("no_context_match", 0.0f0)
    dt = _pair_delta(p)
    regime == "discrete" && return ("contributed", SPIKE_VALUE * SPIKE_VALUE)
    w = SPIKE_VALUE * SPIKE_VALUE * temporal_weight(dt, τ)
    regime == "temporal" && return ("contributed", w)
    return abs(dt) <= WINDOW ? ("contributed", w) : ("rejected_by_window", 0.0f0)
end

"""
    _expected_mass(regime) -> Vector{Float32}

Per-neuron mass reconstructed from the labeled pair table.
"""
function _expected_mass(regime::AbstractString)
    mass = zeros(Float32, N_NEURONS)
    for p in PAIRS
        _, w = _pair_outcome(regime, p)
        mass[p.neuron_id] += w
    end
    return mass
end

_share(mass::AbstractVector{Float32}) =
    (total = sum(mass); total > 0f0 ? mass ./ total : zero(mass))

"""
    _temporal_underflow_ratio() -> Float32

Smallest whole `|Δt|/τ` at which `temporal_weight` underflows to exactly `0.0f0`.
Measured, not assumed, so the summary never claims a stronger guarantee than
`Float32` actually provides.
"""
function _temporal_underflow_ratio()
    ratio = 0.0f0
    while temporal_weight(ratio, 1.0f0) > 0.0f0
        ratio += 1.0f0
    end
    return ratio
end

# --- Run the three kernels over the shared scene --------------------------

source_train, context_train = _build_scene()
readout = _identity_readout(N_NEURONS)

# Same events, wrapped in buffers so the continuous kernel can apply its
# window. No `prune!` is applied: temporal and continuous therefore differ by
# exactly one thing — the kernel's own `|Δt| <= window` test.
source_buffer = TemporalBuffer(WINDOW, source_train.events)
context_buffer = TemporalBuffer(WINDOW, context_train.events)

const MASS = Dict(
    "discrete" => spike_attention_discrete(source_train, context_train, readout),
    "temporal" => spike_attention_temporal(source_train, context_train, readout; τ = τ),
    "continuous" =>
        spike_attention_continuous(source_buffer, context_buffer, readout; τ = τ),
)

const SHARE = Dict(regime => _share(MASS[regime]) for regime in REGIMES)
const WINNER = Dict(regime => argmax(MASS[regime]) for regime in REGIMES)

# Self-check: the labeled pair table must reproduce the kernel output exactly.
for regime in REGIMES
    got, want = MASS[regime], _expected_mass(regime)
    isapprox(got, want; rtol = 1.0f-5, atol = 1.0f-7) || error(
        "pair bookkeeping disagrees with the $(regime) kernel: got $(got), want $(want)",
    )
end

# --- Console report -------------------------------------------------------

println("=== Three Regimes — same spikes, three notions of focus ===")
println("τ = ", τ, ", continuous window = ", WINDOW, ", neurons = ", N_NEURONS)
println("source events  = ", length(source_train.events))
println("context events = ", length(context_train.events))
println()
println("Per-neuron attention mass (identity readout):")
@printf("%-4s %-18s %12s %12s %12s\n", "id", "role", "discrete", "temporal", "continuous")
for i in 1:N_NEURONS
    @printf(
        "%-4d %-18s %12.6f %12.6f %12.6f\n",
        i, NEURON_ROLES[i], MASS["discrete"][i], MASS["temporal"][i], MASS["continuous"][i]
    )
end
println()
for regime in REGIMES
    @printf("winner(%s) = neuron %d (share %.3f)\n",
            regime, WINNER[regime], SHARE[regime][WINNER[regime]])
end
println()
println("Pair contributions:")
@printf("%-18s %-4s %8s %12s %12s %12s\n",
        "pair", "id", "dt", "discrete", "temporal", "continuous")
for p in PAIRS
    cells = map(REGIMES) do regime
        status, w = _pair_outcome(regime, p)
        status == "contributed" ? @sprintf("%.6f", w) :
        status == "rejected_by_window" ? "rejected" : "no match"
    end
    dt = _pair_delta(p)
    @printf("%-18s %-4d %8s %12s %12s %12s\n",
            p.label, p.neuron_id, isnan(dt) ? "-" : @sprintf("%.3f", dt),
            cells[1], cells[2], cells[3])
end
println()

# --- Artifacts: config ----------------------------------------------------

config = Dict{String,Any}(
    "slug" => SLUG,
    "description" => "discrete vs temporal vs continuous attention on one shared spike scene",
    "n_neurons" => N_NEURONS,
    "tau" => τ,
    "tau_note" => "shared by spike_attention_temporal and spike_attention_continuous",
    "continuous_window" => WINDOW,
    "continuous_window_note" =>
        "TemporalBuffer window on both buffers; kernel uses min(source, context)",
    "prune_applied" => false,
    "spike_value" => SPIKE_VALUE,
    "reference_source_time" => T_REF,
    "readout" => "identity",
    "randomness" => "none — the scene is fully specified, no seed required",
    "regimes" => collect(REGIMES),
    "neuron_roles" => collect(NEURON_ROLES),
    "pair_labels" => [p.label for p in PAIRS],
    "pair_notes" => [p.note for p in PAIRS],
    "pair_neuron_ids" => [p.neuron_id for p in PAIRS],
    "pair_source_times" => [p.t_source for p in PAIRS],
    "pair_context_times" => [p.t_context === nothing ? NaN32 : p.t_context for p in PAIRS],
    "pair_deltas" => [_pair_delta(p) for p in PAIRS],
)
config_path = write_config(SLUG, config)

# --- Artifacts: metrics ---------------------------------------------------
#
# Long format: one row per (regime, labeled pair). Neuron-level aggregates
# (`neuron_mass`, `mass_share`, `winner_neuron`) are denormalized onto each
# row so the file is self-contained. `dt = NaN` marks a source spike with no
# context partner.
#
# Every temporal/spike quantity stays `Float32` all the way to the CSV — no
# widening to `Float64` and no lossy pre-rounding. The landed harness writes
# each float in its shortest round-tripping decimal form, so the file carries
# the exact computed `Float32` value.

metric_rows = NamedTuple[]
for regime in REGIMES, p in PAIRS
    status, w = _pair_outcome(regime, p)
    dt = _pair_delta(p)
    push!(metric_rows, (
        regime = regime,
        pair_label = p.label,
        neuron_id = p.neuron_id,
        neuron_role = NEURON_ROLES[p.neuron_id],
        dt = dt,
        within_window = !isnan(dt) && abs(dt) <= WINDOW,
        pair_weight = w,
        status = status,
        neuron_mass = MASS[regime][p.neuron_id],
        mass_share = SHARE[regime][p.neuron_id],
        winner_neuron = WINNER[regime],
    ))
end
metrics_path = write_metrics(SLUG, metric_rows)

# --- Artifacts: figure ----------------------------------------------------

const C_SOURCE = RGBf(0.13, 0.16, 0.24)
const C_KEPT = RGBf(0.11, 0.51, 0.36)
const C_DROPPED = RGBf(0.78, 0.25, 0.22)
const C_NONE = RGBf(0.55, 0.57, 0.60)
const C_REGIME = (RGBf(0.35, 0.42, 0.58), RGBf(0.86, 0.60, 0.20), RGBf(0.11, 0.51, 0.36))

"""
    _pair_color(p) -> RGBf

Color a labeled pair by its fate under the continuous kernel: kept, rejected
by the window, or never matched at all.
"""
function _pair_color(p)
    status, _ = _pair_outcome("continuous", p)
    status == "contributed" && return C_KEPT
    status == "rejected_by_window" && return C_DROPPED
    return C_NONE
end

"""
    _pair_row_offsets() -> Vector{Tuple{Float64,Float64}}

Vertical `(line, label)` nudges for the raster. Neurons carrying more than one
labeled pair (the window-boundary neuron) get their pairs fanned out from the
shared source spike so neither the lines nor the Δt annotations overdraw.
"""
function _pair_row_offsets()
    counts = Dict{Int,Int}()
    for p in PAIRS
        counts[p.neuron_id] = get(counts, p.neuron_id, 0) + 1
    end
    seen = Dict{Int,Int}()
    return map(PAIRS) do p
        k = get(seen, p.neuron_id, 0) + 1
        seen[p.neuron_id] = k
        counts[p.neuron_id] == 1 && return (0.0, -0.17)
        isodd(k) ? (-0.12, -0.28) : (0.12, 0.30)
    end
end

function _build_figure()
    fig = Figure(size = (1180, 1060), backgroundcolor = :white)

    Label(fig[0, 1:3], "Same spikes, three notions of focus";
          fontsize = 26, font = :bold, padding = (0, 0, 4, 0))
    Label(fig[1, 1:3],
          "One deterministic scene · identity readout · τ = $(τ) · continuous window = ±$(WINDOW)";
          fontsize = 14, color = :gray35, padding = (0, 0, 0, 8))

    # --- Panel A: the shared scene ---
    ax = Axis(fig[2, 1:3];
        title = "A · the shared spike scene (identical input to all three kernels)",
        subtitle = "shaded band = the ±$(WINDOW) continuous window around each row's own source spike " *
                   "(the kernel tests |Δt| per pair, not one absolute-time interval)",
        titlealign = :left, subtitlesize = 12, subtitlecolor = :gray35,
        xlabel = "time", ylabel = "neuron",
        yticks = (1:N_NEURONS, ["n$(i) · $(NEURON_ROLES[i])" for i in 1:N_NEURONS]),
        yreversed = true)

    # The continuous kernel's window is per pair (`|Δt| <= window`), not a
    # global slice of absolute time, so each row gets its own admissible band
    # centred on that row's source spike.
    for i in 1:N_NEURONS
        t_src = Float64(_source_time(i))
        poly!(ax, Rect2f(t_src - WINDOW, i - 0.42, 2 * WINDOW, 0.84);
              color = (C_KEPT, 0.10), strokecolor = (C_KEPT, 0.55),
              strokewidth = 1.2, linestyle = :dash)
    end

    for (p, (line_dy, label_dy)) in zip(PAIRS, _pair_row_offsets())
        y = Float64(p.neuron_id)
        col = _pair_color(p)
        if p.t_context !== nothing
            lines!(ax, [Float64(p.t_context), Float64(p.t_source)], [y + line_dy, y];
                   color = (col, 0.85), linewidth = 2.5)
            scatter!(ax, [Float64(p.t_context)], [y + line_dy];
                     color = col, marker = :circle, markersize = 13)
            text!(ax, (Float64(p.t_context) + Float64(p.t_source)) / 2, y + label_dy;
                  text = @sprintf("Δt = %.2f", _pair_delta(p)),
                  align = (:center, label_dy < 0 ? :bottom : :top),
                  fontsize = 11, color = col)
        else
            text!(ax, Float64(p.t_source) + 0.02, y; text = "no context partner",
                  align = (:left, :center), fontsize = 11, color = C_NONE)
        end
        scatter!(ax, [Float64(p.t_source)], [y];
                 color = C_SOURCE, marker = :rect, markersize = 11)
    end

    xlims!(ax, 0.0, 1.30)
    ylims!(ax, N_NEURONS + 0.6, 0.4)

    Legend(fig[3, 1:3],
        [MarkerElement(color = C_SOURCE, marker = :rect, markersize = 11),
         MarkerElement(color = C_KEPT, marker = :circle, markersize = 13),
         MarkerElement(color = C_DROPPED, marker = :circle, markersize = 13),
         MarkerElement(color = C_NONE, marker = :circle, markersize = 13),
         PolyElement(color = (C_KEPT, 0.14))],
        ["source spike", "context spike: inside window", "context spike: outside window",
         "no matching partner", "continuous window (per source spike)"];
        framevisible = false, labelsize = 12, patchsize = (14, 14),
        orientation = :horizontal, tellheight = true, tellwidth = false,
        padding = (0, 0, 0, 0))

    # --- Panels B–D: per-neuron attention mass, one panel per regime ---
    subtitles = (
        "ignores timing — every match counts fully",
        "keeps every match, decays the stale ones",
        "decays *and* rejects anything outside ±$(WINDOW)",
    )
    for (col, regime) in enumerate(REGIMES)
        mass = MASS[regime]
        axb = Axis(fig[4, col];
            title = "$("BCD"[col]) · $(regime)\n$(subtitles[col])",
            titlealign = :left, titlesize = 13,
            xlabel = "attention mass",
            yticks = (1:N_NEURONS, ["n$(i)" for i in 1:N_NEURONS]),
            yreversed = true)
        barplot!(axb, 1:N_NEURONS, Float64.(mass);
                 direction = :x, color = C_REGIME[col], width = 0.62)
        top = max(maximum(mass), 1.0f-3)
        for i in 1:N_NEURONS
            text!(axb, Float64(mass[i]) + 0.05 * top, Float64(i);
                  text = @sprintf("%.3f", mass[i]), align = (:left, :center),
                  fontsize = 11, color = :gray25)
        end
        text!(axb, 0.98 * top, Float64(N_NEURONS) + 0.55;
              text = "winner: n$(WINNER[regime])", align = (:right, :center),
              fontsize = 12, font = :bold, color = C_REGIME[col])
        xlims!(axb, 0.0, 1.32 * top)
        ylims!(axb, N_NEURONS + 0.9, 0.4)
        col == 1 || hideydecorations!(axb; grid = false)
    end

    # --- Panel E: normalized share, the only directly comparable view ---
    axe = Axis(fig[5, 1:3];
        title = "E · normalized share of total attention mass (comparable across regimes)",
        titlealign = :left, xlabel = "neuron", ylabel = "share of total mass",
        xticks = (1:N_NEURONS, ["n$(i)\n$(NEURON_ROLES[i])" for i in 1:N_NEURONS]),
        xticklabelsize = 11)
    for (gi, regime) in enumerate(REGIMES)
        barplot!(axe, 1:N_NEURONS, Float64.(SHARE[regime]);
                 dodge = fill(gi, N_NEURONS), n_dodge = length(REGIMES),
                 color = C_REGIME[gi], label = regime)
    end
    ylims!(axe, 0.0, max(maximum(maximum(SHARE[r]) for r in REGIMES) * 1.35, 0.1))
    axislegend(axe; position = :lt, framevisible = false, labelsize = 12, orientation = :horizontal)

    Label(fig[6, 1:3],
        "Neuron 3 (stale) survives discrete attention intact, is crushed to $(@sprintf("%.3f", MASS["temporal"][3])) " *
        "by temporal decay, and is removed outright by the continuous window. " *
        "Neuron 5 carries two context spikes straddling the window edge: discrete and temporal keep both, " *
        "continuous keeps only the Δt = $(@sprintf("%.2f", _pair_delta(PAIRS[5]))) one.";
        fontsize = 12, color = :gray30, justification = :left, lineheight = 1.2,
        padding = (0, 0, 6, 0), word_wrap = true, tellwidth = false)

    rowsize!(fig.layout, 2, Relative(0.32))
    rowsize!(fig.layout, 4, Relative(0.26))
    rowsize!(fig.layout, 5, Relative(0.22))
    return fig
end

CairoMakie.activate!(type = "png")
fig_path = figure_path(SLUG)
save(fig_path, _build_figure(); px_per_unit = 1.5)

# --- Artifacts: summary ---------------------------------------------------

function _contribution_cell(regime, p)
    status, w = _pair_outcome(regime, p)
    status == "contributed" && return @sprintf("%.3f", w)
    status == "rejected_by_window" && return "— *(window)*"
    return "— *(no match)*"
end

function _summary_markdown()
    io = IOBuffer()
    println(io, "# Three Regimes — same spikes, three notions of focus")
    println(io)
    println(io, "**Slug:** `", SLUG, "` · **Run:** `julia --project=experiments experiments/three_regimes.jl`")
    println(io)
    println(io, "## Research question")
    println(io)
    println(io, "Given exactly the same synthetic spike scene, how do `spike_attention_discrete`,")
    println(io, "`spike_attention_temporal`, and `spike_attention_continuous` differ in what they")
    println(io, "preserve, decay, and reject?")
    println(io)
    println(io, "## Hypothesis")
    println(io)
    println(io, "1. **Discrete** ignores timing and accumulates every matching-neuron interaction regardless of age.")
    println(io, "2. **Temporal** keeps every matching-neuron interaction but exponentially downweights stale ones.")
    println(io, "3. **Continuous** applies the same decay *and* an explicit temporal window, rejecting pairs outside it.")
    println(io)
    println(io, "## Setup")
    println(io)
    println(io, "- One deterministic scene of ", length(source_train.events), " source and ",
            length(context_train.events), " context spikes across ", N_NEURONS, " neurons; no randomness, no seed needed.")
    println(io, "- Identity readout (", N_NEURONS, "×", N_NEURONS, "), so the kernel output *is* the per-neuron attention vector.")
    println(io, "- `τ = ", τ, "` for both temporal and continuous.")
    println(io, "- `TemporalBuffer` window `= ", WINDOW, "` on both buffers (the kernel uses `min(source, context)`).")
    println(io, "- `prune!` is deliberately **not** applied, so temporal and continuous differ by exactly one thing: the kernel's `|Δt| <= window` test.")
    println(io)
    println(io, "## Which events contributed, per regime")
    println(io)
    println(io, "| pair | neuron | Δt | discrete | temporal | continuous |")
    println(io, "| --- | --- | --- | --- | --- | --- |")
    for p in PAIRS
        dt = _pair_delta(p)
        println(io, "| `", p.label, "` | n", p.neuron_id, " · ", NEURON_ROLES[p.neuron_id],
                " | ", isnan(dt) ? "—" : @sprintf("%.3f", dt),
                " | ", _contribution_cell("discrete", p),
                " | ", _contribution_cell("temporal", p),
                " | ", _contribution_cell("continuous", p), " |")
    end
    println(io)
    println(io, "## Per-neuron attention mass (share of total)")
    println(io)
    println(io, "| neuron | role | discrete | temporal | continuous |")
    println(io, "| --- | --- | --- | --- | --- |")
    for i in 1:N_NEURONS
        cells = map(REGIMES) do regime
            @sprintf("%.4f (%.1f%%)", MASS[regime][i], 100 * SHARE[regime][i])
        end
        println(io, "| n", i, " | ", NEURON_ROLES[i], " | ", cells[1], " | ", cells[2],
                " | ", cells[3], " |")
    end
    println(io)
    println(io, "| | discrete | temporal | continuous |")
    println(io, "| --- | --- | --- | --- |")
    println(io, "| **winner neuron** | n", WINNER["discrete"], " | n", WINNER["temporal"],
            " | n", WINNER["continuous"], " |")
    println(io, "| **total mass** | ", @sprintf("%.4f", sum(MASS["discrete"])), " | ",
            @sprintf("%.4f", sum(MASS["temporal"])), " | ",
            @sprintf("%.4f", sum(MASS["continuous"])), " |")
    println(io)
    println(io, "## Observation")
    println(io)
    println(io, "- **Discrete is timing-blind.** The stale pair (n3, Δt = ",
            @sprintf("%.2f", _pair_delta(PAIRS[3])), ") contributes ",
            @sprintf("%.3f", MASS["discrete"][3]), " — exactly as much as the freshly aligned pair on n1. ",
            "It cannot tell a ", @sprintf("%.2f", abs(_pair_delta(PAIRS[3]))),
            "-old coincidence from a ", @sprintf("%.2f", abs(_pair_delta(PAIRS[1]))), "-old one.")
    println(io, "- **Temporal decays but has no cutoff.** n3 drops to ",
            @sprintf("%.4f", MASS["temporal"][3]), " (",
            @sprintf("%.1f%%", 100 * SHARE["temporal"][3]),
            " of total mass) and is still counted: `spike_attention_temporal` applies no window, ",
            "so no pair is excluded *by rule*. Note this is not a guarantee of a strictly positive ",
            "weight — `temporal_weight` is `exp(-|Δt|/τ)` in `Float32`, which underflows to exactly ",
            "`0.0` once `|Δt|/τ` reaches ", @sprintf("%.0f", _temporal_underflow_ratio()),
            " (measured on this machine). That is a numeric floor, not a semantic cutoff.")
    println(io, "- **Continuous rejects.** n3 falls to exactly ",
            @sprintf("%.4f", MASS["continuous"][3]),
            " because |Δt| = ", @sprintf("%.2f", abs(_pair_delta(PAIRS[3]))),
            " exceeds the window of ", WINDOW, ".")
    println(io, "- **The boundary is sharp.** n5 carries two context spikes at Δt = ",
            @sprintf("%.2f", _pair_delta(PAIRS[5])), " and Δt = ",
            @sprintf("%.2f", _pair_delta(PAIRS[6])), ". Discrete counts both (mass ",
            @sprintf("%.3f", MASS["discrete"][5]), "), temporal counts both with decay (",
            @sprintf("%.4f", MASS["temporal"][5]), "), continuous keeps only the inner one (",
            @sprintf("%.4f", MASS["continuous"][5]), ").")
    println(io, "- **Focus itself moves.** The winner is n", WINNER["discrete"],
            " under discrete (pair *count* wins), but n", WINNER["temporal"],
            " under temporal and continuous (*recency* wins).")
    println(io, "- **Unrelated stays silent.** n4 spikes but has no context partner, so it holds ",
            @sprintf("%.4f", MASS["discrete"][4]), " mass in every regime.")
    println(io)
    println(io, "## Verdict")
    println(io)
    println(io, "**Supported.** All three predicted behaviors are reproduced on identical input.")
    println(io, "The regimes form a strict refinement chain on this scene: discrete ⊇ temporal (same")
    println(io, "support, reweighted) ⊇ continuous (support truncated at ±", WINDOW, ").")
    println(io, "Choose *discrete* when only coincidence identity matters, *temporal* when old")
    println(io, "evidence should fade continuously with no explicit cutoff, and *continuous* when")
    println(io, "old evidence must be provably excluded.")
    println(io)
    println(io, "## Artifacts")
    println(io)
    println(io, "- `config.toml` — every τ / window / timing value used by this run")
    println(io, "- `metrics.csv` — one row per (regime, labeled pair) with pair weight, status, neuron mass, share, winner")
    println(io, "- `figure.png` — the scene plus all three regime outputs")
    return String(take!(io))
end

summary_path = write_summary(SLUG, _summary_markdown())

# --- Done -----------------------------------------------------------------

println("Wrote artifacts:")
for path in (config_path, metrics_path, fig_path, summary_path)
    println("  ", relpath(path, repo_root()))
end
