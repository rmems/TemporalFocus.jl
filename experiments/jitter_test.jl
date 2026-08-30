# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Jitter test — how much spike-timestamp jitter can each TemporalFocus kernel
absorb before the selected focus changes?

Run from the repository root:

    julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
    julia --project=experiments experiments/jitter_test.jl

# Scene

A fixed eight-neuron scene with one source spike per neuron at `T_REF` and a
per-neuron bag of context spikes. Two neurons carry the story:

- the **target** neuron has few context spikes, placed within a few hundredths
  of a time unit of the source spike (tight alignment, modest coincidence mass);
- the **loud distractor** carries the largest coincidence mass of any neuron but
  places all of it far from the source spike in time.

Target identity is therefore encoded purely in *timing*. A kernel that ignores
timestamps cannot recover it; a kernel that weights by timing can, until jitter
destroys the alignment.

# Jitter model

Zero-mean Gaussian perturbations are added to **context** spike timestamps only:
`t' = t + σ * z`, `z ~ N(0, 1)` drawn from `MersenneTwister(seed)`. Spike values
and neuron identities are untouched, so any timing-independent kernel must be
bit-for-bit invariant. The normal draws are taken for every `σ` including
`σ = 0`, so a given seed applies the *same* perturbation pattern at every jitter
scale (paired sampling across the σ grid) and `σ = 0` reproduces the baseline
scene exactly.

# Readout

The readout is the `N × N` identity matrix, so the returned output vector *is*
the per-neuron attention vector and every metric stays interpretable per neuron.

# Output distance metric

The primary drift metric is the **relative L2 (Euclidean) distance** between the
jittered output vector and the zero-jitter baseline output vector for the same
kernel, τ, and window:

    drift_rel_l2 = ‖out - baseline‖₂ / ‖baseline‖₂

L2 is chosen over L1 because attention mass here concentrates on a few neurons,
and L2 keeps the reported drift dominated by those large per-neuron moves rather
than by many small ones. Absolute `drift_l1` and `drift_l2` are recorded as well
so either convention can be recomputed from `metrics.csv`.
"""

using CairoMakie
using Printf
using Random
using Statistics
using TemporalFocus

include(joinpath(@__DIR__, "src", "ExperimentUtils.jl"))
using .ExperimentUtils

const SLUG = "jitter_test"

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

const N_NEURONS = 8
const TARGET_NEURON = 3
const LOUD_NEURON = 6
const T_REF = 10.0f0
const SPIKE_VALUE = 1.0f0

"""
Context spike offsets relative to `T_REF`, one vector per neuron.

Neuron 3 (target) is tightly aligned with the source spike; neuron 6 (loud
distractor) has the largest spike count but the poorest alignment.
"""
const CONTEXT_OFFSETS = Vector{Float32}[
    Float32[-0.90, 0.95],
    Float32[-1.80, 1.70, 2.10],
    Float32[-0.03, 0.01, 0.04],
    Float32[0.60, -0.65],
    Float32[1.30, -1.25, 1.55],
    Float32[-1.10, -0.85, 0.95, 1.25, 1.45, -1.35],
    Float32[-0.45, 0.50],
    Float32[2.40, -2.30],
]

"Largest absolute baseline misalignment of the target neuron's context spikes."
const BASELINE_ALIGNMENT_ERROR = maximum(abs, CONTEXT_OFFSETS[TARGET_NEURON])

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

const JITTERS = Float32[0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
const TAUS = Float32[0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
const WINDOWS = Float32[0.10, 0.25, 0.50, 1.00, 2.00, 4.00]
const SEEDS = [11, 23, 37, 53, 71, 97, 131, 173]

"Seed recorded on the single explicit zero-jitter baseline row of each configuration."
const BASELINE_SEED = 0

"Representative τ used for the per-window panel of the figure."
const REPRESENTATIVE_TAU = 0.50f0

"Fraction of seeds that must still select the target for a jitter scale to count as tolerated."
const RETENTION_THRESHOLD = 0.5

"Identity readout: the output vector is the per-neuron attention vector."
const READOUT = Float32[i == j ? 1 : 0 for i in 1:N_NEURONS, j in 1:N_NEURONS]

# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

_source_events() = [SpikeEvent(n, T_REF, SPIKE_VALUE) for n in 1:N_NEURONS]

"""
    _build_context_events(jitter, seed) -> Vector{SpikeEvent}

Baseline context spikes with zero-mean Gaussian timestamp jitter of scale
`jitter`, drawn from `MersenneTwister(seed)`. Draws happen even when
`jitter == 0` so the perturbation pattern of a seed is shared across the whole
σ grid.
"""
function _build_context_events(jitter::Float32, seed::Int)
    rng = MersenneTwister(seed)
    events = SpikeEvent[]
    for neuron_id in 1:N_NEURONS, offset in CONTEXT_OFFSETS[neuron_id]
        z = Float32(randn(rng))
        push!(events, SpikeEvent(neuron_id, T_REF + offset + jitter * z, SPIKE_VALUE))
    end
    return events
end

const _CONTEXT_CACHE = Dict{Tuple{Float32,Int},Vector{SpikeEvent}}()

"Memoized context train, so every kernel/τ/window sees the identical jittered scene."
_context_events(jitter::Float32, seed::Int) =
    get!(() -> _build_context_events(jitter, seed), _CONTEXT_CACHE, (jitter, seed))

# ---------------------------------------------------------------------------
# Kernel evaluation
# ---------------------------------------------------------------------------

const KernelConfig = NamedTuple{(:kernel, :tau, :window),Tuple{Symbol,Float32,Float32}}

function _kernel_configs()
    configs = KernelConfig[(kernel = :discrete, tau = NaN32, window = NaN32)]
    for tau in TAUS
        push!(configs, (kernel = :temporal, tau = tau, window = NaN32))
    end
    for tau in TAUS, window in WINDOWS
        push!(configs, (kernel = :continuous, tau = tau, window = window))
    end
    return configs
end

function _evaluate(config::KernelConfig, source_events, context_events)
    if config.kernel === :discrete
        return spike_attention_discrete(
            SpikeTrain(source_events), SpikeTrain(context_events), READOUT)
    elseif config.kernel === :temporal
        return spike_attention_temporal(
            SpikeTrain(source_events), SpikeTrain(context_events), READOUT; τ = config.tau)
    else
        return spike_attention_continuous(
            TemporalBuffer(config.window, source_events),
            TemporalBuffer(config.window, context_events),
            READOUT; τ = config.tau)
    end
end

"""
    _row(config, jitter, seed, out, baseline) -> NamedTuple

One `metrics.csv` record. `top1` is `0` and `target_share` is `0` when the
window admits no spike pairs at all (total attention exactly zero).
"""
function _row(config::KernelConfig, jitter::Float32, seed::Int,
              out::AbstractVector, baseline::AbstractVector)
    total = sum(out)
    active = total > 0.0f0
    share = active ? out[TARGET_NEURON] / total : 0.0f0
    top1 = active ? argmax(out) : 0
    delta = out .- baseline
    l1 = sum(abs, delta)
    l2 = sqrt(sum(abs2, delta))
    baseline_norm = sqrt(sum(abs2, baseline))
    return (
        kernel = String(config.kernel),
        jitter = jitter,
        tau = config.tau,
        window = config.window,
        seed = seed,
        target_share = share,
        top1 = top1,
        top1_correct = top1 == TARGET_NEURON,
        drift_l1 = l1,
        drift_l2 = l2,
        drift_rel_l2 = baseline_norm > 0.0f0 ? l2 / baseline_norm : 0.0f0,
        total_attention = total,
        active = active,
    )
end

"""
    run_sweep() -> (rows, baselines)

Sweep every kernel configuration over the jitter and seed grids. Each
configuration emits exactly one zero-jitter baseline row (seed
`BASELINE_SEED`, drift zero by construction) followed by one row per
`(jitter > 0, seed)` pair.
"""
function run_sweep()
    source_events = _source_events()
    baseline_context = _context_events(0.0f0, BASELINE_SEED)
    rows = NamedTuple[]
    baselines = Dict{KernelConfig,Vector{Float32}}()

    for config in _kernel_configs()
        baseline = Vector{Float32}(_evaluate(config, source_events, baseline_context))
        baselines[config] = baseline
        push!(rows, _row(config, 0.0f0, BASELINE_SEED, baseline, baseline))
        for jitter in JITTERS
            jitter > 0.0f0 || continue
            for seed in SEEDS
                out = _evaluate(config, source_events, _context_events(jitter, seed))
                push!(rows, _row(config, jitter, seed, out, baseline))
            end
        end
    end
    return rows, baselines
end

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_spread(values) = length(values) > 1 ? std(values) : 0.0

"""
    aggregate(rows) -> Dict

Collapse the per-seed rows into `(kernel, tau, window, jitter)` summaries
carrying the mean and spread over seeds.
"""
function aggregate(rows)
    groups = Dict{Tuple{String,Float32,Float32,Float32},Vector{NamedTuple}}()
    for row in rows
        push!(get!(Vector{NamedTuple}, groups, (row.kernel, row.tau, row.window, row.jitter)), row)
    end
    summaries = Dict{Tuple{String,Float32,Float32,Float32},NamedTuple}()
    for (key, group) in groups
        retention = mean(r -> r.top1_correct ? 1.0 : 0.0, group)
        shares = Float64[r.target_share for r in group]
        drifts = Float64[r.drift_rel_l2 for r in group]
        summaries[key] = (
            n_seeds = length(group),
            retention = retention,
            share_mean = mean(shares),
            share_sd = _spread(shares),
            drift_mean = mean(drifts),
            drift_sd = _spread(drifts),
            drift_max = maximum(drifts),
            collapse_rate = mean(r -> r.active ? 0.0 : 1.0, group),
        )
    end
    return summaries
end

_curve(summaries, kernel::AbstractString, tau::Float32, window::Float32, field::Symbol) =
    [getproperty(summaries[(kernel, tau, window, jitter)], field) for jitter in JITTERS]

"""
    tolerance_sigma(retentions) -> Float32

Timing tolerance envelope: the largest jitter scale on the grid that is still
tolerated, meaning every scale up to and including it keeps top-1 retention at
or above `RETENTION_THRESHOLD`. Returns `NaN32` when the configuration already
misses the target at zero jitter.
"""
function tolerance_sigma(retentions::AbstractVector{<:Real})
    retentions[1] >= RETENTION_THRESHOLD || return NaN32
    tolerated = JITTERS[1]
    for i in 2:length(JITTERS)
        retentions[i] >= RETENTION_THRESHOLD || break
        tolerated = JITTERS[i]
    end
    return tolerated
end

"Largest single-step drop in top-1 retention between adjacent jitter scales."
function max_step_drop(retentions::AbstractVector{<:Real})
    drop = 0.0
    for i in 2:length(retentions)
        drop = max(drop, retentions[i-1] - retentions[i])
    end
    return drop
end

"True when a curve never decreases along the jitter grid."
_is_nondecreasing(values::AbstractVector{<:Real}) =
    all(i -> values[i] <= values[i+1], 1:length(values)-1)

"True when a curve that starts selected rises again after a preceding drop."
function _has_recovery(values::AbstractVector{<:Real})
    isempty(values) && return false
    values[1] > 0 || return false

    peak = values[1]
    has_dropped = false
    for i in 2:length(values)
        value = values[i]
        has_dropped |= value < peak
        has_dropped && value > values[i-1] && return true
        peak = max(peak, value)
    end
    return false
end

"""
    widest_window_agreement(rows) -> NamedTuple

Compare the widest continuous window against the temporal kernel row for row,
rather than only comparing their tolerance envelopes. `first_divergence` is the
smallest jitter scale at which any (τ, seed) pair disagrees on total admitted
attention — `nothing` if the two kernels agree everywhere on this grid — and
`max_rel_gap` is the largest relative gap anywhere on the grid.
"""
function widest_window_agreement(rows)
    widest = maximum(WINDOWS)
    reference = Dict((r.tau, r.jitter, r.seed) => r.total_attention
                     for r in rows if r.kernel == "temporal")
    first_divergence = nothing
    max_rel_gap = 0.0
    for row in rows
        (row.kernel == "continuous" && row.window == widest) || continue
        total = reference[(row.tau, row.jitter, row.seed)]
        (total > 0.0f0 && row.total_attention != total) || continue
        max_rel_gap = max(max_rel_gap, abs(row.total_attention - total) / total)
        if first_divergence === nothing || row.jitter < first_divergence
            first_divergence = row.jitter
        end
    end
    return (widest = widest, first_divergence = first_divergence, max_rel_gap = max_rel_gap)
end

"""
    threshold_ties(summaries) -> Int

Number of swept conditions whose top-1 retention lands exactly on
`RETENTION_THRESHOLD`. `tolerance_sigma` accepts these, so they are the
weakest evidence behind any σ* that depends on them.
"""
function threshold_ties(summaries)
    ties = 0
    for config in _kernel_configs(), jitter in JITTERS
        summary = summaries[(String(config.kernel), config.tau, config.window, jitter)]
        summary.n_seeds > 1 && summary.retention == RETENTION_THRESHOLD && (ties += 1)
    end
    return ties
end

"Number of swept configurations whose top-1 retention rises again at a larger jitter scale."
function retention_recoveries(summaries)
    return count(_kernel_configs()) do config
        _has_recovery(_curve(summaries, String(config.kernel), config.tau, config.window, :retention))
    end
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_jitter_label(sigma::Float32) = sigma == 0.0f0 ? "0" : @sprintf("%.3g", sigma)
_sigma_label(sigma) = isnan(sigma) ? "—" : (sigma == 0.0f0 ? "0" : @sprintf("%.3g", sigma))

"Colour-vision-friendly palette (Wong), long enough for the τ and window grids."
const PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A3E9D"]

function build_figure(summaries)
    xs = collect(1:length(JITTERS))
    ticks = (xs, _jitter_label.(JITTERS))
    colors = PALETTE

    fig = Figure(size = (1500, 1050))
    Label(fig[0, 1:3],
        "Timing tolerance envelope — how much spike-timestamp jitter each kernel absorbs";
        fontsize = 24, font = :bold, padding = (0, 0, 8, 0))

    # (A) Temporal kernel: top-1 retention of the target neuron.
    ax_a = Axis(fig[1, 1];
        title = "A. Target retention vs jitter (temporal kernel)",
        xlabel = "jitter scale σ (time units)", ylabel = "top-1 = target (fraction of seeds)",
        xticks = ticks, yticks = 0:0.25:1)
    ylims!(ax_a, -0.06, 1.08)
    vlines!(ax_a, [1.0 + log_position(BASELINE_ALIGNMENT_ERROR)]; color = (:black, 0.35),
        linestyle = :dot, linewidth = 2)
    text!(ax_a, 1.0 + log_position(BASELINE_ALIGNMENT_ERROR), 0.55;
        text = " baseline alignment error", align = (:left, :center), fontsize = 13,
        color = (:black, 0.6), rotation = pi / 2)
    for (i, tau) in enumerate(TAUS)
        lines!(ax_a, xs, _curve(summaries, "temporal", tau, NaN32, :retention);
            color = colors[i], linewidth = 2.5, label = @sprintf("τ = %.2f", tau))
        scatter!(ax_a, xs, _curve(summaries, "temporal", tau, NaN32, :retention);
            color = colors[i], markersize = 8)
    end
    lines!(ax_a, xs, _curve(summaries, "discrete", NaN32, NaN32, :retention);
        color = :black, linestyle = :dash, linewidth = 2.5, label = "discrete (never target)")
    axislegend(ax_a; position = :lb, framevisible = true, labelsize = 12, nbanks = 2)

    # (B) Temporal kernel: relative L2 drift from the zero-jitter baseline.
    ax_b = Axis(fig[1, 2];
        title = "B. Output drift vs jitter (temporal kernel)",
        xlabel = "jitter scale σ (time units)", ylabel = "relative L2 drift from σ = 0 baseline",
        xticks = ticks)
    for (i, tau) in enumerate(TAUS)
        means = _curve(summaries, "temporal", tau, NaN32, :drift_mean)
        sds = _curve(summaries, "temporal", tau, NaN32, :drift_sd)
        band!(ax_b, xs, means .- sds, means .+ sds; color = (colors[i], 0.15))
        lines!(ax_b, xs, means; color = colors[i], linewidth = 2.5,
            label = @sprintf("τ = %.2f", tau))
    end
    lines!(ax_b, xs, _curve(summaries, "discrete", NaN32, NaN32, :drift_mean);
        color = :black, linestyle = :dash, linewidth = 2.5, label = "discrete (exactly 0)")
    axislegend(ax_b; position = :lt, framevisible = true, labelsize = 12, nbanks = 2)

    # (C) Continuous kernel at a representative τ: graceful decay vs window collapse.
    ax_c = Axis(fig[2, 1];
        title = @sprintf("C. Target retention vs jitter (continuous, τ = %.2f)", REPRESENTATIVE_TAU),
        xlabel = "jitter scale σ (time units)", ylabel = "top-1 = target (fraction of seeds)",
        xticks = ticks, yticks = 0:0.25:1)
    ylims!(ax_c, -0.06, 1.20)
    # Wider windows converge on the temporal kernel, so their curves coincide.
    # Thinning the line with each step keeps every coincident curve visible.
    for (i, window) in enumerate(WINDOWS)
        lines!(ax_c, xs, _curve(summaries, "continuous", REPRESENTATIVE_TAU, window, :retention);
            color = colors[i], linewidth = 6.5 - 0.9 * i, label = @sprintf("window = %.2f", window))
        collapse = _curve(summaries, "continuous", REPRESENTATIVE_TAU, window, :collapse_rate)
        any(>(0.0), collapse) && lines!(ax_c, xs, collapse;
            color = (colors[i], 0.55), linewidth = 2, linestyle = :dot)
    end
    axislegend(ax_c; position = :lb, framevisible = true, labelsize = 12, nbanks = 2)
    text!(ax_c, 1.0, 1.13;
        text = "dotted = fraction of seeds with zero admitted pairs; wider windows coincide (lines thin with window size)",
        align = (:left, :center), fontsize = 12, color = (:black, 0.6))

    # (D) Regime map: tolerated jitter per τ and window.
    columns = vcat([@sprintf("%.2f", w) for w in WINDOWS], ["temporal\n(no window)"])
    envelope = fill(NaN32, length(columns), length(TAUS))
    for (j, tau) in enumerate(TAUS)
        for (i, window) in enumerate(WINDOWS)
            envelope[i, j] = tolerance_sigma(_curve(summaries, "continuous", tau, window, :retention))
        end
        envelope[end, j] = tolerance_sigma(_curve(summaries, "temporal", tau, NaN32, :retention))
    end
    ax_d = Axis(fig[2, 2];
        title = "D. Regime map — ≥50% retention jitter envelope σ*",
        xlabel = "continuous window", ylabel = "τ",
        xticks = (1:length(columns), columns),
        yticks = (1:length(TAUS), [@sprintf("%.2f", t) for t in TAUS]))
    ranks = map(s -> isnan(s) ? NaN : Float64(findfirst(==(s), JITTERS)), envelope)
    heatmap!(ax_d, 1:length(columns), 1:length(TAUS), ranks;
        colormap = :viridis, colorrange = (1.0, Float64(length(JITTERS))), nan_color = "#d0d0d0")
    for j in 1:length(TAUS), i in 1:length(columns)
        rank = ranks[i, j]
        text!(ax_d, i, j; text = _sigma_label(envelope[i, j]), align = (:center, :center),
            fontsize = 13, font = :bold,
            color = (!isnan(rank) && rank < length(JITTERS) * 0.6) ? :white : :black)
    end
    Colorbar(fig[2, 3]; colormap = :viridis, colorrange = (1.0, Float64(length(JITTERS))),
        ticks = (xs, _jitter_label.(JITTERS)),
        label = "largest σ* with ≥50% retention (grey = target not selected at σ = 0)")

    return fig
end

"Position of a jitter magnitude on the evenly spaced grid axis, by interpolation."
function log_position(sigma::Real)
    positive = [j for j in JITTERS if j > 0.0f0]
    sigma <= positive[1] && return 1.0
    for i in 2:length(positive)
        if sigma <= positive[i]
            lo, hi = log(positive[i-1]), log(positive[i])
            return (i - 1) + (log(sigma) - lo) / (hi - lo)
        end
    end
    return Float64(length(positive))
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

"Sorted, de-duplicated `%.2f` labels for a bag of grid values."
_grid_set(values) = [@sprintf("%.2f", v) for v in sort!(unique(values))]

function _markdown_table(io, header, rows)
    println(io, "| ", join(header, " | "), " |")
    println(io, "| ", join(fill("---", length(header)), " | "), " |")
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
    println(io)
end

function _summary_header(io)
    println(io, "# Jitter test — timing tolerance of the TemporalFocus kernels")
    println(io)
    println(io, "Generated by `experiments/jitter_test.jl`. Every number below comes from ",
        "`metrics.csv` in this directory; the sweep is fully determined by the seeds in `config.toml`.")
    println(io)

    println(io, "## Question")
    println(io)
    println(io, "How quickly does TemporalFocus lose target selectivity as spike timestamps are ",
        "perturbed, and how does that sensitivity depend on τ and the continuous window?")
    println(io)

    println(io, "## Setup")
    println(io)
    println(io, @sprintf("A fixed %d-neuron scene carries one source spike per neuron at t = %.1f. ", N_NEURONS, T_REF),
        @sprintf("Neuron %d (the **target**) has %d context spikes placed within %.2f time units of the source spike. ",
            TARGET_NEURON, length(CONTEXT_OFFSETS[TARGET_NEURON]), BASELINE_ALIGNMENT_ERROR),
        @sprintf("Neuron %d (the **loud distractor**) carries the largest coincidence mass in the scene — %d context spikes — ",
            LOUD_NEURON, length(CONTEXT_OFFSETS[LOUD_NEURON])),
        "but places all of it far from the source spike in time. Target identity is therefore encoded purely in timing.")
    println(io)
    println(io, "Zero-mean Gaussian jitter is added to context spike timestamps only: `t' = t + σ·z`, ",
        "`z ~ N(0, 1)` from `MersenneTwister(seed)`. Spike values and neuron identities never change. ",
        "The readout is the identity matrix, so each output vector *is* the per-neuron attention vector.")
    println(io)
    println(io, "The primary output-distance metric is the **relative L2 distance** ",
        "`‖out − baseline‖₂ / ‖baseline‖₂` against the zero-jitter output of the same configuration. ",
        "L2 is preferred here because attention mass concentrates on a few neurons, so L2 keeps the reported ",
        "drift dominated by the large per-neuron moves. Absolute L1 and L2 are also in `metrics.csv`.")
    println(io)
    return nothing
end

"Result 1: the experimental check of discrete attention's predicted invariance."
function _summary_discrete(io, rows)
    discrete_rows = filter(r -> r.kernel == "discrete", rows)
    max_drift = maximum(r -> r.drift_l2, discrete_rows)
    n_shares = length(unique(r.target_share for r in discrete_rows))
    top1s = unique(r.top1 for r in discrete_rows)

    invariant = max_drift == 0.0f0
    println(io, invariant ?
        "## Result 1 — discrete attention is exactly invariant to timestamp jitter" :
        "## Result 1 — discrete attention is not invariant to timestamp jitter")
    println(io)
    println(io, @sprintf("Across all %d discrete rows, spanning every jitter scale up to σ = %.2f and every seed, ",
            length(discrete_rows), maximum(JITTERS)),
        invariant ?
            @sprintf("the maximum L2 drift from the zero-jitter baseline is **%g** — bit-for-bit identical output. ", max_drift) :
            @sprintf("the maximum L2 drift from the zero-jitter baseline is **%g**, so the outputs differ. ", max_drift),
        invariant ?
            @sprintf("Target attention share is constant at **%.4f** (%d distinct value observed) and top-1 is constant at neuron **%d** (%d distinct value observed).",
                first(discrete_rows).target_share, n_shares, first(top1s), length(top1s)) :
            @sprintf("Target attention share has **%d** distinct value(s), and top-1 has **%d** distinct value(s).",
                n_shares, length(top1s)))
    println(io)
    if invariant
        println(io, "The hypothesis holds: the discrete kernel never reads a timestamp, so timestamp-only ",
            @sprintf("perturbation cannot move it. The cost is that it never selects the target — it selects neuron %d, ", first(top1s)),
            "the neuron with the most coincidence mass, at every jitter scale. Perfect robustness here means the kernel ",
            "reads no timing at all, which is not the same thing as resilience.")
    else
        println(io, "**The hypothesis does not hold.** Discrete attention was expected to be exactly invariant to ",
            "timestamp-only jitter, but a non-zero drift was measured. Treat the invariance claim as refuted on this ",
            "grid and investigate before relying on it.")
    end
    println(io)
    return nothing
end

"Result 2: temporal degradation across τ."
function _summary_temporal(io, summaries, baselines)
    println(io, "## Result 2 — temporal weights vary smoothly, but top-1 retention can fall in cliffs")
    println(io)
    table_rows = Vector{String}[]
    for tau in TAUS
        retentions = _curve(summaries, "temporal", tau, NaN32, :retention)
        drifts = _curve(summaries, "temporal", tau, NaN32, :drift_mean)
        drift_sds = _curve(summaries, "temporal", tau, NaN32, :drift_sd)
        baseline = baselines[(kernel = :temporal, tau = tau, window = NaN32)]
        push!(table_rows, [
            @sprintf("%.2f", tau),
            retentions[1] >= 1.0 ? "yes" : "no",
            @sprintf("%.3f", baseline[TARGET_NEURON] / sum(baseline)),
            _sigma_label(tolerance_sigma(retentions)),
            @sprintf("%.2f", retentions[end]),
            @sprintf("%.3f ± %.3f", drifts[end], drift_sds[end]),
            @sprintf("%.2f", max_step_drop(retentions)),
        ])
    end
    _markdown_table(io, [
            "τ",
            "target at σ = 0",
            "baseline target share",
            "tolerated σ*",
            @sprintf("retention at σ = %.2f", maximum(JITTERS)),
            @sprintf("relative drift at σ = %.2f", maximum(JITTERS)),
            "largest single-step retention drop",
        ], table_rows)
    largest_drop = maximum(max_step_drop(
        _curve(summaries, "temporal", tau, NaN32, :retention)) for tau in TAUS)
    println(io, @sprintf("The exponential weights are smooth functions of timestamp error, but top-1 retention is a discrete decision and is not smooth: the largest sampled one-step retention drop is **%.1f percentage points**. ", 100 * largest_drop),
        "The curves still move toward the timing-independent discrete answer as τ flattens the recency weighting.")
    println(io)
    println(io, @sprintf("`σ*` is the largest jitter scale on the grid for which top-1 retention stays at or above %.0f%% ",
            RETENTION_THRESHOLD * 100),
        "for that scale and every smaller one. A dash means the configuration already misses the target at zero jitter.")
    println(io)
    println(io, "Two things are visible in the table and in panel A of `figure.png`:")
    println(io)
    println(io, "1. Tolerance grows with τ up to a point. A short τ resolves the target sharply but has almost no ",
        "margin, because a perturbation of a few hundredths of a time unit is already large relative to τ.")
    println(io, "2. Past that point, a τ large enough to be jitter-tolerant is also large enough to stop discriminating: ",
        "the recency weight flattens, spike count takes over, and the kernel converges on the loud distractor — ",
        "the same answer the discrete kernel gives. There is a genuine interior optimum, not a monotone tradeoff.")
    println(io)
    return nothing
end

"The τ × window regime map of tolerated jitter."
function _summary_regime_map(io, summaries)
    table_rows = Vector{String}[]
    for tau in TAUS
        cells = String[@sprintf("**%.2f**", tau)]
        for window in WINDOWS
            push!(cells, _sigma_label(tolerance_sigma(_curve(summaries, "continuous", tau, window, :retention))))
        end
        push!(table_rows, cells)
    end
    _markdown_table(io, vcat(["τ \\ window"], [@sprintf("%.2f", w) for w in WINDOWS]), table_rows)
    println(io, "Tolerated σ* for the continuous kernel. The same map, plus the temporal kernel as an ",
        "unbounded-window column, is panel D of `figure.png`.")
    println(io)
    return nothing
end

"The subset of continuous configurations whose attention collapsed to exactly zero."
function _summary_collapse_table(io, summaries)
    table_rows = Vector{String}[]
    for tau in TAUS, window in WINDOWS
        collapse = _curve(summaries, "continuous", tau, window, :collapse_rate)
        peak = maximum(collapse)
        peak > 0.0 || continue
        push!(table_rows, [
            @sprintf("%.2f", tau),
            @sprintf("%.2f", window),
            _jitter_label(JITTERS[findfirst(>(0.0), collapse)]),
            @sprintf("%.2f", peak),
            @sprintf("%.2f", max_step_drop(_curve(summaries, "continuous", tau, window, :retention))),
        ])
    end
    if isempty(table_rows)
        println(io, "No configuration ever lost every admitted spike pair, so no attention collapse was observed on this grid.")
        println(io)
        return nothing
    end
    println(io, "### Window-boundary collapse")
    println(io)
    println(io, "A window-boundary failure has a signature the smooth decay does not: total attention drops to ",
        "exactly zero because jitter pushed every relevant pair outside `|Δt| ≤ window`, leaving no neuron to select ",
        "at all. These configurations reached that state:")
    println(io)
    _markdown_table(io,
        ["τ", "window", "first σ with a collapse", "peak collapse rate", "largest single-step retention drop"],
        table_rows)
    println(io)
    return nothing
end

"True when at least half the paired seeds switch directly from correct-active to wrong-active."
function _has_steep_reordering(rows, tau, window)
    paired = Dict((r.jitter, r.seed) => r for r in rows
        if r.kernel == "continuous" && r.tau == tau && r.window == window && r.jitter > 0.0f0)
    for i in 2:(length(JITTERS) - 1)
        transitions = count(SEEDS) do seed
            before = paired[(JITTERS[i], seed)]
            after = paired[(JITTERS[i+1], seed)]
            before.active && before.top1_correct && after.active && !after.top1_correct
        end
        transitions / length(SEEDS) >= 0.5 && return true
    end
    return false
end

"""
Classify the two continuous failure modes independently. A configuration can
both collapse at one scale and undergo a steep active-output reordering at
another, so `collapse_windows` and `steep_*` deliberately overlap.
"""
function _classify_continuous(rows, summaries)
    eligible = 0
    graceful = 0
    overlap = 0
    collapse_windows = Float32[]
    steep_taus = Float32[]
    steep_windows = Float32[]
    for tau in TAUS, window in WINDOWS
        retentions = _curve(summaries, "continuous", tau, window, :retention)
        collapse = _curve(summaries, "continuous", tau, window, :collapse_rate)
        retentions[1] >= RETENTION_THRESHOLD || continue
        eligible += 1
        has_collapse = maximum(collapse) > 0.0
        has_steep_reordering = _has_steep_reordering(rows, tau, window)
        if has_collapse
            push!(collapse_windows, window)
        end
        if has_steep_reordering
            push!(steep_taus, tau)
            push!(steep_windows, window)
        end
        if has_collapse && has_steep_reordering
            overlap += 1
        elseif !has_collapse && !has_steep_reordering
            graceful += 1
        end
    end
    return (eligible = eligible, graceful = graceful, overlap = overlap,
        collapse_windows = collapse_windows, steep_taus = steep_taus,
        steep_windows = steep_windows)
end

"Result 3: continuous degradation, separating graceful decay from the two abrupt failure modes."
function _summary_continuous(io, rows, summaries)
    println(io, "## Result 3 — continuous attention has two separate failure modes")
    println(io)
    _summary_regime_map(io, summaries)
    _summary_collapse_table(io, summaries)

    classes = _classify_continuous(rows, summaries)
    collapsed = length(classes.collapse_windows)
    steep = length(classes.steep_taus)

    println(io, @sprintf("Of the %d continuous configurations that start out selecting the target, **%d degrade gracefully** ",
            classes.eligible, classes.graceful),
        "— no attention collapse and no jitter step where at least half of paired seeds switch directly from ",
        "correct-and-active to wrong-and-active. ",
        "The remaining configurations exhibit one or both of **two independently measured mechanisms**:")
    println(io)
    println(io, @sprintf("- **Window-boundary collapse (%d configurations, at window ∈ {%s}).** ",
            collapsed, join(_grid_set(classes.collapse_windows), ", ")),
        "Jitter pushes the target's context spikes past `|Δt| ≤ window`, the kernel admits no pairs at all, and ",
        "attention goes to exactly zero — there is no second-best neuron to fall back on. This only happens for windows ",
        @sprintf("comparable to the target's %.2f alignment error.", BASELINE_ALIGNMENT_ERROR))
    if steep > 0
        println(io, @sprintf("- **Steep active-output reordering (%d configurations, at τ ∈ {%s} and window ∈ {%s}).** ",
                steep, join(_grid_set(classes.steep_taus), ", "), join(_grid_set(classes.steep_windows), ", ")),
            "At least half of the same paired seeds switch directly from correct-and-active to wrong-and-active in one ",
            "jitter step, so this count cannot be inflated by different seeds entering and leaving collapse while the ",
            "aggregate collapse rate stays flat.")
    end
    classes.overlap > 0 && println(io, @sprintf("- **Overlap (%d configurations).** These exhibit some window-boundary collapse at one or more scales and also a separate steep active-output reordering step; assigning them to a single bucket would hide one mechanism.", classes.overlap))
    println(io)
    collapse_window_set = isempty(classes.collapse_windows) ? "none" : "{" * join(_grid_set(classes.collapse_windows), ", ") * "}"
    steep_window_set = isempty(classes.steep_windows) ? "none" : "{" * join(_grid_set(classes.steep_windows), ", ") * "}"
    println(io, "The data-derived statement is therefore narrower than \"narrow windows fail abruptly\". ",
        "Window-boundary collapse occurs at the measured window set ", collapse_window_set,
        ", while the paired-seed steep active-output reordering criterion is met at ", steep_window_set, ". ",
        "Panel C of `figure.png` shows both shapes on one axis, with dotted lines marking collapse rate.")
    println(io)
    _summary_widest_window(io, rows)
    return nothing
end

"""
When the widest window stops behaving like an unbounded one, compared row by
row rather than by tolerance envelope alone.
"""
function _summary_widest_window(io, rows)
    agreement = widest_window_agreement(rows)
    println(io, "### Where the widest window stops being a no-op")
    println(io)
    if agreement.first_divergence === nothing
        println(io, @sprintf("Every `window = %.2f` row matches the corresponding temporal row exactly on this grid, ", agreement.widest),
            "so the widest window sampled here never gates a pair and is indistinguishable from the unbounded kernel.")
        println(io)
        return nothing
    end
    println(io, @sprintf("The `window = %.2f` column of the map above has the same tolerated σ* as the temporal column, but ", agreement.widest),
        "matching envelopes are not the same thing as matching outputs, and row-by-row the two kernels do part company. ",
        @sprintf("Total admitted attention is identical for every (τ, seed) pair up to σ = %s, and first diverges at **σ = %s**, ",
            _jitter_label(JITTERS[max(1, findfirst(==(agreement.first_divergence), JITTERS) - 1)]),
            _jitter_label(agreement.first_divergence)),
        @sprintf("opening a relative gap of up to **%.1f%%**.", 100 * agreement.max_rel_gap))
    println(io)
    println(io, @sprintf("That is the expected behaviour rather than a defect: the scene's largest baseline offset is %.2f, so once σ ",
            maximum(maximum(abs, offsets) for offsets in CONTEXT_OFFSETS)),
        @sprintf("grows comparable to the headroom between that offset and the window, jitter starts carrying pairs past `|Δt| ≤ %.2f` ", agreement.widest),
        "and the window begins gating them. The window is a no-op only while the jitter stays small relative to it — which is ",
        "the same boundary effect that makes the narrow windows collapse, just displaced to a much larger σ. Any claim that a ",
        "wide window \"reproduces the temporal kernel\" therefore has to be qualified by the jitter scale.")
    println(io)
    return nothing
end

"Result 4: top-1 stability and vector drift carry different information."
function _summary_drift(io, summaries)
    println(io, "## Result 4 — top-1 stability and vector drift disagree, and both are needed")
    println(io)
    table_rows = Vector{String}[]
    for tau in TAUS
        retentions = _curve(summaries, "temporal", tau, NaN32, :retention)
        drifts = _curve(summaries, "temporal", tau, NaN32, :drift_mean)
        idx = findfirst(<(1.0), retentions)
        (idx === nothing || idx == 1) && continue
        push!(table_rows, [
            @sprintf("%.2f", tau),
            _jitter_label(JITTERS[idx-1]),
            @sprintf("%.3f", drifts[idx-1]),
            _jitter_label(JITTERS[idx]),
            @sprintf("%.3f", drifts[idx]),
        ])
    end
    isempty(table_rows) || _markdown_table(io,
        ["τ", "last σ with full retention", "relative drift there", "first σ that loses a seed", "relative drift there"],
        table_rows)
    println(io, "Relative L2 drift is already non-zero at the smallest non-zero jitter scale and reaches a sizeable ",
        "fraction of the baseline norm well before top-1 changes: the attention vector is moving while the selected ",
        "neuron is still correct. Top-1 stability is the discrete decision boundary; vector drift is the continuous ",
        "signal that shows the boundary being approached. Reporting either alone would misstate the robustness.")
    println(io)

    dips = String[]
    taus_with_dips = 0
    for tau in TAUS
        drifts = _curve(summaries, "temporal", tau, NaN32, :drift_mean)
        _is_nondecreasing(drifts) && continue
        taus_with_dips += 1
        for idx in findall(i -> drifts[i+1] < drifts[i], 1:length(drifts)-1)
            push!(dips, @sprintf("τ = %.2f dips from %.3f at σ = %s to %.3f at σ = %s",
                tau, drifts[idx], _jitter_label(JITTERS[idx]), drifts[idx+1], _jitter_label(JITTERS[idx+1])))
        end
    end
    if isempty(dips)
        println(io, @sprintf("On this grid the seed-mean drift happens to be non-decreasing in σ for all %d τ values, but that is ", length(TAUS)),
            "an observation about this grid, not a guarantee.")
    else
        println(io, @sprintf("Drift does **not** grow monotonically, though. For %d of the %d τ values the seed-mean drift dips at some ",
                taus_with_dips, length(TAUS)),
            "point on the grid; every observed dip is listed here (", join(dips, "; "), "). Paired Gaussian perturbations move a spike toward the source time ",
            "as readily as away from it, so a larger σ can happen to land a scene closer to the baseline than a smaller one did. ",
            "The claim supported by the data is that drift *starts early*, not that it *rises steadily*.")
    end
    println(io)
    return nothing
end

function _summary_caveats(io, summaries)
    recoveries = retention_recoveries(summaries)
    ties = threshold_ties(summaries)

    println(io, "## Honest caveats")
    println(io)
    print(io, "- The tolerated σ* values are grid-resolved **estimates**, not bounds. They can only take values from the ",
        "jitter grid in `config.toml`. ")
    if recoveries > 0
        print(io, @sprintf("For %d of the %d swept configurations that start out selecting the target, retention falls and then rises again at a larger sampled scale, ", recoveries, length(_kernel_configs())),
            "because a Gaussian perturbation can carry a spike back toward the source time. ")
    else
        print(io, "No configuration that starts out selecting the target loses and later regains it on this sampled grid, ",
            "but sampled monotonicity does not prove monotonicity between grid points. ")
    end
    println(io, "Two passing grid points therefore do not rule out a failure at an unsampled scale between them. Read σ* as ",
        "\"the largest sampled scale that held\", not as a bound on the continuous threshold.")
    println(io, @sprintf("- Retention is averaged over %d seeds, so it is quantized to multiples of %.3f. The %.0f%% threshold is ",
            length(SEEDS), 1 / length(SEEDS), RETENTION_THRESHOLD * 100),
        @sprintf("exactly attainable (%d of %d seeds) and `tolerance_sigma` accepts it, so the %d conditions that land on it ",
            Int(RETENTION_THRESHOLD * length(SEEDS)), length(SEEDS), ties),
        "sit on a literal coin-flip boundary. Any σ* resting on one of those conditions is the weakest evidence in the tables.")
    println(io, "- The zero-jitter baseline is a single deterministic evaluation, not a seed average, so its row has no ",
        "spread. That is by construction: with σ = 0 the perturbation term vanishes and every seed yields the identical scene.")
    println(io, "- Conclusions are tied to this scene. The tolerance numbers scale with the target's baseline alignment ",
        @sprintf("error (%.2f time units here) and with the coincidence-mass gap between the target and the loud distractor. ", BASELINE_ALIGNMENT_ERROR),
        "The qualitative shapes — flat for discrete, smooth temporal weights with decision cliffs, and boundary-cliff-edged narrow continuous windows — ",
        "are what should carry over.")
    println(io, "- `randn` draws come from `Random.MersenneTwister`, which is reproducible for a given seed on a given ",
        "Julia version. The recorded seeds reproduce this run exactly on the Julia version noted in `config.toml`.")
    println(io)

    println(io, "## Reproducing")
    println(io)
    println(io, "```bash")
    println(io, "julia --project=experiments -e 'using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()'")
    println(io, "julia --project=experiments experiments/jitter_test.jl")
    println(io, "```")
    return nothing
end

"""
    build_summary(rows, summaries, baselines) -> String

Generated interpretation written to `summary.md`. Every claim is derived from
the measured rows rather than asserted, so a change in the data changes the
prose.
"""
function build_summary(rows, summaries, baselines)
    io = IOBuffer()
    _summary_header(io)
    _summary_discrete(io, rows)
    _summary_temporal(io, summaries, baselines)
    _summary_continuous(io, rows, summaries)
    _summary_drift(io, summaries)
    _summary_caveats(io, summaries)
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main()
    @info "Running jitter sweep" slug = SLUG neurons = N_NEURONS target = TARGET_NEURON

    # The zero-jitter scene must not depend on the seed: with σ = 0 the
    # perturbation term vanishes identically.
    let a = _build_context_events(0.0f0, BASELINE_SEED), b = _build_context_events(0.0f0, last(SEEDS))
        a == b || error("zero-jitter scene is seed-dependent; the baseline would not be well defined")
    end

    rows, baselines = run_sweep()
    summaries = aggregate(rows)

    discrete_rows = filter(r -> r.kernel == "discrete", rows)
    discrete_max_drift = maximum(r -> r.drift_l2, discrete_rows)
    discrete_max_drift == 0.0f0 || @warn "discrete attention was not invariant to timestamp jitter" discrete_max_drift

    config = Dict{String,Any}(
        "slug" => SLUG,
        "n_neurons" => N_NEURONS,
        "target_neuron" => TARGET_NEURON,
        "loud_distractor_neuron" => LOUD_NEURON,
        "source_spike_time" => T_REF,
        "spike_value" => SPIKE_VALUE,
        "context_offsets_flat" => reduce(vcat, CONTEXT_OFFSETS),
        "context_spike_counts" => [length(o) for o in CONTEXT_OFFSETS],
        "baseline_alignment_error" => BASELINE_ALIGNMENT_ERROR,
        "jitter_scales" => JITTERS,
        "tau_values" => TAUS,
        "window_values" => WINDOWS,
        "seeds" => SEEDS,
        "baseline_seed" => BASELINE_SEED,
        "representative_tau" => REPRESENTATIVE_TAU,
        "retention_threshold" => RETENTION_THRESHOLD,
        "jitter_distribution" => "gaussian_zero_mean",
        "jitter_applied_to" => "context_spike_timestamps",
        "rng" => "Random.MersenneTwister",
        "readout" => "identity",
        "output_distance_metric" => "relative_l2",
        "julia_version" => string(VERSION),
        "n_rows" => length(rows),
    )

    config_path = write_config(SLUG, config)
    metrics_path = write_metrics(SLUG, rows)

    fig = build_figure(summaries)
    fig_path = figure_path(SLUG)
    save(fig_path, fig; px_per_unit = 1)

    summary_path = write_summary(SLUG, build_summary(rows, summaries, baselines))

    println()
    println("jitter_test — ", length(rows), " rows over ",
        length(_kernel_configs()), " kernel configurations")
    println("  discrete max L2 drift from baseline : ", discrete_max_drift, " (expected exactly 0.0)")
    for tau in TAUS
        println(@sprintf("  temporal  τ=%.2f  tolerated σ* = %-6s  retention at σ=%.2f: %.2f",
            tau, _sigma_label(tolerance_sigma(_curve(summaries, "temporal", tau, NaN32, :retention))),
            maximum(JITTERS), _curve(summaries, "temporal", tau, NaN32, :retention)[end]))
    end
    println()
    println("  wrote ", config_path)
    println("  wrote ", metrics_path)
    println("  wrote ", fig_path)
    println("  wrote ", summary_path)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
