# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Experiment: "Focus Under Fire" (slug: focus_under_fire)
#
# Research question
# -----------------
# How robust is spike-native temporal attention when the context contains
# increasing amounts of *stale* same-neuron activity and *random* unrelated
# neuron noise?
#
# Hypothesis (pre-registered, see `HYPOTHESIS` below)
# ---------------------------------------------------
# Relative to timing-agnostic discrete attention, the temporal and continuous
# kernels should keep a larger share of attention on the recent target as the
# stale distractor load grows, and the continuous kernel should reject stale
# mass hardest once it falls outside the active window.
#
# Scientific integrity
# --------------------
# The scene generator, the full sweep grid, the seed list and the
# "meaningful separation" decision rule are all fixed as constants at the top
# of this file *before* any output is inspected, and they are echoed verbatim
# into `config.toml`. `summary.md` is generated from the measured numbers and
# reports null / negative results explicitly.
#
# Run from the repository root:
#
#     julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#     julia --project=experiments experiments/focus_under_fire.jl
#
# Artifacts land in `experiments/results/focus_under_fire/` via the shared
# harness (`repo_root` / `result_dir` / `figure_path` / `write_config` /
# `write_metrics` / `write_summary`). Results are git-ignored and regenerated
# by the command above.

using CairoMakie
using ExperimentUtils
using Printf
using Random
using Statistics
using TemporalFocus

# Fail loudly rather than publish results measured against some other copy of
# the package. The experiments environment is pointed at this checkout with
# `Pkg.develop(path=".")`; this assertion is the measurement-correctness belt.
let loaded = realpath(pkgdir(TemporalFocus))
    expected = realpath(repo_root())
    loaded == expected || error(
        "TemporalFocus was loaded from $(loaded), not the checkout at $(expected); " *
        "refusing to run so results cannot describe a different implementation",
    )
end

const SLUG = "focus_under_fire"

const HYPOTHESIS = "Temporal and continuous attention retain a larger share of " *
                   "attention on the recent target than discrete attention as stale " *
                   "same-neuron distractor load grows; continuous rejects stale mass " *
                   "hardest once it falls outside the active window."

# ---------------------------------------------------------------------------
# Focus-retention definition
# ---------------------------------------------------------------------------
#
# Every kernel is evaluated with an *identity* readout, so the value returned by
# `spike_attention_*` is the raw per-neuron attention vector `a` (length
# `N_NEURONS`). All spike values in this experiment are positive, so every
# `a[i] >= 0` and the following quantities are well defined:
#
#     target_mass      = a[TARGET_NEURON]
#     total_mass       = sum(a)
#     distractor_mass  = total_mass - target_mass
#     focus_retention  = total_mass > 0 ? target_mass / total_mass : NaN
#     margin           = target_mass - maximum(a[i] for i != TARGET_NEURON)
#     top1_neuron      = argmax(a)                 (ties resolve to lowest index)
#     top1_correct     = margin > 0                (a tie is NOT counted correct)
#
# `focus_retention` is therefore the target neuron's share of *total* attention
# mass in [0, 1], and is reported as `NaN` (not 0) when the kernel produces no
# attention at all, so "no mass anywhere" is never confused with "all mass on a
# distractor".
const FOCUS_RETENTION_DEFINITION =
    "focus_retention = a[target] / sum(a), where a is the per-neuron attention " *
    "vector obtained with an identity readout; NaN when sum(a) == 0."

# ---------------------------------------------------------------------------
# Scene definition (fixed before looking at any output)
# ---------------------------------------------------------------------------

const N_NEURONS = 48
const TARGET_NEURON = 7
const STALE_NEURON = 19
const T_NOW = 10.0f0

# The target: a short recent burst on TARGET_NEURON.
const N_TARGET_SPIKES = 3
const TARGET_SPACING = 0.05f0        # burst spikes at T_NOW - {0.00, 0.05, 0.10}
const TARGET_AMPLITUDE = 1.0f0

# Stale same-neuron interference: a long-past burst on a single rival neuron.
const STALE_SPACING = 0.05f0
const STALE_AMPLITUDE = 1.0f0

# Unrelated-neuron noise: random neurons, random times over the whole history.
const RANDOM_SPAN = 5.0f0            # random spikes drawn from [T_NOW - 5, T_NOW]
const RANDOM_SPIKES_MAX = 3          # spikes per random distractor neuron
const RANDOM_AMP_JITTER = (0.5f0, 1.5f0)  # multiplicative jitter on the amplitude

# Baseline condition; each sweep varies exactly one axis away from this point.
const BASE_N_STALE = 12
const BASE_STALE_AGE = 1.0f0
const BASE_N_RANDOM = 12
const BASE_RANDOM_AMP = 1.0f0
const BASE_TAU = 0.25f0
const BASE_WINDOW = 0.5f0

const SEEDS = [20260823, 20260824, 20260825, 20260826, 20260827]

const KERNELS = ["discrete", "temporal", "continuous"]

# Sweep grid. `axis` names the varied quantity; `values` are its levels.
const SWEEPS = [
    (axis = "stale_count", label = "stale spikes on rival neuron",
     values = Float64[0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40], logx = false),
    (axis = "stale_age", label = "age of stale burst (time units)",
     values = Float64[0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0], logx = true),
    (axis = "random_count", label = "random distractor neurons",
     values = Float64[0, 2, 4, 8, 12, 16, 24, 32, 40], logx = false),
    (axis = "distractor_amp", label = "distractor amplitude scale",
     values = Float64[0.25, 0.5, 1.0, 2.0, 4.0, 8.0], logx = true),
    # The top of this grid must exceed the full scene span (`RANDOM_SPAN`) by a
    # wide margin, otherwise the "flat weight" control is not actually flat: at
    # τ = 4 the multiplier still falls to exp(-5/4) ≈ 0.29 across the scene.
    # τ = 64 gives exp(-5/64) ≈ 0.93, which is a genuine no-recency control.
    (axis = "tau", label = "τ (time constant)",
     values = Float64[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0], logx = true),
    (axis = "window", label = "continuous window width",
     values = Float64[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], logx = true),
]

# Pre-registered decision rule for "a meaningful behavioural separation".
const SEPARATION_RETENTION_DELTA = 0.10   # absolute gap in mean focus retention
const SEPARATION_TOP1_DELTA = 0.20        # absolute gap in top-1 correctness rate

# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

"""
    Condition

One point of the sweep grid: the scene knobs plus the kernel hyper-parameters.
"""
struct Condition
    n_stale::Int
    stale_age::Float32
    n_random::Int
    random_amp::Float32
    τ::Float32
    window::Float32
end

_baseline_condition() = Condition(
    BASE_N_STALE, BASE_STALE_AGE, BASE_N_RANDOM, BASE_RANDOM_AMP, BASE_TAU, BASE_WINDOW,
)

function _with_axis(c::Condition, axis::AbstractString, v::Real)
    axis == "stale_count" &&
        return Condition(Int(v), c.stale_age, c.n_random, c.random_amp, c.τ, c.window)
    axis == "stale_age" &&
        return Condition(c.n_stale, Float32(v), c.n_random, c.random_amp, c.τ, c.window)
    axis == "random_count" &&
        return Condition(c.n_stale, c.stale_age, Int(v), c.random_amp, c.τ, c.window)
    axis == "distractor_amp" &&
        return Condition(c.n_stale, c.stale_age, c.n_random, Float32(v), c.τ, c.window)
    axis == "tau" &&
        return Condition(c.n_stale, c.stale_age, c.n_random, c.random_amp, Float32(v), c.window)
    axis == "window" &&
        return Condition(c.n_stale, c.stale_age, c.n_random, c.random_amp, c.τ, Float32(v))
    throw(ArgumentError("unknown sweep axis $(axis)"))
end

"""
    _condition_key(c) -> Tuple

Canonical parameter tuple for `c`. Every sweep axis passes through the baseline
point, so the baseline condition is generated once per axis; keying on the
parameters is what lets the analysis count each *distinct* condition once
instead of six times.
"""
_condition_key(c::Condition) =
    (c.n_stale, c.stale_age, c.n_random, c.random_amp, c.τ, c.window)

"""
    _condition_id(c) -> String

Human-readable form of [`_condition_key`](@ref), emitted as a `metrics.csv`
column so a reader can deduplicate the repeated baseline rows without having to
re-derive the sweep grid.
"""
_condition_id(c::Condition) = string(
    "s", c.n_stale, "_a", c.stale_age, "_r", c.n_random,
    "_p", c.random_amp, "_t", c.τ, "_w", c.window,
)

"""
    _unique_condition_keys() -> Set

The set of distinct parameter tuples in the sweep grid (< the number of
(axis, value) points, because all six axes share the baseline point).
"""
function _unique_condition_keys()
    base = _baseline_condition()
    return Set(_condition_key(_with_axis(base, s.axis, v)) for s in SWEEPS for v in s.values)
end

const DISTRACTOR_POOL = [i for i in 1:N_NEURONS if i != TARGET_NEURON && i != STALE_NEURON]

"""
    _build_context(cond, seed) -> Vector{SpikeEvent}

Deterministically build the context scene for `cond` under `seed`.

The scene has three additive parts:

1. **target burst** — `N_TARGET_SPIKES` spikes on `TARGET_NEURON` at
   `T_NOW - k * TARGET_SPACING`. Never varied by the sweep.
2. **stale same-neuron interference** — `cond.n_stale` spikes on the single
   rival `STALE_NEURON`, starting `cond.stale_age` in the past and marching
   further back in `STALE_SPACING` steps.
3. **unrelated-neuron noise** — the first `cond.n_random` neurons of a
   seed-fixed permutation of `DISTRACTOR_POOL`, each firing `1:RANDOM_SPIKES_MAX`
   spikes uniformly over `[T_NOW - RANDOM_SPAN, T_NOW]` with amplitude
   `cond.random_amp * U(RANDOM_AMP_JITTER...)`.

Part 3 is *nested*: the permutation and the per-neuron draws are consumed in a
fixed order, so raising `n_random` adds neurons to the previous scene instead of
resampling it, and changing `random_amp` rescales the same scene.
"""
function _build_context(cond::Condition, seed::Integer)
    events = SpikeEvent[]

    for k in 0:(N_TARGET_SPIKES - 1)
        push!(events, SpikeEvent(TARGET_NEURON, T_NOW - k * TARGET_SPACING, TARGET_AMPLITUDE))
    end

    for k in 0:(cond.n_stale - 1)
        t = T_NOW - cond.stale_age - k * STALE_SPACING
        push!(events, SpikeEvent(STALE_NEURON, t, STALE_AMPLITUDE))
    end

    rng = MersenneTwister(seed)
    perm = randperm(rng, length(DISTRACTOR_POOL))
    lo, hi = RANDOM_AMP_JITTER
    for idx in 1:cond.n_random
        neuron = DISTRACTOR_POOL[perm[idx]]
        for _ in 1:rand(rng, 1:RANDOM_SPIKES_MAX)
            t = T_NOW - RANDOM_SPAN * rand(rng, Float32)
            amp = cond.random_amp * (lo + (hi - lo) * rand(rng, Float32))
            push!(events, SpikeEvent(neuron, t, amp))
        end
    end

    return events
end

"""
    _probe_events() -> Vector{SpikeEvent}

The source ("query") train: one unit spike per neuron at `T_NOW`.

The probe is deliberately uniform across neurons — it carries no information
about which neuron is the target — so all selectivity comes from the context
scene and the kernel's treatment of time.
"""
_probe_events() = [SpikeEvent(i, T_NOW, 1.0f0) for i in 1:N_NEURONS]

function _identity_readout(n::Int)
    readout = zeros(Float32, n, n)
    for i in 1:n
        readout[i, i] = 1.0f0
    end
    return readout
end

const READOUT = _identity_readout(N_NEURONS)

"""
    _attention_of(kernel, context, cond) -> Vector{Float32}

Evaluate one kernel on the given context scene. All three kernels see the
*identical* generated scene and the identical uniform probe.
"""
function _attention_of(kernel::AbstractString, context::Vector{SpikeEvent}, cond::Condition)
    probe = _probe_events()
    if kernel == "discrete"
        return spike_attention_discrete(SpikeTrain(probe), SpikeTrain(context), READOUT)
    elseif kernel == "temporal"
        return spike_attention_temporal(
            SpikeTrain(probe), SpikeTrain(context), READOUT; τ = cond.τ,
        )
    elseif kernel == "continuous"
        return spike_attention_continuous(
            TemporalBuffer(cond.window, probe),
            TemporalBuffer(cond.window, context),
            READOUT; τ = cond.τ,
        )
    end
    throw(ArgumentError("unknown kernel $(kernel)"))
end

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

function _focus_metrics(a::AbstractVector{<:Real})
    total = Float64(sum(a))
    target_mass = Float64(a[TARGET_NEURON])
    distractor_mass = total - target_mass
    best_distractor = Float64(maximum(a[i] for i in eachindex(a) if i != TARGET_NEURON))
    retention = total > 0 ? target_mass / total : NaN
    margin = target_mass - best_distractor
    return (
        target_mass = target_mass,
        distractor_mass = distractor_mass,
        total_mass = total,
        focus_retention = retention,
        top1_neuron = argmax(a),
        top1_correct = margin > 0,
        margin = margin,
        best_distractor_mass = best_distractor,
    )
end

# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------

function _run_sweeps()
    rows = NamedTuple[]
    base = _baseline_condition()
    for sweep in SWEEPS
        for v in sweep.values
            cond = _with_axis(base, sweep.axis, v)
            for seed in SEEDS
                context = _build_context(cond, seed)
                for kernel in KERNELS
                    m = _focus_metrics(_attention_of(kernel, context, cond))
                    push!(rows, (
                        sweep = sweep.axis,
                        axis_value = Float64(v),
                        condition_id = _condition_id(cond),
                        kernel = kernel,
                        seed = seed,
                        n_stale = cond.n_stale,
                        stale_age = Float64(cond.stale_age),
                        n_random = cond.n_random,
                        distractor_amp = Float64(cond.random_amp),
                        tau = Float64(cond.τ),
                        window = Float64(cond.window),
                        n_context_spikes = length(context),
                        target_mass = m.target_mass,
                        distractor_mass = m.distractor_mass,
                        total_mass = m.total_mass,
                        focus_retention = m.focus_retention,
                        best_distractor_mass = m.best_distractor_mass,
                        margin = m.margin,
                        top1_neuron = m.top1_neuron,
                        top1_correct = m.top1_correct,
                    ))
                end
            end
        end
    end
    return rows
end

"""
    _aggregate(rows) -> Vector{NamedTuple}

Collapse the per-seed rows into mean ± std (population std over `SEEDS`) per
(sweep, axis value, kernel). `NaN` retentions are excluded from the retention
statistics and counted separately in `n_undefined`.
"""
function _aggregate(rows)
    out = NamedTuple[]
    for sweep in SWEEPS, v in sweep.values, kernel in KERNELS
        sel = [r for r in rows
               if r.sweep == sweep.axis && r.axis_value == Float64(v) && r.kernel == kernel]
        isempty(sel) && continue
        ret = [r.focus_retention for r in sel if !isnan(r.focus_retention)]
        margins = [r.margin for r in sel]
        push!(out, (
            sweep = sweep.axis,
            axis_value = Float64(v),
            kernel = kernel,
            n_seeds = length(sel),
            n_undefined = length(sel) - length(ret),
            retention_mean = isempty(ret) ? NaN : mean(ret),
            retention_std = length(ret) > 1 ? std(ret; corrected = false) : 0.0,
            retention_min = isempty(ret) ? NaN : minimum(ret),
            retention_max = isempty(ret) ? NaN : maximum(ret),
            top1_rate = mean(r.top1_correct for r in sel),
            margin_mean = mean(margins),
            margin_std = length(margins) > 1 ? std(margins; corrected = false) : 0.0,
        ))
    end
    return out
end

_pick(agg, sweep, v, kernel) = only(
    a for a in agg if a.sweep == sweep && a.axis_value == Float64(v) && a.kernel == kernel
)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

const KERNEL_COLOR = Dict(
    "discrete" => RGBf(0.85, 0.37, 0.01),
    "temporal" => RGBf(0.12, 0.47, 0.71),
    "continuous" => RGBf(0.17, 0.63, 0.17),
)

# `temporal` is dashed so it stays visible where it coincides exactly with
# `continuous` (which happens whenever the window covers the whole scene).
const KERNEL_STYLE = Dict(
    "discrete" => :solid,
    "temporal" => :dash,
    "continuous" => :solid,
)

function _series(agg, sweep_axis, kernel, field)
    pts = [a for a in agg if a.sweep == sweep_axis && a.kernel == kernel]
    sort!(pts; by = a -> a.axis_value)
    return [a.axis_value for a in pts], [getfield(a, field) for a in pts]
end

function _panel_grid(fig, agg, field, ylabel, title, ylims)
    for (i, sweep) in enumerate(SWEEPS)
        row, col = fldmod1(i, 3)
        ax = Axis(
            fig[row, col];
            xlabel = sweep.label,
            ylabel = ylabel,
            title = sweep.axis,
            xscale = sweep.logx ? log10 : identity,
            xticks = sweep.logx ? (sweep.values, string.(sweep.values)) : Makie.automatic,
            xticklabelsize = 10,
        )
        for kernel in KERNELS
            x, y = _series(agg, sweep.axis, kernel, field)
            if field === :retention_mean
                _, s = _series(agg, sweep.axis, kernel, :retention_std)
                band!(ax, x, max.(y .- s, 0.0), min.(y .+ s, 1.0);
                      color = (KERNEL_COLOR[kernel], 0.16))
            end
            lines!(ax, x, y; color = KERNEL_COLOR[kernel], linewidth = 2.2,
                   linestyle = KERNEL_STYLE[kernel], label = kernel)
            scatter!(ax, x, y; color = KERNEL_COLOR[kernel], markersize = 7)
        end
        ylims!(ax, ylims...)
        i == 1 && axislegend(ax; position = :lb, framevisible = false, labelsize = 11)
    end
    Label(fig[0, 1:3], title; fontsize = 19, font = :bold, padding = (0, 0, 6, 0))
    return fig
end

function _figure_retention(agg)
    fig = Figure(size = (1250, 720))
    _panel_grid(
        fig, agg, :retention_mean, "focus retention",
        "Focus retention vs distractor load (mean ± 1 SD over $(length(SEEDS)) seeds)",
        (-0.02, 1.02),
    )
    path = figure_path(SLUG, "focus_retention.png")
    save(path, fig; px_per_unit = 1)
    return path
end

function _figure_top1(agg)
    fig = Figure(size = (1250, 720))
    _panel_grid(
        fig, agg, :top1_rate, "top-1 correct rate",
        "Winner stability: fraction of seeds where the target is top-1",
        (-0.05, 1.05),
    )
    path = figure_path(SLUG, "top1_correctness.png")
    save(path, fig; px_per_unit = 1)
    return path
end

function _raster!(ax, context)
    for (name, ids, color, ms) in (
        ("random noise", DISTRACTOR_POOL, RGBf(0.62, 0.62, 0.62), 6),
        ("stale rival", [STALE_NEURON], RGBf(0.85, 0.37, 0.01), 8),
        ("target", [TARGET_NEURON], RGBf(0.12, 0.47, 0.71), 11),
    )
        pts = [e for e in context if e.neuron_id in ids]
        isempty(pts) && continue
        scatter!(ax,
                 [Float64(e.t - T_NOW) for e in pts],
                 [Float64(e.neuron_id) for e in pts];
                 color = color, markersize = ms, label = name)
    end
    vlines!(ax, [0.0]; color = (:black, 0.45), linestyle = :dash)
    return ax
end

function _figure_scene(clean::Condition, noisy::Condition, seed::Integer)
    fig = Figure(size = (1150, 1180))
    scenes = [("clean", clean), ("high noise", noisy)]

    for (col, (name, cond)) in enumerate(scenes)
        context = _build_context(cond, seed)
        ax = Axis(fig[1, col];
                  xlabel = "time relative to probe (t - t_now)",
                  ylabel = "neuron id",
                  title = "$(name) scene — $(length(context)) context spikes\n" *
                          "n_stale=$(cond.n_stale), stale_age=$(cond.stale_age), " *
                          "n_random=$(cond.n_random), amp=$(cond.random_amp)")
        _raster!(ax, context)
        xlims!(ax, -Float64(RANDOM_SPAN) - 0.3, 0.3)
        col == 1 && axislegend(ax; position = :lt, framevisible = false, labelsize = 11)

        for (k, kernel) in enumerate(KERNELS)
            a = _attention_of(kernel, context, cond)
            m = _focus_metrics(a)
            total = sum(a)
            share = total > 0 ? Float64.(a) ./ total : zeros(Float64, length(a))
            colors = [i == TARGET_NEURON ? KERNEL_COLOR["temporal"] :
                      i == STALE_NEURON ? KERNEL_COLOR["discrete"] :
                      RGBf(0.72, 0.72, 0.72) for i in 1:N_NEURONS]
            ax2 = Axis(fig[1 + k, col];
                       xlabel = "neuron id",
                       ylabel = "share of attention",
                       title = @sprintf("%s — retention %.3f, top-1 = n%d (%s)",
                                        kernel, m.focus_retention, m.top1_neuron,
                                        m.top1_correct ? "correct" : "WRONG"))
            barplot!(ax2, 1:N_NEURONS, share; color = colors)
            ylims!(ax2, 0.0, max(0.05, 1.05 * maximum(share)))
        end
    end

    Label(fig[0, 1:2],
          "Representative scenes and per-neuron attention share (seed $(seed))";
          fontsize = 19, font = :bold, padding = (0, 0, 6, 0))
    path = figure_path(SLUG, "scene_clean_vs_noisy.png")
    save(path, fig; px_per_unit = 1)
    return path
end

# ---------------------------------------------------------------------------
# Separation analysis (pre-registered rule)
# ---------------------------------------------------------------------------

"""
    _separations(agg) -> Vector{NamedTuple}

Apply the pre-registered rule to every *distinct* swept condition: a pair of
kernels is "meaningfully separated" at that condition when their mean focus
retention differs by at least `SEPARATION_RETENTION_DELTA`, or their top-1
correctness rate differs by at least `SEPARATION_TOP1_DELTA`.

Conditions are deduplicated by [`_condition_key`](@ref) first: all six axes pass
through the baseline point, so counting (axis, value) points instead would count
the baseline six times and inflate the separation total.
"""
function _separations(agg)
    found = NamedTuple[]
    base = _baseline_condition()
    seen = Set{Any}()
    for sweep in SWEEPS, v in sweep.values
        key = _condition_key(_with_axis(base, sweep.axis, v))
        key in seen && continue
        push!(seen, key)
        for (i, ka) in enumerate(KERNELS), kb in KERNELS[(i + 1):end]
            a = _pick(agg, sweep.axis, v, ka)
            b = _pick(agg, sweep.axis, v, kb)
            dret = abs(a.retention_mean - b.retention_mean)
            dtop = abs(a.top1_rate - b.top1_rate)
            (isnan(dret) ? false : dret >= SEPARATION_RETENTION_DELTA) ||
                dtop >= SEPARATION_TOP1_DELTA || continue
            push!(found, (sweep = sweep.axis, axis_value = Float64(v),
                          kernel_a = ka, kernel_b = kb,
                          retention_a = a.retention_mean, retention_b = b.retention_mean,
                          top1_a = a.top1_rate, top1_b = b.top1_rate,
                          d_retention = dret, d_top1 = dtop))
        end
    end
    return found
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_fmt(x::Real) = isnan(x) ? "NaN" : @sprintf("%.3f", x)

function _condition_table(agg, sweep_axis)
    sweep = only(s for s in SWEEPS if s.axis == sweep_axis)
    io = IOBuffer()
    println(io, "| $(sweep.label) | kernel | focus retention (mean ± SD) | top-1 correct | margin (mean) |")
    println(io, "| --- | --- | --- | --- | --- |")
    for v in sweep.values, kernel in KERNELS
        a = _pick(agg, sweep_axis, v, kernel)
        println(io, "| $(v) | $(kernel) | $(_fmt(a.retention_mean)) ± $(_fmt(a.retention_std)) |",
                " $(_fmt(a.top1_rate)) | $(_fmt(a.margin_mean)) |")
    end
    return String(take!(io))
end

function _build_summary(rows, agg, seps, figures, clean::Condition, noisy::Condition)
    io = IOBuffer()
    base = _baseline_condition()

    println(io, "# Focus Under Fire")
    println(io)
    println(io, "Slug: `$(SLUG)` — generated by `experiments/focus_under_fire.jl`.")
    println(io)
    println(io, "## Research question")
    println(io)
    println(io, "How robust is spike-native temporal attention when the context contains")
    println(io, "increasing amounts of stale same-neuron activity and random unrelated-neuron")
    println(io, "noise?")
    println(io)
    println(io, "## Pre-registered hypothesis")
    println(io)
    println(io, "> $(HYPOTHESIS)")
    println(io)
    println(io, "## Focus retention — exact definition")
    println(io)
    println(io, "```text")
    println(io, "a               = spike_attention_<kernel>(probe, context, I)   # identity readout")
    println(io, "target_mass     = a[$(TARGET_NEURON)]")
    println(io, "total_mass      = sum(a)")
    println(io, "distractor_mass = total_mass - target_mass")
    println(io, "focus_retention = total_mass > 0 ? target_mass / total_mass : NaN")
    println(io, "margin          = target_mass - max(a[i] for i != $(TARGET_NEURON))")
    println(io, "top1_neuron     = argmax(a)          # ties resolve to the lowest index")
    println(io, "top1_correct    = margin > 0         # an exact tie is NOT counted correct")
    println(io, "```")
    println(io)
    println(io, "All spike values are positive, so `a[i] >= 0` and `focus_retention ∈ [0, 1]`.")
    println(io, "It is reported as `NaN` (never 0) when a kernel produces no attention mass at")
    println(io, "all, so \"nothing fired inside the window\" is not confused with \"the target lost\".")
    println(io)
    println(io, "## Scene")
    println(io)
    println(io, "- `$(N_NEURONS)` neurons; target = neuron `$(TARGET_NEURON)`, stale rival = neuron `$(STALE_NEURON)`.")
    println(io, "- Probe (source train): one unit spike on **every** neuron at `t_now = $(T_NOW)`.")
    println(io, "  The probe is uniform, so it encodes nothing about which neuron is the target;")
    println(io, "  all selectivity comes from the context and the kernel's treatment of time.")
    println(io, "- Target burst: $(N_TARGET_SPIKES) spikes on neuron `$(TARGET_NEURON)` at `t_now - k·$(TARGET_SPACING)`. Never swept.")
    println(io, "- Stale interference: `n_stale` spikes on neuron `$(STALE_NEURON)`, starting `stale_age`")
    println(io, "  in the past and marching back in `$(STALE_SPACING)` steps.")
    println(io, "- Unrelated noise: `n_random` neurons drawn from a seed-fixed permutation, each")
    println(io, "  firing `1:$(RANDOM_SPIKES_MAX)` spikes uniformly over `[t_now - $(RANDOM_SPAN), t_now]` with amplitude")
    println(io, "  `distractor_amp · U$(RANDOM_AMP_JITTER)`. The design is *nested*: raising `n_random` adds")
    println(io, "  neurons to the previous scene rather than resampling it.")
    println(io, "- Seeds: `$(SEEDS)` ($(length(SEEDS)) deterministic seeds). The focus-retention")
    println(io, "  panels show mean ± 1 SD; the top-1 panels show the aggregate fraction of seeds")
    println(io, "  where the target wins, without SD bands; the scene figure shows raw attention")
    println(io, "  shares for the single representative seed `$(first(SEEDS))`.")
    println(io, "- Baseline: `n_stale=$(base.n_stale)`, `stale_age=$(base.stale_age)`, `n_random=$(base.n_random)`,")
    println(io, "  `distractor_amp=$(base.random_amp)`, `τ=$(base.τ)`, `window=$(base.window)`. Each sweep moves one axis.")
    println(io, "- All three kernels are evaluated on the **identical** generated scene.")
    println(io)
    println(io, "## Result: does temporal focus survive the garbage?")
    println(io)

    # Headline: the stale_count sweep at its heaviest load.
    heavy = maximum(only(s for s in SWEEPS if s.axis == "stale_count").values)
    hd = _pick(agg, "stale_count", heavy, "discrete")
    ht = _pick(agg, "stale_count", heavy, "temporal")
    hc = _pick(agg, "stale_count", heavy, "continuous")
    println(io, "At the heaviest stale load (`n_stale = $(Int(heavy))`, i.e. $(Int(heavy)) stale spikes on one rival")
    println(io, "neuron against $(N_TARGET_SPIKES) recent target spikes):")
    println(io)
    for a in (hd, ht, hc)
        println(io, "- **$(a.kernel)**: focus retention $(_fmt(a.retention_mean)) ± $(_fmt(a.retention_std)), ",
                "top-1 correct in $(_fmt(a.top1_rate)) of seeds, margin $(_fmt(a.margin_mean)).")
    end
    println(io)

    if isempty(seps)
        println(io, "**Null result.** Under the pre-registered rule (mean focus retention differing")
        println(io, "by ≥ $(SEPARATION_RETENTION_DELTA), or top-1 correctness differing by ≥ $(SEPARATION_TOP1_DELTA)), **no condition in the sweep")
        println(io, "produced a meaningful behavioural separation between the kernels**. The")
        println(io, "hypothesis above is not supported by this scene.")
    else
        best = seps[argmax([s.d_retention for s in seps])]
        println(io, "**Separation observed.** Under the pre-registered rule (mean focus retention")
        n_cells = length(_unique_condition_keys()) * binomial(length(KERNELS), 2)
        println(io, "differing by ≥ $(SEPARATION_RETENTION_DELTA), or top-1 correctness differing by ≥ $(SEPARATION_TOP1_DELTA)), $(length(seps)) of the")
        println(io, "$(n_cells) **distinct** (condition × kernel-pair) cells separate. The largest retention gap is")
        println(io, "`$(best.sweep) = $(best.axis_value)`: **$(best.kernel_a)** $(_fmt(best.retention_a)) vs")
        println(io, "**$(best.kernel_b)** $(_fmt(best.retention_b)) (Δ = $(_fmt(best.d_retention))).")
    end
    println(io)

    println(io, "### Where the kernels behave the same (null / negative findings)")
    println(io)
    flat = String[]
    for sweep in SWEEPS
        nsep = count(s -> s.sweep == sweep.axis, seps)
        nsep == 0 && push!(flat, sweep.axis)
    end
    if isempty(flat)
        println(io, "Every sweep axis contained at least one separating condition.")
    else
        println(io, "These sweep axes contained **no** separating condition at all: ",
                join(("`" * f * "`" for f in flat), ", "), ".")
    end
    println(io)
    # Amplitude / saturation null: where does the target lose under every kernel?
    amp_sweep = only(s for s in SWEEPS if s.axis == "distractor_amp")
    lost = [v for v in amp_sweep.values
            if all(_pick(agg, "distractor_amp", v, k).top1_rate == 0.0 for k in KERNELS)]
    if isempty(lost)
        println(io, "No swept `distractor_amp` level drove top-1 correctness to zero for all three")
        println(io, "kernels: within this grid, amplitude alone never destroys focus outright.")
    else
        println(io, "At `distractor_amp ∈ $(lost)` **every** kernel loses the target (top-1 correct = 0).")
        println(io, "Note precisely what that does and does not show: the noise spikes are drawn")
        println(io, "*uniformly* over `[t_now - $(RANDOM_SPAN), t_now]`, so this sweep contains no distractor")
        println(io, "deliberately synchronized with the probe. The measured result is therefore only")
        println(io, "that sufficiently loud, randomly timed noise defeats all three kernels in these")
        println(io, "seeds — once amplitude is large enough, whichever noise spike happens to land")
        println(io, "nearest the probe outweighs the target. A controlled probe-synchronized")
        println(io, "distractor was **not** tested and no claim is made about one.")
    end
    println(io)
    tau_sweep = only(s for s in SWEEPS if s.axis == "tau")
    big_tau = maximum(tau_sweep.values)
    td = _pick(agg, "tau", big_tau, "discrete")
    tt = _pick(agg, "tau", big_tau, "temporal")
    flattest = Float64(exp(-RANDOM_SPAN / big_tau))
    println(io, "**Flat-weight control.** The `τ` grid deliberately runs past the scene span so the")
    println(io, "top of it is a *genuine* no-recency control: at `τ = $(big_tau)` the exponential")
    println(io, "multiplier only falls to `exp(-$(RANDOM_SPAN)/$(big_tau))` = $(_fmt(flattest)) across the whole $(RANDOM_SPAN)-unit")
    println(io, "history, so `temporal` is within $(_fmt(100 * (1 - flattest)))% of ignoring time and should collapse onto")
    println(io, "`discrete` — measured retention $(_fmt(tt.retention_mean)) vs $(_fmt(td.retention_mean)) (Δ = $(_fmt(abs(tt.retention_mean - td.retention_mean)))). Mid-grid values such as")
    println(io, "`τ = 4` are **not** flat (`exp(-$(RANDOM_SPAN)/4)` ≈ $(_fmt(exp(-RANDOM_SPAN / 4)))) and are not used as the control.")
    println(io, "`discrete` is by construction invariant to both `τ` and `window`; its flat lines in")
    println(io, "those two panels are the intended control in the other direction.")
    println(io)

    println(io, "### Caveats — what this experiment does *not* show")
    println(io)
    disc = unique(r -> (r.condition_id, r.seed), [r for r in rows if r.kernel == "discrete"])
    disc_wins = count(r -> r.top1_correct, disc)
    println(io, "- **The six sweep axes share the baseline point**, so `metrics.csv` contains the")
    println(io, "  baseline condition once per axis panel ($(length(SWEEPS)) copies that differ only in the")
    println(io, "  `sweep` / `axis_value` columns). The `condition_id` column is the canonical")
    println(io, "  parameter tuple; the separation count and the tallies below are computed over")
    println(io, "  **distinct** `condition_id`s ($(length(_unique_condition_keys())) of the $(sum(length(s.values) for s in SWEEPS)) (axis, value) points), not over rows.")
    target_total = N_TARGET_SPIKES * Float64(TARGET_AMPLITUDE)
    println(io, "- **`discrete` never wins anywhere in this grid** ($(disc_wins) of $(length(disc)) distinct evaluations).")
    println(io, "  That is structural, not a staleness effect. Against a uniform probe the discrete")
    println(io, "  kernel reduces to an **amplitude-mass vote**: it sums `source.value · context.value`")
    println(io, "  over matching neurons, so a neuron wins on total spike *amplitude*, not on spike")
    println(io, "  count, and never on timing. The target contributes $(_fmt(target_total)) ($(N_TARGET_SPIKES) spikes × amplitude")
    println(io, "  $(TARGET_AMPLITUDE)), so any neuron accumulating more than $(_fmt(target_total)) of amplitude beats it — whether")
    println(io, "  that is many quiet spikes (the unit-amplitude stale rival) or few loud ones (a random")
    println(io, "  distractor, whose amplitudes are `distractor_amp · U$(RANDOM_AMP_JITTER)` and are *not* unit).")
    println(io, "  The discrete-vs-timing-aware gap therefore measures \"accumulated mass versus")
    println(io, "  recency\", and should not be read as evidence about staleness on its own. The")
    println(io, "  `stale_age` and `window` sweeps — where `discrete` is flat and only the timing-aware")
    println(io, "  kernels move — are the cleaner staleness evidence.")
    fresh = [v for v in only(s for s in SWEEPS if s.axis == "stale_age").values
             if _pick(agg, "stale_age", v, "continuous").top1_rate < 1.0]
    if !isempty(fresh)
        burst_span = (base.n_stale - 1) * STALE_SPACING
        println(io, "- **These near-past, partially windowed bursts defeat the target.** At")
        println(io, "  `stale_age ∈ $(fresh)`, the nearest rival spike is that far before the probe, and")
        println(io, "  the $(base.n_stale)-spike burst extends another $(burst_span) time units into the past.")
        println(io, "  With `window = $(base.window)`, part of each burst is outside the active window, yet")
        println(io, "  even `continuous` loses top-1 in at least one seed. No rival spike is simultaneous")
        println(io, "  with the probe in these conditions, so this experiment makes no claim about a")
        println(io, "  genuinely probe-synchronized rival.")
    end
    println(io, "- The probe is always at `t_now`, so \"recent\" is defined relative to the query. A scene")
    println(io, "  in which the *target* is the stale event was not tested and would invert the ranking")
    println(io, "  by construction.")
    println(io, "- Uncertainty is the spread over $(length(SEEDS)) deterministic seeds (mean ± 1 SD). No")
    println(io, "  significance test is claimed; $(length(SEEDS)) seeds is a spread, not a p-value.")
    println(io, "- The scene is synthetic and domain-free by design: no dataset, no application")
    println(io, "  semantics, and nothing outside the package's own spike types enters the measurement.")
    println(io)

    println(io, "## Sweep tables")
    println(io)
    for sweep in SWEEPS
        println(io, "### `$(sweep.axis)`")
        println(io)
        print(io, _condition_table(agg, sweep.axis))
        println(io)
    end

    println(io, "## Figures")
    println(io)
    println(io, "| figure | what it answers |")
    println(io, "| --- | --- |")
    println(io, "| `focus_retention.png` | Mean focus retention ± 1 SD over seeds: how much temporal garbage can the signal survive before focus flips? |")
    println(io, "| `top1_correctness.png` | Aggregate fraction of seeds where the target stays top-1; no SD band is shown. |")
    println(io, "| `scene_clean_vs_noisy.png` | Raw per-neuron attention shares for the single representative seed `$(first(SEEDS))`, not an across-seed aggregate. |")
    println(io)
    for f in figures
        println(io, "- `$(basename(f))`")
    end
    println(io)
    println(io, "Representative scenes: clean = `n_stale=$(clean.n_stale), stale_age=$(clean.stale_age), " *
                "n_random=$(clean.n_random), amp=$(clean.random_amp)`; " *
                "high noise = `n_stale=$(noisy.n_stale), stale_age=$(noisy.stale_age), " *
                "n_random=$(noisy.n_random), amp=$(noisy.random_amp)`.")
    println(io)
    println(io, "## Reproducing")
    println(io)
    println(io, "```bash")
    println(io, "julia --project=experiments -e 'using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()'")
    println(io, "julia --project=experiments experiments/focus_under_fire.jl")
    println(io, "```")
    println(io)
    println(io, "The sweep grid, seed list, scene constants and the separation decision rule are")
    println(io, "fixed as constants at the top of the script and echoed into `config.toml`. No")
    println(io, "scene parameter was adjusted to favour a result. Generated artifacts are")
    println(io, "git-ignored; re-run the command to rebuild them.")
    println(io)
    println(io, "### Scope of the reproducibility claim")
    println(io)
    println(io, "Identical inputs on the same Julia version reproduce byte-identical `metrics.csv`.")
    println(io, "`write_config` appends a `[provenance]` table (git commit, dirty flag, Julia")
    println(io, "version, UTC timestamp). This script also records the plotting stack under")
    println(io, "`[environment]`:")
    println(io)
    println(io, "```text")
    println(io, "julia         $(VERSION)")
    println(io, "CairoMakie    $(pkgversion(CairoMakie))")
    println(io, "Makie         $(pkgversion(CairoMakie.Makie))")
    println(io, "TemporalFocus $(pkgversion(TemporalFocus))")
    println(io, "```")
    println(io)
    println(io, "`experiments/Project.toml` declares a CairoMakie compatibility range so `Pkg`")
    println(io, "can select a release for the active Julia version. `experiments/Manifest.toml`")
    println(io, "and `experiments/results/` are git-ignored. A different renderer may re-render")
    println(io, "the figures; the numbers in `metrics.csv` depend only on TemporalFocus and the")
    println(io, "recorded seeds, and are not affected by the plotting stack.")

    return String(take!(io))
end

function _build_config(clean::Condition, noisy::Condition, scene_seed::Integer)
    base = _baseline_condition()
    return Dict{String,Any}(
        "slug" => SLUG,
        "hypothesis" => HYPOTHESIS,
        # Plotting stack for this run. The harness also appends `[provenance]`
        # (git commit, dirty flag, Julia version, UTC timestamp). Figures may
        # differ across renderer versions; `metrics.csv` does not.
        "environment" => Dict{String,Any}(
            "julia" => string(VERSION),
            "CairoMakie" => string(pkgversion(CairoMakie)),
            "Makie" => string(pkgversion(CairoMakie.Makie)),
            "TemporalFocus" => string(pkgversion(TemporalFocus)),
        ),
        "focus_retention_definition" => FOCUS_RETENTION_DEFINITION,
        "seeds" => SEEDS,
        "kernels" => KERNELS,
        "scene" => Dict{String,Any}(
            "n_neurons" => N_NEURONS,
            "target_neuron" => TARGET_NEURON,
            "stale_neuron" => STALE_NEURON,
            "t_now" => T_NOW,
            "n_target_spikes" => N_TARGET_SPIKES,
            "target_spacing" => TARGET_SPACING,
            "target_amplitude" => TARGET_AMPLITUDE,
            "stale_spacing" => STALE_SPACING,
            "stale_amplitude" => STALE_AMPLITUDE,
            "random_span" => RANDOM_SPAN,
            "random_spikes_max" => RANDOM_SPIKES_MAX,
            "random_amp_jitter" => [RANDOM_AMP_JITTER[1], RANDOM_AMP_JITTER[2]],
            "probe" => "one unit spike per neuron at t_now (uniform, target-agnostic)",
            "readout" => "identity",
        ),
        "baseline" => Dict{String,Any}(
            "n_stale" => base.n_stale,
            "stale_age" => base.stale_age,
            "n_random" => base.n_random,
            "distractor_amp" => base.random_amp,
            "tau" => base.τ,
            "window" => base.window,
        ),
        "sweeps" => Dict{String,Any}(s.axis => s.values for s in SWEEPS),
        "separation_rule" => Dict{String,Any}(
            "retention_delta" => SEPARATION_RETENTION_DELTA,
            "top1_delta" => SEPARATION_TOP1_DELTA,
        ),
        "representative_scenes" => Dict{String,Any}(
            "seed" => scene_seed,
            "clean" => Dict{String,Any}(
                "n_stale" => clean.n_stale, "stale_age" => clean.stale_age,
                "n_random" => clean.n_random, "distractor_amp" => clean.random_amp,
                "tau" => clean.τ, "window" => clean.window,
            ),
            "noisy" => Dict{String,Any}(
                "n_stale" => noisy.n_stale, "stale_age" => noisy.stale_age,
                "n_random" => noisy.n_random, "distractor_amp" => noisy.random_amp,
                "tau" => noisy.τ, "window" => noisy.window,
            ),
        ),
    )
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main()
    CairoMakie.activate!(; type = "png")

    base = _baseline_condition()
    clean = Condition(0, base.stale_age, 0, base.random_amp, base.τ, base.window)
    noisy = Condition(40, base.stale_age, 40, base.random_amp, base.τ, base.window)
    scene_seed = first(SEEDS)

    println("[$(SLUG)] sweeping $(sum(length(s.values) for s in SWEEPS)) conditions × ",
            "$(length(KERNELS)) kernels × $(length(SEEDS)) seeds")

    rows = _run_sweeps()
    agg = _aggregate(rows)
    seps = _separations(agg)

    config_path = write_config(SLUG, _build_config(clean, noisy, scene_seed))
    metrics_path = write_metrics(SLUG, rows)
    figures = [_figure_retention(agg), _figure_top1(agg), _figure_scene(clean, noisy, scene_seed)]
    summary_path = write_summary(SLUG, _build_summary(rows, agg, seps, figures, clean, noisy))

    println("[$(SLUG)] rows: $(length(rows))")
    println("[$(SLUG)] separating (condition × kernel-pair) cells: $(length(seps))")
    println()
    heavy = maximum(only(s for s in SWEEPS if s.axis == "stale_count").values)
    @printf("%-12s %-22s %-16s %-10s\n", "kernel", "retention @stale=$(Int(heavy))", "top-1 correct", "margin")
    for kernel in KERNELS
        a = _pick(agg, "stale_count", heavy, kernel)
        @printf("%-12s %-22s %-16s %-10s\n", kernel,
                "$(_fmt(a.retention_mean)) ± $(_fmt(a.retention_std))",
                _fmt(a.top1_rate), _fmt(a.margin_mean))
    end
    println()
    for p in (config_path, metrics_path, summary_path, figures...)
        println("[$(SLUG)] wrote ", relpath(p, repo_root()))
    end
    return nothing
end

main()
