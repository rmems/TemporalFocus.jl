# SPDX-License-Identifier: MIT OR Apache-2.0
# Experiment: Temporal Lens — sweep Δt × τ and visualize the recency field.
# Run: julia --project=experiments experiments/temporal_lens.jl
#
# Research question
#   How does the temporal attention contribution change jointly with spike
#   separation Δt and time constant τ?
#
# The sweep is a pure characterization of `temporal_weight`; no part of this
# script changes or wraps the package API.

using CairoMakie
using ExperimentUtils
using Printf
using Random
using TemporalFocus

const SLUG = "temporal_lens"

# The sweep consumes no randomness — every sample is a fixed grid point — but
# the seed is set and recorded so the run stays reproducible by contract.
const SEED = 20260823

# Δt spans ±1.2 s so that even the longest τ decays visibly inside the window.
# It is a spike separation, so it is stored in Float32 like every other temporal
# quantity, and widened explicitly where the analysis needs Float64.
const DT_LIMIT = 1.2f0
# τ spans three orders of magnitude: 10 ms (coincidence) to 10 s (long context).
const TAU_LOG10_MIN = -2.0
const TAU_LOG10_MAX = 1.0

# Dense field for the figures and for the numerical checks.
const FIELD_DT_N = 481
const FIELD_TAU_N = 241
# Coarser sweep for the machine-readable error table (121 × 21 = 2541 rows).
const SWEEP_DT_N = 121
const SWEEP_TAU_N = 21

# Representative short-, medium-, and long-memory time constants. The same
# colors identify them in both figures.
const TAU_MARKS = (
    (label = "short", τ = 0.02f0, color = :deepskyblue),
    (label = "medium", τ = 0.2f0, color = :darkorange),
    (label = "long", τ = 2.0f0, color = :mediumseagreen),
)

# Iso-weight contours; these are the edges of the focus cone.
const CONTOUR_LEVELS = [0.01f0, 0.1f0, 0.5f0, 0.9f0]

# Reference lag used in the summary to make τ concrete.
const REFERENCE_LAG = 0.05f0

"""
    _dt_grid(n) -> Vector{Float32}

Symmetric grid of spike separations covering negative and positive lag. The grid
is laid out in `Float64` and rounded before narrowing, so the sampled
separations are the clean decimals they look like rather than accumulated
`Float32` range error.
"""
_dt_grid(n::Integer) =
    Float32.(round.(range(-Float64(DT_LIMIT), Float64(DT_LIMIT); length = n), digits = 6))

"""
    _tau_grid(n) -> Vector{Float32}

Logarithmically spaced time constants spanning `TAU_LOG10_MIN:TAU_LOG10_MAX`.
"""
_tau_grid(n::Integer) = Float32.(exp10.(range(TAU_LOG10_MIN, TAU_LOG10_MAX; length = n)))

"""
    _analytic_weight(dt, τ) -> Float64

Independent reference for the recency law. It stays in `Float64` all the way
through the error arithmetic: narrowing it back to `Float32` first would hide
the kernel's own final rounding, since both values would land on the same
`Float32` and report zero error.
"""
_analytic_weight(dt::Float32, τ::Float32) = exp(-abs(Float64(dt)) / Float64(τ))

"""
    _weight_field(dts, taus) -> Matrix{Float32}

Evaluate `temporal_weight` over the full `Δt × τ` grid.
"""
function _weight_field(dts::AbstractVector{Float32}, taus::AbstractVector{Float32})
    field = Matrix{Float32}(undef, length(dts), length(taus))
    for (j, τ) in enumerate(taus), (i, dt) in enumerate(dts)
        field[i, j] = temporal_weight(dt, τ)
    end
    return field
end

"""
    _analytic_field(dts, taus) -> Matrix{Float64}

Same grid as [`_weight_field`](@ref), evaluated with [`_analytic_weight`](@ref).
"""
function _analytic_field(dts::AbstractVector{Float32}, taus::AbstractVector{Float32})
    field = Matrix{Float64}(undef, length(dts), length(taus))
    for (j, τ) in enumerate(taus), (i, dt) in enumerate(dts)
        field[i, j] = _analytic_weight(dt, τ)
    end
    return field
end

"""
    _symmetry_deviation(dts, taus) -> Float32

Largest `|w(Δt, τ) - w(-Δt, τ)|` over the grid. The recency law depends on
`abs(Δt)`, so this must be exactly zero.
"""
function _symmetry_deviation(dts::AbstractVector{Float32}, taus::AbstractVector{Float32})
    worst = 0.0f0
    for τ in taus, dt in dts
        worst = max(worst, abs(temporal_weight(dt, τ) - temporal_weight(-dt, τ)))
    end
    return worst
end

"""
    _ratio_deviation(taus, ratios) -> Float32

Largest deviation from the ratio law: for a fixed `r = |Δt| / τ` the weight
must be `exp(-r)` regardless of the absolute time scale.
"""
function _ratio_deviation(taus::AbstractVector{Float32}, ratios)
    worst = 0.0f0
    for r in ratios, τ in taus
        expected = Float32(exp(-Float64(r)))
        worst = max(worst, abs(temporal_weight(Float32(r) * τ, τ) - expected))
    end
    return worst
end

"""
    _rel_error_bound(r) -> Float64

First-order bound on the relative error of `temporal_weight` at `r = |Δt| / τ`.

Two roundings contribute. Forming the quotient in `Float32` perturbs `r` by up
to `eps / 2` relatively, i.e. by `r · eps / 2` in absolute terms, and since
`d(exp(-x))/exp(-x) = -dx` that lands on the result multiplied by `r`. Rounding
the `Float32` result of `exp` itself then adds up to another `eps / 2`. Hence
`(r + 1) · eps(Float32) / 2` — dropping the final term would understate the
bound near `r = 0`, where measured errors of about `1.1e-7` really do occur.
"""
_rel_error_bound(r::Real) = (Float64(r) + 1) * eps(Float32) / 2

"""
    _error_stats(dts, taus, measured, reference) -> NamedTuple

Compare the `Float32` field against the `Float64` reference field, in `Float64`.

Grid points are bucketed by what the `Float32` result can still represent,
because that — not the kernel — sets the achievable accuracy:

- *normal*: the weight is a normal `Float32`, so relative error is a real
  measurement of the kernel. `worst_ratio` is the `|Δt| / τ` where it peaks and
  `rel_bound` is [`_rel_error_bound`](@ref) there. `bound_headroom` is the
  largest measured-error-to-bound ratio anywhere in this bucket: at most `1`
  means the bound held over the whole grid, not just at its worst point.
- *subnormal*: the weight has lost significand bits, so relative error is
  bounded by the float layout.
- *underflowed*: the weight is exactly zero against a tiny but non-zero
  reference, so relative error is `1` by definition.
"""
function _error_stats(
    dts::AbstractVector{Float32},
    taus::AbstractVector{Float32},
    measured::AbstractMatrix{Float32},
    reference::AbstractMatrix{Float64},
)
    max_abs = 0.0
    max_rel_normal = 0.0
    max_rel_subnormal = 0.0
    worst_ratio = 0.0
    bound_headroom = 0.0
    normal = 0
    subnormal = 0
    underflow = 0
    for (j, τ) in enumerate(taus), (i, dt) in enumerate(dts)
        weight = Float64(measured[i, j])
        ref = reference[i, j]
        err = abs(weight - ref)
        max_abs = max(max_abs, err)
        rel = ref > 0.0 ? err / ref : 0.0
        if weight == 0.0
            underflow += 1
        elseif weight < Float64(floatmin(Float32))
            subnormal += 1
            max_rel_subnormal = max(max_rel_subnormal, rel)
        else
            normal += 1
            ratio = abs(Float64(dt)) / Float64(τ)
            bound_headroom = max(bound_headroom, rel / _rel_error_bound(ratio))
            if rel > max_rel_normal
                max_rel_normal = rel
                worst_ratio = ratio
            end
        end
    end
    return (
        max_abs = max_abs,
        max_rel_normal = max_rel_normal,
        max_rel_subnormal = max_rel_subnormal,
        worst_ratio = worst_ratio,
        rel_bound = _rel_error_bound(worst_ratio),
        bound_headroom = bound_headroom,
        normal = normal,
        subnormal = subnormal,
        underflow = underflow,
    )
end

"""
    _metric_rows(dts, taus) -> Vector{NamedTuple}

One row per sweep point: separation and time constant in `Float32` as the
kernel sees them, the measured `Float32` weight, and the `Float64` analytic
weight with the errors against it (relative error is `NaN` only if the analytic
weight itself underflows `Float64`, which this grid never reaches).
"""
function _metric_rows(dts::AbstractVector{Float32}, taus::AbstractVector{Float32})
    rows = Vector{
        NamedTuple{
            (:dt, :tau, :weight, :analytic, :abs_error, :rel_error),
            Tuple{Float32, Float32, Float32, Float64, Float64, Float64},
        },
    }()
    sizehint!(rows, length(dts) * length(taus))
    for τ in taus, dt in dts
        weight = temporal_weight(dt, τ)
        analytic = _analytic_weight(dt, τ)
        abs_error = abs(Float64(weight) - analytic)
        rel_error = analytic > 0.0 ? abs_error / analytic : NaN
        push!(rows, (
            dt = dt,
            tau = τ,
            weight = weight,
            analytic = analytic,
            abs_error = abs_error,
            rel_error = rel_error,
        ))
    end
    return rows
end

"""
    _lag_at_weight(τ, w) -> Float64

Separation at which the weight has fallen to `w`: `|Δt| = τ * log(1 / w)`.
"""
_lag_at_weight(τ::Real, w::Real) = Float64(τ) * log(1 / Float64(w))

"""
    _widen(x) -> Float64

Widen a `Float32` sweep parameter for display and layout without dragging the
binary32 representation error along: `Float64(1.2f0)` is `1.2000000476837158`,
which reads badly in `config.toml` and renders as an unusable axis tick label.
Six digits is far beyond `Float32` precision, so nothing meaningful is lost.
"""
_widen(x::Real) = round(Float64(x), digits = 6)

"""
    _dt_ticks()

Axis ticks across the Δt window, widened to the clean decimals a reader expects
to see on an axis.
"""
_dt_ticks() = range(-_widen(DT_LIMIT), _widen(DT_LIMIT); step = 0.4)

"""
    _window_contrast(τ) -> Float64

How much more a zero-lag spike is worth than one at the edge of the observation
window, `w(0) / w(DT_LIMIT) = exp(DT_LIMIT / τ)`. This is the honest measure of
"does τ still rank context by time inside this window": it is astronomically
large for short τ and approaches 1 as τ grows past the window.
"""
_window_contrast(τ::Real) = exp(Float64(DT_LIMIT) / Float64(τ))

"""
    _render_lens(dts, taus, field, path) -> String

Heatmap of the recency field with the focus cone annotated. `τ` is drawn on a
`log10` axis so three decades fit in one view; the iso-weight contours are the
walls of the lens.
"""
function _render_lens(
    dts::AbstractVector{Float32},
    taus::AbstractVector{Float32},
    field::AbstractMatrix{Float32},
    path::AbstractString,
)
    log_taus = log10.(taus)

    fig = Figure(size = (1100, 700))
    ax = Axis(
        fig[1, 1],
        title = "Temporal lens: the recency field of temporal_weight(Δt, τ)",
        subtitle = "Attention depends only on |Δt| / τ — τ sets how wide the focus cone opens",
        xlabel = "spike separation Δt (s)",
        ylabel = "time constant τ (s, log scale)",
        xticks = _dt_ticks(),
        yticks = ([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
                  ["0.01", "0.032", "0.1", "0.32", "1.0", "3.2", "10"]),
    )

    plot = heatmap!(ax, dts, log_taus, field; colormap = :magma, colorrange = (0.0f0, 1.0f0))
    contour!(
        ax,
        dts,
        log_taus,
        field;
        levels = CONTOUR_LEVELS,
        color = (:white, 0.65),
        linewidth = 1.2,
    )
    vlines!(ax, [0.0]; color = (:white, 0.5), linestyle = :dot, linewidth = 1.0)

    for mark in TAU_MARKS
        y = log10(mark.τ)
        hlines!(ax, [y]; color = mark.color, linestyle = :dash, linewidth = 2.0)
        text!(
            ax,
            -_widen(DT_LIMIT) + 0.03,
            y + 0.06;
            text = @sprintf("%s memory  τ = %.3g s  (half weight at ±%.0f ms)",
                            mark.label, mark.τ, 1000 * _lag_at_weight(mark.τ, 0.5)),
            color = mark.color,
            fontsize = 13,
            align = (:left, :bottom),
        )
    end

    Colorbar(fig[1, 2], plot; label = "temporal_weight(Δt, τ)")
    Label(
        fig[2, 1:2],
        "White contours: weight = " * join(string.(CONTOUR_LEVELS), ", ") *
        ". Every contour satisfies |Δt| = τ · ln(1/w), so the cone widens in " *
        "direct proportion to τ.",
        fontsize = 12,
        padding = (0, 0, 0, 6),
    )

    save(path, fig; px_per_unit = 1.5)
    return path
end

"""
    _render_curves(dts, path) -> String

Decay curves for the representative τ values, linear and semi-logarithmic.
"""
function _render_curves(dts::AbstractVector{Float32}, path::AbstractString)
    # The log panel shows this many decades; curves keep their true values and
    # simply leave the panel below it, rather than being clamped onto a floor
    # that would fake a flat tail.
    log_floor = 1.0e-9

    fig = Figure(size = (1100, 480))
    ax_lin = Axis(
        fig[1, 1],
        title = "Decay curves",
        xlabel = "spike separation Δt (s)",
        ylabel = "temporal_weight",
        xticks = _dt_ticks(),
    )
    ax_log = Axis(
        fig[1, 2],
        title = "Same curves, log weight (|slope| = 1/τ)",
        xlabel = "spike separation Δt (s)",
        ylabel = "temporal_weight (log scale)",
        xticks = _dt_ticks(),
        yscale = log10,
    )

    for mark in TAU_MARKS
        weights = [temporal_weight(dt, mark.τ) for dt in dts]
        label = @sprintf("%s: τ = %.3g s", mark.label, mark.τ)
        lines!(ax_lin, dts, weights; color = mark.color, linewidth = 2.5, label = label)
        # A log axis cannot show an underflowed zero; NaN leaves a gap instead
        # of inventing a value. Every other point is plotted unmodified.
        lines!(ax_log, dts, [w > 0.0f0 ? Float64(w) : NaN for w in weights];
               color = mark.color, linewidth = 2.5, label = label)
    end

    ylims!(ax_lin, -0.02, 1.05)
    ylims!(ax_log, log_floor, 2.0)
    axislegend(ax_lin; position = :rt, framevisible = true)
    Label(
        fig[2, 1:2],
        @sprintf("Log panel clipped at %.0e; the short-τ curve keeps falling below the panel \
                  (it reaches %.1e at Δt = %.1f s).",
                 log_floor,
                 temporal_weight(DT_LIMIT, first(TAU_MARKS).τ),
                 DT_LIMIT),
        fontsize = 12,
        padding = (0, 0, 0, 6),
    )

    save(path, fig; px_per_unit = 1.5)
    return path
end

"""
    _build_summary(...) -> String

Generated interpretation of the observed regimes.
"""
function _build_summary(;
    dts_field,
    taus_field,
    dts_sweep,
    taus_sweep,
    stats,
    symmetry,
    ratio_dev,
    exact_fraction,
    figure_file,
    curves_file,
    supported,
)
    io = IOBuffer()
    println(io, "# Temporal Lens — sweep Δt × τ and visualize the recency field")
    println(io)
    println(io, """
Characterization of `temporal_weight(Δt, τ) = exp(-|Δt| / τ)` over a symmetric
grid of spike separations and three decades of time constants. Generated by
`experiments/temporal_lens.jl`; the package API is unchanged.""")
    println(io)
    println(io, "## Hypothesis")
    println(io)
    println(io, """
Attention is governed by the ratio `|Δt| / τ`. Small τ should create a narrow
focus around `Δt = 0`; large τ should keep weight over longer delays; the field
should be exactly symmetric under `Δt → -Δt`; and the Float32 kernel should
match `exp(-|Δt|/τ)` bit for bit, with every normal-range relative error inside
the first-order bound `(r + 1) · eps(Float32) / 2`.""")
    println(io)
    println(io, "## Sweep")
    println(io)
    @printf(io, "- Δt: %d points on [%.2f, %.2f] s (figures and checks), %d points for the error table\n",
            length(dts_field), -DT_LIMIT, DT_LIMIT, length(dts_sweep))
    @printf(io, "- τ: %d log-spaced points on [%.3g, %.3g] s (figures and checks), %d for the error table\n",
            length(taus_field), first(taus_field), last(taus_field), length(taus_sweep))
    @printf(io, "- grid points evaluated: %d\n", length(dts_field) * length(taus_field))
    println(io)
    println(io, "## What τ means in concrete temporal terms")
    println(io)
    @printf(io, "| regime | τ | half weight (w = 0.5) | 10%% weight (w = 0.1) | 1%% weight (w = 0.01) | weight at Δt = %.0f ms | contrast w(0)/w(%.1f s) |\n",
            1000 * REFERENCE_LAG, DT_LIMIT)
    println(io, "|--------|---|----------------------|----------------------|----------------------|----------------------|------------------------|")
    for mark in TAU_MARKS
        @printf(io, "| %s memory | %.3g s | ±%.1f ms | ±%.1f ms | ±%.1f ms | %.4f | %.3g× |\n",
                mark.label,
                mark.τ,
                1000 * _lag_at_weight(mark.τ, 0.5),
                1000 * _lag_at_weight(mark.τ, 0.1),
                1000 * _lag_at_weight(mark.τ, 0.01),
                temporal_weight(REFERENCE_LAG, mark.τ),
                _window_contrast(mark.τ))
    end
    println(io)
    short_mark, medium_mark, long_mark = TAU_MARKS
    @printf(io, """
Read the table as a memory span. With **τ = %.3g s** a context spike only counts
while it lands inside roughly ±%.0f ms of the source spike — the kernel behaves
like a coincidence detector, and a %.0f ms lag already costs it %.1f%% of its
contribution. With **τ = %.3g s** the same %.0f ms lag costs only %.1f%%, so a whole
burst of recent spikes still contributes; this is a working-memory-width focus.
With **τ = %.3g s** a spike at the ±%.1f s edge of the window still carries
weight %.3f, so the near/far contrast across the whole window is only %.3g× —
against %.3g× at τ = %.3g s. Time still ranks context at this τ, but weakly, and
the ranking keeps flattening as τ grows past the window.
""",
            short_mark.τ,
            1000 * _lag_at_weight(short_mark.τ, 0.5),
            1000 * REFERENCE_LAG,
            100 * (1 - temporal_weight(REFERENCE_LAG, short_mark.τ)),
            medium_mark.τ,
            1000 * REFERENCE_LAG,
            100 * (1 - temporal_weight(REFERENCE_LAG, medium_mark.τ)),
            long_mark.τ,
            DT_LIMIT,
            temporal_weight(DT_LIMIT, long_mark.τ),
            _window_contrast(long_mark.τ),
            _window_contrast(medium_mark.τ),
            medium_mark.τ)
    println(io)
    println(io, "## Observed regimes")
    println(io)
    println(io, """
1. **Ratio law.** The field is a function of `|Δt| / τ` alone, so the heatmap is
   self-similar along rays from the origin. Every iso-weight contour is the line
   `|Δt| = τ · ln(1/w)`, which is why the focus cone opens in direct proportion
   to τ.
2. **Short τ — narrow lens.** Weight collapses within a few multiples of τ; the
   usable window is `±τ · ln(10) ≈ ±2.3τ` down to 10% weight.
3. **Long τ — flat field.** Once τ exceeds the observation window the weight
   tends to 1 everywhere and temporal ordering carries almost no signal.
4. **Symmetry.** Past and future context are treated identically: the kernel
   weights `abs(Δt)`, so it is a recency filter, not a causal one.""")
    println(io)
    println(io, "## Numerical checks")
    println(io)
    @printf(io, "- symmetry `max |w(Δt,τ) - w(-Δt,τ)|` = %.3g (exact zero expected)\n", symmetry)
    @printf(io, "- ratio law `max |w(rτ,τ) - exp(-r)|` = %.3g\n", ratio_dev)
    @printf(io, "- max absolute error vs the `Float64` reference = %.3g\n", stats.max_abs)
    @printf(io, "- max relative error over normal `Float32` weights = %.3g at |Δt|/τ = %.1f\n",
            stats.max_rel_normal, stats.worst_ratio)
    @printf(io, "- first-order bound at that ratio, `(r + 1) · eps(Float32) / 2` = %.3g\n", stats.rel_bound)
    @printf(io, "- worst measured-error-to-bound ratio anywhere on the grid = %.3f (≤ 1 ⇒ the bound holds everywhere)\n",
            stats.bound_headroom)
    @printf(io, "- max relative error over subnormal `Float32` weights = %.3g\n",
            stats.max_rel_subnormal)
    @printf(io, "- grid points matching `exp(-abs(dt)/tau)` bit for bit in `Float32`: %.2f%%\n",
            100 * exact_fraction)
    @printf(io, "- grid points by `Float32` range: %d normal, %d subnormal, %d underflowed to zero\n",
            stats.normal, stats.subnormal, stats.underflow)
    println(io)
    println(io, """
Errors are taken against a `Float64` evaluation of the law and computed in
`Float64`, so the kernel's own final rounding is included rather than absorbed
by rounding the reference.

The kernel reproduces `exp(-abs(dt)/tau)` in `Float32` bit for bit, so it loses
nothing against the naive same-precision formula. Its residual error is not flat
at `eps(Float32)` either: rounding the ratio `r = |Δt| / τ` to `Float32` costs
`eps / 2` relatively and `exp` amplifies that by `r`, on top of the `eps / 2`
from rounding `exp`'s own `Float32` result. Relative accuracy
therefore degrades in proportion to how many time constants apart the spikes
are — harmless in practice, because the weights that lose relative precision are
the ones already indistinguishable from zero in an attention sum. That
degradation ends in the float layout itself: `exp(-x)` leaves the `Float32`
normal range near `x ≈ 87.3` and reaches zero near `x ≈ 103.9`, so past roughly
104 τ of separation the weight is simply zero.""")
    println(io)
    println(io, "## Artifacts")
    println(io)
    println(io, "- `", figure_file, "` — the recency field with the focus cone annotated")
    println(io, "- `", curves_file, "` — decay curves for the representative τ values")
    println(io, "- `metrics.csv` — per-point measured/analytic weights and errors")
    println(io, "- `config.toml` — the exact sweep parameters")
    println(io)
    println(io, "## Verdict")
    println(io)
    println(
        io,
        "The observation **$(supported ? "supports" : "does not support")** the hypothesis.",
    )
    println(io)
    println(io, "## Reproduce")
    println(io)
    println(io, "```bash")
    println(io, "julia --project=experiments experiments/temporal_lens.jl")
    println(io, "```")
    return String(take!(io))
end

function main()
    # The sweep is a fixed grid; the seed is constructed and recorded so the
    # run satisfies the harness determinism contract.
    Random.seed!(SEED)
    CairoMakie.activate!(type = "png")

    dts_field = _dt_grid(FIELD_DT_N)
    taus_field = _tau_grid(FIELD_TAU_N)
    dts_sweep = _dt_grid(SWEEP_DT_N)
    taus_sweep = _tau_grid(SWEEP_TAU_N)

    field = _weight_field(dts_field, taus_field)
    reference = _analytic_field(dts_field, taus_field)
    stats = _error_stats(dts_field, taus_field, field, reference)
    symmetry = _symmetry_deviation(dts_field, taus_field)
    ratio_dev = _ratio_deviation(taus_field, (0.5f0, 1.0f0, 2.0f0, 5.0f0))

    exact = count(
        temporal_weight(dt, τ) === exp(-abs(dt) / τ)
        for τ in taus_field, dt in dts_field
    )
    exact_fraction = exact / (length(dts_field) * length(taus_field))
    supported = iszero(symmetry) && exact_fraction == 1 && stats.bound_headroom <= 1

    figure_file = "figure.png"
    curves_file = "decay_curves.png"
    lens_path = _render_lens(dts_field, taus_field, field, figure_path(SLUG, figure_file))
    curves_path = _render_curves(dts_field, figure_path(SLUG, curves_file))

    rows = _metric_rows(dts_sweep, taus_sweep)
    metrics_path = write_metrics(SLUG, rows)

    config_path = write_config(SLUG, Dict(
        "slug" => SLUG,
        "seed" => SEED,
        "dt_limit_s" => _widen(DT_LIMIT),
        "dt_points_field" => FIELD_DT_N,
        "dt_points_sweep" => SWEEP_DT_N,
        "tau_log10_min" => TAU_LOG10_MIN,
        "tau_log10_max" => TAU_LOG10_MAX,
        "tau_points_field" => FIELD_TAU_N,
        "tau_points_sweep" => SWEEP_TAU_N,
        "tau_marks_s" => [_widen(mark.τ) for mark in TAU_MARKS],
        "contour_levels" => _widen.(CONTOUR_LEVELS),
        "reference_lag_s" => _widen(REFERENCE_LAG),
    ))

    summary = _build_summary(;
        dts_field,
        taus_field,
        dts_sweep,
        taus_sweep,
        stats,
        symmetry,
        ratio_dev,
        exact_fraction,
        figure_file,
        curves_file,
        supported,
    )
    summary_path = write_summary(SLUG, summary)

    println("Temporal Lens — Δt × τ recency field")
    @printf("  grid: %d Δt × %d τ = %d evaluations\n",
            length(dts_field), length(taus_field), length(dts_field) * length(taus_field))
    @printf("  Δt range: [%.2f, %.2f] s   τ range: [%.3g, %.3g] s\n",
            first(dts_field), last(dts_field), first(taus_field), last(taus_field))
    @printf("  symmetry max |w(Δt,τ) - w(-Δt,τ)| = %.3g\n", symmetry)
    @printf("  ratio law max |w(rτ,τ) - exp(-r)| = %.3g\n", ratio_dev)
    @printf("  max abs error vs Float64 reference = %.3g\n", stats.max_abs)
    @printf("  max rel error (normal Float32 weights) = %.3g at |Δt|/τ = %.1f (bound %.3g)\n",
            stats.max_rel_normal, stats.worst_ratio, stats.rel_bound)
    @printf("  worst error/bound ratio on the grid = %.3f\n", stats.bound_headroom)
    @printf("  max rel error (subnormal Float32 weights) = %.3g\n", stats.max_rel_subnormal)
    @printf("  bitwise Float32 agreement = %.2f%%\n", 100 * exact_fraction)
    @printf("  grid points normal / subnormal / underflowed = %d / %d / %d\n",
            stats.normal, stats.subnormal, stats.underflow)
    for mark in TAU_MARKS
        @printf("  %-6s memory τ = %6.3g s → half weight at ±%7.1f ms, 10%% at ±%7.1f ms\n",
                mark.label, mark.τ,
                1000 * _lag_at_weight(mark.τ, 0.5),
                1000 * _lag_at_weight(mark.τ, 0.1))
    end
    println("  hypothesis supported: ", supported)
    println("  wrote ", lens_path)
    println("  wrote ", curves_path)
    println("  wrote ", metrics_path, " (", length(rows), " rows)")
    println("  wrote ", config_path)
    println("  wrote ", summary_path)
    supported || error("temporal lens experiment did not reproduce the expected recency field")
    return nothing
end

main()
