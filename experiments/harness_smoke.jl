# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Harness smoke experiment — τ sweep from sharp coincidence to timing-insensitive attention.

Hypothesis: as the time constant τ grows, `spike_attention_temporal` loses its
timing selectivity and converges to the timing-agnostic `spike_attention_discrete`
result on the same spike scene.

This is the reference implementation of the artifact contract in
`experiments/src/ExperimentUtils.jl`: deterministic seed, recorded config,
machine-readable metrics, one figure, and a summary that states whether the
observation supports the hypothesis.

    julia --project=experiments experiments/harness_smoke.jl
"""

using CairoMakie
using ExperimentUtils
using Printf
using Random
using TemporalFocus

const SLUG = "harness-smoke"
const RNG_SEED = 42
const N_NEURONS = 16
const N_EVENTS = 48
const N_OUT = 4
const T_MAX = 1.0f0
const TAUS = Float32[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0]
const CONVERGENCE_TOL = 0.01f0

_make_train(rng::AbstractRNG) = SpikeTrain([
    SpikeEvent(rand(rng, 1:N_NEURONS), rand(rng, Float32) * T_MAX, 1.0f0) for _ in 1:N_EVENTS
])

_norm(v::AbstractVector{<:Real}) = sqrt(sum(abs2, v))

function _figure(taus::Vector{Float64}, ratios::Vector{Float64}, figure_file::AbstractString)
    CairoMakie.activate!(type = "png")
    fig = Figure(size = (760, 460))
    ax = Axis(
        fig[1, 1];
        title = "Temporal attention loses timing selectivity as τ grows",
        xlabel = "τ (time constant, same units as spike times)",
        ylabel = "‖temporal readout‖ / ‖discrete readout‖",
        xscale = log10,
    )
    hlines!(ax, [1.0]; color = (:grey30, 0.9), linestyle = :dash, label = "discrete baseline")
    lines!(ax, taus, ratios; color = :dodgerblue, linewidth = 2.5)
    scatter!(
        ax,
        taus,
        ratios;
        color = :dodgerblue,
        markersize = 9,
        label = "spike_attention_temporal",
    )
    ylims!(ax, 0.0, 1.1)
    axislegend(ax; position = :rb)
    save(figure_file, fig; px_per_unit = 2)
    return figure_file
end

function main()
    rng = MersenneTwister(RNG_SEED)
    source = _make_train(rng)
    context = _make_train(rng)
    readout = rand(rng, Float32, N_NEURONS, N_OUT)

    discrete = spike_attention_discrete(source, context, readout)
    discrete_norm = _norm(discrete)
    peak = maximum(abs, discrete)
    peak > 0 || error("degenerate spike scene: discrete readout is all zeros")

    # Compare the full readout vectors, not just their norms: two different
    # vectors can share a norm, so norm-only checks could claim convergence that
    # did not happen. `slack` is a Float32 rounding allowance at this magnitude.
    temporals = [spike_attention_temporal(source, context, readout; τ = τ) for τ in TAUS]
    slack = 1.0f-5 * peak

    rows = map(zip(TAUS, temporals)) do (τ, temporal)
        deviation = temporal .- discrete
        return (
            tau = τ,
            temporal_norm = _norm(temporal),
            discrete_norm = discrete_norm,
            ratio = _norm(temporal) / discrete_norm,
            max_abs_deviation = maximum(abs, deviation),
            max_rel_deviation = maximum(abs, deviation) / peak,
        )
    end

    ratios = [row.ratio for row in rows]
    # Every readout component grows with τ, no component overshoots the discrete
    # readout, and the last one matches it elementwise.
    monotonic = all(i -> all(temporals[i + 1] .>= temporals[i] .- slack), 1:(length(temporals) - 1))
    bounded = all(temporal -> maximum(temporal .- discrete) <= slack, temporals)
    converged = last(rows).max_rel_deviation <= CONVERGENCE_TOL
    supported = monotonic && bounded && converged

    config_file = write_config(
        SLUG,
        Dict(
            "seed" => RNG_SEED,
            "n_neurons" => N_NEURONS,
            "n_events" => N_EVENTS,
            "n_out" => N_OUT,
            "t_max" => T_MAX,
            "tau_values" => TAUS,
            "convergence_tol" => CONVERGENCE_TOL,
        ),
    )
    metrics_file = write_metrics(SLUG, rows)
    figure_file = _figure(Float64.(TAUS), Float64.(ratios), figure_path(SLUG))

    table = join(
        (
            @sprintf(
                "| %g | %.4f | %.4f | %.4f |",
                row.tau,
                row.temporal_norm,
                row.ratio,
                row.max_rel_deviation
            ) for row in rows
        ),
        "\n",
    )
    # Format Float32 constants explicitly: string interpolation of a Float32
    # renders a `f0` suffix on Julia < 1.12.
    tol_s = @sprintf("%g", CONVERGENCE_TOL)
    t_max_s = @sprintf("%g", T_MAX)
    tau_min_s = @sprintf("%g", first(TAUS))
    tau_max_s = @sprintf("%g", last(TAUS))
    ratio_min_s = @sprintf("%.1f%%", 100 * first(ratios))
    ratio_max_s = @sprintf("%.4f", last(ratios))
    deviation_max_s = @sprintf("%.4f", last(rows).max_rel_deviation)
    summary_file = write_summary(
        SLUG,
        """
        # Harness smoke — τ sweep from coincidence to timing-insensitive attention

        ## Question

        What happens to `spike_attention_temporal` on a fixed spike scene as the
        time constant τ is swept over four orders of magnitude?

        ## Hypothesis

        Larger τ flattens the exponential recency weight `exp(-|Δt| / τ)` toward
        1, so every component of the temporal readout should grow with τ, stay
        at or below the timing-agnostic `spike_attention_discrete` readout, and
        converge to it elementwise at the largest τ — within $(tol_s) of the
        largest discrete readout component.

        ## Setup

        Seed $(RNG_SEED) (`MersenneTwister`), $(N_EVENTS) source and $(N_EVENTS)
        context events over $(N_NEURONS) neurons, spike times uniform on
        `[0, $(t_max_s))`, non-negative $(N_NEURONS)×$(N_OUT) readout, τ from
        $(tau_min_s) to $(tau_max_s). Full configuration in `config.toml`.

        ## Result

        | τ | ‖temporal readout‖ | ratio to discrete | max elementwise deviation |
        |---:|---:|---:|---:|
        $(table)

        Deviations are `max|temporal - discrete|` divided by the largest
        discrete readout component.

        - every component non-decreasing in τ: **$(monotonic)**
        - no component exceeds the discrete baseline: **$(bounded)**
        - at τ = $(tau_max_s): norm ratio **$(ratio_max_s)**, max elementwise
          deviation **$(deviation_max_s)** (converged: **$(converged)**)

        At the shortest τ the readout keeps only near-coincident pairs
        ($(ratio_min_s) of the discrete magnitude); by τ = $(tau_max_s) the
        kernel is effectively timing-insensitive and reproduces coincidence
        attention.

        ## Verdict

        The observation **$(supported ? "supports" : "does not support")** the hypothesis.

        ## Artifacts

        `config.toml`, `metrics.csv`, `figure.png`, `summary.md` in this directory.

        ## Reproduce

        ```bash
        julia --project=experiments experiments/harness_smoke.jl
        ```
        """,
    )

    println("harness smoke experiment (slug: ", SLUG, ")")
    println("  monotonic in τ (elementwise): ", monotonic)
    println("  bounded by discrete:          ", bounded)
    println(
        "  converged at τ=", tau_max_s, ": ", converged,
        " (norm ratio ", ratio_max_s, ", max elementwise deviation ", deviation_max_s, ")",
    )
    println("  hypothesis supported: ", supported)
    for file in (config_file, metrics_file, figure_file, summary_file)
        println("  wrote ", file)
    end

    supported || error("harness smoke experiment did not reproduce the expected τ behavior")
    return nothing
end

main()
