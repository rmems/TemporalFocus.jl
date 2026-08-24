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

    discrete_norm = _norm(spike_attention_discrete(source, context, readout))
    discrete_norm > 0 || error("degenerate spike scene: discrete readout is all zeros")

    rows = map(TAUS) do τ
        temporal_norm = _norm(spike_attention_temporal(source, context, readout; τ = τ))
        return (
            tau = τ,
            temporal_norm = temporal_norm,
            discrete_norm = discrete_norm,
            ratio = temporal_norm / discrete_norm,
        )
    end

    ratios = [row.ratio for row in rows]
    monotonic = issorted(ratios)
    bounded = all(<=(1 + eps(Float32)), ratios)
    converged = last(ratios) >= 1 - CONVERGENCE_TOL
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
        (@sprintf("| %g | %.4f | %.4f |", row.tau, row.temporal_norm, row.ratio) for row in rows),
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
    summary_file = write_summary(
        SLUG,
        """
        # Harness smoke — τ sweep from coincidence to timing-insensitive attention

        ## Question

        What happens to `spike_attention_temporal` on a fixed spike scene as the
        time constant τ is swept over four orders of magnitude?

        ## Hypothesis

        Larger τ flattens the exponential recency weight `exp(-|Δt| / τ)` toward
        1, so the temporal readout should grow monotonically with τ, stay at or
        below the timing-agnostic `spike_attention_discrete` readout, and
        converge to it (within $(tol_s)) at the largest τ.

        ## Setup

        Seed $(RNG_SEED) (`MersenneTwister`), $(N_EVENTS) source and $(N_EVENTS)
        context events over $(N_NEURONS) neurons, spike times uniform on
        `[0, $(t_max_s))`, non-negative $(N_NEURONS)×$(N_OUT) readout, τ from
        $(tau_min_s) to $(tau_max_s). Full configuration in `config.toml`.

        ## Result

        | τ | ‖temporal readout‖ | ratio to discrete |
        |---:|---:|---:|
        $(table)

        - monotonically non-decreasing in τ: **$(monotonic)**
        - never exceeds the discrete baseline: **$(bounded)**
        - ratio at τ = $(tau_max_s): **$(ratio_max_s)** (converged: **$(converged)**)

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
    println("  monotonic in τ:      ", monotonic)
    println("  bounded by discrete: ", bounded)
    println("  converged at τ=", tau_max_s, ": ", converged, " (ratio ", ratio_max_s, ")")
    println("  hypothesis supported: ", supported)
    for file in (config_file, metrics_file, figure_file, summary_file)
        println("  wrote ", file)
    end

    supported || error("harness smoke experiment did not reproduce the expected τ behavior")
    return nothing
end

main()
