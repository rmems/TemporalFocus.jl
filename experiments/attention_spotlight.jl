# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Attention Spotlight — replay a recorded spike stream through `TemporalBuffer`s
# and watch continuous attention move between neurons.
#
# Run:
#   julia --project=experiments experiments/attention_spotlight.jl
#
# Optional animation (deterministic, opt-in, artifact intentionally not committed):
#   SPOTLIGHT_ANIMATE=1 julia --project=experiments experiments/attention_spotlight.jl
#
# The replay is a pure experiment-side loop over the existing package primitives
# (`TemporalBuffer`, `prune!`, `spike_attention_continuous`, `normalize_l1!`).
# No scheduler, event loop, or runtime API is added to TemporalFocus itself.

import Pkg

const EXPERIMENTS_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(EXPERIMENTS_DIR, ".."))
const SLUG = "attention_spotlight"

"""
    ensure_environment()

Make sure the `experiments/` environment is active and instantiated so that a
bare `julia --project=experiments experiments/attention_spotlight.jl` works from
a fresh checkout (TemporalFocus is `dev`ed from the repo root, never added as a
root dependency).
"""
function ensure_environment()
    project = joinpath(EXPERIMENTS_DIR, "Project.toml")
    Base.active_project() == project || Pkg.activate(EXPERIMENTS_DIR)
    needs_setup = !isfile(joinpath(EXPERIMENTS_DIR, "Manifest.toml")) ||
                  Base.find_package("TemporalFocus") === nothing ||
                  Base.find_package("CairoMakie") === nothing
    if needs_setup
        @info "Instantiating the experiments environment (first run may take a few minutes)"
        Pkg.develop(path = REPO_ROOT)
        Pkg.instantiate()
    end
    return nothing
end

ensure_environment()

using Printf
using CairoMakie
using TemporalFocus

include(joinpath(EXPERIMENTS_DIR, "src", "ExperimentUtils.jl"))
using .ExperimentUtils

# ---------------------------------------------------------------------------
# Configuration — the whole scenario is a pure function of these numbers.
# ---------------------------------------------------------------------------

const CONFIG = (
    slug = SLUG,
    n_neurons = 6,
    # Retention window of both buffers; also bounds |dt| inside the kernel.
    buffer_window = 0.35f0,
    # Recency time constant: exp(-|dt| / τ). Short relative to the burst spacing
    # so that the freshest coincidences dominate the score.
    tau = 0.10f0,
    sample_dt = 0.02f0,
    t_end = 4.80f0,
    # Four phases, each spotlighting one neuron => three handoffs.
    phase_length = 1.20f0,
    phase_neurons = (2, 5, 3, 6),
    bursts_per_phase = 7,
    burst_start = 0.10f0,
    burst_spacing = 0.15f0,
    # Lag between the context spike and the source spike of one pattern pair.
    pattern_lag = 0.03f0,
    pattern_value = 1.0f0,
    # Low-amplitude chatter that keeps every other neuron slightly active.
    background_period = 0.15f0,
    background_lag = 0.07f0,
    background_value = 0.35f0,
    # Deterministic golden-ratio jitter (no RNG, identical on every platform).
    jitter = 0.012f0,
    # A focus segment must persist this many samples before it counts as a handoff.
    min_hold_steps = 5,
)

# ---------------------------------------------------------------------------
# Scenario: a recorded, causally ordered event stream.
# ---------------------------------------------------------------------------

"""
    ReplayEvent(t, neuron_id, value, stream, role)

One recorded spike in the replay scenario. `stream` is `:source` or `:context`
(which buffer it is delivered to) and `role` is `:pattern` or `:background`
(bookkeeping for the raster only).
"""
struct ReplayEvent
    t::Float32
    neuron_id::Int
    value::Float32
    stream::Symbol
    role::Symbol
end

# Low-discrepancy, RNG-free jitter: reproducible across Julia versions/platforms.
_jitter(k::Integer, amplitude::Float32) =
    amplitude * (2.0f0 * Float32(mod(k * 0.6180339887498949, 1.0)) - 1.0f0)

"""
    dominant_neuron(cfg, t) -> Int

Neuron that the scenario spotlights at time `t`.
"""
function dominant_neuron(cfg, t::Real)
    phase = 1 + floor(Int, Float32(t) / cfg.phase_length)
    return cfg.phase_neurons[clamp(phase, 1, length(cfg.phase_neurons))]
end

"""
    build_stream(cfg) -> Vector{ReplayEvent}

Build the recorded scenario: per-phase bursts on the spotlighted neuron plus
weak background chatter on the others, sorted into causal arrival order.

Both halves of a context/source pair must fall inside the replay horizon
`[0, cfg.t_end]`, so the recorded stream is exactly the stream the replay
consumes — no event is written to `scenario.csv` that never reaches a buffer.
Jitter counters are advanced before the horizon check, so filtering never
shifts the timing of the events that are kept.
"""
function build_stream(cfg)
    events = ReplayEvent[]

    burst_index = 0
    for (phase, dominant) in enumerate(cfg.phase_neurons)
        phase_start = Float32(phase - 1) * cfg.phase_length
        for burst in 0:(cfg.bursts_per_phase - 1)
            burst_index += 1
            t = phase_start + cfg.burst_start + Float32(burst) * cfg.burst_spacing +
                _jitter(burst_index, cfg.jitter)
            (0.0f0 <= t && t + cfg.pattern_lag <= cfg.t_end) || continue
            push!(events, ReplayEvent(t, dominant, cfg.pattern_value, :context, :pattern))
            push!(events, ReplayEvent(t + cfg.pattern_lag, dominant, cfg.pattern_value,
                                      :source, :pattern))
        end
    end

    n_ticks = floor(Int, cfg.t_end / cfg.background_period)
    for tick in 0:(n_ticks - 1)
        t = Float32(tick) * cfg.background_period + _jitter(1_000 + tick, cfg.jitter)
        (0.0f0 <= t && t + cfg.background_lag <= cfg.t_end) || continue
        neuron = 1 + mod(tick, cfg.n_neurons)
        if neuron == dominant_neuron(cfg, t)
            neuron = 1 + mod(neuron, cfg.n_neurons)
        end
        push!(events, ReplayEvent(t, neuron, cfg.background_value, :context, :background))
        push!(events, ReplayEvent(t + cfg.background_lag, neuron, cfg.background_value,
                                  :source, :background))
    end

    sort!(events; alg = MergeSort,
          by = e -> (e.t, e.stream === :source ? 1 : 0, e.neuron_id, e.value))
    return events
end

# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

"""
    ReplayStep

Everything recorded at one simulation timestamp.
"""
struct ReplayStep
    step::Int
    t::Float32
    ingested_source::Int
    ingested_context::Int
    pruned_source::Int
    pruned_context::Int
    n_source::Int
    n_context::Int
    attention::Vector{Float32}
    shares::Vector{Float32}
    top1_neuron::Int
    top1_attention::Float32
    top1_share::Float32
    top1_margin::Float32
end

"""
    identity_readout(n) -> Matrix{Float32}

Identity readout, so `spike_attention_continuous` returns the raw per-neuron
attention vector (`transpose(I) * attention == attention`) instead of a
projected readout.
"""
function identity_readout(n::Integer)
    readout = zeros(Float32, n, n)
    for i in 1:n
        readout[i, i] = 1.0f0
    end
    return readout
end

"""
    replay(cfg, events) -> Vector{ReplayStep}

Drive the buffers through the scenario. At every timestamp: ingest the events
that have already happened (`event.t <= t`), `prune!` both buffers against the
current time, evaluate continuous attention, and record the state.
"""
function replay(cfg, events)
    readout = identity_readout(cfg.n_neurons)
    source_buffer = TemporalBuffer(cfg.buffer_window)
    context_buffer = TemporalBuffer(cfg.buffer_window)

    steps = ReplayStep[]
    n_steps = max(1, round(Int, cfg.t_end / cfg.sample_dt))
    next_event = 1

    for step in 0:n_steps
        # Sample on the `sample_dt` grid, but never past the declared horizon,
        # and finish exactly at `t_end`. That keeps the two guarantees intact
        # when `t_end` is not an exact multiple of `sample_dt`: every recorded
        # event is consumed, and nothing is recorded after the horizon.
        t = step == n_steps ? cfg.t_end : min(Float32(step) * cfg.sample_dt, cfg.t_end)

        # 1. Causal ingest: nothing from the future ever enters a buffer.
        ingested_source = 0
        ingested_context = 0
        while next_event <= length(events) && events[next_event].t <= t
            event = events[next_event]
            spike = SpikeEvent(event.neuron_id, event.t, event.value)
            if event.stream === :source
                push!(source_buffer.events, spike)
                ingested_source += 1
            else
                push!(context_buffer.events, spike)
                ingested_context += 1
            end
            next_event += 1
        end

        # 2. Prune both buffers against the current simulation time.
        before_source = length(source_buffer.events)
        before_context = length(context_buffer.events)
        prune!(source_buffer, t)
        prune!(context_buffer, t)
        pruned_source = before_source - length(source_buffer.events)
        pruned_context = before_context - length(context_buffer.events)

        # 3. Evaluate continuous attention over what is still in the window.
        attention = spike_attention_continuous(source_buffer, context_buffer, readout;
                                               τ = cfg.tau)
        shares = normalize_l1!(copy(attention))

        # 4. Record.
        top1_neuron = 0
        top1_attention = 0.0f0
        top1_share = 0.0f0
        top1_margin = 0.0f0
        # `attention` always holds exactly `cfg.n_neurons` (≥ 1) entries, so
        # `findmax` is total here. Before any spike is in range the vector is
        # all zeros; that is reported as "no focus" (`top1_neuron == 0`) rather
        # than as an arbitrary winner.
        peak_attention, peak = findmax(attention)
        if peak_attention > 0.0f0
            top1_neuron = peak
            top1_attention = peak_attention
            top1_share = shares[peak]
            runner_up = 0.0f0
            for (i, share) in pairs(shares)
                i == peak && continue
                share > runner_up && (runner_up = share)
            end
            top1_margin = top1_share - runner_up
        end

        push!(steps, ReplayStep(step, t, ingested_source, ingested_context,
                                pruned_source, pruned_context,
                                length(source_buffer.events), length(context_buffer.events),
                                attention, shares,
                                top1_neuron, top1_attention, top1_share, top1_margin))
    end

    return steps
end

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

"""
    focus_segments(steps, min_hold) -> Vector{NamedTuple}

Contiguous runs of the same top-1 neuron that last at least `min_hold` samples.
Short flickers are dropped so that only real spotlight moves are reported.
"""
function focus_segments(steps, min_hold::Integer)
    runs = Tuple{Int,Int,Int}[]
    for (i, step) in pairs(steps)
        if !isempty(runs) && runs[end][1] == step.top1_neuron
            runs[end] = (step.top1_neuron, runs[end][2], i)
        else
            push!(runs, (step.top1_neuron, i, i))
        end
    end
    return [(neuron = neuron,
             t_start = steps[first_i].t,
             t_end = steps[last_i].t,
             n_samples = last_i - first_i + 1)
            for (neuron, first_i, last_i) in runs
            if neuron != 0 && (last_i - first_i + 1) >= min_hold]
end

"""
    handoffs(segments) -> Vector{NamedTuple}

Spotlight handoffs: each transition between consecutive stable focus segments.
"""
handoffs(segments) = [(from = segments[i].neuron, to = segments[i + 1].neuron,
                       t = segments[i + 1].t_start)
                      for i in 1:(length(segments) - 1)
                      if segments[i].neuron != segments[i + 1].neuron]

"""
    metric_rows(cfg, steps) -> Vector{<:NamedTuple}

Flatten the replay into CSV-ready rows (one per simulation timestamp).
"""
function metric_rows(cfg, steps)
    attention_columns = Tuple(Symbol("attention_n", i) for i in 1:cfg.n_neurons)
    share_columns = Tuple(Symbol("share_n", i) for i in 1:cfg.n_neurons)
    return [merge((step = s.step,
                   t = s.t,
                   n_source_events = s.n_source,
                   n_context_events = s.n_context,
                   n_ingested_source = s.ingested_source,
                   n_ingested_context = s.ingested_context,
                   n_pruned_source = s.pruned_source,
                   n_pruned_context = s.pruned_context),
                  NamedTuple{attention_columns}(Tuple(s.attention)),
                  NamedTuple{share_columns}(Tuple(s.shares)),
                  (top1_neuron = s.top1_neuron,
                   top1_attention = s.top1_attention,
                   top1_share = s.top1_share,
                   top1_margin = s.top1_margin))
            for s in steps]
end

"""
    write_scenario(dir, events) -> String

Record the exact event stream that was replayed, so the run can be audited
without re-deriving it from the config.

Times and values are written with Julia's shortest round-tripping `Float32`
representation (not a fixed number of decimals), so parsing a row back into
`Float32` reproduces the replayed event bit-for-bit.
"""
function write_scenario(dir::AbstractString, events)
    path = joinpath(dir, "scenario.csv")
    open(path, "w") do io
        println(io, "t,stream,neuron_id,value,role")
        for event in events
            println(io, event.t, ",", event.stream, ",", event.neuron_id, ",",
                    event.value, ",", event.role)
        end
    end
    return path
end

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

const CONTEXT_COLOR = RGBf(0.20, 0.45, 0.70)
const SOURCE_COLOR = RGBf(0.90, 0.45, 0.10)
const PRUNE_COLOR = RGBf(0.80, 0.20, 0.30)

"""
    share_matrix(cfg, steps) -> Matrix{Float32}

`length(steps) × n_neurons` matrix of normalized attention shares.
"""
function share_matrix(cfg, steps)
    shares = Matrix{Float32}(undef, length(steps), cfg.n_neurons)
    for (i, step) in pairs(steps)
        shares[i, :] .= step.shares
    end
    return shares
end

"""
    build_figure(cfg, events, steps, moves) -> Figure

Four stacked panels: event raster, time × neuron attention heatmap with the
top-1 trace overlaid, top-1 focus/margin trace, and buffer occupancy plus
per-step prune counts.
"""
function build_figure(cfg, events, steps, moves)
    times = Float32[s.t for s in steps]
    shares = share_matrix(cfg, steps)
    top1 = Float32[s.top1_neuron == 0 ? NaN32 : Float32(s.top1_neuron) for s in steps]
    move_times = Float32[m.t for m in moves]
    phase_edges = Float32[Float32(p) * cfg.phase_length
                          for p in 1:(length(cfg.phase_neurons) - 1)]

    fig = Figure(size = (1180, 1480), fontsize = 15)
    Label(fig[0, 1:2],
          "Attention spotlight — streaming TemporalBuffer replay " *
          "(window = $(cfg.buffer_window) s, τ = $(cfg.tau) s)";
          fontsize = 21, font = :bold, padding = (0, 0, 6, 0))

    # Panel 1 — event raster.
    ax1 = Axis(fig[1, 1]; ylabel = "neuron",
               title = "1 · Recorded event stream (ingested causally, event.t ≤ t)")
    context_events = [e for e in events if e.stream === :context]
    source_events = [e for e in events if e.stream === :source]
    scatter!(ax1, Float32[e.t for e in context_events],
             Float32[e.neuron_id - 0.16f0 for e in context_events];
             marker = :dtriangle, markersize = 9, color = CONTEXT_COLOR, label = "context spike")
    scatter!(ax1, Float32[e.t for e in source_events],
             Float32[e.neuron_id + 0.16f0 for e in source_events];
             marker = :utriangle, markersize = 9, color = SOURCE_COLOR, label = "source spike")
    vlines!(ax1, phase_edges; color = (:black, 0.35), linestyle = :dash)
    for (phase, dominant) in enumerate(cfg.phase_neurons)
        centre = (Float32(phase) - 0.5f0) * cfg.phase_length
        text!(ax1, centre, cfg.n_neurons + 0.55f0; text = "phase $phase · neuron $dominant",
              align = (:center, :bottom), fontsize = 12, color = (:black, 0.7))
    end
    ylims!(ax1, 0.3, cfg.n_neurons + 1.2)
    ax1.yticks = 1:cfg.n_neurons
    axislegend(ax1; position = :lb, orientation = :horizontal, framevisible = false,
               labelsize = 12)

    # Panel 2 — the spotlight itself.
    ax2 = Axis(fig[2, 1]; ylabel = "neuron",
               title = "2 · Attention spotlight: normalized share per neuron over time")
    heat = heatmap!(ax2, times, Float32.(1:cfg.n_neurons), shares;
                    colormap = :magma, colorrange = (0.0f0, 1.0f0))
    lines!(ax2, times, top1; color = (:white, 0.85), linewidth = 2.5)
    vlines!(ax2, move_times; color = (:white, 0.55), linestyle = :dot)
    ax2.yticks = 1:cfg.n_neurons
    Colorbar(fig[2, 2], heat; label = "attention share (L1)")

    # Panel 3 — top-1 trace and margin.
    ax3 = Axis(fig[3, 1]; ylabel = "top-1 neuron",
               title = "3 · Top-1 focus trace (colour = margin over runner-up)")
    stairs!(ax3, times, top1; step = :post, color = (:black, 0.3))
    margins = scatter!(ax3, times, top1; color = Float32[s.top1_margin for s in steps],
                       colormap = :viridis, colorrange = (0.0f0, 1.0f0), markersize = 8)
    vlines!(ax3, move_times; color = (PRUNE_COLOR, 0.7), linestyle = :dot)
    for move in moves
        text!(ax3, move.t, cfg.n_neurons + 0.35f0;
              text = @sprintf("%d→%d @ %.2fs", move.from, move.to, move.t),
              align = (:center, :bottom), fontsize = 11, color = PRUNE_COLOR)
    end
    ylims!(ax3, 0.3, cfg.n_neurons + 1.1)
    ax3.yticks = 1:cfg.n_neurons
    Colorbar(fig[3, 2], margins; label = "top-1 margin")

    # Panel 4 — occupancy and pruning.
    ax4 = Axis(fig[4, 1]; xlabel = "simulation time (s)", ylabel = "events",
               title = "4 · Buffer occupancy after prune! and events dropped per step")
    barplot!(ax4, times, Float32[s.pruned_source + s.pruned_context for s in steps];
             width = cfg.sample_dt, gap = 0.0, color = (PRUNE_COLOR, 0.45),
             label = "events pruned this step")
    lines!(ax4, times, Float32[s.n_source for s in steps];
           color = SOURCE_COLOR, linewidth = 2, label = "source events retained")
    lines!(ax4, times, Float32[s.n_context for s in steps];
           color = CONTEXT_COLOR, linewidth = 2, label = "context events retained")
    axislegend(ax4; position = :rt, framevisible = false, labelsize = 12)

    linkxaxes!(ax1, ax2, ax3, ax4)
    for ax in (ax1, ax2, ax3)
        hidexdecorations!(ax; grid = false)
    end
    xlims!(ax4, 0.0f0, cfg.t_end)
    rowgap!(fig.layout, 8)

    return fig
end

"""
    render_animation(cfg, steps, path; stride=4, framerate=12) -> String

Opt-in GIF: the spotlight heatmap with a moving time cursor plus a live bar
chart of the current attention shares. Returns the written path; recording
errors (missing FFMPEG backend, full disk, Makie failure) propagate.
"""
function render_animation(cfg, steps, path; stride::Integer = 4, framerate::Integer = 12)
    times = Float32[s.t for s in steps]
    shares = share_matrix(cfg, steps)
    top1 = Float32[s.top1_neuron == 0 ? NaN32 : Float32(s.top1_neuron) for s in steps]

    cursor = Observable(times[1])
    bars = Observable(collect(shares[1, :]))

    fig = Figure(size = (960, 480), fontsize = 15)
    ax_map = Axis(fig[1, 1]; xlabel = "simulation time (s)", ylabel = "neuron",
                  title = "attention share")
    heatmap!(ax_map, times, Float32.(1:cfg.n_neurons), shares;
             colormap = :magma, colorrange = (0.0f0, 1.0f0))
    lines!(ax_map, times, top1; color = (:white, 0.8), linewidth = 2)
    vlines!(ax_map, cursor; color = :white, linewidth = 2)
    ax_map.yticks = 1:cfg.n_neurons

    ax_bar = Axis(fig[1, 2]; xlabel = "neuron", ylabel = "share", title = "now")
    barplot!(ax_bar, Float32.(1:cfg.n_neurons), bars;
             color = bars, colormap = :magma, colorrange = (0.0f0, 1.0f0))
    ylims!(ax_bar, 0.0f0, 1.0f0)
    ax_bar.xticks = 1:cfg.n_neurons
    colsize!(fig.layout, 2, Relative(0.3))

    frames = 1:stride:length(steps)
    # Recording failures are not swallowed: the animation is opt-in, so a
    # missing FFMPEG backend, a full disk, or a Makie error should surface
    # loudly instead of silently producing no GIF.
    record(fig, path, frames; framerate = framerate) do i
        cursor[] = times[i]
        bars[] = collect(shares[i, :])
        ax_bar.title = @sprintf("t = %.2f s", times[i])
    end
    return path
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

function summary_markdown(cfg, events, steps, segments, moves, artifacts)
    peak_source = maximum(s.n_source for s in steps)
    peak_context = maximum(s.n_context for s in steps)
    total_pruned = sum(s.pruned_source + s.pruned_context for s in steps)
    prune_steps = count(s -> s.pruned_source + s.pruned_context > 0, steps)
    focused = count(s -> s.top1_neuron != 0, steps)

    io = IOBuffer()
    println(io, "# Attention Spotlight")
    println(io)
    println(io, "Replay of a recorded spike stream through two `TemporalBuffer`s. At every")
    println(io, "simulation timestamp the replay ingests the events that have already")
    println(io, "happened (`event.t <= t`), calls `prune!` on both buffers, evaluates")
    println(io, "`spike_attention_continuous`, and records the per-neuron attention.")
    println(io, "Everything lives in `experiments/`; the package itself gains no runtime,")
    println(io, "scheduler, or event-loop API.")
    println(io)
    println(io, "## Scenario")
    println(io)
    println(io, "- neurons: $(cfg.n_neurons)")
    println(io, "- buffer window: $(cfg.buffer_window) s (also bounds `|dt|` inside the kernel)")
    println(io, "- τ: $(cfg.tau) s")
    println(io, "- sampled every $(cfg.sample_dt) s from 0.0 s to $(cfg.t_end) s " *
                "($(length(steps)) timesteps)")
    println(io, "- $(length(cfg.phase_neurons)) phases of $(cfg.phase_length) s, " *
                "spotlighting neurons $(join(cfg.phase_neurons, ", "))")
    println(io, "- $(length(events)) recorded events " *
                "($(count(e -> e.role === :pattern, events)) pattern, " *
                "$(count(e -> e.role === :background, events)) background)")
    println(io)
    println(io, "## Focus timeline")
    println(io)
    println(io, "| segment | top-1 neuron | from (s) | to (s) | samples |")
    println(io, "|---|---|---|---|---|")
    for (i, segment) in pairs(segments)
        @printf(io, "| %d | %d | %.2f | %.2f | %d |\n",
                i, segment.neuron, segment.t_start, segment.t_end, segment.n_samples)
    end
    println(io)
    println(io, "$(length(moves)) spotlight handoff(s):")
    println(io)
    for move in moves
        @printf(io, "- neuron %d → neuron %d at t = %.2f s\n", move.from, move.to, move.t)
    end
    println(io)
    println(io, "## Buffer activity")
    println(io)
    println(io, "- peak retained source events: $(peak_source)")
    println(io, "- peak retained context events: $(peak_context)")
    println(io, "- steps where `prune!` dropped at least one event: " *
                "$(prune_steps) of $(length(steps))")
    println(io, "- total events pruned across the replay: $(total_pruned)")
    println(io, "- steps with a non-zero attention peak: $(focused) of $(length(steps))")
    println(io)
    println(io, "## Artifacts")
    println(io)
    for artifact in artifacts
        println(io, "- `$(basename(artifact))`")
    end
    println(io)
    println(io, "## Reproduce")
    println(io)
    println(io, "```bash")
    println(io, "julia --project=experiments experiments/attention_spotlight.jl")
    println(io, "```")
    println(io)
    println(io, "The optional GIF is not committed; it is regenerated on demand,")
    println(io, "deterministically, from the same scenario:")
    println(io)
    println(io, "```bash")
    println(io, "SPOTLIGHT_ANIMATE=1 julia --project=experiments experiments/attention_spotlight.jl")
    println(io, "```")
    return String(take!(io))
end

"""
    config_float(x) -> Float64

`Float32` parameter widened for TOML output without the binary-representation
noise of a bare `Float64` conversion.
"""
config_float(x::Real) = round(Float64(x); digits = 6)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main(cfg = CONFIG)
    events = build_stream(cfg)
    steps = replay(cfg, events)
    segments = focus_segments(steps, cfg.min_hold_steps)
    moves = handoffs(segments)

    dir = result_dir(cfg.slug)
    config_path = write_config(cfg.slug, Dict{String,Any}(
        "n_neurons" => cfg.n_neurons,
        "buffer_window" => config_float(cfg.buffer_window),
        "tau" => config_float(cfg.tau),
        "sample_dt" => config_float(cfg.sample_dt),
        "t_end" => config_float(cfg.t_end),
        "phase_length" => config_float(cfg.phase_length),
        "phase_neurons" => collect(cfg.phase_neurons),
        "bursts_per_phase" => cfg.bursts_per_phase,
        "burst_start" => config_float(cfg.burst_start),
        "burst_spacing" => config_float(cfg.burst_spacing),
        "pattern_lag" => config_float(cfg.pattern_lag),
        "pattern_value" => config_float(cfg.pattern_value),
        "background_period" => config_float(cfg.background_period),
        "background_lag" => config_float(cfg.background_lag),
        "background_value" => config_float(cfg.background_value),
        "jitter" => config_float(cfg.jitter),
        "min_hold_steps" => cfg.min_hold_steps,
        "n_events" => length(events),
        "n_steps" => length(steps),
    ))
    scenario_path = write_scenario(dir, events)
    metrics_path = write_metrics(cfg.slug, metric_rows(cfg, steps))

    CairoMakie.activate!(type = "png")
    figure = build_figure(cfg, events, steps, moves)
    fig_path = figure_path(cfg.slug)
    save(fig_path, figure)

    artifacts = [config_path, scenario_path, metrics_path, fig_path]

    if get(ENV, "SPOTLIGHT_ANIMATE", "0") == "1"
        push!(artifacts, render_animation(cfg, steps, figure_path(cfg.slug, "spotlight.gif")))
    end

    summary_path = write_summary(cfg.slug, summary_markdown(cfg, events, steps, segments,
                                                            moves, artifacts))
    push!(artifacts, summary_path)

    println("Attention spotlight replay")
    println("  events replayed : ", length(events))
    println("  timesteps       : ", length(steps),
            " (dt = ", cfg.sample_dt, " s, t_end = ", cfg.t_end, " s)")
    println("  peak occupancy  : ", maximum(s.n_source for s in steps), " source / ",
            maximum(s.n_context for s in steps), " context events")
    println("  events pruned   : ", sum(s.pruned_source + s.pruned_context for s in steps),
            " over ", count(s -> s.pruned_source + s.pruned_context > 0, steps), " steps")
    println("  focus segments  :")
    for segment in segments
        @printf("    neuron %d  %.2f s → %.2f s  (%d samples)\n",
                segment.neuron, segment.t_start, segment.t_end, segment.n_samples)
    end
    println("  handoffs        : ", length(moves))
    for move in moves
        @printf("    neuron %d → neuron %d at t = %.2f s\n", move.from, move.to, move.t)
    end
    println("  artifacts       :")
    for artifact in artifacts
        println("    ", relpath(artifact, repo_root()))
    end

    return artifacts
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
