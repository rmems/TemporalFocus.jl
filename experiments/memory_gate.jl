# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Experiment: memory_gate
#
# Maps the trade-off surface between the two independent notions of memory in
# `spike_attention_continuous`:
#
#   * τ                      — soft exponential decay (`temporal_weight`)
#   * TemporalBuffer.window  — a hard, per-pair admissibility boundary
#
# Setup (once):
#   julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#   julia --project=experiments experiments/memory_gate.jl
#
# Artifacts land in experiments/results/memory_gate/.

using Printf
using Random
using Statistics

using CairoMakie
using TemporalFocus

include(joinpath(@__DIR__, "src", "ExperimentUtils.jl"))
using .ExperimentUtils

const SLUG = "memory_gate"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#
# Times are in milliseconds. TemporalFocus is unit-agnostic (all temporal
# quantities are plain `Float32`); ms is chosen only to keep the numbers
# readable. Nothing in the experiment depends on the unit.

# The scene contains no random draws: every spike time and amplitude is a fixed
# constant. The seed is recorded and applied for artifact-contract compliance so
# that the script stays deterministic if noise is ever added.
const SEED = 20260823

const N_NEURONS = 3
const SIGNAL = 1      # carries the target AND the stale same-neuron distractors
const COMPETITOR = 2  # unrelated neuron, recent but weak
const UNRELATED = 3   # unrelated neuron, far in the past

const T_NOW = 0.0f0          # every source ("query") spike fires now
const SOURCE_VALUE = 1.0f0

const TARGET_LAG = 8.0f0     # the interaction we want preserved
const TARGET_VALUE = 1.0f0

const STALE_LAGS = (60.0f0, 95.0f0)  # same-neuron distractors we want rejected
const STALE_VALUE = 1.0f0

const COMPETITOR_LAG = 2.5f0
const COMPETITOR_VALUE = 0.30f0

const UNRELATED_LAG = 150.0f0
const UNRELATED_VALUE = 0.90f0

# Sweep grids (logarithmic in both axes).
const TAU_MIN, TAU_MAX, N_TAU = 1.0f0, 300.0f0, 25
# The window grid deliberately starts below TARGET_LAG and ends beyond the
# largest stale lag (and beyond UNRELATED_LAG).
const WINDOW_MIN, WINDOW_MAX, N_WINDOW = 3.0f0, 240.0f0, 25

# Regime thresholds (documented in summary.md; only LEAK_MAX is a soft choice).
const TARGET_FLOOR = 0.05f0  # target mass below this ⇒ the target is effectively forgotten
const LEAK_MAX = 0.20f0      # stale leakage at/above this ⇒ "over-retentive"

# `Inf32` window ⇒ the hard gate admits every pair, isolating the pure τ effect.
const WINDOW_OPEN = Inf32

const REGIMES = ("window_clipped", "decay_starved", "selective_gate", "soft_decay", "over_retentive")
const REGIME_LABELS = ("hard-clipped", "decay-starved", "selective gate", "soft decay", "over-retentive")
# Okabe-Ito qualitative palette: an established color-vision-deficiency-safe
# set. Identity is never carried by color alone here — every region is also
# directly labeled and listed in the legend.
const REGIME_COLORS = ("#56B4E9", "#CC79A7", "#009E73", "#E69F00", "#D55E00")

# Okabe-Ito subset for the 1-D slices.
const SLICE_COLORS = ("#0072B2", "#009E73", "#E69F00")

const INK_PRIMARY = "#1a1a1a"
const INK_MUTED = "#5c5c5c"

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

logspace(lo::Real, hi::Real, n::Integer) =
    Float32[round(Float32(x); sigdigits = 4)
            for x in exp10.(range(log10(Float64(lo)), log10(Float64(hi)); length = n))]

"""Source volley: one query spike per neuron at `T_NOW`."""
source_events() = [SpikeEvent(i, T_NOW, SOURCE_VALUE) for i in 1:N_NEURONS]

target_events() = [SpikeEvent(SIGNAL, T_NOW - TARGET_LAG, TARGET_VALUE)]
stale_events() = [SpikeEvent(SIGNAL, T_NOW - lag, STALE_VALUE) for lag in STALE_LAGS]
competitor_events() = [SpikeEvent(COMPETITOR, T_NOW - COMPETITOR_LAG, COMPETITOR_VALUE)]
unrelated_events() = [SpikeEvent(UNRELATED, T_NOW - UNRELATED_LAG, UNRELATED_VALUE)]

all_context_events() =
    vcat(target_events(), stale_events(), competitor_events(), unrelated_events())

# Transparent (identity) readout, so the kernel output *is* the per-neuron
# attention vector and nothing about the mapping needs to be inverted.
const READOUT = Float32[i == j for i in 1:N_NEURONS, j in 1:N_NEURONS]

"""
    attention(context_events, window, τ) -> Vector{Float32}

Per-neuron attention for a context sub-scene, using the real kernel. Both
buffers carry the same `window`; `spike_attention_continuous` admits a pair
only when `abs(dt) <= min(source.window, context.window)`.
"""
function attention(context_events::Vector{SpikeEvent}, window::Float32, τ::Float32)
    source = TemporalBuffer(window, source_events())
    context = TemporalBuffer(window, context_events)
    return spike_attention_continuous(source, context, READOUT; τ = τ)
end

"""Exactly the kernel's admissibility test, on the same `Float32` values."""
in_window(lag::Float32, window::Float32) = abs(lag) <= window

# ---------------------------------------------------------------------------
# Derived metrics (zero handling is documented in summary.md)
# ---------------------------------------------------------------------------

# share of *all* attention mass in the scene that sits on the target interaction
safe_share(part::Float32, total::Float32) = total > 0f0 ? part / total : 0f0

# fraction of the signal neuron's own mass contributed by stale spikes
function stale_leakage(target_mass::Float32, stale_mass::Float32)
    denom = target_mass + stale_mass
    return denom > 0f0 ? stale_mass / denom : 0f0
end

# `Inf` when the gate admits the target but no stale mass; `NaN` when neither
# the target nor any stale spike contributes.
function target_stale_ratio(target_mass::Float32, stale_mass::Float32)
    stale_mass > 0f0 && return target_mass / stale_mass
    target_mass > 0f0 && return Inf32
    return NaN32
end

function classify(target_in::Bool, target_mass::Float32, n_stale_in::Int, leakage::Float32)
    target_in || return 1                     # the hard window removed the target outright
    target_mass < TARGET_FLOOR && return 2    # in-window, but exponential decay erased it
    n_stale_in == 0 && return 3               # target kept, every stale spike gated out
    leakage < LEAK_MAX && return 4            # stale admitted, but τ suppresses it
    return 5                                  # stale activity retains substantial mass
end

_r6(x::Real) = isfinite(x) ? round(Float32(x); sigdigits = 6) : Float32(x)

# Float32 → Float64 widening exposes binary noise (0.3f0 becomes
# 0.30000001192092896). Round it back off so config.toml stays readable.
_f64(x::Real) = round(Float64(x); sigdigits = 6)

# Two fixed precisions rather than `@sprintf("%.*f", digits, x)`: dynamic
# precision in Printf needs Julia 1.10, and this repo targets 1.9+.
function _fmt(x::Real)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    return @sprintf("%.4f", x)
end

function _fmt2(x::Real)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    return @sprintf("%.2f", x)
end

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

Random.seed!(SEED)

const TAUS = logspace(TAU_MIN, TAU_MAX, N_TAU)
const WINDOWS = logspace(WINDOW_MIN, WINDOW_MAX, N_WINDOW)

rows = Vector{NamedTuple}(undef, length(TAUS) * length(WINDOWS))
share_map = fill(NaN32, length(TAUS), length(WINDOWS))
leak_map = fill(NaN32, length(TAUS), length(WINDOWS))
# Same quantity as `leak_map`, recomputed with the hard gate disabled. The
# difference between the two is exactly the hard window's contribution.
leak_open_map = fill(NaN32, length(TAUS), length(WINDOWS))
regime_map = zeros(Int, length(TAUS), length(WINDOWS))
top1_map = zeros(Int, length(TAUS), length(WINDOWS))

let k = 0
    for (i, τ) in pairs(TAUS), (j, window) in pairs(WINDOWS)
        att_full = attention(all_context_events(), window, τ)
        att_target = attention(target_events(), window, τ)
        att_stale = attention(stale_events(), window, τ)
        att_comp = attention(competitor_events(), window, τ)
        att_unrel = attention(unrelated_events(), window, τ)

        target_mass = att_target[SIGNAL]
        stale_mass = att_stale[SIGNAL]
        competitor_mass = att_comp[COMPETITOR]
        unrelated_mass = att_unrel[UNRELATED]
        total_mass = sum(att_full)

        # The kernel accumulates additively over context events, so the
        # ablations must sum back to the full scene. Guards against a silent
        # mis-decomposition.
        parts = target_mass + stale_mass + competitor_mass + unrelated_mass
        isapprox(parts, total_mass; atol = 1f-5, rtol = 1f-5) ||
            error("attention decomposition is not additive at τ=$(τ), window=$(window)")

        # Counterfactual with the hard gate disabled: isolates the τ effect.
        target_mass_open = attention(target_events(), WINDOW_OPEN, τ)[SIGNAL]
        stale_mass_open = attention(stale_events(), WINDOW_OPEN, τ)[SIGNAL]

        target_in = in_window(TARGET_LAG, window)
        stale_flags = [in_window(lag, window) for lag in STALE_LAGS]
        n_stale_in = count(stale_flags)

        # Fraction of the τ-only stale mass that the hard window removed.
        stale_clipped_fraction =
            stale_mass_open > 0f0 ? 1f0 - stale_mass / stale_mass_open : 0f0

        share = safe_share(target_mass, total_mass)
        leakage = stale_leakage(target_mass, stale_mass)
        ratio = target_stale_ratio(target_mass, stale_mass)

        # `argmax` breaks ties toward the lowest neuron id; 0 means "no mass
        # anywhere, so no winner exists".
        top1 = maximum(att_full) > 0f0 ? argmax(att_full) : 0
        regime = classify(target_in, target_mass, n_stale_in, leakage)

        share_map[i, j] = share
        leak_map[i, j] = leakage
        leak_open_map[i, j] = stale_leakage(target_mass_open, stale_mass_open)
        regime_map[i, j] = regime
        top1_map[i, j] = top1

        k += 1
        rows[k] = (
            tau = τ,
            window = window,
            target_lag = TARGET_LAG,
            stale_lag_1 = STALE_LAGS[1],
            stale_lag_2 = STALE_LAGS[2],
            target_in_window = target_in,
            stale_1_in_window = stale_flags[1],
            stale_2_in_window = stale_flags[2],
            n_stale_in_window = n_stale_in,
            competitor_in_window = in_window(COMPETITOR_LAG, window),
            unrelated_in_window = in_window(UNRELATED_LAG, window),
            target_mass = _r6(target_mass),
            stale_mass = _r6(stale_mass),
            competitor_mass = _r6(competitor_mass),
            unrelated_mass = _r6(unrelated_mass),
            total_mass = _r6(total_mass),
            target_mass_open_window = _r6(target_mass_open),
            stale_mass_open_window = _r6(stale_mass_open),
            stale_clipped_fraction = _r6(stale_clipped_fraction),
            target_share = _r6(share),
            stale_leakage = _r6(leakage),
            target_stale_ratio = _r6(ratio),
            top1_neuron = top1,
            top1_correct = top1 == SIGNAL,
            regime = REGIMES[regime],
        )
    end
end

metrics_path = write_metrics(SLUG, rows)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

config = Dict{String,Any}(
    "slug" => SLUG,
    "seed" => SEED,
    "deterministic" => true,
    "random_draws" => false,
    "time_unit" => "ms (nominal; TemporalFocus is unit-agnostic Float32)",
    "julia_version" => string(VERSION),
    "kernel" => "spike_attention_continuous",
    "readout" => "identity (transparent)",
    "n_neurons" => N_NEURONS,
    "n_conditions" => length(rows),
    "tau_grid" => _f64.(TAUS),
    "window_grid" => _f64.(WINDOWS),
    "scene" => Dict{String,Any}(
        "t_now" => _f64(T_NOW),
        "source_value" => _f64(SOURCE_VALUE),
        "signal_neuron" => SIGNAL,
        "competitor_neuron" => COMPETITOR,
        "unrelated_neuron" => UNRELATED,
        "target_lag" => _f64(TARGET_LAG),
        "target_value" => _f64(TARGET_VALUE),
        "stale_lags" => _f64.(collect(STALE_LAGS)),
        "stale_value" => _f64(STALE_VALUE),
        "competitor_lag" => _f64(COMPETITOR_LAG),
        "competitor_value" => _f64(COMPETITOR_VALUE),
        "unrelated_lag" => _f64(UNRELATED_LAG),
        "unrelated_value" => _f64(UNRELATED_VALUE),
    ),
    "thresholds" => Dict{String,Any}(
        "target_floor" => _f64(TARGET_FLOOR),
        "leak_max" => _f64(LEAK_MAX),
    ),
)
config_path = write_config(SLUG, config)

# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------

nearest_index(grid, value) = argmin(abs.(log10.(Float64.(grid)) .- log10(Float64(value))))

const TAU_SLICE_TARGETS = (2.0, 20.0, 200.0)
const WINDOW_SLICE_TARGETS = (5.0, 40.0, 200.0)

tau_slice_idx = [nearest_index(TAUS, v) for v in TAU_SLICE_TARGETS]
window_slice_idx = [nearest_index(WINDOWS, v) for v in WINDOW_SLICE_TARGETS]

row_at(i, j) = rows[(i - 1) * length(WINDOWS) + j]

# ---------------------------------------------------------------------------
# Figure — the "memory control panel"
# ---------------------------------------------------------------------------

CairoMakie.activate!(type = "png", px_per_unit = 2)

xs = Float64.(log10.(TAUS))
ys = Float64.(log10.(WINDOWS))

function log_ticks(vals)
    candidates = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300]
    keep = filter(t -> minimum(vals) <= t <= maximum(vals), candidates)
    return (log10.(Float64.(keep)), string.(keep))
end

function reference_lines!(ax)
    for lag in (TARGET_LAG, STALE_LAGS...)
        hlines!(ax, [log10(Float64(lag))]; color = (:white, 0.85), linewidth = 3.0)
        hlines!(ax, [log10(Float64(lag))]; color = (:black, 0.65), linewidth = 1.2,
                linestyle = :dash)
    end
end

function phase_axis(pos, title, subtitle)
    ax = Axis(pos;
        title = title,
        subtitle = subtitle,
        titlesize = 15,
        subtitlesize = 11,
        subtitlecolor = INK_MUTED,
        xlabel = "τ  (decay time constant, ms)",
        ylabel = "window  (hard gate, ms)",
        xlabelsize = 12,
        ylabelsize = 12,
        xticks = log_ticks(TAUS),
        yticks = log_ticks(WINDOWS),
        xgridvisible = false,
        ygridvisible = false,
        xticklabelsize = 10,
        yticklabelsize = 10,
    )
    return ax
end

ink_for(hex) = begin
    c = parse(RGBf, hex)
    0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b > 0.55 ? :black : :white
end

fig = Figure(size = (1720, 1010), backgroundcolor = :white)

Label(fig[1, 1:6], "Memory Gate — where continuous attention keeps a recent target and drops a stale one";
      fontsize = 21, font = :bold, color = INK_PRIMARY, halign = :left, padding = (8, 0, 0, 4))
Label(fig[2, 1:6],
      "Scene (fixed): query volley at t=0 · target on neuron 1 at lag $(Int(TARGET_LAG)) ms · " *
      "stale same-neuron distractors at $(Int(STALE_LAGS[1])) and $(Int(STALE_LAGS[2])) ms · " *
      "weak competitor on neuron 2 at $(COMPETITOR_LAG) ms · unrelated neuron 3 at $(Int(UNRELATED_LAG)) ms. " *
      "Dashed lines mark window = target / stale lags.";
      fontsize = 11.5, color = INK_MUTED, halign = :left, padding = (8, 0, 0, 8))

ax_share = phase_axis(fig[3, 1], "A · target attention share",
                      "target mass ÷ total attention mass")
hm_share = heatmap!(ax_share, xs, ys, share_map;
                    colormap = :Blues, colorrange = (0.0, 0.8))
reference_lines!(ax_share)
Colorbar(fig[3, 2], hm_share; label = "target share", labelsize = 11, ticklabelsize = 10,
         width = 12)

ax_leak = phase_axis(fig[3, 3], "B · stale leakage",
                     "stale ÷ (target + stale) on the signal neuron")
hm_leak = heatmap!(ax_leak, xs, ys, leak_map;
                   colormap = :Reds, colorrange = (0.0, 1.0))
reference_lines!(ax_leak)
Colorbar(fig[3, 4], hm_leak; label = "stale leakage", labelsize = 11, ticklabelsize = 10,
         width = 12)

ax_open = phase_axis(fig[3, 5], "C · stale leakage with the hard gate OFF",
                     "counterfactual at window = ∞ — the τ-only half of panel B")
hm_open = heatmap!(ax_open, xs, ys, leak_open_map;
                   colormap = :Reds, colorrange = (0.0, 1.0))
reference_lines!(ax_open)
Colorbar(fig[3, 6], hm_open; label = "stale leakage (τ only)", labelsize = 11,
         ticklabelsize = 10, width = 12)

ax_regime = phase_axis(fig[4, 1:2], "D · regime map",
                       "black outline = region where top-1 is the target neuron")
heatmap!(ax_regime, xs, ys, Float64.(regime_map);
         colormap = cgrad([parse(RGBf, c) for c in REGIME_COLORS], 5; categorical = true),
         colorrange = (0.5, 5.5))
reference_lines!(ax_regime)

regime_counts = [count(==(k), regime_map) for k in 1:length(REGIMES)]
for k in 1:length(REGIMES)
    idx = findall(==(k), regime_map)
    isempty(idx) && continue
    regime_counts[k] < 0.03 * length(regime_map) && continue
    kx = [xs[I[1]] for I in idx]
    ky = [ys[I[2]] for I in idx]
    cx, cy = median(kx), median(ky)
    m = argmin((kx .- cx) .^ 2 .+ (ky .- cy) .^ 2)
    # Tall, narrow bands get a rotated label so it stays inside its own region.
    tall = (maximum(kx) - minimum(kx)) < 0.5 * (maximum(ky) - minimum(ky))
    text!(ax_regime, kx[m], ky[m]; text = REGIME_LABELS[k], align = (:center, :center),
          fontsize = 11, font = :bold, rotation = tall ? π / 2 : 0.0,
          color = ink_for(REGIME_COLORS[k]))
end

# Where the target neuron actually wins top-1. Deliberately *not* the same shape
# as the regime map: a selective gate does not guarantee a correct top-1.
contour!(ax_regime, xs, ys, Float64.(top1_map .== SIGNAL);
         levels = [0.5], color = (:black, 0.9), linewidth = 2.5)

ax_slice_w = Axis(fig[4, 3:4];
    title = "E · slice at fixed τ",
    subtitle = "solid = target share · dashed = stale leakage",
    titlesize = 15, subtitlesize = 11, subtitlecolor = INK_MUTED,
    xlabel = "window (ms)", ylabel = "fraction",
    xlabelsize = 12, ylabelsize = 12, xscale = log10,
    xticks = ([3, 10, 30, 100, 240], ["3", "10", "30", "100", "240"]),
    xticklabelsize = 10, yticklabelsize = 10,
    xgridvisible = false, ygridcolor = (:black, 0.08), ygridwidth = 1.0,
)
for lag in (TARGET_LAG, STALE_LAGS...)
    vlines!(ax_slice_w, [Float64(lag)]; color = (:black, 0.28), linewidth = 1.0,
            linestyle = :dot)
end
for (n, i) in pairs(tau_slice_idx)
    lines!(ax_slice_w, Float64.(WINDOWS), Float64.(share_map[i, :]);
           color = SLICE_COLORS[n], linewidth = 2.0, label = "τ = $(TAUS[i]) ms")
    lines!(ax_slice_w, Float64.(WINDOWS), Float64.(leak_map[i, :]);
           color = SLICE_COLORS[n], linewidth = 2.0, linestyle = :dash)
end
ylims!(ax_slice_w, -0.03, 1.03)
axislegend(ax_slice_w; position = :lt, framevisible = false, labelsize = 10, patchsize = (18, 2))

ax_slice_t = Axis(fig[4, 5:6];
    title = "F · slice at fixed window",
    subtitle = "solid = target share · dashed = stale leakage",
    titlesize = 15, subtitlesize = 11, subtitlecolor = INK_MUTED,
    xlabel = "τ (ms)", ylabel = "fraction",
    xlabelsize = 12, ylabelsize = 12, xscale = log10,
    xticks = ([1, 3, 10, 30, 100, 300], ["1", "3", "10", "30", "100", "300"]),
    xticklabelsize = 10, yticklabelsize = 10,
    xgridvisible = false, ygridcolor = (:black, 0.08), ygridwidth = 1.0,
)
for (n, j) in pairs(window_slice_idx)
    lines!(ax_slice_t, Float64.(TAUS), Float64.(share_map[:, j]);
           color = SLICE_COLORS[n], linewidth = 2.0, label = "window = $(WINDOWS[j]) ms")
    lines!(ax_slice_t, Float64.(TAUS), Float64.(leak_map[:, j]);
           color = SLICE_COLORS[n], linewidth = 2.0, linestyle = :dash)
end
ylims!(ax_slice_t, -0.03, 1.03)
axislegend(ax_slice_t; position = :lt, framevisible = false, labelsize = 10, patchsize = (18, 2))

Legend(fig[5, 1:6],
    [PolyElement(color = parse(RGBf, REGIME_COLORS[k]), strokecolor = :white, strokewidth = 1)
     for k in 1:length(REGIMES)],
    ["$(REGIME_LABELS[k]) — $(round(100 * regime_counts[k] / length(regime_map); digits = 1))% of plane"
     for k in 1:length(REGIMES)],
    "Regimes (panel D)";
    orientation = :horizontal, framevisible = false, labelsize = 11, titlesize = 12,
    titleposition = :left, patchsize = (16, 12))

for gap in (1, 3, 5)          # axis → its own colorbar
    colgap!(fig.layout, gap, 6)
end
for gap in (2, 4)             # between panel groups
    colgap!(fig.layout, gap, 30)
end
rowgap!(fig.layout, 3, 20)
rowsize!(fig.layout, 3, Relative(0.42))
rowsize!(fig.layout, 4, Relative(0.42))

figure_file = figure_path(SLUG)
save(figure_file, fig)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

function slice_table_over_windows(i)
    io = IOBuffer()
    println(io, "| window (ms) | target in win | stale in win | target mass | stale mass | target share | stale leakage | target÷stale | top-1 | regime |")
    println(io, "|---:|:--:|:--:|---:|---:|---:|---:|---:|:--:|:--|")
    for j in eachindex(WINDOWS)
        r = row_at(i, j)
        @printf(io, "| %.6g | %s | %d/2 | %s | %s | %s | %s | %s | %s | %s |\n",
            r.window, r.target_in_window ? "yes" : "no", r.n_stale_in_window,
            _fmt(r.target_mass), _fmt(r.stale_mass), _fmt(r.target_share),
            _fmt(r.stale_leakage), _fmt2(r.target_stale_ratio),
            r.top1_neuron == 0 ? "—" : string(r.top1_neuron), r.regime)
    end
    return rstrip(String(take!(io)))
end

function slice_table_over_taus(j)
    io = IOBuffer()
    println(io, "| τ (ms) | target mass | stale mass | target mass (no window) | stale mass (no window) | target share | stale leakage | top-1 | regime |")
    println(io, "|---:|---:|---:|---:|---:|---:|---:|:--:|:--|")
    for i in eachindex(TAUS)
        r = row_at(i, j)
        @printf(io, "| %.6g | %s | %s | %s | %s | %s | %s | %s | %s |\n",
            r.tau, _fmt(r.target_mass), _fmt(r.stale_mass),
            _fmt(r.target_mass_open_window), _fmt(r.stale_mass_open_window),
            _fmt(r.target_share), _fmt(r.stale_leakage),
            r.top1_neuron == 0 ? "—" : string(r.top1_neuron), r.regime)
    end
    return rstrip(String(take!(io)))
end

n_cond = length(rows)
top1_correct_frac = count(r -> r.top1_correct, rows) / n_cond
best = rows[argmax([r.target_share for r in rows])]
worst_leak = rows[argmax([r.stale_leakage for r in rows])]

# Cells where the hard window (not decay) is what removed the target.
window_killed = count(r -> !r.target_in_window && r.target_mass_open_window >= TARGET_FLOOR, rows)
# Cells where decay (not the window) is what removed the target.
decay_killed = count(r -> r.target_in_window && r.target_mass < TARGET_FLOOR, rows)
# Cells where the gate is selective but top-1 is still wrong.
selective_but_wrong = count(r -> r.regime == "selective_gate" && !r.top1_correct, rows)

regime_summary = join(
    ["- `$(REGIMES[k])` (**$(REGIME_LABELS[k])**): $(regime_counts[k]) / $(n_cond) conditions " *
     "($(round(100 * regime_counts[k] / n_cond; digits = 1))% of the plane)"
     for k in 1:length(REGIMES)], "\n")

summary = """
# Memory Gate — the τ × window trade-off

**Slug:** `memory_gate` · **Kernel:** `spike_attention_continuous` · **Readout:** identity
(transparent, so the kernel output *is* the per-neuron attention vector).

## Research question

Across the `τ × window` plane, where does continuous attention preserve a recent
target interaction while suppressing a stale same-neuron distractor?

## What the two parameters actually do

`spike_attention_continuous` applies two independent mechanisms:

1. **`τ` — soft exponential decay.** Every admitted pair is scaled by
   `temporal_weight(dt, τ) = exp(-abs(dt)/τ)`. This is smooth and never reaches zero.
2. **`TemporalBuffer.window` — a hard admissibility boundary.** A `(source, context)`
   pair contributes only when `abs(dt) <= min(source.window, context.window)`. This is a
   step function: a pair is either fully counted or fully discarded.

Note that the kernel does **not** call `prune!`; admissibility is decided per event
*pair*, not by buffer age. Because both buffers carry the same swept window here, the
effective gate is exactly that window.

## Scene (deterministic, fixed across the whole sweep)

| role | neuron | lag from `t=0` | value |
|:--|--:|--:|--:|
| query volley (source) | 1, 2, 3 | 0 | $(SOURCE_VALUE) |
| **target** (want kept) | $(SIGNAL) | $(TARGET_LAG) ms | $(TARGET_VALUE) |
| **stale distractor** (want rejected) | $(SIGNAL) | $(STALE_LAGS[1]) ms | $(STALE_VALUE) |
| **stale distractor** (want rejected) | $(SIGNAL) | $(STALE_LAGS[2]) ms | $(STALE_VALUE) |
| weak recent competitor | $(COMPETITOR) | $(COMPETITOR_LAG) ms | $(COMPETITOR_VALUE) |
| unrelated far-past neuron | $(UNRELATED) | $(UNRELATED_LAG) ms | $(UNRELATED_VALUE) |

The stale distractors sit on the **same neuron** as the target, so they corrupt the
target neuron's own score rather than competing for top-1. Neurons 2 and 3 make top-1
meaningful.

Because the kernel accumulates additively over context events, target / stale /
competitor / unrelated masses are measured by re-running the **real kernel** on context
sub-scenes; the script asserts that the parts sum back to the full-scene total at every
one of the $(n_cond) conditions.

Times are nominal milliseconds. TemporalFocus is unit-agnostic (`Float32` throughout);
no unit semantics are attached, and in particular none are financial.

## Sweep

- τ: $(N_TAU) log-spaced values, $(TAUS[1]) → $(TAUS[end]) ms (short-, medium- and
  long-memory relative to the $(TARGET_LAG) / $(STALE_LAGS[1]) / $(STALE_LAGS[2]) ms lags).
- window: $(N_WINDOW) log-spaced values, $(WINDOWS[1]) → $(WINDOWS[end]) ms — starting
  **below** the target lag ($(TARGET_LAG) ms) and ending **beyond** the largest stale lag
  ($(STALE_LAGS[2]) ms) and the unrelated lag ($(UNRELATED_LAG) ms).
- $(n_cond) conditions total. Determinism: the scene contains no random draws at all
  (seed `$(SEED)` is recorded and applied for contract compliance only), so re-running
  reproduces `metrics.csv` and `figure.png` exactly.

## Zero and tie handling

- `target_share = target_mass / total_mass`, defined as `0` when `total_mass == 0`.
- `stale_leakage = stale_mass / (target_mass + stale_mass)`, defined as `0` when the
  denominator is `0`.
- `target_stale_ratio = target_mass / stale_mass`; `Inf` when the gate admits the target
  but no stale mass, `NaN` when neither contributes. It is reported in `metrics.csv` and
  in the slice tables; the figure plots the bounded `stale_leakage` instead, because the
  ratio is `Inf` across the entire selective band and would render as one flat color.
- `top1_neuron` is `argmax` of the full attention vector (ties break toward the lowest
  neuron id) and is `0` when no neuron has any mass.

## Regimes observed

Classification order (first match wins):

1. `window_clipped` — the target lag is outside the window.
2. `decay_starved` — the target is inside the window but `target_mass < $(TARGET_FLOOR)`.
3. `selective_gate` — the target is in, **every** stale spike is out.
4. `soft_decay` — stale spikes are admitted but leakage `< $(LEAK_MAX)`.
5. `over_retentive` — stale spikes are admitted and leakage `>= $(LEAK_MAX)`.

$(regime_summary)

Boundaries 1 and 3 are **exact structural facts** — they follow from `abs(dt) <= window`
and nothing else. The `soft_decay` / `over_retentive` split is the one *soft* boundary:
it moves if `LEAK_MAX` is changed. The `decay_starved` cut depends on `TARGET_FLOOR` in
the same way. Both thresholds are recorded in `config.toml`.

## Results

All four hypothesized regimes are present, and they are separated by the boundaries the
hypothesis predicted:

- **Too narrow / hard-clipped.** For window < $(TARGET_LAG) ms the target pair is discarded
  outright and the signal neuron's attention is exactly `0`, for *every* τ including the
  longest. τ cannot compensate: this is a step, not a gradient.
- **Selective gate.** For window between $(TARGET_LAG) ms and $(STALE_LAGS[1]) ms the target is
  admitted and both stale spikes are structurally excluded — `stale_mass` is exactly `0`
  and `target_stale_ratio` is `Inf` across that whole band, again independent of τ.
- **Soft-decay regime.** Above window ≈ $(STALE_LAGS[1]) ms the stale spikes become admissible
  and leakage is set purely by τ.
- **Over-retentive.** With a wide window *and* a long τ, stale mass approaches and then
  passes the target: peak stale leakage in the sweep is
  **$(_fmt(worst_leak.stale_leakage))** at τ = $(worst_leak.tau) ms, window =
  $(worst_leak.window) ms.

Best target share on the plane: **$(_fmt(best.target_share))** at τ = $(best.tau) ms,
window = $(best.window) ms (regime `$(best.regime)`). Top-1 is the target neuron in
**$(round(100 * top1_correct_frac; digits = 1))%** of conditions.

## Separating decay from the hard window

This is the point of the experiment, and the two mechanisms are cleanly separable in the
data:

- `target_mass_open_window` / `stale_mass_open_window` are the same masses recomputed with
  the gate disabled (`window = Inf`). They depend on **τ only**.
- The difference between those and the gated masses is attributable to the **window only**.

**Panels B and C are the gated and un-gated versions of the same ratio, side by side.** C
switches the hard gate off, so it is a function of τ alone — perfectly flat in the window
direction — and any cell where B is pale while C is red is a cell where the hard window,
not decay, is doing the suppressing.

`stale_leakage` is a **ratio**, though, so B and C are a counterfactual *comparison*, not
an additive split: the hard gate moves both the numerator and the denominator, and `B - C`
is not itself a "window contribution". The additive decomposition lives in the mass
columns, quantified next.

Concretely:

- **$(window_killed)** conditions have `target_mass == 0` while the τ-only counterfactual
  would have kept `>= $(TARGET_FLOOR)` of it. That loss is 100% the hard window.
- **$(decay_killed)** conditions admit the target through the gate yet still end below
  `$(TARGET_FLOOR)`. That loss is 100% exponential decay.
- `stale_clipped_fraction` gives the same decomposition for the stale side: it is exactly
  `1.0` while both stale spikes are gated out, drops to a partial value once one of the
  two is admitted, and reaches `0.0` when the window covers both — a staircase in the
  window direction, with a smooth τ gradient inside each step.

The signature difference is visible directly in the slices (panels E and F, tables below):
along the **window** axis the curves move in flat steps that snap at $(TARGET_LAG),
$(STALE_LAGS[1]) and $(STALE_LAGS[2]) ms; along the **τ** axis the same quantities move as
smooth sigmoids with no discontinuity anywhere.

## Contrary / unexpected findings (kept, not tuned away)

1. **A fifth regime that the hypothesis did not list.** At τ well below the target lag the
   target is admitted by the window yet decays to nothing anyway. Calling that "selective"
   would be wrong — the model is not being selective, it has forgotten everything — so it
   is labeled `decay_starved` and reported separately. It occupies
   $(round(100 * regime_counts[2] / n_cond; digits = 1))% of the plane.
2. **A selective gate does not guarantee correct top-1.** In **$(selective_but_wrong)**
   conditions the gate is doing exactly what it should (target in, all stale out) and the
   target still loses top-1 to the weak but nearer competitor on neuron 2, because at short
   τ a lag-$(COMPETITOR_LAG) ms spike worth $(COMPETITOR_VALUE) outweighs a lag-$(TARGET_LAG) ms
   spike worth $(TARGET_VALUE). Regime membership describes the *memory gate*; it is not a
   proxy for task accuracy. The black outline in panel D is the top-1-correct boundary, and
   it deliberately does not follow the regime borders.
3. **The "optimum" is a plateau, not a peak.** Target share is essentially flat in the
   window direction across the entire selective band, so there is no sharp best window —
   only a wide safe corridor bounded below by the target lag and above by the nearest stale
   lag. Any downstream tuning should treat this as an interval, not a point.

## Scope boundary vs. issue #48 (Focus Under Fire)

This experiment sweeps **parameters** (τ × window) against a *fixed*, minimal scene. It
deliberately does **not** sweep distractor load, count, rate, or amplitude, and it draws no
conclusions about robustness under increasing interference — that is issue #48's question.
The single overlapping concept is "a distractor should be suppressed"; here that is only
the readout used to locate parameter boundaries, not the independent variable.

Per the issue's non-goals: nothing here proposes auto-tuning τ or window in the core API,
and no domain-specific (in particular no financial) meaning is attached to either
parameter.

## Slices

### Fixed τ = $(TAUS[tau_slice_idx[2]]) ms — sweeping the window (medium memory)

$(slice_table_over_windows(tau_slice_idx[2]))

### Fixed window = $(WINDOWS[window_slice_idx[3]]) ms — sweeping τ (window wide enough for everything)

$(slice_table_over_taus(window_slice_idx[3]))

Panels E and F plot these plus τ = $(TAUS[tau_slice_idx[1]]) / $(TAUS[tau_slice_idx[3]]) ms
and window = $(WINDOWS[window_slice_idx[1]]) / $(WINDOWS[window_slice_idx[2]]) ms. Every row
above is a verbatim row of `metrics.csv`.

## Artifacts

- `config.toml` — the exact configuration used
- `metrics.csv` — $(n_cond) rows, one per `(τ, window)` condition
- `summary.md` — this file
- `figure.png` — the memory control panel:
  - **A** target attention share over the plane
  - **B** stale leakage over the plane (τ *and* window)
  - **C** the same leakage with the hard gate off (τ only) — B's counterfactual
  - **D** regime map, with the top-1-correct boundary outlined
  - **E** slice at fixed τ, sweeping the window — flat steps at the event lags
  - **F** slice at fixed window, sweeping τ — smooth sigmoids, no discontinuity

## Reproduce

```bash
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=experiments experiments/memory_gate.jl
```
"""

summary_path = write_summary(SLUG, summary)

# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

println("memory_gate — ", n_cond, " conditions over ", N_TAU, " τ × ", N_WINDOW, " window")
println("  regime counts:")
for k in 1:length(REGIMES)
    @printf("    %-16s %4d  (%4.1f%%)\n", REGIMES[k], regime_counts[k],
            100 * regime_counts[k] / n_cond)
end
@printf("  best target share    : %.4f at τ=%g ms, window=%g ms\n",
        best.target_share, best.tau, best.window)
@printf("  peak stale leakage   : %.4f at τ=%g ms, window=%g ms\n",
        worst_leak.stale_leakage, worst_leak.tau, worst_leak.window)
@printf("  top-1 correct        : %.1f%% of conditions\n", 100 * top1_correct_frac)
@printf("  target lost to window: %d conditions;  lost to decay: %d conditions\n",
        window_killed, decay_killed)
println("  wrote ", relpath(config_path, repo_root()))
println("  wrote ", relpath(metrics_path, repo_root()))
println("  wrote ", relpath(figure_file, repo_root()))
println("  wrote ", relpath(summary_path, repo_root()))
