# SPDX-License-Identifier: MIT OR Apache-2.0

@inline _temporal_weight_unchecked(dt::Float32, τ::Float32) = exp(-abs(dt) / τ)

@inline function _require_finite_dt(dt::Float32)
    isfinite(dt) || throw(ArgumentError("spike time difference must be finite, got $dt"))
    return dt
end

@inline function _require_positive_tau(τ::Real)
    τ_f32 = Float32(τ)
    isfinite(τ_f32) && τ_f32 > 0f0 ||
        throw(ArgumentError("τ must be finite and positive, got $τ"))
    return τ_f32
end

"""
    temporal_weight(dt, τ) -> Float32

Exponential recency weight for a time difference.

Computes `exp(-abs(dt) / τ)`. Closer-in-time spikes receive higher weight.

# Arguments
- `dt::Real`: time difference between spikes
- `τ::Real`: positive time constant (converted to `Float32`)

# Returns
- `Float32` weight in `(0, 1]`

# Throws
- `ArgumentError` if `τ` is not finite and positive after conversion to
  `Float32` (values that overflow to `Inf32`, such as `1e39`, are rejected),
  or if `dt` is not finite
"""
@inline function temporal_weight(dt::Real, τ::Real)
    τ_f32 = _require_positive_tau(τ)
    dt_f32 = Float32(dt)
    isfinite(dt_f32) || throw(ArgumentError("dt must be finite, got $dt_f32"))
    return _temporal_weight_unchecked(dt_f32, τ_f32)
end

"""
    spike_attention_temporal(source_spikes, context_spikes, readout; τ=1.0f0)

Coincidence attention with exponential temporal decay.

For each source event, accumulates products of source and context spike values
on matching neuron IDs, scaled by [`temporal_weight`](@ref) of their time
difference, then applies the readout matrix.

# Arguments
- `source_spikes::SpikeTrain`: source spike train
- `context_spikes::SpikeTrain`: context spike train
- `readout::AbstractMatrix`: readout with one row per neuron (row count must be ≥ 1)
- `τ::Real`: positive time constant for recency weighting (default `1.0f0`)

# Returns
- readout vector `transpose(readout) * attention`, where `attention` is a
  `Vector{Float32}` of length `size(readout, 1)`

# Throws
- `ArgumentError` if `τ` is not finite and positive (including values that
  overflow `Float32` to `Inf32`), the readout has no rows, a source `neuron_id`
  is outside `1:size(readout, 1)`, or any spike time, spike value, or readout
  entry is `NaN` or `Inf`. A spike-time difference that overflows `Float32`
  also raises `ArgumentError`
"""
function spike_attention_temporal(
    source_spikes::SpikeTrain,
    context_spikes::SpikeTrain,
    readout::AbstractMatrix;
    τ::Real = 1.0f0,
)
    τ_f32 = _require_positive_tau(τ)
    n = _check_positive_rows(readout)
    _require_finite_events(source_spikes.events)
    _require_finite_events(context_spikes.events)
    attention = zeros(Float32, n)

    @inbounds for source_event in source_spikes.events
        source_id = _check_neuron_id(source_event.neuron_id, n, "source")
        for context_event in context_spikes.events
            if source_id == context_event.neuron_id
                dt = _require_finite_dt(source_event.t - context_event.t)
                attention[source_id] += source_event.value * context_event.value *
                                        _temporal_weight_unchecked(dt, τ_f32)
            end
        end
    end

    return _apply_readout(attention, readout)
end
