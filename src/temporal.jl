# SPDX-License-Identifier: MIT OR Apache-2.0

@inline _temporal_weight_unchecked(dt::Float32, τ::Float32) = exp(-abs(dt) / τ)

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
- `ArgumentError` if `τ` is not positive, or if `dt` is `NaN` or `Inf`
"""
@inline function temporal_weight(dt::Real, τ::Real)
    τ_f32 = Float32(τ)
    τ_f32 > 0f0 || throw(ArgumentError("τ must be positive"))
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
- `ArgumentError` if `τ ≤ 0`, the readout has no rows, a source `neuron_id`
  is outside `1:size(readout, 1)`, or any spike time, spike value, or readout
  entry is `NaN` or `Inf`
"""
function spike_attention_temporal(
    source_spikes::SpikeTrain,
    context_spikes::SpikeTrain,
    readout::AbstractMatrix;
    τ::Real = 1.0f0,
)
    τ_f32 = Float32(τ)
    τ_f32 > 0f0 || throw(ArgumentError("τ must be positive"))
    n = _check_positive_rows(readout)
    _require_finite_events(source_spikes.events)
    _require_finite_events(context_spikes.events)
    attention = zeros(Float32, n)

    @inbounds for source_event in source_spikes.events
        source_id = _check_neuron_id(source_event.neuron_id, n, "source")
        for context_event in context_spikes.events
            if source_id == context_event.neuron_id
                attention[source_id] += source_event.value * context_event.value *
                                        _temporal_weight_unchecked(source_event.t - context_event.t, τ_f32)
            end
        end
    end

    return _apply_readout(attention, readout)
end
