# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    spike_attention_continuous(source_buffer, context_buffer, readout; τ=1.0f0)

Windowed temporal attention over [`TemporalBuffer`](@ref)s.

Like [`spike_attention_temporal`](@ref), but only pairs of events whose absolute
time difference is within `min(source_buffer.window, context_buffer.window)`
contribute. Matching neuron IDs are required; contributions are scaled by
[`temporal_weight`](@ref).

# Arguments
- `source_buffer::TemporalBuffer`: source events and window
- `context_buffer::TemporalBuffer`: context events and window
- `readout::AbstractMatrix`: readout with one row per neuron (row count must be ≥ 1)
- `τ::Real`: positive time constant for recency weighting (default `1.0f0`)

# Returns
- readout vector `transpose(readout) * attention`, where `attention` is a
  `Vector{Float32}` of length `size(readout, 1)`

# Throws
- `ArgumentError` if `τ ≤ 0`, the readout has no rows, a source `neuron_id`
  is outside `1:size(readout, 1)`, either buffer window is `NaN`, or any
  spike time, spike value, or readout entry is `NaN` or `Inf`. A `+Inf`
  window is allowed and admits every finite pair (`abs(dt) <= Inf`).
"""
function spike_attention_continuous(
    source_buffer::TemporalBuffer,
    context_buffer::TemporalBuffer,
    readout::AbstractMatrix;
    τ::Real = 1.0f0,
)
    τ_f32 = Float32(τ)
    τ_f32 > 0f0 || throw(ArgumentError("τ must be positive"))
    n = _check_positive_rows(readout)
    isnan(source_buffer.window) && throw(ArgumentError("source window must not be NaN"))
    isnan(context_buffer.window) && throw(ArgumentError("context window must not be NaN"))
    _require_finite_events(source_buffer.events)
    _require_finite_events(context_buffer.events)
    attention = zeros(Float32, n)
    window = min(source_buffer.window, context_buffer.window)

    @inbounds for source_event in source_buffer.events
        source_id = _check_neuron_id(source_event.neuron_id, n, "source")
        for context_event in context_buffer.events
            if source_id == context_event.neuron_id
                dt = source_event.t - context_event.t
                if abs(dt) <= window
                    attention[source_id] += source_event.value * context_event.value *
                                            _temporal_weight_unchecked(dt, τ_f32)
                end
            end
        end
    end

    return _apply_readout(attention, readout)
end
