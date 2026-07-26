# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    SpikeEvent(neuron_id, t, value=1.0f0)

A single spike event at a neuron.

# Arguments
- `neuron_id::Integer`: 1-based neuron index that produced the spike
- `t::Real`: spike time (stored as `Float32`)
- `value::Real`: spike amplitude or weight (default `1.0f0`, stored as `Float32`)

# Fields
- `neuron_id::Int`
- `t::Float32`
- `value::Float32`
"""
struct SpikeEvent
    neuron_id::Int
    t::Float32
    value::Float32
end

SpikeEvent(neuron_id::Integer, t::Real, value::Real = 1.0f0) =
    SpikeEvent(Int(neuron_id), Float32(t), Float32(value))

"""
    SpikeTrain(events=SpikeEvent[])

A collection of [`SpikeEvent`](@ref)s stored in vector order.

Attention kernels treat trains as bags of events (order does not affect
scores), but `==` / `isequal` / `hash` compare the underlying `events` vector
element-wise, so order is part of equality. Callers that need order-invariant
identity should normalize event order (or use a multiset) themselves.

**Hash keys:** `events` is a mutable vector. Do not mutate a `SpikeTrain`
(including pushing/replacing events) while it is stored as a key in a `Dict`
or element of a `Set` — that changes its hash and breaks lookup/deletion.

# Arguments
- `events`: vector of `SpikeEvent` (copied into a `Vector{SpikeEvent}`)

# Fields
- `events::Vector{SpikeEvent}`
"""
struct SpikeTrain
    events::Vector{SpikeEvent}
end

SpikeTrain() = SpikeTrain(SpikeEvent[])
SpikeTrain(events::AbstractVector{<:SpikeEvent}) = SpikeTrain(collect(events))

"""
    TemporalBuffer(window, events=SpikeEvent[])

A sliding temporal window of spike events.

Events older than `window` relative to a reference time can be removed with
[`prune!`](@ref).

**Hash keys:** `events` is mutated by [`prune!`](@ref) and may be edited
directly. Do not mutate a `TemporalBuffer` while it is stored as a key in a
`Dict` or element of a `Set` — that changes its hash and breaks
lookup/deletion.

# Arguments
- `window::Real`: retention window length (stored as `Float32`)
- `events`: initial events (copied into a `Vector{SpikeEvent}`)

# Fields
- `window::Float32`
- `events::Vector{SpikeEvent}`
"""
struct TemporalBuffer
    window::Float32
    events::Vector{SpikeEvent}
end

TemporalBuffer(window::Real, events::AbstractVector{<:SpikeEvent} = SpikeEvent[]) =
    TemporalBuffer(Float32(window), collect(events))

Base.isempty(train::SpikeTrain) = isempty(train.events)
Base.isempty(buffer::TemporalBuffer) = isempty(buffer.events)

"""
    prune!(buffer::TemporalBuffer, current_time) -> TemporalBuffer

Remove events from `buffer` that fall outside the retention window.

Keeps only events satisfying `(current_time - event.t) <= buffer.window`.
Mutates `buffer.events` in place and returns `buffer`.

# Arguments
- `buffer::TemporalBuffer`: buffer to prune
- `current_time::Real`: reference time (converted to `Float32`)

# Returns
- the same `buffer` after pruning
"""
function prune!(buffer::TemporalBuffer, current_time::Real)
    current_time_f32 = Float32(current_time)
    filter!(event -> (current_time_f32 - event.t) <= buffer.window, buffer.events)
    return buffer
end

function Base.show(io::IO, e::SpikeEvent)
    print(io, "SpikeEvent(neuron_id=", e.neuron_id, ", t=", e.t, ", value=", e.value, ")")
end

function Base.show(io::IO, train::SpikeTrain)
    n = length(train.events)
    print(io, "SpikeTrain(", n, n == 1 ? " event)" : " events)")
end

function Base.show(io::IO, buffer::TemporalBuffer)
    n = length(buffer.events)
    print(io, "TemporalBuffer(window=", buffer.window, ", ", n, n == 1 ? " event)" : " events)")
end

# `==` uses Float32 `==` (±0.0 equal, NaNs not equal).
# `isequal`/`hash` follow Julia float rules (±0.0 distinct, NaNs equal) for Set/Dict.
# Mutable containers (`SpikeTrain` / `TemporalBuffer`) must not be mutated while
# stored as Dict keys or Set elements (hash is content-based over `events`).
Base.:(==)(a::SpikeEvent, b::SpikeEvent) =
    a.neuron_id == b.neuron_id && a.t == b.t && a.value == b.value

Base.isequal(a::SpikeEvent, b::SpikeEvent) =
    a.neuron_id == b.neuron_id && isequal(a.t, b.t) && isequal(a.value, b.value)

Base.hash(a::SpikeEvent, h::UInt) =
    hash(a.value, hash(a.t, hash(a.neuron_id, h)))

Base.:(==)(a::SpikeTrain, b::SpikeTrain) = a.events == b.events

Base.isequal(a::SpikeTrain, b::SpikeTrain) = isequal(a.events, b.events)

Base.hash(a::SpikeTrain, h::UInt) = hash(a.events, h)

Base.:(==)(a::TemporalBuffer, b::TemporalBuffer) =
    a.window == b.window && a.events == b.events

Base.isequal(a::TemporalBuffer, b::TemporalBuffer) =
    isequal(a.window, b.window) && isequal(a.events, b.events)

Base.hash(a::TemporalBuffer, h::UInt) =
    hash(a.events, hash(a.window, h))
