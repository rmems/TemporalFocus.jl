# SPDX-License-Identifier: MIT OR Apache-2.0

function _require_finite_weights!(weights::AbstractVector{<:AbstractFloat})
    for w in weights
        isfinite(w) || throw(ArgumentError("weights must be finite, got $w"))
    end
    return weights
end

"""
    normalize_l1!(weights) -> weights

In-place L1 normalization of a weight vector.

Divides each element by `sum(weights)` when the sum is positive. If the
vector is empty or the sum is zero or negative, `weights` is left unchanged
and no division is performed (so a zero sum cannot produce `NaN`).

# Arguments
- `weights::AbstractVector{<:AbstractFloat}`: vector mutated in place

# Returns
- the same `weights` vector after normalization (or unchanged if empty or sum ≤ 0)

# Throws
- `ArgumentError` if any element is `NaN` or `Inf`, or if a finite vector
  overflows to a non-finite sum
"""
function normalize_l1!(weights::AbstractVector{<:AbstractFloat})
    _require_finite_weights!(weights)
    isempty(weights) && return weights
    total = sum(weights)
    isfinite(total) || throw(ArgumentError("weight sum must be finite, got $total"))
    if total > zero(total)
        weights ./= total
    end
    return weights
end

"""
    normalize_max!(weights) -> weights

In-place max normalization of a weight vector.

Divides each element by `maximum(weights)` when the peak is positive. If the
vector is empty or the peak is zero or negative, `weights` is left unchanged
and no division is performed.

# Arguments
- `weights::AbstractVector{<:AbstractFloat}`: vector mutated in place

# Returns
- the same `weights` vector after normalization (or unchanged if empty or peak ≤ 0)

# Throws
- `ArgumentError` if any element is `NaN` or `Inf`
"""
function normalize_max!(weights::AbstractVector{<:AbstractFloat})
    _require_finite_weights!(weights)
    isempty(weights) && return weights
    peak = maximum(weights)
    if peak > zero(peak)
        weights ./= peak
    end
    return weights
end
