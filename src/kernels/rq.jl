# Rational Quadratic Covariance Function

include("rq_iso.jl")
include("rq_ard.jl")

"""
    RQ(ll::Union{Real,Vector{Real}}, lσ::Real, lα::Real)

Create Rational Quadratic kernel with length scale `exp.(ll)`, signal standard deviation
`exp(lσ)`, and shape parameter `exp(lα)`.

See also [`RQIso`](@ref) and [`RQArd`](@ref).
"""
RQ(ll::Real, lσ::Real, lα::Real) = RQIso(ll, lσ, lα)
RQ(ll::Vector{Real}, lσ::Real, lα::Real) = RQArd(ll, lσ, lα)
