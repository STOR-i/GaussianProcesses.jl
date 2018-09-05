# Rational Quadratic Covariance Function

include("rq_iso.jl")
include("rq_ard.jl")

"""
    RQ(ll::Union{Float64,Vector{Float64}}, lσ::Float64, lα::Float64)

Create Rational Quadratic kernel with length scale `exp.(ll)`, signal standard deviation
`exp(lσ)`, and shape parameter `exp(lα)`.

See also [`RQIso`](@ref) and [`RQArd`](@ref).
"""
RQ(ll::Float64, lσ::Float64, lα::Float64) = RQIso(ll, lσ, lα)
RQ(ll::Vector{Float64}, lσ::Float64, lα::Float64) = RQArd(ll, lσ, lα)
