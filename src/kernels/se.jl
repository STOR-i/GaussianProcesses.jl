# Squared Exponential Covariance Function

include("se_iso.jl")
include("se_ard.jl")

"""
    SE(ll::Union{Real,Vector{Real}}, lσ::Real)

Create squared exponential kernel with length scale `exp.(ll)` and signal standard deviation
`exp(lσ)`.

See also [`SEIso`](@ref) and [`SEArd`](@ref).
"""
SE(ll::Real, lσ::Real) = SEIso(ll, lσ)
SE(ll::Vector{Real}, lσ::Real) = SEArd(ll, lσ)
