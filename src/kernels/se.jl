# Squared Exponential Covariance Function

include("se_iso.jl")
include("se_ard.jl")

"""
    SE(ll::Union{Float64,Vector{Float64}}, lσ::Float64)

Create squared exponential kernel with length scale `exp.(ll)` and signal standard deviation
`exp(lσ)`.

See also [`SEIso`](@ref) and [`SEArd`](@ref).
"""
SE(ll::Float64, lσ::Float64) = SEIso(ll, lσ)
SE(ll::Vector{Float64}, lσ::Float64) = SEArd(ll, lσ)
