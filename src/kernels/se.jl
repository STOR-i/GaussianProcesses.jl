# Squared Exponential Covariance Function 

include("se_iso.jl")
include("se_ard.jl")

@doc """
# Description
Constructor the Squared Exponential kernel

# See also
SEIso, SEArd
""" ->
SE(ll::Float64, lσ::Float64) = SEIso(ll, lσ)
SE(ll::Vector{Float64}, lσ::Float64) = SEArd(ll, lσ)
