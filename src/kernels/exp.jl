# Exponential Covariance Function 

include("exp_iso.jl")
include("exp_ard.jl")

@doc """
# Description
Constructor the Exponential kernel

# See also
ExpIso, ExpArd
""" ->
ExpKern(ll::Float64, lσ::Float64) = ExpIso(ll, lσ)
ExpKern(ll::Vector{Float64}, lσ::Float64) = ExpArd(ll, lσ)
