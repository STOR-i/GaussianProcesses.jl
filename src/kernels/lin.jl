# Linear covariance function

include("lin_iso.jl")
include("lin_ard.jl")


@doc """
# Description
Constructors for linear kernel

# See also
LinIso, LinArd
""" ->
Lin(ll::Float64) = LinIso(ll)
Lin(ll::Vector{Float64}) = LinArd(ll)
