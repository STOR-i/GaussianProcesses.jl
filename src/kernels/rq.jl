# Rational Quadratic Covariance Function 

include("rq_iso.jl")
include("rq_ard.jl")

@doc """
# Description
Constructor for the Rational Quadratic kernel

# See also
RQIso, RQArd
""" ->
RQ(ll::Float64, lσ::Float64, lα::Float64) = RQIso(ll, lσ, lα)
RQ(ll::Vector{Float64}, lσ::Float64, lα::Float64) = RQArd(ll, lσ, lα)
