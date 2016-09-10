# Linear covariance function

@inline dotij(X::Matrix{Float64}, i::Int, j::Int, dim::Int) = sum((X[d,i]*X[d,j]) for d in 1:dim)
@inline dotijp(X::Matrix{Float64}, i::Int, j::Int, p::Int) = X[p,i]*X[p,j]
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
