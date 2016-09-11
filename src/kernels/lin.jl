# Linear covariance function

@inline dotijp(X::Matrix{Float64}, i::Int, j::Int, p::Int) = X[p,i]*X[p,j]
@inline function dotij(X::Matrix, i::Int, j::Int, dim::Int)
	s=zero(eltype(X))
	@inbounds @simd for p in 1:dim
		s+=dotijp(X,i,j,p)
	end
	return s
end
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
