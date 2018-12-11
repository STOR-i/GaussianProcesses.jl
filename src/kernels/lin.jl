# Linear covariance function

@inline dotijp(X::AbstractMatrix, i::Int, j::Int, p::Int) = X[p,i]*X[p,j]
@inline function dotij(X::AbstractMatrix, i::Int, j::Int, dim::Int)
	s=zero(eltype(X))
	@inbounds @simd for p in 1:dim
		s+=dotijp(X,i,j,p)
	end
	return s
end
include("lin_iso.jl")
include("lin_ard.jl")


"""
    Lin(ll::Union{Real,Vector{Real}})

Create linear kernel with length scale `exp.(ll)`.

See also [`LinIso`](@ref) and [`LinArd`](@ref).
"""
Lin(ll::Real) = LinIso(ll)
Lin(ll::AbstractVector) = LinArd(ll)
