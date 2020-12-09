# Linear covariance function

@inline dotijp(X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, p::Int) = X1[p,i]*X2[p,j]
@inline function dotij(X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int)
	s=zero(promote_type(eltype(X1), eltype(X2)))
	@inbounds @simd for p in 1:dim
		s+=dotijp(X1,X2,i,j,p)
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
Lin(ll::AbstractVector{<:Real}) = LinArd(ll)
