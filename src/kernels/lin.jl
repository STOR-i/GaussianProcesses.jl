# Linear covariance function

@inline dotijp(X::MatF64, i::Int, j::Int, p::Int) = X[p,i]*X[p,j]
@inline function dotij(X::MatF64, i::Int, j::Int, dim::Int)
	s=zero(eltype(X))
	@inbounds @simd for p in 1:dim
		s+=dotijp(X,i,j,p)
	end
	return s
end
include("lin_iso.jl")
include("lin_ard.jl")


"""
    Lin(ll::Union{Float64,Vector{Float64}})

Create linear kernel with length scale `exp.(ll)`.

See also [`LinIso`](@ref) and [`LinArd`](@ref).
"""
Lin(ll::Float64) = LinIso(ll)
Lin(ll::VecF64) = LinArd(ll)
