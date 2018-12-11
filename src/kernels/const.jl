"""
    Const <: Kernel

Constant kernel
```math
k(x,x') = σ²
```
with signal standard deviation ``σ``.
"""
mutable struct Const{T} <: Kernel where {T<:Real}
    "Signal variance"
    σ2::T
    "Priors for kernel parameters"
    priors::Array
end

"""
    Const(lσ::T)

Create `Const` with signal standard deviation `exp(lσ)`.
"""
Const(lσ::T) where T = Const{T}(exp(2 * lσ), [])

function set_params!(cons::Const, hyp::AbstractVector)
    length(hyp) == 1 || throw(ArgumentError("Constant kernel has one parameter, received $(length(hyp))."))
    cons.σ2 = exp(2.0*hyp[1])
end

get_params(cons::Const{T}) where T = T[log(cons.σ2)/2.0]
get_param_names(cons::Const) = [:lσ]
num_params(cons::Const) = 1

cov(cons::Const) = cons.σ2
function cov(cons::Const, x::AbstractVector, y::AbstractVector)
    return cov(cons)
end

@inline dk_dlσ(cons::Const) = 2.0*cov(cons)
@inline function dKij_dθp(cons::Const, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    if p == 1
        return dk_dlσ(cons)
    else
        return NaN
    end
end
@inline function dKij_dθp(cons::Const, X::AbstractMatrix, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    if p == 1
        return dKij_dθp(cons, X, i, j, p, dim)
    else
        return NaN
    end
end
