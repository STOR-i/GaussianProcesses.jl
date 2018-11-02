"""
    Const <: Kernel

Constant kernel
```math
k(x,x') = σ²
```
with signal standard deviation ``σ``.
"""
mutable struct Const <: Kernel
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Const(lσ::Float64)

    Create `Const` with signal standard deviation `exp(lσ)`.
    """
    Const(lσ::Float64) = new(exp(2 * lσ), [])
end

function set_params!(cons::Const, hyp::AbstractVector)
    length(hyp) == 1 || throw(ArgumentError("Constant kernel only has one parameters"))
    cons.σ2 = exp(2.0*hyp[1])
end

get_params(cons::Const) = Float64[log(cons.σ2)/2.0]
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
