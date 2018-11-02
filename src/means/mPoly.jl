# Polynomial mean function

"""
    MeanPoly <: Mean

Polynomial mean function
```math
m(x) = ∑ᵢⱼ βᵢⱼxᵢʲ
```
with polynomial coefficients ``βᵢⱼ`` of shape ``d × D`` where ``d`` is the dimension of
observations and ``D`` is the degree of the polynomial.
"""
mutable struct MeanPoly <: Mean
    "Polynomial coefficients"
    β::Matrix{Float64}
    "Priors for mean parameters"
    priors::Array

    """
        MeanPoly(β::Matrix{Float64})

    Create `MeanPoly` with polynomial coefficients `β`.
    """
    MeanPoly(β::Matrix{Float64}) = new(β, [])
end

function mean(mPoly::MeanPoly, x::AbstractVector)
    dim = length(x)
    deg = size(mPoly.β, 2)
    dim == size(mPoly.β, 1) || throw(ArgumentError("Observations and mean function have inconsistent dimensions"))
    return sum(dot(x.^j, mPoly.β[:,j]) for j in 1:deg)
end


get_params(mPoly::MeanPoly) = vec(mPoly.β)
get_param_names(mPoly::MeanPoly) = [:β]
num_params(mPoly::MeanPoly) = length(mPoly.β)

function set_params!(mPoly::MeanPoly, hyp::AbstractVector)
    length(hyp) == num_params(mPoly) || throw(ArgumentError("Polynomial mean function has $(num_param) parameters"))
    copyto!(mPoly.β, hyp)
end


function grad_mean(mPoly::MeanPoly, x::AbstractVector)
    dim = length(x)
    deg = size(mPoly.β, 2)
    dM_theta = Array{Float64}(undef, dim, deg)
    for i in 1:dim
        for j in 1:deg
            dM_theta[i,j] = x[i].^j
        end
    end
    return vec(dM_theta)
end
