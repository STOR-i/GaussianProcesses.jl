# Linear mean function

"""
    MeanLin <: Mean

Linear mean function
```math
m(x) = xᵀβ
```
with linear coefficients ``β``.
"""
mutable struct MeanLin <: Mean
    "Linear coefficients"
    β::Vector{Float64}
    "Priors for mean parameters"
    priors::Array

    """
        MeanLin(β::Vector{Float64})

    Create `MeanLin` with linear coefficients `β`.
    """
    MeanLin(β::Vector{Float64}) = new(β, [])
end

Statistics.mean(mLin::MeanLin, x::VecF64) = dot(mLin.β, x)
Statistics.mean(mLin::MeanLin, X::MatF64) = X'mLin.β

get_params(mLin::MeanLin) = mLin.β
get_param_names(::MeanLin) = [:β]
num_params(mLin::MeanLin) = length(mLin.β)

function set_params!(mLin::MeanLin, hyp::VecF64)
    length(hyp) == length(mLin.β) || throw(ArgumentError("Linear mean function only has $(mLin.dim) parameters"))
    copyto!(mLin.β, hyp)
end

function grad_mean(mLin::MeanLin, x::VecF64)
    dM_theta = x
    return dM_theta
end
