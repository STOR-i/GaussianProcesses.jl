#Constant mean function

"""
    MeanConst <: Mean

Constant mean function
```math
m(x) = β
```
with constant ``β``.
"""
mutable struct MeanConst <: Mean
    "Constant"
    β::Float64
    "Priors for mean parameters"
    priors::Array

    """
        MeanConst(β::Float64)

    Create `MeanConst` with constant `β`.
    """
    MeanConst(β::Float64) = new(β, [])
end

mean(mConst::MeanConst, x::AbstractVector) = mConst.β
mean(mConst::MeanConst, X::AbstractMatrix) = fill(mConst.β, size(X,2))

get_params(mConst::MeanConst) = Float64[mConst.β]
get_param_names(::MeanConst) = [:β]
num_params(mConst::MeanConst) = 1
function set_params!(mConst::MeanConst, hyp::AbstractVector)
    length(hyp) == 1 || throw(ArgumentError("Constant mean function only has 1 parameter"))
    mConst.β = hyp[1]
end
function grad_mean(mConst::MeanConst, x::AbstractVector)
    dM_theta = ones(1)
    return dM_theta
end
