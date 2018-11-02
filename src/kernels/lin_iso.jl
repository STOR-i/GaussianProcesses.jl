# Linear Isotropic Covariance Function

"""
    LinIso <: Kernel

Isotropic linear kernel (covariance)
```math
k(x, x') = xᵀx'/ℓ²
```
with length scale ``ℓ``.
"""
mutable struct LinIso <: Kernel
    "Squared length scale"
    ℓ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        LinIso(ll::Float64)

    Create `LinIso` with length scale `exp(ll)`.
    """
    LinIso(ll::Float64) = new(exp(2 * ll), [])
end

struct LinIsoData{D} <: KernelData
    XtX::D
end

function KernelData(k::LinIso, X::AbstractMatrix)
    XtX=X'*X
    LinearAlgebra.copytri!(XtX, 'U') # make sure it's symmetric
    LinIsoData(XtX)
end
kernel_data_key(k::LinIso, X::AbstractMatrix) = "LinIsoData"

_cov(lin::LinIso, xTy) = xTy ./ lin.ℓ2
function cov(lin::LinIso, x::AbstractVector, y::AbstractVector)
    K = _cov(lin, dot(x,y))
    return K
end

@inline @inbounds function cov_ij(lin::LinIso, X::AbstractMatrix, data::LinIsoData, i::Int, j::Int, dim::Int)
    return _cov(lin, data.XtX[i, j])
end
function cov(lin::LinIso, X::AbstractMatrix, data::LinIsoData)
    K = _cov(lin, data.XtX)
    return K
end
function cov!(cK::AbstractMatrix, lin::LinIso, X::AbstractMatrix, data::LinIsoData)
    iℓ2 = 1/lin.ℓ2
    @inbounds @simd for I in eachindex(cK,data.XtX)
        cK[I] = data.XtX[I]*iℓ2
    end
    return cK
end

get_params(lin::LinIso) = Float64[log(lin.ℓ2) / 2]
get_param_names(::LinIso) = [:ll]
num_params(lin::LinIso) = 1

function set_params!(lin::LinIso, hyp::AbstractVector)
    length(hyp) == 1 || throw(ArgumentError("Linear isotropic kernel only has one parameter"))
    lin.ℓ2 = exp(2 * hyp[1])
end

@inline dk_dll(lin::LinIso, xTy::Float64) = -2 * _cov(lin,xTy)
@inline function dKij_dθp(lin::LinIso, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    if p==1
        return dk_dll(lin, dotij(X,i,j,dim))
    else
        return NaN
    end
end
@inline function dKij_dθp(lin::LinIso, X::AbstractMatrix, data::LinIsoData, i::Int, j::Int, p::Int, dim::Int)
    if p==1
        return dk_dll(lin, data.XtX[i,j])
    else
        return NaN
    end
end
