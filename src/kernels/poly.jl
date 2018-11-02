# Polynomial covariance function

"""
    Poly <: Kernel

Polynomial kernel (covariance)
```math
k(x,x') = σ²(xᵀx' + c)ᵈ
```
with signal standard deviation ``σ``, additive constant ``c``, and degree ``d``.
"""
mutable struct Poly <: Kernel
    "Constant"
    c::Float64
    "Signal variance"
    σ2::Float64
    "Degree of polynomial"
    deg::Int
    "Priors for kernel parameters"
    priors::Array

    """
        Poly(lc::Float64, lσ::Float64, deg::Int)

    Create `Poly` with signal standard deviation `exp(lσ)`, additive constant `exp(lc)`,
    and degree `deg`.
    """
    Poly(lc::Float64, lσ::Float64, deg::Int) = new(exp(lc), exp(2 * lσ), deg, [])
end

function KernelData(k::Poly, X::AbstractMatrix)
    XtX=X'*X
    LinearAlgebra.copytri!(XtX, 'U')
    LinIsoData(XtX)
end
kernel_data_key(k::Poly, X::AbstractMatrix) = "LinIsoData"

_cov(poly::Poly, xTy) = poly.σ2*(poly.c.+xTy).^poly.deg
function Statistics.cov(poly::Poly, x::AbstractVector, y::AbstractVector)
    K = _cov(poly, dot(x,y))
end
function cov!(cK::AbstractMatrix, poly::Poly, X::AbstractMatrix, data::LinIsoData)
    cK .= _cov(poly, data.XtX)
end
@inline @inbounds function cov_ij(poly::Poly, X::AbstractMatrix, data::LinIsoData, i::Int, j::Int, dim::Int)
    return _cov(poly, data.XtX[i, j])
end
Statistics.cov(poly::Poly, X::AbstractMatrix, data::LinIsoData) = _cov(poly, data.XtX)

get_params(poly::Poly) = Float64[log(poly.c), log(poly.σ2) / 2]
get_param_names(poly::Poly) = [:lc, :lσ]
num_params(poly::Poly) = 2

function set_params!(poly::Poly, hyp::AbstractVector)
    length(hyp) == 2 || throw(ArgumentError("Polynomial function has two parameters"))
    poly.c = exp(hyp[1])
    poly.σ2 = exp(2 * hyp[2])
end

@inline dk_dlc(poly::Poly, xTy::Float64) = poly.c*poly.deg*poly.σ2*(poly.c+xTy).^(poly.deg-1)
@inline dk_dlσ(poly::Poly, xTy::Float64) = 2 * _cov(poly,xTy)
@inline function dKij_dθp(poly::Poly, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    if p==1
        return dk_dlc(poly, dotij(X,i,j,dim))
    else
        return dk_dlσ(poly, dotij(X,i,j,dim))
    end
end
@inline function dKij_dθp(poly::Poly, X::AbstractMatrix, data::LinIsoData, i::Int, j::Int, p::Int, dim::Int)
    if p==1
        return dk_dlc(poly, data.XtX[i,j])
    else
        return dk_dlσ(poly, data.XtX[i,j])
    end
end
