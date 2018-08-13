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

function KernelData(k::Poly, X::MatF64)
    XtX=X'*X
    LinearAlgebra.copytri!(XtX, 'U')
    LinIsoData(XtX)
end
kernel_data_key(k::Poly, X::MatF64) = "LinIsoData"

_cov(poly::Poly, xTy) = poly.σ2*(poly.c.+xTy).^poly.deg
function Statistics.cov(poly::Poly, x::VecF64, y::VecF64)
    K = _cov(poly, dot(x,y))
end
function cov!(cK::MatF64, poly::Poly, X::MatF64, data::LinIsoData)
    cK[:,:] = _cov(poly, data.XtX)
    return cK
end
function Statistics.cov(poly::Poly, X::MatF64, data::LinIsoData)
    K = _cov(poly, data.XtX)
    return K
end

get_params(poly::Poly) = Float64[log(poly.c), log(poly.σ2)/2.0]
get_param_names(poly::Poly) = [:lc, :lσ]
num_params(poly::Poly) = 2

function set_params!(poly::Poly, hyp::VecF64)
    length(hyp) == 2 || throw(ArgumentError("Polynomial function has two parameters"))
    poly.c = exp(hyp[1])
    poly.σ2 = exp(2.0*hyp[2])
end

@inline dk_dlc(poly::Poly, xTy::Float64) = poly.c*poly.deg*poly.σ2*(poly.c+xTy).^(poly.deg-1)
@inline dk_dlσ(poly::Poly, xTy::Float64) = 2.0*_cov(poly,xTy)
@inline function dKij_dθp(poly::Poly, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    if p==1
        return dk_dlc(poly, dotij(X,i,j,dim))
    else
        return dk_dlσ(poly, dotij(X,i,j,dim))
    end
end
@inline function dKij_dθp(poly::Poly, X::MatF64, data::LinIsoData, i::Int, j::Int, p::Int, dim::Int)
    if p==1
        return dk_dlc(poly, data.XtX[i,j])
    else
        return dk_dlσ(poly, data.XtX[i,j])
    end
end
