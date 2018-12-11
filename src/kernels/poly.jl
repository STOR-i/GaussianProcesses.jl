# Polynomial covariance function

"""
    Poly <: Kernel

Polynomial kernel (covariance)
```math
k(x,x') = σ²(xᵀx' + c)ᵈ
```
with signal standard deviation ``σ``, additive constant ``c``, and degree ``d``.
"""
mutable struct Poly{T<:Real} <: Kernel
    "Constant"
    c::T
    "Signal variance"
    σ2::T
    "Degree of polynomial"
    deg::Int
    "Priors for kernel parameters"
    priors::Array
end

"""
    Poly(lc::Real, lσ::Real, deg::Int)

Create `Poly` with signal standard deviation `exp(lσ)`, additive constant `exp(lc)`,
and degree `deg`.
"""
Poly(lc::T, lσ::T, deg::Int) where T = Poly{T}(exp(lc), exp(2 * lσ), deg, [])

function KernelData(k::Poly, X1::AbstractMatrix, X2::AbstractMatrix)
    XtX=X1'*X2
	if X1==X2
    	LinearAlgebra.copytri!(XtX, 'U')
	end
    LinIsoData(XtX)
end
kernel_data_key(k::Poly, X1::AbstractMatrix, X2::AbstractMatrix) = "LinIsoData"

_cov(poly::Poly, xTy) = poly.σ2*(poly.c.+xTy).^poly.deg
function cov(poly::Poly, x::AbstractVector, y::AbstractVector)
    K = _cov(poly, dot(x,y))
end
function cov!(cK::AbstractMatrix, poly::Poly, X::AbstractMatrix, data::LinIsoData)
    cK .= _cov(poly, data.XtX)
end
@inline @inbounds function cov_ij(poly::Poly, X1::AbstractMatrix, X2::AbstractMatrix, data::LinIsoData, i::Int, j::Int, dim::Int)
    return _cov(poly, data.XtX[i, j])
end
cov(poly::Poly, X::AbstractMatrix, data::LinIsoData) = _cov(poly, data.XtX)

get_params(poly::Poly) = [log(poly.c), log(poly.σ2) / 2]
get_param_names(poly::Poly) = [:lc, :lσ]
num_params(poly::Poly) = 2

function set_params!(poly::Poly, hyp::AbstractVector)
    length(hyp) == 2 || throw(ArgumentError("Polynomial function has two parameters"))
    poly.c = exp(hyp[1])
    poly.σ2 = exp(2 * hyp[2])
end

@inline dk_dlc(poly::Poly, xTy::Real) = poly.c*poly.deg*poly.σ2*(poly.c+xTy).^(poly.deg-1)
@inline dk_dlσ(poly::Poly, xTy::Real) = 2 * _cov(poly,xTy)
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
