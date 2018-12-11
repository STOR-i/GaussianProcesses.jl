#Linear ARD Covariance Function

"""
    LinArd <: Kernel

ARD linear kernel (covariance)
```math
k(x,x') = xᵀL⁻²x'
```
with length scale ``ℓ = (ℓ₁, ℓ₂, …)`` and ``L = diag(ℓ₁, ℓ₂, …)``.
"""
mutable struct LinArd{T<:Real} <: Kernel
    "Length scale"
    ℓ::Vector{T}
    "Priors for kernel parameters"
    priors::Array
end

"""
    LinArd(ll::Vector{T})

Create `LinArd` with length scale `exp.(ll)`.
"""
LinArd(ll::Vector{T}) where T = LinArd{T}(exp.(ll), [])

cov(lin::LinArd, x::AbstractVector, y::AbstractVector) = dot(x./lin.ℓ, y./lin.ℓ)

struct LinArdData{D<:AbstractArray} <: KernelData
    XtX_d::D
end

function KernelData(k::LinArd, X1::AbstractMatrix, X2::AbstractMatrix)
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
	@assert dim1==dim2
	dim = dim1
    XtX_d = Array{eltype(X2)}(undef, nobs1, nobs2, dim)
    @inbounds @simd for d in 1:dim
        for i in 1:nobs1
            for j in 1:nobs2
                XtX_d[i, j, d] = X1[d, i] * X2[d, j]
            end
        end
    end
    LinArdData(XtX_d)
end
kernel_data_key(k::LinArd, X1::AbstractMatrix, X2::AbstractMatrix) = "LinArdData"
function cov(lin::LinArd, X::AbstractMatrix)
    K = (X./lin.ℓ)' * (X./lin.ℓ)
    LinearAlgebra.copytri!(K, 'U')
    return K
end
function cov!(cK::AbstractMatrix, lin::LinArd, X::AbstractMatrix, data::LinArdData)
    dim, nobs = size(X)
    fill!(cK, 0)
    for d in 1:dim
        LinearAlgebra.axpy!(1/lin.ℓ[d]^2, view(data.XtX_d,1:nobs, 1:nobs ,d), cK)
    end
    return cK
end
function cov(lin::LinArd, X::AbstractMatrix, data::LinArdData)
    nobs = size(X,2)
    K = Array{eltype(X)}(undef, nobs, nobs)
    cov!(K, lin, X, data)
end
@inline @inbounds function cov_ij(lin::LinArd, X1::AbstractMatrix, X2::AbstractMatrix, data::LinArdData, i::Int, j::Int, dim::Int)
    ck = 0.0
    for d in 1:dim
        ck += data.XtX_d[i,j,d] * 1/lin.ℓ[d]^2
    end
    return ck
end
get_params(lin::LinArd) = log.(lin.ℓ)
get_param_names(lin::LinArd) = get_param_names(lin.ℓ, :ll)
num_params(lin::LinArd) = length(lin.ℓ)

function set_params!(lin::LinArd, hyp::AbstractVector)
    length(hyp) == num_params(lin) || throw(ArgumentError("Linear ARD kernel has $(num_params(lin)) parameters"))
    @. lin.ℓ = exp(hyp)
end

@inline dk_dll(lin::LinArd, xy::Real, d::Int) = -2 * xy / lin.ℓ[d]^2
@inline function dKij_dθp(lin::LinArd, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    if p<=dim
        return dk_dll(lin, dotijp(X,i,j,p), p)
    else
        return NaN
    end
end
@inline function dKij_dθp(lin::LinArd, X::AbstractMatrix, data::LinArdData, i::Int, j::Int, p::Int, dim::Int)
    if p <= dim
        return dk_dll(lin, data.XtX_d[i,j,p],p)
    else
        return NaN
    end
end
