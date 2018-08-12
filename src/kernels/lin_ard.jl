#Linear ARD Covariance Function

"""
    LinArd <: Kernel

ARD linear kernel (covariance)
```math
k(x,x') = xᵀL⁻²x'
```
with length scale ``ℓ = (ℓ₁, ℓ₂, …)`` and ``L = diag(ℓ₁, ℓ₂, …)``.
"""
mutable struct LinArd <: Kernel
    "Length scale"
    ℓ::VecF64
    "Priors for kernel parameters"
    priors::Array

    """
        LinArd(ll::Vector{Float64})

    Create `LinArd` with length scale `exp.(ll)`.
    """
    LinArd(ll::VecF64) = new(exp.(ll), [])
end

function Statistics.cov(lin::LinArd, x::VecF64, y::VecF64)
    K = dot(x./lin.ℓ,y./lin.ℓ)
    return K
end

struct LinArdData <: KernelData
    XtX_d::Array{Float64,3}
end

function KernelData(k::LinArd, X::MatF64)
    dim, n = size(X)
    XtX_d = Array{Float64}(undef, n, n, dim)
    for d in 1:dim
        XtX_d[:, :, d] .= view(X,d,:) * view(X,d,:)'
        LinearAlgebra.copytri!(view(XtX_d,:,:,d), 'U')
    end
    LinArdData(XtX_d)
end
kernel_data_key(k::LinArd, X::MatF64) = "LinArdData"
function Statistics.cov(lin::LinArd, X::MatF64)
    K = (X./lin.ℓ)' * (X./lin.ℓ)
    LinearAlgebra.copytri!(K, 'U')
    return K
end
function cov!(cK::MatF64, lin::LinArd, X::MatF64, data::LinArdData)
    dim, n = size(X)
    fill!(cK, 0)
    for d in 1:dim
        LinearAlgebra.axpy!(1/lin.ℓ[d]^2, view(data.XtX_d,:,:,d), cK)
    end
    return cK
end
function Statistics.cov(lin::LinArd, X::MatF64, data::LinArdData)
    nobsv=size(X,2)
    K = zeros(Float64,nobsv,nobsv)
    cov!(K,lin,X,data)
    return K
end

get_params(lin::LinArd) = log.(lin.ℓ)
get_param_names(lin::LinArd) = get_param_names(lin.ℓ, :ll)
num_params(lin::LinArd) = length(lin.ℓ)

function set_params!(lin::LinArd, hyp::VecF64)
    length(hyp) == num_params(lin) || throw(ArgumentError("Linear ARD kernel has $(num_params(lin)) parameters"))
    lin.ℓ = exp.(hyp)
end

@inline dk_dll(lin::LinArd, xy::Float64, d::Int) = -2.0*xy/lin.ℓ[d]^2
@inline function dKij_dθp(lin::LinArd, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    if p<=dim
        return dk_dll(lin, dotijp(X,i,j,p), p)
    else
        return NaN
    end
end
@inline function dKij_dθp(lin::LinArd, X::MatF64, data::LinArdData, i::Int, j::Int, p::Int, dim::Int)
    if p<=dim
        return dk_dll(lin, data.XtX_d[i,j,p],p)
    else
        return NaN
    end
end
