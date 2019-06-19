"""
    Masked{K<:Kernel} <: Kernel

A wrapper for kernels so that they are only applied along certain dimensions.

This is similar to the `active_dims` kernel attribute in the python GPy package and to the
`covMask` function in the matlab gpml package.

The implementation is very simple: any function of the kernel that takes an `X::Matrix`
input is delegated to the wrapped kernel along with a view of `X` that only includes the
active dimensions.
"""
struct Masked{K<:Kernel, DIM} <: Kernel
    kernel::K
    active_dims::SVector{DIM, Int}
end
num_dims(masked::Masked{K, DIM}) where {K, DIM} = DIM
function Masked(kern::Kernel, active_dims::AbstractVector)
    dim = length(active_dims)
    K = typeof(kern)
    return Masked{K,dim}(kern, SVector{dim, Int}(active_dims))
end
xview(masked::Masked, X::AbstractMatrix) = view(X, masked.active_dims, :)

################
## KernelData ##
################
struct MaskedData{M1<:AbstractMatrix,M2<:AbstractMatrix,KD<:KernelData} <: KernelData
    X1view::M1
    X2view::M2
    wrappeddata::KD
end
function KernelData(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix)
    X1view = xview(masked,X1)
    if X1 === X2
        X2view = X1view
    else
        X2view = xview(masked,X2)
    end
	wrappeddata = KernelData(masked.kernel, X1view, X2view)
    return MaskedData(X1view, X2view, wrappeddata)
end

@inline @inbounds function cov_ij(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::MaskedData, i::Int, j::Int, dim::Int)
    return cov_ij(masked.kernel, data.X1view, data.X2view, data.wrappeddata, i, j, num_dims(masked))
end
@inline @inbounds function cov_ij(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int)
    return cov_ij(masked.kernel, xview(masked,X1), xview(masked,X2), data, i, j, num_dims(masked))
end

@inline function dKij_dθp(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::MaskedData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(masked.kernel, data.X1view, data.X2view, data.wrappeddata, i, j, p, num_dims(masked))
end
@inline function dKij_dθp(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(masked.kernel, xview(masked,X1), xview(masked,X2), data, i, j, p, num_dims(masked))
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int, npars::Int)
    X1view, X2view = xview(masked,X1), xview(masked,X2)
    return dKij_dθ!(dK, masked.kernel, X1view, X2view, data, i, j, num_dims(masked), npars)
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::MaskedData, i::Int, j::Int, dim::Int, npars::Int)
    return dKij_dθ!(dK, masked.kernel, data.X1view, data.X2view, data.wrappeddata, i, j, num_dims(masked), npars)
end

function kernel_data_key(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix)
    k = kernel_data_key(masked.kernel, xview(masked,X1), xview(masked,X2))
    return @sprintf("%s_active=%s", k, masked.active_dims)
end

get_params(masked::Masked) = get_params(masked.kernel)
get_param_names(masked::Masked) = get_param_names(masked.kernel)
num_params(masked::Masked) = num_params(masked.kernel)
set_params!(masked::Masked, hyp) = set_params!(masked.kernel, hyp)

function cov(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::MaskedData)
    return cov(masked.kernel, data.X1view, data.X2view, data.wrappeddata)
end
function cov!(s::AbstractMatrix, masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::MaskedData)
    return cov!(s, masked.kernel, data.X1view, data.X2view, data.wrappeddata)
end
function grad_slice!(dK::AbstractMatrix, masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, data::MaskedData, iparam::Int)
    return grad_slice!(dK, masked.kernel, data.X1view, data.X2view, data.wrappeddata, iparam)
end

function cov(masked::Masked, x1::AbstractVector, x2::AbstractVector)
    return cov(masked.kernel, view(x1,masked.active_dims), view(x2,masked.active_dims))
end

# priors
get_priors(masked::Masked) = get_priors(masked.kernel)
set_priors!(masked::Masked, priors) = set_priors!(masked.kernel, priors)
