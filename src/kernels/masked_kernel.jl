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
function Masked(kern::Kernel, active_dims::AbstractVector{Int})
    dim = length(active_dims)
    K = typeof(kern)
    return Masked{K,dim}(kern, SVector{dim, Int}(active_dims))
end

@inline function cov_ij(masked::Masked, X::MatF64, i::Int, j::Int, dim::Int)
    return cov_ij(masked.kernel, @view(X[masked.active_dims, :]), i, j, num_dims(masked))
end
@inline function cov_ij(masked::Masked, X::MatF64, data::KernelData, i::Int, j::Int, dim::Int)
    return cov_ij(masked.kernel, @view(X[masked.active_dims, :]), data, i, j, num_dims(masked))
end
@inline function cov_ij(masked::Masked, X::MatF64, data::EmptyData, i::Int, j::Int, dim::Int)
    return cov_ij(masked, X, i, j, dim)
end

@inline function dKij_dθp(masked::Masked, X::MatF64, data::KernelData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(masked.kernel, @view(X[masked.active_dims, :]), data, i, j, p, num_dims(masked))
end
@inline function dKij_dθp(masked::Masked, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(masked.kernel, @view(X[masked.active_dims, :]), i, j, p, num_dims(masked))
end
@inline function dKij_dθp(masked::Masked, X::MatF64, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(masked, X, i, j, p, dim)
end
@inline @inbounds function dKij_dθ!(dK::VecF64, masked::Masked, X::MatF64, data::KernelData, i::Int, j::Int, dim::Int, npars::Int)
    Xmasked = @view(X[masked.active_dims,:])
    return dKij_dθ!(dK, masked.kernel, Xmasked, data, i, j, num_dims(masked), npars)
end

function Statistics.cov(masked::Masked, x1::MatF64, x2::MatF64)
    return cov(masked.kernel, view(x1,masked.active_dims,:), view(x2,masked.active_dims,:))
end
function Statistics.cov(masked::Masked, x1::VecF64, x2::VecF64)
    return cov(masked.kernel, view(x1,masked.active_dims), view(x2,masked.active_dims))
end
function Statistics.cov(masked::Masked, X::MatF64, data::KernelData)
    return cov(masked.kernel, view(X,masked.active_dims,:), data)
end
function cov!(
    s::MatF64, masked::Masked, X::MatF64, data::KernelData)
    return cov!(s, masked.kernel, view(X,masked.active_dims,:), data)
end
function KernelData(masked::Masked, X::MatF64)
    return KernelData(masked.kernel, view(X,masked.active_dims,:))
end
function kernel_data_key(masked::Masked, X::MatF64)
    k = kernel_data_key(masked.kernel, view(X,masked.active_dims,:))
    return @sprintf("%s_active=%s", k, masked.active_dims)
end

get_params(masked::Masked) = get_params(masked.kernel)
get_param_names(masked::Masked) = get_param_names(masked.kernel)
num_params(masked::Masked) = num_params(masked.kernel)
set_params!(masked::Masked, hyp) = set_params!(masked.kernel, hyp)
function grad_slice!(
    dK::MatF64, masked::Masked, X::MatF64, data::KernelData, iparam::Int)
    return grad_slice!(dK, masked.kernel, view(X,masked.active_dims,:), data, iparam)
end

# with EmptyData
function Statistics.cov(masked::Masked, X::MatF64, data::EmptyData)
    return cov(masked.kernel, view(X,masked.active_dims,:), data)
end
function cov!(s::MatF64, masked::Masked, X::MatF64, data::EmptyData)
    return cov!(s, masked.kernel, view(X,masked.active_dims,:), data)
end
function grad_slice!(dK::MatF64, masked::Masked, X::MatF64, data::EmptyData, iparam::Int)
    return grad_slice!(dK, masked.kernel, view(X,masked.active_dims,:), data, iparam)
end

# priors
get_priors(masked::Masked) = get_priors(masked.kernel)
set_priors!(masked::Masked, priors) = set_priors!(masked.kernel, priors)
