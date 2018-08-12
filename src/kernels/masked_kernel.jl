"""
    Masked{K<:Kernel} <: Kernel

A wrapper for kernels so that they are only applied along certain dimensions.

This is similar to the `active_dims` kernel attribute in the python GPy package and to the
`covMask` function in the matlab gpml package.

The implementation is very simple: any function of the kernel that takes an `X::Matrix`
input is delegated to the wrapped kernel along with a view of `X` that only includes the
active dimensions.
"""
struct Masked{K<:Kernel} <: Kernel
    "Wrapped kernel"
    kern::K
    "A vector of dimensions on which the kernel applies"
    active_dims::Vector{Int}
end

function Statistics.cov(masked::Masked, x1::MatF64, x2::MatF64)
    return cov(masked.kern, view(x1,masked.active_dims,:), view(x2,masked.active_dims,:))
end
function Statistics.cov(masked::Masked, x1::VecF64, x2::VecF64)
    return cov(masked.kern, view(x1,masked.active_dims), view(x2,masked.active_dims))
end
function Statistics.cov(masked::Masked, X::MatF64, data::KernelData)
    return cov(masked.kern, view(X,masked.active_dims,:), data)
end
function cov!(s::MatF64, masked::Masked, X::MatF64, data::KernelData)
    return cov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function addcov!(s::MatF64, masked::Masked, X::MatF64, data::KernelData)
    return addcov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function multcov!(s::MatF64, masked::Masked, X::MatF64, data::KernelData)
    return multcov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function KernelData(masked::Masked, X::MatF64)
    return KernelData(masked.kern, view(X,masked.active_dims,:))
end
function kernel_data_key(masked::Masked, X::MatF64)
    k = kernel_data_key(masked.kern, view(X,masked.active_dims,:))
    return @sprintf("%s_active=%s", k, masked.active_dims)
end

get_params(masked::Masked) = get_params(masked.kern)
get_param_names(masked::Masked) = get_param_names(masked.kern)
num_params(masked::Masked) = num_params(masked.kern)
set_params!(masked::Masked, hyp) = set_params!(masked.kern, hyp)
function grad_slice!(dK::MatF64, masked::Masked, X::MatF64, data::KernelData, iparam::Int)
    return grad_slice!(dK, masked.kern, view(X,masked.active_dims,:), data, iparam)
end

# with EmptyData
function Statistics.cov(masked::Masked, X::MatF64, data::EmptyData)
    return cov(masked.kern, view(X,masked.active_dims,:), data)
end
function cov!(s::MatF64, masked::Masked, X::MatF64, data::EmptyData)
    return cov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function addcov!(s::MatF64, masked::Masked, X::MatF64, data::EmptyData)
    return addcov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function multcov!(s::MatF64, masked::Masked, X::MatF64, data::EmptyData)
    return multcov!(s, masked.kern, view(X,masked.active_dims,:), data)
end

function grad_slice!(dK::MatF64, masked::Masked, X::MatF64, data::EmptyData, iparam::Int)
    return grad_slice!(dK, masked.kern, view(X,masked.active_dims,:), data, iparam)
end

# priors
get_priors(masked::Masked) = get_priors(masked.kern)
set_priors!(masked::Masked, priors) = set_priors!(masked.kern, priors)

