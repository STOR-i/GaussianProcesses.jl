""" 
# Description
A `Masked` kernel type is a wrapper for kernels so
that they are only applied along certain dimensions of `X`.
This is similar to the `active_dims` kernel attribute in the python GPy 
package and to the `covMask` function in the matlab gpml package.

The implementation is very simple: any function of the kernel
that takes an `X::Matrix` input is delegated to the wrapped 
kernel along with a view of the X matrix that only includes
the active dimensions.

# Arguments
* `kern::Kernel`: The wrapper kernel
* `active_dims::Vector{Int}`: A vector of dimensions on which the kernel applies
"""
type Masked{K<:Kernel} <: Kernel
    kern::K
    active_dims::Vector{Int}
end

function cov{K<:Kernel,M1<:MatF64,M2<:MatF64}(masked::Masked{K}, x1::M1, x2::M2)
    return cov(masked.kern, view(x1,masked.active_dims,:), view(x2,masked.active_dims,:))
end
function cov{K<:Kernel,V1<:VecF64,V2<:VecF64}(masked::Masked{K}, x1::V1, x2::V2)
    return cov(masked.kern, view(x1,masked.active_dims), view(x2,masked.active_dims))
end
function cov{K<:Kernel,M<:MatF64}(masked::Masked{K}, X::M, data::KernelData)
    return cov(masked.kern, view(X,masked.active_dims,:), data)
end
function cov!{K<:Kernel,M1<:MatF64, M2<:MatF64}(
    s::M1, masked::Masked{K}, X::M2, data::KernelData)
    return cov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function addcov!{K<:Kernel,M1<:MatF64, M2<:MatF64}(
    s::M1, masked::Masked{K}, X::M2, data::KernelData)
    return addcov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function multcov!{K<:Kernel,M1<:MatF64, M2<:MatF64}(
    s::M1, masked::Masked{K}, X::M2, data::KernelData)
    return multcov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function KernelData{K<:Kernel,M<:MatF64}(masked::Masked{K}, X::M)
    return KernelData(masked.kern, view(X,masked.active_dims,:))
end
function kernel_data_key{K<:Kernel,M<:MatF64}(masked::Masked{K}, X::M)
    k = kernel_data_key(masked.kern, view(X,masked.active_dims,:))
    return Symbol(k, "_active=", masked.active_dims)
end


get_params{K<:Kernel}(masked::Masked{K}) = get_params(masked.kern)
get_param_names{K<:Kernel}(masked::Masked{K}) = get_param_names(masked.kern)
num_params{K<:Kernel}(masked::Masked{K}) = num_params(masked.kern)
set_params!{K<:Kernel}(masked::Masked{K}, hyp) = set_params!(masked.kern, hyp)
function grad_slice!{K<:Kernel, M1<:MatF64, M2<:MatF64}(
    dK::M1, masked::Masked{K}, X::M2, data::KernelData, iparam::Int)
    return grad_slice!(dK, masked.kern, view(X,masked.active_dims,:), data, iparam)
end

# with EmptyData
function cov{K<:Kernel,M<:MatF64}(masked::Masked{K}, X::M, data::EmptyData)
    return cov(masked.kern, view(X,masked.active_dims,:), data)
end
function cov!{K<:Kernel,M1<:MatF64, M2<:MatF64}(
    s::M1, masked::Masked{K}, X::M2, data::EmptyData)
    return cov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function addcov!{K<:Kernel,M1<:MatF64, M2<:MatF64}(
    s::M1, masked::Masked{K}, X::M2, data::EmptyData)
    return addcov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function multcov!{K<:Kernel,M1<:MatF64, M2<:MatF64}(
    s::M1, masked::Masked{K}, X::M2, data::EmptyData)
    return multcov!(s, masked.kern, view(X,masked.active_dims,:), data)
end
function grad_slice!{K<:Kernel, M1<:MatF64, M2<:MatF64}(
    dK::M1, masked::Masked{K}, X::M2, data::EmptyData, iparam::Int)
    return grad_slice!(dK, masked.kern, view(X,masked.active_dims,:), data, iparam)
end
