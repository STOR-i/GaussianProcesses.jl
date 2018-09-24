struct FixedKernel{K<:Kernel} <: Kernel
    kernel::K
    free::Vector{Int} # vector of *free* parameters
end

get_params(k::FixedKernel) = get_params(k.kernel)[k.free]
get_param_names(k::FixedKernel) = get_param_names(k.kernel)[k.free]
function set_params!(k::FixedKernel, hyp)
    p = get_params(k.kernel)
    p[k.free] = hyp
    set_params!(k.kernel, p)
end
num_params(k::FixedKernel) = length(k.free)

# convenience functions to fix a parameter
function fix(k::Kernel, par::Symbol)
    npars = num_params(k)
    free = collect(1:npars)
    names = get_param_names(k)
    tofix = findfirst(==(par), names)
    tofix == nothing || deleteat!(free, tofix)
    return FixedKernel(k, free)
end

function fix(k::FixedKernel, par::Symbol)
    free = k.free
    names = get_param_names(k)
    tofix = findfirst(==(par), names)
    tofix == nothing || deleteat!(free, tofix)
    return FixedKernel(k.kernel, free)
end
function fix(k::Kernel)
    return FixedKernel(k, Int[])
end

# convenience functions to free a parameter
function free(k::FixedKernel)
    return k.kernel
end
function free(k::FixedKernel, par::Symbol)
    all_names = get_param_names(k.kernel)
    ipar = findfirst(==(par), all_names)
    if ipar == nothing || ipar ∉ k.free
        free = k.free
    else
        free = sort(unique([k.free; ipar]))
    end

    return FixedKernel(k.kernel, free)
end

function grad_slice!(dK::MatF64, k::FixedKernel, X::MatF64, data::KernelData, p::Int)
    return grad_slice!(dK, k.kernel, X, data, k.free[p])
end
function grad_slice!(dK::MatF64, k::FixedKernel, X::MatF64, data::EmptyData, p::Int)
    return grad_slice!(dK, k.kernel, X, data, k.free[p])
end
@inline function dKij_dθp(fk::FixedKernel,X::MatF64,i::Int,j::Int,p::Int,dim::Int)
    return dKij_dθp(fk.kernel, X, i, j, fk.free[p], dim)
end
@inline function dKij_dθp(fk::FixedKernel,X::MatF64,data::KernelData,i::Int,j::Int,p::Int,dim::Int)
    return dKij_dθp(fk.kernel, X, data, i, j, fk.free[p], dim)
end

# delegate everything else to the wrapped kernel
# (is there a better way to do this?)
cov_ij(fk::FixedKernel, X::MatF64, i::Int, j::Int, dim::Int) = cov_ij(fk.kernel, X, i, j, dim)
cov_ij(fk::FixedKernel, X::MatF64, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(fk.kernel, X, data, i, j, dim)
cov_ij(fk::FixedKernel, X::MatF64, data::EmptyData, i::Int, j::Int, dim::Int) = cov_ij(fk, X, i, j, dim)
Statistics.cov(fk::FixedKernel, x::VecF64, y::VecF64) = cov(fk.kernel, x, y)
KernelData(fk::FixedKernel, args...) = KernelData(fk.kernel, args...)
KernelData(fk::FixedKernel, X::MatF64) = KernelData(fk.kernel, X)
kernel_data_key(fk::FixedKernel, args...) = kernel_data_key(fk.kernel, args...)
kernel_data_key(fk::FixedKernel, X::MatF64) = kernel_data_key(fk.kernel, X)

##########
# Priors #
##########

function get_priors(k::FixedKernel)
    free_priors = get_priors(k.kernel)
    if isempty(free_priors) return []
    else
        return free_priors[k.free]
    end
end

function set_priors!(k::FixedKernel, priors)
    p = get_priors(k.kernel)
    p[k.free] = priors
    set_priors!(k.kernel, p)
end

function prior_logpdf(k::FixedKernel)
    return 0.0
end

function prior_gradlogpdf(k::FixedKernel)
    return zeros(num_params(k))
end
