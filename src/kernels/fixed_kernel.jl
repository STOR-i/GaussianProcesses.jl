struct FixedKernel{K<:Kernel, NFREE} <: Kernel
    kernel::K
    free::SVector{NFREE, Int}
end

@deprecate FixedKern FixedKernel

get_params(k::FixedKernel) = get_params(k.kernel)[k.free]
get_param_names(k::FixedKernel) = get_param_names(k.kernel)[k.free]
function set_params!(k::FixedKernel, hyp)
    p = get_params(k.kernel)
    p[k.free] = hyp
    set_params!(k.kernel, p)
end
num_params(k::FixedKernel{K,NFREE}) where {K,NFREE} = NFREE

# convenience functions to fix a parameter
function FixedKernel(k::Kernel, free::AbstractVector{<:Int})
    npars = length(free)
    sfree = SVector{npars, Int}(free)
    return FixedKernel(k, sfree)
end
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

function grad_slice!(dK::AbstractMatrix, k::FixedKernel, X::AbstractMatrix, data::KernelData, p::Int)
    return grad_slice!(dK, k.kernel, X, data, k.free[p])
end
function grad_slice!(dK::AbstractMatrix, k::FixedKernel, X::AbstractMatrix, data::EmptyData, p::Int)
    return grad_slice!(dK, k.kernel, X, data, k.free[p])
end
@inline function dKij_dθp(fk::FixedKernel,X::AbstractMatrix,i::Int,j::Int,p::Int,dim::Int)
    return dKij_dθp(fk.kernel, X, i, j, fk.free[p], dim)
end
@inline function dKij_dθp(fk::FixedKernel,X::AbstractMatrix,data::EmptyData,i::Int,j::Int,p::Int,dim::Int)
    return dKij_dθp(fk, X, i, j, p, dim)
end
@inline function dKij_dθp(fk::FixedKernel,X::AbstractMatrix,data::KernelData,i::Int,j::Int,p::Int,dim::Int)
    return dKij_dθp(fk.kernel, X, data, i, j, fk.free[p], dim)
end

# delegate everything else to the wrapped kernel
@inline cov_ij(fk::FixedKernel, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int) = cov_ij(fk.kernel, X1, X2, i, j, dim)
@inline cov_ij(fk::FixedKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(fk.kernel, X1, X2, data, i, j, dim)
@inline cov_ij(fk::FixedKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::EmptyData, i::Int, j::Int, dim::Int) = cov_ij(fk, X1, X2, i, j, dim)
cov(fk::FixedKernel, x::AbstractVector, y::AbstractVector) = cov(fk.kernel, x, y)
KernelData(fk::FixedKernel, X1::AbstractMatrix, X2::AbstractMatrix) = KernelData(fk.kernel, X1, X2)
kernel_data_key(fk::FixedKernel, X1::AbstractMatrix, X2::AbstractMatrix) = kernel_data_key(fk.kernel, X1, X2)

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
