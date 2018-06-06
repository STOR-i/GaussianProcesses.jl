type FixedKern{K<:Kernel} <: Kernel
    kern::K
    free::Vector{Int} # vector of *free* parameters
end


get_params(fk::FixedKern) = get_params(fk.kern)[fk.free]
get_param_names(fk::FixedKern) = get_param_names(fk.kern)[fk.free]
function set_params!(fk::FixedKern, hyp)
    p = get_params(fk.kern)
    p[fk.free] = hyp
    set_params!(fk.kern, p)
end
num_params(fk::FixedKern) = length(fk.free)

# convenience functions to fix a parameter
function fix(k::Kernel, par::Symbol)
    npars = num_params(k)
    free = collect(1:npars)
    names = get_param_names(k)
    tofix = find(names.==par)[1]
    deleteat!(free, tofix)
    return FixedKern(k, free)
end

function fix(k::FixedKern, par::Symbol)
    free = k.free
    names = get_param_names(k)
    tofix = find(names.==par)[1]
    deleteat!(free, tofix)
    return FixedKern(k.kern, free)
end
function fix(k::Kernel)
    return FixedKern(k, Int[])
end

# convenience functions to free a parameter
function free(k::Kernel)
    return k.kern
end
function free(k::Kernel, par::Symbol)
    all_names = get_param_names(k.kern)
    ipar = find(all_names.==par)[1]
    free = sort(unique([k.free; ipar]))
    if length(free) == num_params(k.kern)
        return k.kern
    else
        return FixedKern(k.kern, free)
    end
end

function grad_slice!{M1<:MatF64,M2<:MatF64}(dK::M1, fk::FixedKern, X::M2, data::KernelData, p::Int)
    return grad_slice!(dK, fk.kern, X, data, fk.free[p])
end
function grad_slice!{M1<:MatF64,M2<:MatF64}(dK::M1, fk::FixedKern, X::M2, data::EmptyData, p::Int)
    return grad_slice!(dK, fk.kern, X, data, fk.free[p])
end
@inline function dKij_dθp(fk::FixedKern{K},X::M,i::Int,j::Int,p::Int,dim::Int) where {M<:MatF64,K}
    return dKij_dθp(fk.kern, X, i, j, fk.free[p], dim)
end
@inline function dKij_dθp(fk::FixedKern{K},X::M,data::D,i::Int,j::Int,p::Int,dim::Int) where {M<:MatF64,D<:KernelData,K}
    return dKij_dθp(fk.kern, X, data, i, j, fk.free[p], dim)
end

# delegate everything else to the wrapped kernel
# (is there a better way to do this?)
cov_ij(fk::FixedKern{K}, X::MatF64, i::Int, j::Int, dim::Int) where {K<:Kernel} = cov_ij(fk.kern, X, i, j, dim)
cov_ij(fk::FixedKern{K}, X::MatF64, data::KernelData, i::Int, j::Int, dim::Int) where {K<:Kernel} = cov_ij(fk.kern, X, data, i, j, dim)
cov_ij(fk::FixedKern{K}, X::MatF64, data::EmptyData, i::Int, j::Int, dim::Int) where {K<:Kernel} = cov_ij(fk, X, i, j, dim)
cov(fk::FixedKern{K}, x::VecF64, y::VecF64) where {K} = cov(fk.kern, x, y)
KernelData(fk::FixedKern, args...) = KernelData(fk.kern, args...)
KernelData{M<:MatF64}(fk::FixedKern, X::M) = KernelData(fk.kern, X)
kernel_data_key(fk::FixedKern, args...) = kernel_data_key(fk.kern, args...)
kernel_data_key{M<:MatF64}(fk::FixedKern, X::M) = kernel_data_key(fk.kern, X)

##########
# Priors #
##########

function get_priors(fk::FixedKern)
    free_priors = get_priors(fk.kern)
    if isempty(free_priors) return []
    else
        return free_priors[fk.free]
    end
end

function set_priors!(fk::FixedKern, priors)
    p = get_priors(fk.kern)
    p[fk.free] = priors
    set_priors!(fk.kern, p)
end

function prior_logpdf(k::FixedKern)
    return 0.0
end

function prior_gradlogpdf(k::FixedKern)
    return zeros(num_params(k))
end
