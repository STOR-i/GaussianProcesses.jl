struct FixedKern <: Kernel
    kern::Kernel
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
    tofix = findfirst(==(par), names)
    tofix == nothing || deleteat!(free, tofix)
    return FixedKern(k, free)
end

function fix(k::FixedKern, par::Symbol)
    free = k.free
    names = get_param_names(k)
    tofix = findfirst(==(par), names)
    tofix == nothing || deleteat!(free, tofix)
    return FixedKern(k.kern, free)
end
function fix(k::Kernel)
    return FixedKern(k, Int[])
end

# convenience functions to free a parameter
function free(k::FixedKern)
    return k.kern
end
function free(k::FixedKern, par::Symbol)
    all_names = get_param_names(k.kern)
    ipar = findfirst(==(par), all_names)
    if ipar == nothing || ipar ∉ k.free
        free = k.free
    else
        free = sort(unique([k.free; ipar]))
    end
    if length(free) == num_params(k.kern)
        return k.kern
    else
        return FixedKern(k.kern, free)
    end
end

function grad_slice!(dK::MatF64, fk::FixedKern, X::MatF64, data::KernelData, p::Int)
    return grad_slice!(dK, fk.kern, X, data, fk.free[p])
end
function grad_slice!(dK::MatF64, fk::FixedKern, X::MatF64, data::EmptyData, p::Int)
    return grad_slice!(dK, fk.kern, X, data, fk.free[p])
end
@inline function dKij_dθp(fk::FixedKern,X::MatF64,i::Int,j::Int,p::Int,dim::Int)
    return dKij_dθp(fk.kern, X, i, j, fk.free[p], dim)
end
@inline function dKij_dθp(fk::FixedKern,X::MatF64,data::KernelData,i::Int,j::Int,p::Int,dim::Int)
    return dKij_dθp(fk.kern, X, data, i, j, fk.free[p], dim)
end

# delegate everything else to the wrapped kernel
# (is there a better way to do this?)
Statistics.cov(fk::FixedKern, args...) = cov(fk.kern, args...)
Statistics.cov(fk::FixedKern, X₁::MatF64, X₂::MatF64) = cov(fk.kern, X₁, X₂)
Statistics.cov(fk::FixedKern, X::MatF64) = cov(fk.kern, X)
Statistics.cov(fk::FixedKern, X::MatF64, data::EmptyData) = cov(ck.kern,X,data)
KernelData(fk::FixedKern, args...) = KernelData(fk.kern, args...)
KernelData(fk::FixedKern, X::MatF64) = KernelData(fk.kern, X)
kernel_data_key(fk::FixedKern, args...) = kernel_data_key(fk.kern, args...)
kernel_data_key(fk::FixedKern, X::MatF64) = kernel_data_key(fk.kern, X)
cov!(cK::MatF64, fk::FixedKern, args...) = cov!(cK, fk.kern, args...)
cov!(cK::MatF64, fk::FixedKern, X₁::MatF64, X₂::MatF64) = cov!(cK, fk.kern, k, X₁, X₂)
cov!(cK::MatF64, fk::FixedKern, X::MatF64, data::EmptyData)=cov!(cK, fk.kern,X,data)
cov!(cK::MatF64,fk::FixedKern,X::MatF64) = cov!(cK, fk.kern, X)
addcov!(cK::MatF64, fk::FixedKern, X::MatF64) = addcov!(cK, fk.kern, X)
addcov!(cK::MatF64, fk::FixedKern, X1::MatF64, X2::MatF64) = addcov!(cK, fk.kern, X1, X2)
multcov!(cK::MatF64, fk::FixedKern, X::MatF64) = multcov!(cK, fk.kern,X)
multcov!(cK::MatF64, fk::FixedKern, X1::MatF64, X2::MatF64) = multcov!(cK, fk.kern,X1, X2)
function multcov!(cK::MatF64, fk::FixedKern, X::MatF64, data::KernelData)
    return multcov!(cK, fk.kern, X, data)
end
function addcov!(cK::MatF64, fk::FixedKern, X::MatF64, data::KernelData)
    return addcov!(cK, fk.kern, X, data)
end

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
