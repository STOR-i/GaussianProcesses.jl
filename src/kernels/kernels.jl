# This file contains a list of the currently available covariance functions

abstract type Kernel end

"""
    KernelData

Data to be used with a kernel object to calculate a covariance matrix,
which is independent of kernel hyperparameters.

See also [`EmptyData`](@ref).
"""
abstract type KernelData end

"""
    EmptyData <: KernelData

Default empty `KernelData`.
"""
struct EmptyData <: KernelData end

KernelData(k::Kernel, X::MatF64) = EmptyData()
kernel_data_key(k::Kernel, X::MatF64) = "EmptyData"

"""
    cov(k::Kernel, X₁::Matrix{Float64}, X₂::Matrix{Float64})

Create covariance matrix from kernel `k` and matrices of observations `X₁` and `X₂`, where
each column is an observation.
"""
function Statistics.cov(k::Kernel, X₁::MatF64, X₂::MatF64)
    d(x1,x2) = cov(k, x1, x2)
    return map_column_pairs(d, X₁, X₂)
end

"""
    cov!(cK::Matrix{Float64}, k::Kernel, X₁::Matrix{Float64}, X₂::Matrix{Float64})

Like [`cov(k, X₁, X₂)`](@ref), but stores the result in `cK` rather than a new matrix.
"""
function cov!(cK::MatF64, k::Kernel, X₁::MatF64, X₂::MatF64)
    d(x1,x2) = cov(k, x1, x2)
    return map_column_pairs!(cK, d, X₁, X₂)
end

"""
    cov(k::Kernel, X::Matrix{Float64}[, data::KernelData = KernelData(k, X)])

Create covariance function from kernel `k`, matrix of observations `X`, where each column is
an observation, and kernel data `data` constructed from input observations.
"""
function Statistics.cov(k::Kernel, X::MatF64, data::EmptyData)
    d(x,y) = cov(k, x, y)
    return map_column_pairs(d, X)
end

Statistics.cov(k::Kernel, X::MatF64) = cov(k, X, KernelData(k, X))

"""
    cov!(cK::Matrix{Float64}, k::Kernel,
         X::Matrix{Float64}[, data::KernelData = KernelData(k, X)])

Like [`cov(k, X, data)`](@ref), but stores the result in `cK` rather than a new matrix.
"""
function cov!(cK::MatF64, k::Kernel, X::MatF64, data::EmptyData)
    d(x,y) = cov(k, x, y)
    return map_column_pairs!(cK, d, X)
end

cov!(cK:: MatF64, k::Kernel, X::MatF64) = cov!(cK, k, X, KernelData(k, X))
#=function cov!(cK::Matrix{Float64}, k::Kernel, X::Matrix{Float64}, data::KernelData)=#
#=    cK[:,:] = cov(k,X,data)=#
#=end=#
function addcov!(cK::MatF64, k::Kernel, X1::MatF64, X2::MatF64)
    cK .+= cov(k, X1, X2)
    return cK
end
function addcov!(cK::MatF64, k::Kernel, X::MatF64)
    cK .+= cov(k, X, KernelData(k, X))
    return cK
end
function addcov!(cK::MatF64, k::Kernel, X::MatF64, data::KernelData)
    cK .+= cov(k, X, data)
    return cK
end
function multcov!(cK::MatF64, k::Kernel, X1::MatF64, X2::MatF64)
    cK .*= cov(k, X1, X2)
    return cK
end
function multcov!(cK::MatF64, k::Kernel, X::MatF64)
    cK .*= cov(k, X, KernelData(k, X))
    return cK
end
function multcov!(cK::MatF64, k::Kernel, X::MatF64, data::KernelData)
    cK .*= cov(k, X, data)
    return cK
end

function grad_slice!(dK::MatF64, k::Kernel, X::MatF64, data::KernelData, p::Int)
    dim = size(X,1)
    nobsv = size(X,2)
    @inbounds for j in 1:nobsv
        @simd for i in 1:j
            dK[i,j] = dKij_dθp(k,X,data,i,j,p,dim)
            dK[j,i] = dK[i,j]
        end
    end
    return dK
end

function grad_slice!(dK::MatF64, k::Kernel, X::MatF64, data::EmptyData, p::Int)
    dim = size(X,1)
    nobsv = size(X,2)
    @inbounds for j in 1:nobsv
        @simd for i in 1:j
            dK[i,j] = dKij_dθp(k,X,i,j,p,dim)
            dK[j,i] = dK[i,j]
        end
    end
    return dK
end

# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack!(stack::AbstractArray, k::Kernel, X::MatF64, data::KernelData)
    npars = num_params(k)
    for p in 1:npars
        grad_slice!(view(stack,:,:,p),k,X,data,p)
    end
    return stack
end

function grad_stack!(stack::AbstractArray, k::Kernel, X::MatF64)
    grad_stack!(stack, k, X, KernelData(k, X))
end

grad_stack(k::Kernel, X::MatF64) = grad_stack(k, X, KernelData(k, X))

function grad_stack(k::Kernel, X::MatF64, data::KernelData)
    n = num_params(k)
    n_obsv = size(X, 2)
    stack = Array{Float64}( n_obsv, n_obsv, n)
    grad_stack!(stack, k, X, data)
    return stack
end


##############################
# Parameter name definitions #
##############################

# This generates names like [:ll_1, :ll_2, ...] for parameter vectors
get_param_names(n::Int, prefix::Symbol) = [Symbol(prefix, :_, i) for i in 1:n]
get_param_names(v::Vector, prefix::Symbol) = get_param_names(length(v), prefix)

# Fallback. Yields names like :Matl2Iso_param_1 => 0.5
# Ideally this is never used, because the names are uninformative.
get_param_names(obj::Union{Kernel, Mean}) =
    get_param_names(num_params(obj),
                    Symbol(typeof(obj).name.name, :_param_))

"""
    composite_param_names(objects, prefix)

Call `get_param_names` on each element of `objects` and prefix the returned name of the
element at index `i` with `prefix * i * '_'`.

# Examples
```jldoctest
julia> get_param_names(ProdKernel(Mat(1/2, 1/2, 1/2), SEArd([0.0, 1.0], 0.0)))
5-element Array{Symbol,1}:
 :pk1_ll
 :pk1_lσ
 :pk2_ll_1
 :pk2_ll_2
 :pk2_lσ
```
"""
function composite_param_names(objects, prefix)
    p = Symbol[]
    for (i, obj) in enumerate(objects)
        append!(p, [Symbol(prefix, i, :_, sym) for sym in get_param_names(obj)])
    end
    p
end

function Base.show(io::IO, k::Kernel, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(k)), Params: ")
    # params_dict = zip(get_param_names(k), get_params(k))
    # for (k, val) in params_dict
    #     print(io, "$(k)=$(val) ")
    # end
    show(io, get_params(k))
    print(io, "\n")
end

num_params(k::Kernel) = throw(MethodError(num_params, ()))

##########
# Priors #
##########

get_priors(k::Kernel) = k.priors

function set_priors!(k::Kernel, priors::Array)
    length(priors) == num_params(k) || throw(ArgumentError("$(typeof(k)) has exactly $(num_params(k)) parameters"))
    k.priors = priors
end

function prior_logpdf(k::Kernel)
    priors = get_priors(k)
    return isempty(priors) ? 0.0 : sum(logpdf(prior,param) for (prior, param) in zip(priors,get_params(k)))
end

function prior_gradlogpdf(k::Kernel)
    priors = get_priors(k)
    return isempty(priors) ? zeros(num_params(k)) : Float64[gradlogpdf(prior,param) for (prior, param) in zip(priors,get_params(k))]
end

include("stationary.jl")
include("distance.jl")
include("lin.jl")               # Linear covariance function
include("se.jl")                # Squared exponential covariance function
include("rq.jl")                # Rational quadratic covariance function
include("mat.jl")               # Matern covariance function
include("periodic.jl")          # Periodic covariance function
include("poly.jl")              # Polnomial covariance function
include("noise.jl")             # White noise covariance function
include("const.jl")             # Constant (bias) covariance function

# Composite kernels
include("composite_kernel.jl")  # Composite kernels

# Wrapped kernels
include("masked_kernel.jl")     # Masked kernels (apply to subset of X dims)
include("fixed_kernel.jl")      # Fixed kernels (fix some hyperparameters)
