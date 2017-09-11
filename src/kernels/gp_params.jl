import Base: start, next, done, length, iteratoreltype, iteratorsize
@compat abstract type GPParameter end

struct OptimParamIterable
    params::NamedTuples.NamedTuple
end
const ParamState = Tuple{Int,Int}
function start(piter::OptimParamIterable)
    iθ = 0
    for (iθ,θ) in enumerate(piter.params)
        if !θ.fixed
            break
        end
    end
    return ParamState((iθ, 1))
end
function next(piter::OptimParamIterable, state::ParamState)
    iθ, θp = state
    θsym = keys(piter.params)[iθ]
    θp_next = θp
    θ = piter.params[iθ]
    if θp < num_optim_params(θ)
        θp_next += 1
    else
        iθ += 1
        θp_next = 1
    end
    return ((θsym, θp), ParamState((iθ, θp_next)))
end
function done(piter::OptimParamIterable, state::ParamState)
    iθ, θp = state
    if done(piter.params, iθ)
        return true
    end
    θ = piter.params[iθ]
    if θp < num_optim_params(θ)
        return false
    end
    # look ahead: any non-fixed parameters?
    while !done(piter.params, iθ)
        θ, iθ = next(piter.params, iθ)
        if !θ.fixed
            # we're not done
            return false
        end
    end
    # run out of parameters, we're done
    return true
end
function length(piter::OptimParamIterable)
    n = 0
    for θ in piter.params
        n += num_optim_params(θ)
    end
    return n
end
iteratoreltype(piter::OptimParamIterable) = EltypeUnknown()
iteratorsize(piter::OptimParamIterable) = HasLength()

@compat abstract type UnivariateGPParameter <: GPParameter end
@inline function get_value{P<:UnivariateGPParameter}(par::P)
    return par.val
end
function show(io::IO, par::UnivariateGPParameter)
    print(io, "$(typeof(par))($(get_value(par)))")
end
@inline function get_optim_val(par::UnivariateGPParameter, i::Int)
    if i==1
        return get_optim_val(par)
    else
        throw(ArgumentError("$(typeof(par)) has exactly $(num_optim_params(par)) parameters"))
    end
end
@inline function set_value!(par::UnivariateGPParameter, i::Int, val::Float64)
    return set_value!(par, val)
    # i==1 || throw(ArgumentError("$(typeof(par)) has exactly $(num_optim_params(par)) parameters"))
end
@inline function set_optim_val!(par::UnivariateGPParameter, i::Int, x::Float64)
    return set_optim_val!(par, x)
    # i==1 || throw(ArgumentError("$(typeof(par)) has exactly $(num_optim_params(par)) parameters"))
end
@inline function num_optim_params(par::UnivariateGPParameter)
    if par.fixed
        return 0
    else
        return 1
    end
end

@compat mutable struct ContinuousGPParam <: UnivariateGPParameter
    val::Float64
    fixed::Bool
end   
function ContinuousGPParam(val::Float64)
    return ContinuousGPParam(val, false)
end
@inline function set_value!(par::ContinuousGPParam, val::Float64)
    par.val = val
end
@inline function set_optim_val!(par::ContinuousGPParam, x::Float64)
    par.val = x
end
@inline function get_optim_val(par::ContinuousGPParam)
    return par.val
end


@compat mutable struct PositiveGPParam <: UnivariateGPParameter
    val::Float64  # should be Number?
    optim_val::Float64
    fixed::Bool
end
function PositiveGPParam(val::Float64)
    val > 0 || throw(ArgumentError("Parameter value must be positive"))
    return PositiveGPParam(val, log(val), false)
end
@inline function set_value!(par::PositiveGPParam, val::Float64)
    par.val = val
    par.optim_val = log(val)
end
@inline function set_optim_val!(par::PositiveGPParam, x::Float64)
    par.val = exp(x)
    par.optim_val = x
end
@inline function get_optim_val(par::PositiveGPParam)
    return par.optim_val
end

#=
 = Vector Parameters
 =#

@compat abstract type VectorGPParam <: GPParameter end
@inline get_value{P<:VectorGPParam}(par::P, i::Int) = par.val[i]
@inline get_value{P<:VectorGPParam}(par::P) = par.val
@inline function num_optim_params(par::VectorGPParam)
    if par.fixed
        return 0
    else
        return par.dim
    end
end

@compat mutable struct ContinuousGPVector <: VectorGPParam
    dim::Int
    val::Vector{Float64}
    fixed::Bool
end
function ContinuousGPVector(val::Vector{Float64})
    return ContinuousGPVector(length(val), val, false)
end
@inline function set_value!(par::ContinuousGPVector, i::Int, val::Float64)
    par.val[i] = val
end
@inline set_value!(par::ContinuousGPVector, val::Vector{Float64}) = copy!(par.val, val)

@inline set_optim_val!(par::ContinuousGPVector, i::Int, x::Float64) = set_value!(par, i, x)
@inline get_optim_val(par::ContinuousGPVector, i::Int) = get_value(par, i)
@inline set_optim_val!(par::ContinuousGPVector, x::Vector{Float64}) = set_value!(par, x)

@compat mutable struct PositiveGPVector <: VectorGPParam
    dim::Int
    val::Vector{Float64}
    optim_val::Vector{Float64}
    fixed::Bool
end
function PositiveGPVector(val::Vector{Float64})
    all(val.>0) || throw(ArgumentError("Parameter values must be positive"))
    return PositiveGPVector(length(val), val, log.(val), false)
end
@inline function set_value!(par::PositiveGPVector, i::Int, val::Float64)
    par.val[i] = val
    par.optim_val[i] = log(val)
end
@inline function set_value!(par::PositiveGPVector, val::Vector{Float64})
    for i in 1:par.dim
        set_value!(par, i, val[i])
    end
end

@inline function set_optim_val!(par::PositiveGPVector, i::Int, x::Float64)
    par.val[i] = exp(x)
    par.optim_val[i] = x
end
@inline get_optim_val(par::PositiveGPVector, i::Int) = par.optim_val[i]
@inline function set_optim_val!(par::PositiveGPVector, x::Vector{Float64})
    # should check length(x) == par.dim, but that's slow
    for i in 1:par.dim
        set_optim_val!(par, i, x[i])
    end
end
