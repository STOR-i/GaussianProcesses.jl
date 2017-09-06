@compat abstract type GPParameter end
@compat abstract type UnivariateGPParameter <: GPParameter end
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
@inline function set_optim_val(par::UnivariateGPParameter, x::Float64, i::Int)
    if (i==1) & !(par.fixed)
        return set_optim_val!(par, x)
    else
        throw(ArgumentError("$(typeof(par)) has exactly $(num_optim_params(par)) parameters"))
    end
end
@inline function num_optim_pars(par::UnivariateGPParameter)
    if par.fixed
        return 1
    else
        return 0
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
@inline function get_value(par::ContinuousGPParam)
    return par.val
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
    return PositiveGPParam(val, log(val), false)
end
@inline function set_value!(par::PositiveGPParam, val::Float64)
    par.val = val
    par.optim_val = log(val)
end
@inline function get_value(par::PositiveGPParam)
    return par.val
end
@inline function set_optim_val!(par::PositiveGPParam, x::Float64)
    par.val = exp(x)
    par.optim_val = x
end
@inline function get_optim_val(par::PositiveGPParam)
    return par.optim_val
end
