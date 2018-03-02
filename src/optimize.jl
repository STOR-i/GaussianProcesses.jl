get_params_kwargs(::Type{GPE}; kwargs...) = delete!(Dict(kwargs), :lik)
get_params_kwargs{T<:Real}(::Type{GPMC{T}}; kwargs...) = delete!(Dict(kwargs), :noise)

@doc """
    # Description
    A function for optimising the GP hyperparameters based on type II maximum likelihood estimation. This function performs gradient based optimisation using the Optim pacakge to which the user is referred to for further details.

    # Arguments:
    * `gp::GPBase`: Predefined Gaussian process type
    * `mean::Bool`: Mean function hyperparameters should be optmized
    * `kern::Bool`: Kernel function hyperparameters should be optmized
    * `noise::Bool`: Observation noise hyperparameter should be optimized (GPE only)
    * `lik::Bool`: Likelihood hyperparameters should be optimized (GPMC only)
    * `kwargs`: Keyword arguments for the optimize function from the Optim package

    # Return:
    * `::Optim.MultivariateOptimizationResults{Float64,1}`: optimization results object
    """ 
function optimize!(gp::GPBase; method=LBFGS(), mean::Bool=true, kern::Bool=true, noise::Bool=true, lik::Bool=true, kwargs...)
    params_kwargs = get_params_kwargs(typeof(gp); mean=mean, kern=kern, noise=noise, lik=lik)
    # println(params_kwargs)
    func = get_optim_target(gp; params_kwargs...)
    init = get_params(gp; params_kwargs...)  # Initial hyperparameter values
    results = optimize(func, init; method=method, kwargs...)     # Run optimizer
    set_params!(gp, Optim.minimizer(results); params_kwargs...)
    update_target!(gp)
    return results
end
    
function get_optim_target(gp::GPBase; params_kwargs...)
    
    function ltarget(hyp::Vector{Float64})
        prev = get_params(gp; params_kwargs...)
        try
            set_params!(gp, hyp; params_kwargs...)
            update_target!(gp)
            return -gp.target
        catch err
            # reset parameters to remove any NaNs
            set_params!(gp, prev; params_kwargs...)
            
            if !all(isfinite.(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end        
    end

    function ltarget_and_dltarget!(grad::Vector{Float64}, hyp::Vector{Float64})
        prev = get_params(gp; params_kwargs...)
        try
            set_params!(gp, hyp; params_kwargs...)
            update_target_and_dtarget!(gp; params_kwargs...)
            grad[:] = -gp.dtarget
            return -gp.target
        catch err
            # reset parameters to remove any NaNs
            set_params!(gp, prev; params_kwargs...)
            if !all(isfinite.(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end
    end
    
    function dltarget!(grad::Vector{Float64}, hyp::Vector{Float64})
        ltarget_and_dltarget!(grad::Vector{Float64}, hyp::Vector{Float64})
    end

    xinit = get_params(gp; params_kwargs...)
    func = OnceDifferentiable(ltarget, dltarget!, ltarget_and_dltarget!, xinit)
    return func
end

