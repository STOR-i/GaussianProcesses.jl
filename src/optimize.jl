@doc """
    # Description
    A function for optimising the GP hyperparameters based on type II maximum likelihood estimation. This function performs gradient based optimisation using the Optim pacakge to which the user is referred to for further details.

    # Arguments:
    * `gp::GPMC`: Predefined Gaussian process type
    * `mean::Bool`: Mean function hyperparameters should be optmized
    * `kern::Bool`: Kernel function hyperparameters should be optmized
    * `kwargs`: Keyword arguments for the optimize function from the Optim package

    # Return:
    * `::Optim.MultivariateOptimizationResults{Float64,1}`: optimization results object
    """ ->
function optimize!(gp; method=LBFGS(), kwargs...)
    func = get_optim_target(gp)
    init = get_params(gp)  # Initial hyperparameter values
    results = optimize(func,init; method=method, kwargs...)                     # Run optimizer
    set_params!(gp, Optim.minimizer(results))
    update_target!(gp)
    return results
end

function get_optim_target(gp)
    
    function ltarget(hyp::Vector{Float64})
        try
            set_params!(gp, hyp)
            update_target!(gp)
            return -gp.lp
        catch err
            if !all(isfinite(hyp))
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

    function ltarget_and_dltarget!(hyp::Vector{Float64}, grad::Vector{Float64})
        try
            set_params!(gp, hyp)
            update_target_and_dtarget!(gp)
            grad[:] = -gp.dlp
            return -gp.lp
        catch err
            if !all(isfinite(hyp))
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
    
    function dltarget!(hyp::Vector{Float64}, grad::Vector{Float64})
        ltarget_and_dltarget!(hyp::Vector{Float64}, grad::Vector{Float64})
    end

    func = OnceDifferentiable(ltarget, dltarget!, ltarget_and_dltarget!)
    return func
end
