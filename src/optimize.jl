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
function optimize!(gp::GPMC; lik::Bool=true, mean::Bool=true, kern::Bool=true,
                   method=LBFGS(), kwargs...)
    func = get_optim_target(gp, lik=lik, mean=mean, kern=kern)
    init = get_params(gp;  lik=lik, mean=mean, kern=kern)  # Initial hyperparameter values
    results = optimize(func,init; method=method, kwargs...)                     # Run optimizer
    set_params!(gp, Optim.minimizer(results), lik=lik,mean=mean,kern=kern)
    update_lpost!(gp)
    return results
end

function get_optim_target(gp::GPMC; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    Kgrad_buffer = Array(Float64, gp.nobsv, gp.nobsv)
    
    function lpost(hyp::Vector{Float64})
        try
            set_params!(gp, hyp; lik=lik, mean=mean, kern=kern)
            update_lpost!(gp)
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

    function lpost_and_dlpost!(hyp::Vector{Float64}, grad::Vector{Float64})
        try
            set_params!(gp, hyp; lik=lik, mean=mean, kern=kern)
            update_lpost_and_dlpost!(gp, Kgrad_buffer; lik=lik, mean=mean, kern=kern)
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
    
    function dlpost!(hyp::Vector{Float64}, grad::Vector{Float64})
        lpost_and_dlpost!(hyp::Vector{Float64}, grad::Vector{Float64})
    end

    func = OnceDifferentiable(lpost, dlpost!, lpost_and_dlpost!)
    return func
end
