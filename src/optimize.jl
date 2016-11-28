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
function optimize!(gp::GPMC; lik::Bool=false, mean::Bool=true, kern::Bool=true,
                   method=BFGS(), kwargs...)
    func = get_optim_target(gp, lik=lik, mean=mean, kern=kern)
    init = get_params(gp;  lik=lik, mean=mean, kern=kern)  # Initial hyperparameter values
    results = optimize(func,init; method=method, kwargs...)                     # Run optimizer
    set_params!(gp, results.minimum, lik=lik,mean=mean,kern=kern)
    ll!(gp)
    return results
end

function get_optim_target(gp::GPMC; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    Kgrad_buffer = Array(Float64, gp.nobsv, gp.nobsv)
    
    function ll(hyp::Vector{Float64})
        try
            set_params!(gp, hyp; lik=lik, mean=mean, kern=kern)
            ll!(gp)
            return -gp.ll
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

    function ll_and_dll!(hyp::Vector{Float64}, grad::Vector{Float64})
        try
            set_params!(gp, hyp; lik=lik, mean=mean, kern=kern)
            dll!(gp, Kgrad_buffer; lik=lik, mean=mean, kern=kern)
            grad[:] = -gp.dll
            return -gp.ll
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
    
    function dLL!(hyp::Vector{Float64}, grad::Vector{Float64})
        ll_and_dll!(hyp::Vector{Float64}, grad::Vector{Float64})
    end

    func = DifferentiableFunction(ll, dLL!, ll_and_dll!)
    return func
end
