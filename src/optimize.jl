@doc """
# Description
A function for optimising the GP hyperparameters based on type II maximum likelihood estimation. This function performs gradient based optimisation using the Optim pacakge to which the user is referred to for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
* `noise::Bool`: Noise hyperparameters should be optmized
* `mean::Bool`: Mean function hyperparameters should be optmized
* `kern::Bool`: Kernel function hyperparameters should be optmized
* `kwargs`: Keyword arguments for the optimize function from the Optim package

# Return:
* `::Optim.MultivariateOptimizationResults{Float64,1}`: optimization results object
""" ->
function optimize!(gp::GP; noise::Bool=true, mean::Bool=true, kern::Bool=true,
                    method=ConjugateGradient(), kwargs...)
    func = get_optim_target(gp, noise=noise, mean=mean, kern=kern)
    init = get_params(gp;  noise=noise, mean=mean, kern=kern)  # Initial hyperparameter values
    results=optimize(func,init; method=method, kwargs...)                     # Run optimizer
    set_params!(gp, Optim.minimizer(results), noise=noise,mean=mean,kern=kern)
    update_mll!(gp)
    return results
end

function get_optim_target(gp::GP; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    Kgrad_buffer = Array(Float64, gp.nobsv, gp.nobsv)
    ααinvcKI_buffer = Array(Float64, gp.nobsv, gp.nobsv)
    function mll(hyp::Vector{Float64})
        try
            set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
            update_mll!(gp)
            return -gp.mLL
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

    function mll_and_dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        try
            set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
            update_mll_and_dmll!(gp, Kgrad_buffer, ααinvcKI_buffer; noise=noise, mean=mean, kern=kern)
            grad[:] = -gp.dmLL
            return -gp.mLL
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
    function dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        mll_and_dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
    end

    func = DifferentiableFunction(mll, dmll!, mll_and_dmll!)
    return func
end
